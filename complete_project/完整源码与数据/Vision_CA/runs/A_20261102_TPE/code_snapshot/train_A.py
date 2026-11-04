"""A协议训练入口：固定名单、TPE分类头搜索、微调、手检测与最终评价。

本程序不调用数据划分函数。训练只读取train，选型只读取val；test仅在evaluate阶段读取。
所有截止时间均使用本机带时区时间。达到截止时间时保存当前状态，不把半轮称为完整训练轮。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF

LABELS = {"peace": 1, "peace_inverted": 1, "stop": 2,
          "stop_inverted": 2, "palm": 2, "fist": 3, "like": 4, "dislike": 5}
CLASS_NAMES = ["other", "peace", "palm", "fist", "like", "dislike"]
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
SPLITS = ("train", "val", "test")


def seed_all(seed=42):
    # 固定数据顺序和随机数；GPU卷积允许选择高效算法，但不承诺跨硬件逐位相同。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_json(path, obj):
    # 先写临时文件，再替换，避免进程中断留下半个JSON。
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch(path, obj):
    # checkpoint同样采用完整写入后替换，不删除已有最优权重。
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def deadline_reached(deadline, reserve=0):
    return time.time() + reserve >= datetime.fromisoformat(deadline).timestamp() if deadline else False


def log_event(out, **fields):
    # 日志按行追加，保留失败、部分轮和完整轮的真实状态。
    fields = {"time": datetime.now().astimezone().isoformat(), **fields}
    print(json.dumps(fields, ensure_ascii=False), flush=True)
    with (Path(out) / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(fields, ensure_ascii=False) + "\n")


def load_split(split_root, split):
    # 每张照片在固定manifest中只出现一次，框和标签从对应原类别JSON读取。
    folder = Path(split_root) / split
    with (folder / "manifest.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    annotations = {p.stem: read_json(p) for p in (folder / "annotations").glob("*.json")}
    result = []
    for row in rows:
        annotation = annotations[row["class_name"]][row["image_id"]]
        boxes = annotation["bboxes"]
        raw_labels = annotation["labels"]
        if len(boxes) != len(raw_labels):
            raise ValueError(f"框与标签不一致：{row['image_id']}")
        result.append({**row, "path": str(folder / row["image_path"]),
                       "bboxes": boxes, "raw_labels": raw_labels,
                       "labels": [LABELS.get(x, 0) for x in raw_labels]})
    return result


def box_xyxy(box, width, height):
    # HaGRID原框为相对坐标x、y、宽、高，先转换到原照片像素。
    x, y, w, h = box
    return [max(0.0, x * width), max(0.0, y * height),
            min(float(width), (x + w) * width), min(float(height), (y + h) * height)]


def crop_hand(image, xyxy, margin=0.10):
    # 对长边留10%边缘，补成正方形；训练与预测均用同一裁图函数。
    x1, y1, x2, y2 = xyxy
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    if side <= 0:
        raise ValueError("手框没有有效面积")
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rect = (math.floor(cx - side / 2), math.floor(cy - side / 2),
            math.ceil(cx + side / 2), math.ceil(cy + side / 2))
    # PIL越界区域补黑，避免在图边缘把手图拉伸成不同比例。
    return image.crop(rect).resize((224, 224), Image.Resampling.BILINEAR), rect


def prepare_crops(args):
    # 只生成派生裁图和索引，绝不移动或覆盖A的原始三组照片。
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audit = {"split_root": str(Path(args.split_root).resolve()), "margin": 0.10,
             "label_mapping": LABELS, "default_other": 0, "splits": {}}
    previous_ids = set()
    for split in SPLITS:
        records = load_split(args.split_root, split)
        ids = {r["image_id"] for r in records}
        if ids & previous_ids:
            raise ValueError("同一原图跨组，停止准备")
        previous_ids |= ids
        dest = out / "crops" / split
        dest.mkdir(parents=True, exist_ok=True)

        def one(record):
            with Image.open(record["path"]) as source:
                image = source.convert("RGB")
            rows = []
            for index, (bbox, raw, label) in enumerate(zip(record["bboxes"], record["raw_labels"], record["labels"])):
                xyxy = box_xyxy(bbox, *image.size)
                cropped, rect = crop_hand(image, xyxy)
                name = f"{record['image_id']}__hand{index}.jpg"
                target = dest / name
                if not target.exists():
                    cropped.save(target, quality=95, subsampling=0)
                rows.append({"image_id": record["image_id"], "hand_index": index,
                             "path": str(target), "source_path": record["path"],
                             "raw_label": raw, "label": label, "user_id": record["user_id"],
                             "bbox_xyxy": xyxy, "crop_rectangle": rect})
            return rows

        with ThreadPoolExecutor(max_workers=args.workers or 8) as pool:
            crop_rows = [item for group in pool.map(one, records) for item in group]
        atomic_json(out / f"crops_{split}.json", crop_rows)
        audit["splits"][split] = {"photos": len(records), "hand_crops": len(crop_rows),
                                  "class_counts": dict(Counter(r["label"] for r in crop_rows)),
                                  "manifest_sha256": hashlib.sha256((Path(args.split_root) / split / "manifest.csv").read_bytes()).hexdigest()}
        log_event(out, stage="prepare", split=split, **audit["splits"][split])
    atomic_json(out / "data_audit.json", audit)


class CropDataset(Dataset):
    """已固定的手框裁图；随机增强仅允许用于训练。"""
    def __init__(self, root, split, augment=False):
        self.rows = read_json(Path(root) / f"crops_{split}.json")
        self.transform = transforms.Compose([
            *([transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
               transforms.ColorJitter(0.2, 0.2, 0.15, 0.03)] if augment else []),
            transforms.ToTensor(), NORMALIZE])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as source:
            image = source.convert("RGB")
        return self.transform(image), row["label"], index


class HandDataset(Dataset):
    """完整照片和手框，只训练手/背景检测，不把原手势当检测类别。"""
    def __init__(self, split_root, split, augment=False):
        self.rows = load_split(split_root, split)
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as source:
            image = source.convert("RGB")
        w, h = image.size
        boxes = torch.tensor([box_xyxy(b, w, h) for b in row["bboxes"]], dtype=torch.float32).reshape(-1, 4)
        if self.augment and random.random() < 0.5:
            image = ImageOps.mirror(image)
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        target = {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64),
                  "image_id": torch.tensor(index)}
        return TF.to_tensor(image), target


def detection_collate(batch):
    return tuple(zip(*batch))


def make_loader(dataset, batch_size, workers, shuffle=False, detection=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      collate_fn=detection_collate if detection else None)


def resnet(pretrained=None, dropout=0.0, fine_tune=False):
    # 显式加载本地ImageNet参数，最后替换为512到6的分类层。
    model = models.resnet18(weights=None)
    if pretrained:
        model.load_state_dict(torch.load(pretrained, map_location="cpu", weights_only=True))
    for parameter in model.parameters():
        parameter.requires_grad = False
    if fine_tune:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 6))
    return model


def confusion_metrics(targets, predictions):
    # 固定六类分母，缺失类别F1为0；不删除难例或错误样本。
    cm = np.zeros((6, 6), dtype=np.int64)
    for truth, prediction in zip(targets, predictions):
        cm[int(truth), int(prediction)] += 1
    tp = cm.diagonal().astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(6), where=cm.sum(0) != 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(6), where=cm.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(6), where=precision + recall != 0)
    return {"n": int(cm.sum()), "accuracy": float(tp.sum() / max(1, cm.sum())),
            "macro_f1": float(f1.mean()), "confusion_matrix": cm.tolist(),
            "precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist(),
            "support": cm.sum(1).tolist()}


def feature_cache(args):
    # TPE只用train/val特征；最终test不在搜索阶段读取。
    device = torch.device(args.device)
    net = resnet(args.pretrained).to(device).eval()
    net.fc = nn.Identity()
    for split in ("train", "val"):
        loader = make_loader(CropDataset(args.out, split), args.batch_size, args.workers)
        features, labels = [], []
        started = time.time()
        with torch.inference_mode():
            for batch, target, _ in loader:
                if deadline_reached(args.deadline, 30):
                    raise TimeoutError("特征提取未完整，未保存不完整缓存")
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    values = net(batch.to(device, non_blocking=True))
                features.append(values.float().cpu())
                labels.append(target)
        atomic_torch(Path(args.out) / f"features_{split}.pt",
                     {"x": torch.cat(features), "y": torch.cat(labels),
                      "pretrained_sha256": hashlib.sha256(Path(args.pretrained).read_bytes()).hexdigest()})
        log_event(args.out, stage="features", split=split, n=len(loader.dataset), seconds=time.time() - started)


def optimizer_for(parameters, name, lr, wd):
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=wd)


def search(args):
    """在冻结ImageNet特征上用真实TPE搜索分类头，不声称搜索了整个网络。"""
    import optuna
    out = Path(args.out)
    device = torch.device(args.device)
    train = torch.load(out / "features_train.pt", map_location="cpu", weights_only=True)
    val = torch.load(out / "features_val.pt", map_location="cpu", weights_only=True)
    xtrain, ytrain = train["x"].to(device), train["y"].to(device)
    xval, yval = val["x"].to(device), val["y"].to(device)
    counts = torch.bincount(ytrain, minlength=6).float()
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=4)
    study = optuna.create_study(study_name="A_resnet18_frozen_features_TPE", direction="maximize",
                               sampler=sampler, storage="sqlite:///" + str((out / "tpe.sqlite3").resolve()).replace("\\", "/"),
                               load_if_exists=True)

    def objective(trial):
        if deadline_reached(args.deadline, 60):
            raise optuna.TrialPruned("搜索截止时间已到")
        seed_all(42)
        name = trial.suggest_categorical("optimizer", ["adamw", "sgd"])
        lr = trial.suggest_float("lr", 1e-4, 3e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        smooth = trial.suggest_float("label_smoothing", 0.0, 0.12)
        weighted = trial.suggest_categorical("class_weight", [False, True])
        head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 6)).to(device)
        weights = (counts.sum() / (6 * counts)).sqrt() if weighted else None
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=smooth)
        optimizer = optimizer_for(head.parameters(), name, lr, wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        best = -1.0
        for epoch in range(args.epochs):
            head.train()
            permutation = torch.randperm(len(ytrain), device=device)
            for indices in permutation.split(max(args.batch_size, 128)):
                if deadline_reached(args.deadline, 30):
                    raise optuna.TrialPruned("到达预算，保留已完成trial")
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(head(xtrain[indices]), ytrain[indices])
                loss.backward()
                optimizer.step()
            head.eval()
            with torch.inference_mode():
                validation_predictions = []
                for validation_batch in xval.split(1024):
                    if deadline_reached(args.deadline, 10):
                        raise optuna.TrialPruned("验证未完成，不返回部分验证分数")
                    validation_predictions.append(head(validation_batch).argmax(1).cpu())
                prediction = torch.cat(validation_predictions)
            metrics = confusion_metrics(yval.cpu().numpy(), prediction.numpy())
            score = metrics["macro_f1"]
            trial.report(score, epoch)
            scheduler.step()
            if score > best:
                best = score
                atomic_torch(out / "trials" / f"trial_{trial.number}_best.pt",
                             {"head": head.state_dict(), "params": trial.params, "epoch": epoch + 1,
                              "validation": metrics, "trial": trial.number})
            log_event(out, stage="tpe", trial=trial.number, epoch=epoch + 1,
                      validation_macro_f1=score, validation_accuracy=metrics["accuracy"])
        return best

    study.optimize(objective, n_trials=args.trials, timeout=args.timeout or None,
                   catch=(FloatingPointError,), gc_after_trial=True)
    finished = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not finished:
        raise RuntimeError("没有完成的TPE trial，不能生成最优模型")
    best = study.best_trial
    atomic_json(out / "tpe_summary.json", {"objective": "validation_macro_f1", "scope": "frozen_ImageNet_features_classifier_head",
                "trials": [{"number": t.number, "state": t.state.name, "value": t.value, "params": t.params} for t in study.trials],
                "best_trial": best.number, "best_params": best.params, "best_value": best.value,
                "test_accessed": False})
    checkpoint = torch.load(out / "trials" / f"trial_{best.number}_best.pt", map_location="cpu", weights_only=True)
    model = resnet(args.pretrained, best.params["dropout"])
    model.fc.load_state_dict(checkpoint["head"])
    atomic_torch(out / "classifier_best.pt", {"model": model.state_dict(), "params": best.params,
                 "stage": "frozen_head_TPE", "validation": checkpoint["validation"], "epoch": checkpoint["epoch"]})


def evaluate_classifier(model, loader, device, deadline=None):
    model.eval()
    targets, predictions, probabilities, indices = [], [], [], []
    with torch.inference_mode():
        for batch, label, idx in loader:
            if deadline_reached(deadline, 10):
                raise TimeoutError("分类评价未完成：达到截止时间，不返回部分指标供选型")
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                scores = model(batch.to(device, non_blocking=True))
            prob = scores.float().softmax(1).cpu()
            targets.extend(label.tolist())
            predictions.extend(prob.argmax(1).tolist())
            probabilities.extend(prob.tolist())
            indices.extend(idx.tolist())
    return confusion_metrics(targets, predictions), probabilities, indices


def fine_tune(args):
    # 只解冻最后一组残差块与分类头，前面层固定并保持eval，避免BN统计偷偷变化。
    out = Path(args.out)
    checkpoint = torch.load(out / "classifier_best.pt", map_location="cpu", weights_only=True)
    params = checkpoint["params"]
    device = torch.device(args.device)
    model = resnet(None, params["dropout"], fine_tune=True).to(device)
    model.load_state_dict(checkpoint["model"])
    train = CropDataset(out, "train", augment=True)
    val = CropDataset(out, "val")
    train_loader = make_loader(train, args.batch_size, args.workers, shuffle=True)
    val_loader = make_loader(val, args.batch_size, args.workers)
    counts = torch.bincount(torch.tensor([r["label"] for r in train.rows]), minlength=6).float().to(device)
    weights = (counts.sum() / (6 * counts)).sqrt() if params["class_weight"] else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=params["label_smoothing"])
    optimizer = torch.optim.AdamW([
        {"params": model.layer4.parameters(), "lr": args.lr},
        {"params": model.fc.parameters(), "lr": args.lr * 5}], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = checkpoint["validation"]["macro_f1"]
    bad_epochs = 0
    for epoch in range(args.epochs):
        if deadline_reached(args.deadline, 120):
            log_event(out, stage="finetune", status="deadline_before_epoch", epoch=epoch + 1)
            break
        model.eval()
        model.layer4.train()
        model.fc.train()
        total_loss, samples = 0.0, 0
        started = time.time()
        complete = True
        for batch, target, _ in train_loader:
            if deadline_reached(args.deadline, 90):
                complete = False
                break
            batch, target = batch.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(batch), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(target)
            samples += len(target)
        if not complete:
            atomic_torch(out / "classifier_partial.pt", {"model": model.state_dict(), "params": params,
                         "stage": "partial_finetune", "epoch": epoch + 1, "samples": samples})
            log_event(out, stage="finetune", status="partial_epoch_not_selected", epoch=epoch + 1, samples=samples)
            break
        metrics, _, _ = evaluate_classifier(model, val_loader, device, args.deadline)
        scheduler.step()
        improved = metrics["macro_f1"] > best
        if improved:
            best = metrics["macro_f1"]
            atomic_torch(out / "classifier_best.pt", {"model": model.state_dict(), "params": params,
                         "stage": "layer4_finetune", "validation": metrics, "epoch": epoch + 1,
                         "fine_tune_lr": args.lr, "optimizer": "AdamW", "train_samples": samples})
            bad_epochs = 0
        else:
            bad_epochs += 1
        log_event(out, stage="finetune", status="complete_epoch", epoch=epoch + 1,
                  samples=samples, loss=total_loss / samples, validation=metrics,
                  seconds=time.time() - started, best=best)
        if bad_epochs >= args.patience:
            break


def detector_model(backbone=True):
    # 类别0为背景、1为手；通用COCO权重不会作为现成手检测器。
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if backbone else None
    return models.detection.ssdlite320_mobilenet_v3_large(weights=None,
             weights_backbone=weights, num_classes=2, trainable_backbone_layers=3)


def detection_metrics(predictions, targets, threshold=0.5, iou_threshold=0.5):
    from torchvision.ops import box_iou
    # 此处是指定分数阈值与IoU的召回/精确率，不冒充COCO AP。
    tp = fp = fn = 0
    for pred, target in zip(predictions, targets):
        keep = pred["scores"] >= threshold
        boxes = pred["boxes"][keep]
        gt = target["boxes"]
        used = set()
        for box in boxes:
            if len(gt):
                values = box_iou(box[None], gt)[0]
                order = values.argsort(descending=True).tolist()
                match = next((j for j in order if j not in used and values[j] >= iou_threshold), None)
            else:
                match = None
            if match is not None:
                used.add(match)
                tp += 1
            else:
                fp += 1
        fn += len(gt) - len(used)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn), "score_threshold": threshold, "iou_threshold": iou_threshold}


def validate_detector(model, loader, device, deadline=None):
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for images, truth in loader:
            if deadline_reached(deadline, 10):
                raise TimeoutError("检测验证未完成：达到截止时间，不返回部分指标供选型")
            result = model([im.to(device, non_blocking=True) for im in images])
            predictions.extend([{k: v.cpu() for k, v in p.items()} for p in result])
            targets.extend(truth)
    metrics = detection_metrics(predictions, targets)
    metrics["photos"] = len(targets)
    return metrics


def train_detector(args):
    out = Path(args.out)
    device = torch.device(args.device)
    model = detector_model().to(device)
    loader = make_loader(HandDataset(args.split_root, "train", True), args.batch_size,
                         args.workers, shuffle=True, detection=True)
    val = make_loader(HandDataset(args.split_root, "val"), args.batch_size,
                      args.workers, detection=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0
    for epoch in range(args.epochs):
        if deadline_reached(args.deadline, 150):
            break
        model.train()
        total_loss = samples = 0
        complete = True
        started = time.time()
        for images, targets in loader:
            if deadline_reached(args.deadline, 120):
                complete = False
                break
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"检测器loss不是有限值：{losses}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(images)
            samples += len(images)
        if not complete:
            atomic_torch(out / "detector_partial.pt", {"model": model.state_dict(), "epoch": epoch + 1, "samples": samples})
            log_event(out, stage="detector", status="partial_epoch_not_selected", samples=samples)
            break
        metrics = validate_detector(model, val, device, args.deadline)
        scheduler.step()
        score = 2 * metrics["precision"] * metrics["recall"] / max(1e-8, metrics["precision"] + metrics["recall"])
        if score > best:
            best = score
            atomic_torch(out / "detector_best.pt", {"model": model.state_dict(), "epoch": epoch + 1,
                         "validation": metrics, "optimizer": "SGD", "lr": args.lr, "weight_decay": 5e-4})
        log_event(out, stage="detector", status="complete_epoch", epoch=epoch + 1,
                  samples=samples, loss=total_loss / samples, validation=metrics, seconds=time.time() - started)


def aggregate_commands(labels):
    # 可靠停止优先；0不抵消另一只有效手；不同有效动作冲突时拒绝输出动作。
    if 3 in labels:
        return 3
    active = {int(x) for x in labels if x != 0}
    return next(iter(active)) if len(active) == 1 else 0


def whole_photo_predictions(detector, classifier, rows, device, detector_threshold=0.5, class_threshold=0.5, batch_size=16, deadline=None):
    detector.eval()
    classifier.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            if deadline_reached(deadline, 10):
                raise TimeoutError("整图预测未完成：达到截止时间，不保存为完整评价或阈值依据")
            group = rows[start:start + batch_size]
            images = []
            for row in group:
                with Image.open(row["path"]) as source:
                    images.append(source.convert("RGB"))
            detected = detector([TF.to_tensor(image).to(device) for image in images])
            for image, row, result in zip(images, group, detected):
                keep = result["scores"] >= detector_threshold
                boxes = result["boxes"][keep].cpu().tolist()
                scores = result["scores"][keep].cpu().tolist()
                hand_predictions = []
                if boxes:
                    crops = torch.stack([NORMALIZE(TF.to_tensor(crop_hand(image, box)[0])) for box in boxes]).to(device)
                    probabilities = classifier(crops).float().softmax(1).cpu()
                    for box, detection_score, probs in zip(boxes, scores, probabilities):
                        confidence, label = probs.max(0)
                        accepted = confidence.item() >= class_threshold
                        hand_predictions.append({"bbox": box, "detector_score": detection_score,
                            "label": label.item(), "confidence": confidence.item(), "accepted": accepted,
                            "probabilities": probs.tolist()})
                accepted_labels = [p["label"] for p in hand_predictions if p["accepted"]]
                outputs.append({"image_id": row["image_id"], "truth": aggregate_commands(row["labels"]),
                                "prediction": aggregate_commands(accepted_labels), "hands": hand_predictions,
                                "image_size": list(image.size)})
    return outputs


def command_safety(outputs):
    # 分别统计无效输入误发运动、停止手误发运动，保留各自真实分母。
    moving = {1, 2, 4, 5}
    zero = [r for r in outputs if r["truth"] == 0]
    stop = [r for r in outputs if r["truth"] == 3]
    return {"zero_samples": len(zero),
            "zero_to_motion": sum(r["prediction"] in moving for r in zero),
            "stop_samples": len(stop),
            "stop_to_motion": sum(r["prediction"] in moving for r in stop),
            "zero_to_motion_rate": sum(r["prediction"] in moving for r in zero) / max(1, len(zero)),
            "stop_to_motion_rate": sum(r["prediction"] in moving for r in stop) / max(1, len(stop))}


def checkpoint_hashes(out):
    return {name: hashlib.sha256((Path(out) / name).read_bytes()).hexdigest()
            for name in ("classifier_best.pt", "detector_best.pt")}


def frozen_models(out, device):
    checkpoint = torch.load(Path(out) / "classifier_best.pt", map_location="cpu", weights_only=True)
    classifier = resnet(None, checkpoint["params"]["dropout"]).to(device)
    classifier.load_state_dict(checkpoint["model"])
    detector = detector_model(False).to(device)
    detector.load_state_dict(torch.load(Path(out) / "detector_best.pt", map_location="cpu", weights_only=True)["model"])
    return detector.eval(), classifier.eval()


def calibrate(args):
    # 完整验证集只运行一次网络；阈值搜索只过滤已保存输出，不重复运行检测器。
    out = Path(args.out)
    if (out / "calibration.json").exists():
        raise FileExistsError("阈值已经冻结，不允许无痕覆盖；新实验请使用新的run目录")
    device = torch.device(args.device)
    detector, classifier = frozen_models(out, device)
    hashes = checkpoint_hashes(out)
    rows = load_split(args.split_root, "val")
    outputs = whole_photo_predictions(detector, classifier, rows, device,
                                      detector_threshold=0.3, class_threshold=0.0,
                                      batch_size=args.batch_size, deadline=args.deadline)
    atomic_json(out / "val_whole_raw.json", {"split": "val", "weights": hashes,
                "minimum_recorded_detector_score": 0.3, "predictions": outputs})
    candidates = []
    for det_threshold in (0.3, 0.5, 0.7):
        for cls_threshold in (0.3, 0.5, 0.7, 0.9):
            if deadline_reached(args.deadline, 1):
                raise TimeoutError("阈值搜索未完成：不冻结部分候选结果")
            adjusted = [{"truth": r["truth"], "prediction": aggregate_commands([
                h["label"] for h in r["hands"] if h["detector_score"] >= det_threshold
                and h["confidence"] >= cls_threshold])} for r in outputs]
            metrics = confusion_metrics([r["truth"] for r in adjusted], [r["prediction"] for r in adjusted])
            candidates.append({"detector_threshold": det_threshold, "class_threshold": cls_threshold,
                               "metrics": metrics, "safety": command_safety(adjusted)})
    # 主指标为macro-F1；同分才优先更少误发运动，最后选更高的拒识阈值。
    best = max(candidates, key=lambda c: (c["metrics"]["macro_f1"],
        -c["safety"]["stop_to_motion"], -c["safety"]["zero_to_motion"],
        c["class_threshold"], c["detector_threshold"]))
    atomic_json(out / "calibration.json", {"selection_split": "val", "test_accessed": False,
                "objective": "macro_f1_then_stop_and_zero_false_motion_counts", "weights": hashes,
                "candidates": candidates, "selected": best,
                "frozen_at": datetime.now().astimezone().isoformat()})
    log_event(out, stage="calibrate", selected=best)


def final_evaluate(args):
    # 所有权重和阈值冻结后才进入此阶段；不依据test重新优化。
    out = Path(args.out)
    device = torch.device(args.device)
    # 必须在任何test名单、照片和标签读取之前完成冻结校准与双权重身份核验。
    calibration = read_json(out / "calibration.json")
    if calibration.get("selection_split") != "val" or calibration.get("test_accessed") is not False:
        raise ValueError("校准记录不是仅由验证集选择的冻结记录")
    if calibration["weights"] != checkpoint_hashes(out):
        raise ValueError("冻结阈值所对应的权重已变化，不能访问最终测试")
    det_threshold = calibration["selected"]["detector_threshold"]
    cls_threshold = calibration["selected"]["class_threshold"]
    checkpoint = torch.load(out / "classifier_best.pt", map_location="cpu", weights_only=True)
    classifier = resnet(None, checkpoint["params"]["dropout"]).to(device)
    classifier.load_state_dict(checkpoint["model"])
    loader = make_loader(CropDataset(out, "test"), args.batch_size, args.workers)
    metrics, probabilities, indices = evaluate_classifier(classifier, loader, device, args.deadline)
    rows = loader.dataset.rows
    atomic_json(out / "test_oracle_crop.json", {"scope": "ground_truth_hand_boxes_only", "metrics": metrics,
                "checkpoint_sha256": hashlib.sha256((out / "classifier_best.pt").read_bytes()).hexdigest(),
                "predictions": [{"image_id": rows[i]["image_id"], "hand_index": rows[i]["hand_index"],
                     "raw_label": rows[i]["raw_label"], "truth": rows[i]["label"],
                     "prediction": int(np.argmax(p)), "probabilities": p} for i, p in zip(indices, probabilities)]})
    if (out / "detector_best.pt").exists():
        detector = detector_model(False).to(device)
        det_checkpoint = torch.load(out / "detector_best.pt", map_location="cpu", weights_only=True)
        detector.load_state_dict(det_checkpoint["model"])
        test_rows = load_split(args.split_root, "test")
        outputs = whole_photo_predictions(detector, classifier, test_rows, device, det_threshold,
                                          cls_threshold, args.batch_size, args.deadline)
        whole_metrics = confusion_metrics([r["truth"] for r in outputs], [r["prediction"] for r in outputs])
        atomic_json(out / "test_whole_photo.json", {"scope": "complete_photo_detector_crop_classifier_rules",
                    "detector_threshold": det_threshold, "class_threshold": cls_threshold,
                    "metrics": whole_metrics, "predictions": outputs,
                    "weights": checkpoint_hashes(out), "safety": command_safety(outputs)})
        # 复用完整整图推理的手框，不额外对test重复运行检测网络。
        predicted, targets = [], []
        for row, output in zip(test_rows, outputs):
            predicted.append({"boxes": torch.tensor([h["bbox"] for h in output["hands"]]).reshape(-1, 4),
                              "scores": torch.tensor([h["detector_score"] for h in output["hands"]])})
            targets.append({"boxes": torch.tensor([box_xyxy(b, *output["image_size"]) for b in row["bboxes"]]).reshape(-1, 4)})
        detection = detection_metrics(predicted, targets, det_threshold)
        atomic_json(out / "test_detector.json", {"scope": "hand_detection_precision_recall_at_fixed_threshold",
                    "weights": checkpoint_hashes(out), "metrics": detection,
                    "predictions": [{"image_id": r["image_id"], "boxes": p["boxes"].tolist(),
                        "scores": p["scores"].tolist(), "gt_boxes": t["boxes"].tolist()}
                        for r, p, t in zip(test_rows, predicted, targets)]})
    log_event(out, stage="test", scope="oracle_crop", metrics=metrics)


def cpu_latency(args):
    # 测速不调整权重或阈值；固定前100张test和10次预热，明确包括照片打开解码。
    out = Path(args.out)
    calibration = read_json(out / "calibration.json")
    hashes = checkpoint_hashes(out)
    if hashes != calibration["weights"]:
        raise ValueError("测速权重与冻结阈值不匹配")
    device = torch.device("cpu")
    detector, classifier = frozen_models(out, device)
    rows = load_split(args.split_root, "test")[:100]
    dt = calibration["selected"]["detector_threshold"]
    ct = calibration["selected"]["class_threshold"]
    for i in range(10):
        whole_photo_predictions(detector, classifier, [rows[i % len(rows)]], device, dt, ct, 1)
    times = []
    for row in rows:
        started = time.perf_counter()
        whole_photo_predictions(detector, classifier, [row], device, dt, ct, 1)
        times.append((time.perf_counter() - started) * 1000)
    import platform
    atomic_json(out / "cpu_latency.json", {"scope": "whole_photo_decode_detection_crop_classification_rules",
         "model_loading_included": False, "warmup": 10, "n": len(rows), "batch_size": 1,
         "threads": torch.get_num_threads(), "cpu": platform.processor(), "platform": platform.platform(),
         "weights": hashes, "mean_ms": float(np.mean(times)), "p50_ms": float(np.percentile(times, 50)),
         "p95_ms": float(np.percentile(times, 95)), "all_ms": times,
         "image_ids": [r["image_id"] for r in rows]})


def cpu_smoke(args):
    # 只检查前两个train样本的结构、标签和模型输出，不访问测试来选型。
    records = load_split(args.split_root, "train")
    dataset = HandDataset(args.split_root, "train")
    images, target = dataset[0]
    assert images.shape[0] == 3 and (target["boxes"][:, 2:] > target["boxes"][:, :2]).all()
    model = resnet(None).eval()
    with torch.inference_mode():
        result = model(torch.zeros(2, 3, 224, 224))
    assert result.shape == (2, 6)
    assert aggregate_commands([0, 1]) == 1 and aggregate_commands([1, 4]) == 0
    assert aggregate_commands([1, 3]) == 3 and aggregate_commands([]) == 0
    atomic_json(Path(args.out) / "cpu_smoke.json", {"status": "pass", "train_photos": len(records),
               "first_image_shape": list(images.shape), "first_boxes": target["boxes"].tolist(),
               "classifier_output_shape": list(result.shape), "GPU_training_started": False})
    print("CPU smoke PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "features", "search", "finetune", "detector", "calibrate", "evaluate", "latency", "smoke"])
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pretrained")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=0)
    parser.add_argument("--deadline", help="带时区ISO截止时刻，例如2026-11-02T16:30:00+08:00")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--detector-threshold", type=float, default=0.5)
    parser.add_argument("--class-threshold", type=float, default=0.5)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    seed_all()
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    log_event(args.out, stage=args.stage, status="start", args=vars(args))
    try:
        {"prepare": prepare_crops, "features": feature_cache, "search": search,
         "finetune": fine_tune, "detector": train_detector, "calibrate": calibrate,
         "evaluate": final_evaluate, "latency": cpu_latency,
         "smoke": cpu_smoke}[args.stage](args)
    except TimeoutError as error:
        log_event(args.out, stage=args.stage, status="deadline_incomplete_not_selected", reason=str(error))
        raise
    log_event(args.out, stage=args.stage, status="exit0")


if __name__ == "__main__":
    main()
