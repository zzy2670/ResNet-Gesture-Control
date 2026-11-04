"""在固定A协议上比较ResNet34/50，不修改正在运行的ResNet18主程序。

每个架构使用独立进程与输出目录。三种架构共享原图名单、手裁图和最终手检测器，
TPE只看训练/验证特征；测试前冻结权重和验证阈值。根调度器负责GPU顺序与总截止时间。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

import train_A as base

ARCHITECTURES = {
    "resnet34": {"factory": models.resnet34, "weights": models.ResNet34_Weights.IMAGENET1K_V1,
                 "feature_dim": 512},
    "resnet50": {"factory": models.resnet50, "weights": models.ResNet50_Weights.IMAGENET1K_V1,
                 "feature_dim": 2048},
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def architecture_model(arch, pretrained=None, dropout=0.0, fine_tune=False):
    # 明确按架构构建，再严格加载该架构的官方ImageNet V1权重，禁止跨架构误用。
    specification = ARCHITECTURES[arch]
    model = specification["factory"](weights=None)
    if pretrained:
        model.load_state_dict(torch.load(pretrained, map_location="cpu", weights_only=True), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    if fine_tune:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(specification["feature_dim"], 6))
    return model


def copy_same_or_create(source, target):
    # 派生索引引用原有绝对裁图路径，只复制小索引，不再次复制或重划照片。
    source, target = Path(source), Path(target)
    if target.exists():
        if file_hash(source) != file_hash(target):
            raise ValueError(f"已有文件身份不符，拒绝覆盖：{target}")
        return
    before = file_hash(source)
    shutil.copy2(source, target)
    if file_hash(source) != before or file_hash(target) != before:
        raise ValueError(f"源文件在复制期间变化：{source}")


def prepare(args):
    out = Path(args.out)
    for name in ("crops_train.json", "crops_val.json", "crops_test.json", "data_audit.json"):
        copy_same_or_create(Path(args.shared_run) / name, out / name)
    # 下载使用TorchVision官方枚举及其哈希检查，不导入另一架构的参数。
    weight = ARCHITECTURES[args.arch]["weights"]
    cache = Path(torch.hub.get_dir()) / "checkpoints" / Path(weight.url).name
    torch.hub.load_state_dict_from_url(weight.url, model_dir=str(cache.parent),
                                      map_location="cpu", check_hash=True)
    model = architecture_model(args.arch, cache)
    assert model.fc[1].in_features == ARCHITECTURES[args.arch]["feature_dim"]
    base.atomic_json(out / "pretrained_identity.json", {"architecture": args.arch,
        "weights_enum": str(weight), "url": weight.url, "path": str(cache.resolve()),
        "sha256": file_hash(cache), "bytes": cache.stat().st_size,
        "strict_architecture_load": True, "GPU_training_started": False})
    base.log_event(out, stage="compare_prepare", architecture=args.arch,
                   feature_dim=ARCHITECTURES[args.arch]["feature_dim"], weights_path=str(cache))


def bind_architecture(args):
    # 只在此独立进程内切换工厂；不修改train_A.py源码，也不影响其他训练进程。
    arch = args.arch
    base.resnet = lambda pretrained=None, dropout=0.0, fine_tune=False: architecture_model(
        arch, pretrained=pretrained, dropout=dropout, fine_tune=fine_tune)


def search(args):
    """与ResNet18相同六个超参数范围和选型指标，仅线性头输入维度随架构变化。"""
    import optuna
    out = Path(args.out)
    device = torch.device(args.device)
    train = torch.load(out / "features_train.pt", map_location="cpu", weights_only=True)
    val = torch.load(out / "features_val.pt", map_location="cpu", weights_only=True)
    dim = ARCHITECTURES[args.arch]["feature_dim"]
    if train["x"].shape[1] != dim or val["x"].shape[1] != dim:
        raise ValueError("特征维数不匹配，拒绝混用其他ResNet特征")
    xtrain, ytrain = train["x"].to(device), train["y"].to(device)
    xval, yval = val["x"].to(device), val["y"].to(device)
    counts = torch.bincount(ytrain, minlength=6).float()
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=4)
    study = optuna.create_study(study_name=f"A_{args.arch}_frozen_features_TPE", direction="maximize",
        sampler=sampler, storage="sqlite:///" + str((out / "tpe.sqlite3").resolve()).replace("\\", "/"),
        load_if_exists=True)

    def objective(trial):
        if base.deadline_reached(args.deadline, 60):
            raise optuna.TrialPruned("搜索截止时间已到")
        base.seed_all(42)
        optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "sgd"])
        lr = trial.suggest_float("lr", 1e-4, 3e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        smoothing = trial.suggest_float("label_smoothing", 0.0, 0.12)
        weighted = trial.suggest_categorical("class_weight", [False, True])
        head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 6)).to(device)
        weights = (counts.sum() / (6 * counts)).sqrt() if weighted else None
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing)
        optimizer = base.optimizer_for(head.parameters(), optimizer_name, lr, wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        best = -1.0
        for epoch in range(args.epochs):
            head.train()
            permutation = torch.randperm(len(ytrain), device=device)
            for indices in permutation.split(max(args.batch_size, 128)):
                if base.deadline_reached(args.deadline, 30):
                    raise optuna.TrialPruned("到达预算，保留已完成trial")
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(head(xtrain[indices]), ytrain[indices])
                loss.backward()
                optimizer.step()
            head.eval()
            predictions = []
            with torch.inference_mode():
                for batch in xval.split(1024):
                    if base.deadline_reached(args.deadline, 10):
                        raise optuna.TrialPruned("验证未完成，不返回部分分数")
                    predictions.append(head(batch).argmax(1).cpu())
            metrics = base.confusion_metrics(yval.cpu().numpy(), torch.cat(predictions).numpy())
            score = metrics["macro_f1"]
            trial.report(score, epoch)
            scheduler.step()
            if score > best:
                best = score
                base.atomic_torch(out / "trials" / f"trial_{trial.number}_best.pt", {
                    "head": head.state_dict(), "params": trial.params, "epoch": epoch + 1,
                    "validation": metrics, "trial": trial.number, "architecture": args.arch})
            base.log_event(out, stage="tpe", trial=trial.number, epoch=epoch + 1,
                           validation_macro_f1=score, validation_accuracy=metrics["accuracy"])
        return best

    study.optimize(objective, n_trials=args.trials, timeout=args.timeout or None,
                   catch=(FloatingPointError,), gc_after_trial=True)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError("没有完成的TPE trial，不生成最优权重")
    best = study.best_trial
    base.atomic_json(out / "tpe_summary.json", {"architecture": args.arch,
        "objective": "validation_macro_f1", "scope": "frozen_ImageNet_features_classifier_head",
        "trials": [{"number": t.number, "state": t.state.name, "value": t.value, "params": t.params}
                   for t in study.trials], "best_trial": best.number, "best_params": best.params,
        "best_value": best.value, "test_accessed": False})
    checkpoint = torch.load(out / "trials" / f"trial_{best.number}_best.pt", map_location="cpu", weights_only=True)
    model = architecture_model(args.arch, args.pretrained, best.params["dropout"])
    model.fc.load_state_dict(checkpoint["head"], strict=True)
    base.atomic_torch(out / "classifier_best.pt", {"model": model.state_dict(), "params": best.params,
        "stage": "frozen_head_TPE", "validation": checkpoint["validation"], "epoch": checkpoint["epoch"],
        "architecture": args.arch})


def shared_detector(args):
    # 只有检测器主流程完成后由root调用；验证两个目录里的检测权重完全一致。
    source = Path(args.shared_run) / "detector_best.pt"
    if not source.is_file():
        raise FileNotFoundError("共享ResNet18主流程手检测器尚未完成，不能校准或测试比较模型")
    copy_same_or_create(source, Path(args.out) / "detector_best.pt")
    base.atomic_json(Path(args.out) / "shared_detector_identity.json", {
        "source": str(source.resolve()), "sha256": file_hash(source),
        "bytes": source.stat().st_size, "retrained_for_comparison": False})


def classifier_latency(args):
    # 此指标只测224×224分类网络计算；整图指标仍通过base.cpu_latency测量。
    out = Path(args.out)
    calibration = base.read_json(out / "calibration.json")
    hashes = base.checkpoint_hashes(out)
    if calibration["weights"] != hashes:
        raise ValueError("测速模型与冻结校准身份不一致")
    checkpoint = torch.load(out / "classifier_best.pt", map_location="cpu", weights_only=True)
    model = architecture_model(args.arch, dropout=checkpoint["params"]["dropout"]).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = base.CropDataset(out, "test")
    # 所有架构使用相同索引和照片；预处理在计时之前完成。
    samples = [dataset[i][0].unsqueeze(0) for i in range(min(100, len(dataset)))]
    times = []
    with torch.inference_mode():
        for i in range(20):
            model(samples[i % len(samples)])
        for tensor in samples:
            started = time.perf_counter()
            model(tensor)
            times.append((time.perf_counter() - started) * 1000)
    base.atomic_json(out / "classifier_cpu_latency.json", {"architecture": args.arch,
        "scope": "classifier_only_preprocessed_224x224_tensor", "batch_size": 1,
        "preprocessing_included": False, "model_loading_included": False,
        "warmup": 20, "n": len(samples), "threads": torch.get_num_threads(), "weights": hashes,
        "mean_ms": float(np.mean(times)), "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)), "all_ms": times,
        "sample_ids": [f"{r['image_id']}__hand{r['hand_index']}" for r in dataset.rows[:len(samples)]],
        "parameters": sum(p.numel() for p in model.parameters()),
        "classifier_checkpoint_bytes": (out / "classifier_best.pt").stat().st_size})


def model_info(args):
    model = architecture_model(args.arch, fine_tune=True)
    data = {"architecture": args.arch, "feature_dim": ARCHITECTURES[args.arch]["feature_dim"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "finetune_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "pretraining": "ImageNet1K_V1", "fine_tune": "layer4_and_fc_only",
            "fine_tune_optimizer": "AdamW", "layer4_lr": args.lr, "fc_lr": args.lr * 5,
            "base_training_source_sha256": file_hash(Path(base.__file__)),
            "comparison_source_sha256": file_hash(Path(__file__)),
            "shared_run": str(Path(args.shared_run).resolve()), "split_root": str(Path(args.split_root).resolve())}
    base.atomic_json(Path(args.out) / "model_info.json", data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "features", "search", "finetune", "calibrate", "evaluate", "latency", "classifier-latency", "smoke"])
    parser.add_argument("--arch", required=True, choices=list(ARCHITECTURES))
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--shared-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pretrained")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=0)
    parser.add_argument("--deadline")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()
    out = Path(args.out)
    if out.resolve() == Path(args.shared_run).resolve():
        raise ValueError("比较输出不得与ResNet18主目录相同")
    out.mkdir(parents=True, exist_ok=True)
    base.seed_all(42)
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    bind_architecture(args)
    model_info(args)
    if args.stage not in ("prepare", "smoke"):
        identity = base.read_json(out / "pretrained_identity.json")
        if identity["architecture"] != args.arch:
            raise ValueError("输出目录属于另一架构")
        args.pretrained = args.pretrained or identity["path"]
        if file_hash(args.pretrained) != identity["sha256"]:
            raise ValueError("ImageNet权重身份不符")
    if args.stage in ("calibrate", "evaluate", "latency", "classifier-latency"):
        shared_detector(args)
    base.log_event(out, stage=args.stage, architecture=args.arch, status="start", args=vars(args))
    if args.stage == "smoke":
        # CPU仅用全零张量确认模型输出与特征维数，不读取test指标。
        model = architecture_model(args.arch).eval()
        with torch.inference_mode():
            assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 6)
        base.atomic_json(out / "comparison_cpu_smoke.json", {"status": "pass", "architecture": args.arch,
            "feature_dim": ARCHITECTURES[args.arch]["feature_dim"], "GPU_training_started": False})
    else:
        try:
            {"prepare": prepare, "features": base.feature_cache, "search": search,
             "finetune": base.fine_tune, "calibrate": base.calibrate,
             "evaluate": base.final_evaluate, "latency": base.cpu_latency,
             "classifier-latency": classifier_latency}[args.stage](args)
        except TimeoutError as error:
            base.log_event(out, stage=args.stage, status="deadline_incomplete_not_selected", reason=str(error))
            raise
    base.log_event(out, stage=args.stage, architecture=args.arch, status="exit0")


if __name__ == "__main__":
    main()
