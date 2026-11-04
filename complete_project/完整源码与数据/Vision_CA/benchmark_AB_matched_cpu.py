"""六个冻结模型的同输入CPU测速；默认只显示协议，--run才执行一次。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
A_ROOT = ROOT / "runs/A_20261102_TPE"
B_ROOT = ROOT / "runs/B_20261102_TPE"
OUTPUT = B_ROOT / "matched_cpu_benchmark.json"
PROTOCOL = B_ROOT / "MATCHED_CPU_PROTOCOL.md"
MODEL_KEYS = ["A_resnet18", "B_resnet18", "A_resnet34", "B_resnet34", "A_resnet50", "B_resnet50"]
ORDERS = [MODEL_KEYS, list(reversed(MODEL_KEYS)), MODEL_KEYS[3:] + MODEL_KEYS[:3]]
SCOPES = ["classifier_only_preprocessed_224x224_tensor", "whole_photo_decode_detection_crop_classification_rules"]


def now():
    # 保存本机时区，便于与后台任务和系统事件对照。
    return datetime.now().astimezone().isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_exclusive(path, value, precommit):
    # 禁止覆盖已有结果；先编码，再独占创建并落盘。
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    precommit()
    with Path(path).open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def execute():
    # 标记一次已消耗的运行机会，失败也保留，不自动重新测速。
    journal_path = OUTPUT.with_suffix(".attempt.jsonl")
    if OUTPUT.exists() or journal_path.exists():
        raise FileExistsError("结果或运行日志已存在，拒绝覆盖及自动重跑")
    journal = journal_path.open("x", encoding="utf-8", buffering=1)

    def event(**values):
        journal.write(json.dumps({"time": now(), **values}, ensure_ascii=False) + "\n")
        journal.flush()
        os.fsync(journal.fileno())

    result = {"status": "running", "started_at": now(), "models": []}
    try:
        event(status="started", protocol_sha256=sha(PROTOCOL), script_sha256=sha(__file__))
        # 重型库只在显式执行分支导入，--help和默认调用都不加载模型。
        import numpy as np
        import psutil
        import torch
        import torchvision
        from PIL import Image
        from torchvision import models
        from torchvision.transforms import functional as TF
        import train_A as base
        from deployment_src.gesture_predict import build_hand_detector

        torch.set_num_threads(8)
        device = torch.device("cpu")
        process = psutil.Process()

        def snapshot():
            # 系统CPU利用率与本进程CPU时间用于解释实际背景负载。
            return {"time": now(), "monotonic": time.perf_counter(),
                    "system_cpu_times": dict(psutil.cpu_times()._asdict()),
                    "process_cpu_times": dict(process.cpu_times()._asdict()),
                    "memory_available_bytes": psutil.virtual_memory().available,
                    "process_rss_bytes": process.memory_info().rss,
                    "process_count": len(psutil.pids())}

        def background(before, after):
            elapsed = after["monotonic"] - before["monotonic"]
            cpu_before, cpu_after = before["system_cpu_times"], after["system_cpu_times"]
            delta = {key: cpu_after[key] - cpu_before[key] for key in cpu_before}
            total = sum(delta.values())
            idle = delta.get("idle", 0) + delta.get("iowait", 0)
            own = sum(after["process_cpu_times"].get(k, 0) - before["process_cpu_times"].get(k, 0)
                      for k in ("user", "system"))
            return {"system_busy_percent": 100 * (total - idle) / max(total, 1e-9),
                    "process_cpu_seconds": own, "elapsed_seconds": elapsed,
                    "estimated_other_process_busy_percent": 100 * max(0, total - idle - own) / max(total, 1e-9),
                    "before": before, "after": after}

        audit = read(A_ROOT / "data_audit.json")
        split_root = Path(audit["split_root"])
        photos = sorted(base.load_split(split_root, "val"), key=lambda row: row["image_id"])[:100]
        crop_rows = sorted(read(A_ROOT / "crops_val.json"), key=lambda row: (row["image_id"], int(row["hand_index"])))[:100]
        if len(photos) != 100 or len(crop_rows) != 100:
            raise ValueError("固定输入不足100个")
        tensors, crop_identity = [], []
        for row in crop_rows:
            with Image.open(row["path"]) as image:
                tensor = base.NORMALIZE(TF.to_tensor(image.convert("RGB"))).unsqueeze(0)
            if list(tensor.shape) != [1, 3, 224, 224] or tensor.dtype != torch.float32:
                raise ValueError("分类输入形状或类型不符合协议")
            tensors.append(tensor)
            crop_identity.append({"sample_key": f"{row['image_id']}__hand{row['hand_index']}",
                                  "path": row["path"], "crop_sha256": sha(row["path"]),
                                  "source_path": row["source_path"], "source_sha256": sha(row["source_path"]),
                                  "preprocessed_tensor_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()})
        photo_identity = [{"image_id": row["image_id"], "path": row["path"], "source_sha256": sha(row["path"])} for row in photos]
        result.update({"protocol_sha256": sha(PROTOCOL), "script_sha256": sha(__file__),
                       "source_sha256": {name: sha(ROOT / name) for name in ("train_A.py", "compare_resnets_A.py", "deployment_src/gesture_predict.py")},
                       "selection_split": "A_val", "accuracy_recomputed": False,
                       "used_to_select_weights_or_thresholds": False, "photos": photo_identity,
                       "classifier_inputs": crop_identity, "repeat_orders": ORDERS,
                       "scopes": SCOPES, "batch_size": 1, "dtype": "float32", "device": "cpu",
                       "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                       "warmup_per_model_per_scope": 20, "repeats": 3, "unique_inputs_per_scope": 100,
                       "model_loading_included": False, "classifier_preprocessing_included": False,
                       "whole_photo_preprocessing_included": True,
                       "whole_photo_call": "train_A.whole_photo_predictions(..., rows=[row], batch_size=1)",
                       "whole_photo_batch_note": "one photograph per call; detected hand crops are classified together as in original cpu_latency",
                       "model_factory": "offline deployment_src.build_hand_detector; strict load of original checkpoints; original whole-photo function unchanged",
                       "environment": {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                                       "torchvision": torchvision.__version__, "numpy": np.__version__,
                                       "platform": platform.platform(), "cpu": platform.processor(),
                                       "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
                                       "torch_build": torch.__config__.show(), "pid": os.getpid(),
                                       "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}})
        # 冻结样本索引与原标注身份，提交前一并核验，避免只核模型。
        identity_files = [A_ROOT / "data_audit.json", A_ROOT / "crops_val.json", split_root / "val/manifest.csv"]
        identity_files.extend(sorted((split_root / "val/annotations").glob("*.json")))
        result["input_index_sha256"] = {str(path): sha(path) for path in identity_files}
        loaded = {}
        # 所有模型提前载入内存；加载、哈希及输入准备均不计入时延。
        for key in MODEL_KEYS:
            protocol, architecture = key.split("_", 1)
            folder = A_ROOT if protocol == "A" else B_ROOT
            if architecture != "resnet18":
                folder = folder / ("compare_" + architecture)
            calibration = read(folder / "calibration.json")
            weights = base.checkpoint_hashes(folder)
            if calibration["weights"] != weights or calibration["selection_split"] != "val" or calibration["test_accessed"] is not False:
                raise ValueError(f"冻结记录核验失败：{key}")
            checkpoint = torch.load(folder / "classifier_best.pt", map_location="cpu", weights_only=True)
            classifier = getattr(models, architecture)(weights=None)
            classifier.fc = torch.nn.Sequential(torch.nn.Dropout(checkpoint["params"]["dropout"]), torch.nn.Linear(classifier.fc.in_features, 6))
            classifier.load_state_dict(checkpoint["model"], strict=True)
            detector = build_hand_detector()
            detector.load_state_dict(torch.load(folder / "detector_best.pt", map_location="cpu", weights_only=True)["model"], strict=True)
            detector.eval()
            classifier.eval()
            item = {"key": key, "architecture": architecture, "protocol": protocol,
                    "run_folder": str(folder), "weights": weights,
                    "calibration_sha256": sha(folder / "calibration.json"),
                    "detector_threshold": calibration["selected"]["detector_threshold"],
                    "classifier_threshold": calibration["selected"]["class_threshold"],
                    "classifier_parameters": sum(p.numel() for p in classifier.parameters()),
                    "measurements": {scope: [] for scope in SCOPES}}
            result["models"].append(item)
            loaded[key] = (detector, classifier, item)
            event(status="model_loaded", key=key, weights=weights)

        def whole(key, row):
            detector, classifier, item = loaded[key]
            return base.whole_photo_predictions(detector, classifier, [row], device,
                item["detector_threshold"], item["classifier_threshold"], 1)[0]

        with torch.inference_mode():
            for scope in SCOPES:
                for key in MODEL_KEYS:
                    for index in range(20):
                        if scope == SCOPES[0]:
                            loaded[key][1](tensors[index])
                        else:
                            whole(key, photos[index])
                    event(status="warmup_complete", key=key, scope=scope, count=20)
                for repeat, order in enumerate(ORDERS):
                    for key in order:
                        before = snapshot()
                        values, workloads = [], []
                        started_at = now()
                        for index in range(100):
                            start = time.perf_counter_ns()
                            if scope == SCOPES[0]:
                                loaded[key][1](tensors[index])
                            else:
                                output = whole(key, photos[index])
                            values.append((time.perf_counter_ns() - start) / 1e6)
                            if scope == SCOPES[1]:
                                workloads.append({"image_id": photos[index]["image_id"],
                                    "detected_boxes_after_frozen_threshold": len(output["hands"]),
                                    "accepted_hands": sum(bool(h["accepted"]) for h in output["hands"])})
                        block = {"repeat": repeat + 1, "started_at": started_at, "ended_at": now(),
                                 "all_ms": values, "mean_ms": float(np.mean(values)),
                                 "workloads": workloads, "cpu_background": background(before, snapshot())}
                        loaded[key][2]["measurements"][scope].append(block)
                        event(status="block_complete", key=key, scope=scope, **block)
                        print(f"{key} {scope} repeat={repeat + 1} mean_ms={block['mean_ms']:.4f}", flush=True)
        for item in result["models"]:
            item["summary"] = {}
            for scope, blocks in item["measurements"].items():
                values = [value for block in blocks for value in block["all_ms"]]
                item["summary"][scope] = {"n": len(values), "mean_ms": float(np.mean(values)),
                    "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
                    "sd_ms": float(np.std(values, ddof=1)), "sd_definition": "sample standard deviation ddof=1",
                    "repeat_means_ms": [block["mean_ms"] for block in blocks]}
        result.update(status="complete", ended_at=now())

        def precommit():
            # 结果编码后紧邻提交复核全部活身份，提交使用已编码的同一份字节。
            for item in result["models"]:
                if base.checkpoint_hashes(item["run_folder"]) != item["weights"] or sha(Path(item["run_folder"]) / "calibration.json") != item["calibration_sha256"]:
                    raise ValueError("执行期间权重或阈值变化，保留日志但不提交成功结果")
            frozen_files = {str(PROTOCOL): result["protocol_sha256"], str(Path(__file__)): result["script_sha256"],
                            **{str(ROOT / name): digest for name, digest in result["source_sha256"].items()},
                            **result["input_index_sha256"]}
            for row in photo_identity:
                frozen_files[row["path"]] = row["source_sha256"]
            for row in crop_identity:
                frozen_files[row["path"]] = row["crop_sha256"]
                frozen_files[row["source_path"]] = row["source_sha256"]
            for path, digest in frozen_files.items():
                if sha(path) != digest:
                    raise ValueError(f"提交前身份发生变化：{path}")

        save_exclusive(OUTPUT, result, precommit)
        event(status="complete", output=str(OUTPUT), output_sha256=sha(OUTPUT))
    except BaseException as error:
        event(status="failed", error=repr(error), traceback=traceback.format_exc(), partial_result=result)
        raise
    finally:
        journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="按冻结协议执行一次全六模型基准；已有结果或尝试日志时拒绝重跑")
    args = parser.parse_args()
    if args.run:
        execute()
    else:
        print(PROTOCOL.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
