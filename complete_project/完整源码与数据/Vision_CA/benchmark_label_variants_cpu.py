"""十二个双标签版本冻结模型的同输入CPU测速；默认只显示协议，--run才执行一次。"""
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
MERGED_ROOT = ROOT / "runs/MERGED_20261102"
REFERENCE = B_ROOT / "matched_cpu_benchmark.json"
REFERENCE_SHA256 = "23f8dd1e2b08ac7e9e2f4f1618534212a209f867410a54873663555a6e3fba68"
OUTPUT = MERGED_ROOT / "label_variants_cpu_benchmark.json"
PROTOCOL = MERGED_ROOT / "LABEL_VARIANTS_CPU_PROTOCOL.md"
MODEL_KEYS = [f"{variant}_{protocol}_{architecture}"
              for architecture in ("resnet18", "resnet34", "resnet50")
              for protocol in ("A", "B") for variant in ("unmerged", "merged")]
ORDERS = [MODEL_KEYS, list(reversed(MODEL_KEYS)), MODEL_KEYS[6:] + MODEL_KEYS[:6]]
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


def scientific_readiness():
    # 先核权威总完成与科学独审；缺任何门禁都不得加载任何一个模型。
    complete_path = MERGED_ROOT / "all_training_evaluation_complete.json"
    aggregate_path = MERGED_ROOT / "aggregate_review.json"
    complete, aggregate = read(complete_path), read(aggregate_path)
    if complete.get("status") != "PASS":
        raise ValueError("新六模型尚未全部完成训练和评价")
    if aggregate.get("status") != "PASS_MERGED_SIX_MODEL_AUDIT_WITH_LIMITATIONS" or aggregate.get("issues") != []:
        raise ValueError("新六模型真实科学汇总独审未通过")
    bound = {}

    def bind(path, digest):
        path = str(Path(path))
        if path in bound and bound[path] != digest:
            raise ValueError(f"科学独审身份互相冲突：{path}")
        bound[path] = digest

    bind(complete_path, sha(complete_path))
    bind(aggregate_path, sha(aggregate_path))
    aggregate_bound = aggregate["bound_file_sha256"]
    for path, digest in aggregate_bound.items():
        bind(path, digest)
    expected = {(protocol, arch) for protocol in ("A", "B") for arch in ("resnet18", "resnet34", "resnet50")}
    aggregate_models = {(row["protocol"], row["architecture"]): row for row in aggregate["models"]}
    if len(aggregate["models"]) != 6 or set(aggregate_models) != expected:
        raise ValueError("科学汇总独审没有完整且唯一的六模型")
    model_reports = []
    for protocol, arch in sorted(expected):
        folder = MERGED_ROOT / protocol / arch
        report_path = folder / "model_review.json"
        report = read(report_path)
        if (report.get("status") != "PASS_MERGED_MODEL_AUDIT_WITH_LIMITATIONS" or report.get("issues") != []
                or report.get("protocol") != protocol or report.get("architecture") != arch):
            raise ValueError(f"逐模型科学独审未通过或身份不符：{protocol}/{arch}")
        if str(report_path) not in aggregate_bound:
            raise ValueError("科学汇总没有绑定真实逐模型独审文件")
        bind(report_path, aggregate_bound[str(report_path)])
        report_bound = report["bound_file_sha256"]
        for name in ("classifier_best.pt", "detector_best.pt", "calibration.json", "val_whole_raw.json",
                     "test_oracle_crop.json", "test_whole_photo.json", "test_detector.json"):
            if str(folder / name) not in report_bound:
                raise ValueError(f"科学独审未绑定模型、校准或原始输出：{folder / name}")
        for path, digest in report_bound.items():
            bind(path, digest)
        calibration = read(folder / "calibration.json")
        summary = aggregate_models[(protocol, arch)]
        if report["weights"] != calibration["weights"] or summary["weights"] != report["weights"]:
            raise ValueError("科学独审与校准的双权重身份不一致")
        if report["calibration_selected"] != calibration["selected"] or summary["calibration_selected"] != report["calibration_selected"]:
            raise ValueError("科学独审与实际冻结阈值不一致")
        for name, digest in report["weights"].items():
            bind(folder / name, digest)
        model_reports.append({"protocol": protocol, "architecture": arch, "path": str(report_path),
                              "sha256": aggregate_bound[str(report_path)], "status": report["status"]})
    # 复核独审实际绑定的原始产物和源码，只做哈希，不重算准确率或模型推理。
    for path, digest in bound.items():
        if sha(path) != digest:
            raise ValueError(f"科学独审绑定产物已变化：{path}")
    return {"status": "PASS", "completion_path": str(complete_path), "aggregate_path": str(aggregate_path),
            "aggregate_status": aggregate["status"], "model_reports": model_reports, "bound_file_sha256": bound}


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
        result["scientific_readiness"] = scientific_readiness()
        event(status="scientific_readiness_pass", model_reports=result["scientific_readiness"]["model_reports"])
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

        if psutil.__version__ != "7.0.0":
            raise RuntimeError("必须使用已预检的psutil 7.0.0，不能静默缺失背景负载证据")
        event(status="dependency_preflight_pass", python=sys.version, executable=sys.executable,
              torch=torch.__version__, torchvision=torchvision.__version__, psutil=psutil.__version__)
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

        # 沿用旧基准的确切身份和顺序，禁止重新取前100或按结果换样本。
        if sha(REFERENCE) != REFERENCE_SHA256:
            raise ValueError("旧共同输入基准身份变化")
        reference = read(REFERENCE)
        if reference["status"] != "complete":
            raise ValueError("旧共同输入基准未完成")
        for path, digest in reference["input_index_sha256"].items():
            if sha(path) != digest:
                raise ValueError(f"旧输入索引身份变化：{path}")
        for name in ("train_A.py", "deployment_src/gesture_predict.py"):
            if sha(ROOT / name) != reference["source_sha256"][name]:
                raise ValueError(f"原计时函数或工厂源码身份变化：{name}")
        audit = read(A_ROOT / "data_audit.json")
        split_root = Path(audit["split_root"])
        photo_lookup = {row["image_id"]: row for row in base.load_split(split_root, "val")}
        photo_identity = reference["photos"]
        crop_identity = reference["classifier_inputs"]
        if len(photo_identity) != 100 or len(crop_identity) != 100:
            raise ValueError("旧共同输入数量不为100")
        photos = [photo_lookup[row["image_id"]] for row in photo_identity]
        for row, identity in zip(photos, photo_identity):
            if Path(row["path"]).resolve() != Path(identity["path"]).resolve() or sha(row["path"]) != identity["source_sha256"]:
                raise ValueError("原照片路径或像素身份变化")
        tensors = []
        for row in crop_identity:
            if sha(row["path"]) != row["crop_sha256"] or sha(row["source_path"]) != row["source_sha256"]:
                raise ValueError("GT手图或其原图身份变化")
            with Image.open(row["path"]) as image:
                tensor = base.NORMALIZE(TF.to_tensor(image.convert("RGB"))).unsqueeze(0)
            if list(tensor.shape) != [1, 3, 224, 224] or tensor.dtype != torch.float32:
                raise ValueError("分类输入形状或类型不符合协议")
            if hashlib.sha256(tensor.numpy().tobytes()).hexdigest() != row["preprocessed_tensor_sha256"]:
                raise ValueError("分类预处理张量与旧共同输入不一致")
            tensors.append(tensor)
        mappings = {"unmerged": dict(base.LABELS),
                    "merged": {**base.LABELS, "two_up": 1, "two_up_inverted": 1}}
        variant_photos = {variant: [{**row, "labels": [mapping.get(raw, 0) for raw in row["raw_labels"]]}
                                    for row in photos] for variant, mapping in mappings.items()}
        result["photo_truth_metadata"] = {variant: [{"image_id": row["image_id"], "raw_labels": row["raw_labels"],
            "labels": row["labels"], "truth": base.aggregate_commands(row["labels"])} for row in rows]
            for variant, rows in variant_photos.items()}
        result["label_mappings"] = mappings
        result["reference_benchmark"] = {"path": str(REFERENCE), "sha256": REFERENCE_SHA256,
                                         "reuse": "exact photo/crop keys, order, pixels and classifier tensor SHA256"}
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
                                       "torchvision": torchvision.__version__, "psutil": psutil.__version__, "numpy": np.__version__,
                                       "platform": platform.platform(), "cpu": platform.processor(),
                                       "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
                                       "total_ram_bytes": psutil.virtual_memory().total,
                                       "torch_build": torch.__config__.show(), "pid": os.getpid(),
                                       "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}})
        # 冻结样本索引与原标注身份，提交前一并核验，避免只核模型。
        identity_files = [A_ROOT / "data_audit.json", A_ROOT / "crops_val.json", split_root / "val/manifest.csv"]
        identity_files.extend(sorted((split_root / "val/annotations").glob("*.json")))
        result["input_index_sha256"] = {str(path): sha(path) for path in identity_files}
        result["input_index_sha256"][str(REFERENCE)] = REFERENCE_SHA256
        result["memory_budget"] = {"expected_process_rss_gib": 3, "host_expected_ram_gib": 32,
                                   "minimum_available_before_load_gib": 6, "before_load": snapshot()}
        if psutil.virtual_memory().available < 6 * 1024**3:
            raise MemoryError("12模型预载之前可用内存不足6GiB，保留失败日志并停止")
        reference_models = {item["key"]: item for item in reference["models"]}
        loaded = {}
        # 全部12模型提前载入内存；加载、哈希及输入准备均不计入时延。
        for key in MODEL_KEYS:
            variant, protocol, architecture = key.split("_", 2)
            old_folder = A_ROOT if protocol == "A" else B_ROOT
            if architecture != "resnet18":
                old_folder = old_folder / ("compare_" + architecture)
            folder = old_folder if variant == "unmerged" else MERGED_ROOT / protocol / architecture
            calibration = read(folder / "calibration.json")
            weights = base.checkpoint_hashes(folder)
            if calibration["weights"] != weights or calibration["selection_split"] != "val" or calibration["test_accessed"] is not False:
                raise ValueError(f"冻结记录核验失败：{key}")
            old_weights = reference_models[f"{protocol}_{architecture}"]["weights"]
            if weights["detector_best.pt"] != old_weights["detector_best.pt"]:
                raise ValueError(f"检测器不是同协议原权重：{key}")
            if variant == "unmerged" and weights != old_weights:
                raise ValueError(f"旧模型身份与历史基准不一致：{key}")
            parameter_files = [old_folder / "tpe_summary.json"]
            if variant == "merged":
                data_identity = read(folder / "merged_data_identity.json")
                if data_identity["mapping"] != mappings["merged"] or data_identity["old_mapping"] != mappings["unmerged"]:
                    raise ValueError(f"合并模型标签语义与协议不一致：{key}")
                parameter_files += [folder / name for name in ("merged_input_identity.json", "merged_data_identity.json", "fixed_head_summary.json", "finetune_summary.json")]
            parameter_sha256 = {str(path): sha(path) for path in parameter_files}
            checkpoint = torch.load(folder / "classifier_best.pt", map_location="cpu", weights_only=True)
            classifier = getattr(models, architecture)(weights=None)
            classifier.fc = torch.nn.Sequential(torch.nn.Dropout(checkpoint["params"]["dropout"]), torch.nn.Linear(classifier.fc.in_features, 6))
            classifier.load_state_dict(checkpoint["model"], strict=True)
            detector = build_hand_detector()
            detector.load_state_dict(torch.load(folder / "detector_best.pt", map_location="cpu", weights_only=True)["model"], strict=True)
            detector.eval()
            classifier.eval()
            item = {"key": key, "label_variant": variant, "architecture": architecture, "protocol": protocol,
                    "classifier_params": checkpoint["params"],
                    "output_classes": 6, "parameter_files_sha256": parameter_sha256,
                    "run_folder": str(folder), "weights": weights,
                    "calibration_sha256": sha(folder / "calibration.json"),
                    "detector_threshold": calibration["selected"]["detector_threshold"],
                    "classifier_threshold": calibration["selected"]["class_threshold"],
                    "classifier_parameters": sum(p.numel() for p in classifier.parameters()),
                    "measurements": {scope: [] for scope in SCOPES}}
            result["models"].append(item)
            loaded[key] = (detector, classifier, item)
            event(status="model_loaded", key=key, weights=weights, memory=snapshot())
        result["memory_budget"]["after_load"] = snapshot()

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
                            whole(key, variant_photos[loaded[key][2]["label_variant"]][index])
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
                                output = whole(key, variant_photos[loaded[key][2]["label_variant"]][index])
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
            for path, digest in result["scientific_readiness"]["bound_file_sha256"].items():
                if sha(path) != digest:
                    raise ValueError(f"提交前科学完成或独审绑定身份变化：{path}")
            for item in result["models"]:
                if base.checkpoint_hashes(item["run_folder"]) != item["weights"] or sha(Path(item["run_folder"]) / "calibration.json") != item["calibration_sha256"]:
                    raise ValueError("执行期间权重或阈值变化，保留日志但不提交成功结果")
            for item in result["models"]:
                for path, digest in item["parameter_files_sha256"].items():
                    if sha(path) != digest:
                        raise ValueError(f"提交前参数或标签定义身份变化：{path}")
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
    parser.add_argument("--run", action="store_true", help="按冻结协议执行一次全十二模型双标签版本基准；已有结果或尝试日志时拒绝重跑")
    args = parser.parse_args()
    if args.run:
        execute()
    else:
        print(PROTOCOL.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
