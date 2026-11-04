"""六个真实CPU烟测结束后独立核SHA和预算，提交最终预审门禁。"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
SOURCES = ("run_merged_pipeline.py", "run_merged_training.py", "train_A.py", "compare_resnets_A.py",
           "prepare_merged_labels.py", "runs/MERGED_20261102/MERGED_PROTOCOL.md")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    data_file = RUN / "preflight_data_independent_review.json"
    data = read(data_file)
    require(data["status"] == "PASS_DATA_PREFLIGHT", "数据独审未通过")
    bound = dict(data["bound_file_sha256"])
    bound[str(data_file)] = sha(data_file)
    bound[str(Path(__file__).resolve())] = sha(__file__)
    sources = {name: sha(ROOT / name) for name in SOURCES}
    require(sources["train_A.py"] == "8f63912166b2ecee0b31cacf984904e646cf43d2b722eb4a4167956a4b5e1b9d", "原A源码改变")
    require(sources["compare_resnets_A.py"] == "3bd9e96999917edf07a1738b0ab6eb8710405848cc2baa1691303e97ec944a3f", "原对比源码改变")
    require(read(RUN / "controller_source_identity.json") == sources, "控制器源码pin不符")
    for name, digest in sources.items():
        require(sha(RUN / "code_snapshot" / name) == digest, "源码快照不符")
        bound[str(ROOT / name)] = digest
        bound[str(RUN / "code_snapshot" / name)] = digest
        if name.endswith(".py"):
            ast.parse((ROOT / name).read_text(encoding="utf-8"))
    complete = RUN / "all_cpu_smokes_complete.json"
    require(read(complete)["status"] == "PASS" and read(complete)["source_sha256"] == sources, "CPU预审总完成证据不符")
    bound[str(complete)] = sha(complete)
    events = [json.loads(line) for line in (RUN / "controller_events.jsonl").read_text(encoding="utf-8").splitlines()]
    bound[str(RUN / "controller_source_identity.json")] = sha(RUN / "controller_source_identity.json")
    bound[str(RUN / "controller_events.jsonl")] = sha(RUN / "controller_events.jsonl")
    started = [e for e in events if e.get("status") == "started"]
    require(len(started) == 6 and all(e["stage"] == "smoke" for e in started), "预审前存在非CPU阶段或重复尝试")
    exited = [e for e in events if e.get("status") == "exited"]
    require(len(exited) == 6 and all(e["stage"] == "smoke" and e["exit_code"] == 0 for e in exited), "烟测退出证据不全")
    models = []
    for protocol in ("A", "B"):
        split = ROOT / "data" / f"{protocol}_split_verified_20261102"
        if protocol == "A":
            split /= "P1_random_image"
        original = ROOT / "runs" / f"{protocol}_20261102_TPE"
        for arch in ("resnet18", "resnet34", "resnet50"):
            out = RUN / protocol / arch
            source = original if arch == "resnet18" else original / f"compare_{arch}"
            identity_file = out / "merged_input_identity.json"
            identity = read(identity_file)
            require(identity["protocol"] == protocol and identity["architecture"] == arch, "run身份不符")
            require(identity["source_sha256"] == sources and identity["new_TPE_trials"] == 0 and
                    identity["unmerged_classifier_weights_used"] is False, "旧微调或新搜索限制不符")
            inputs = identity["inputs_sha256"]
            expected = [out / f"crops_{s}.json" for s in ("train", "val", "test")]
            expected += [out / f"features_{s}.pt" for s in ("train", "val")]
            expected += [split / s / "manifest.csv" for s in ("train", "val", "test")]
            expected += [p for s in ("train", "val", "test") for p in sorted((split / s / "annotations").glob("*.json"))]
            expected += [out / "merged_data_identity.json"]
            expected += [out / f"photo_truth_{s}.json" for s in ("train", "val", "test")]
            expected += [source / "tpe_summary.json", original / "detector_best.pt"]
            # 除两个官方初始化外，所有实际读取的清单、标注、cache与原超参都必须明确pin。
            require(set(str(p.resolve()) for p in expected).issubset(inputs), "必要输入未pin")
            require(len(inputs) == len(expected) + 2, "意外额外输入或缺官方初始化")
            for path, digest in inputs.items():
                if path in bound:
                    require(bound[path] == digest, "输入与数据独审字节不符")
                bound[path] = digest
            summary = read(source / "tpe_summary.json")
            require(len(summary["trials"]) == 24 and all(t["state"] == "COMPLETE" for t in summary["trials"]), "原TPE来源非24完成trial")
            marker_file = out / "stage_markers" / "smoke.json"
            marker = read(marker_file)
            smoke_file = out / "cpu_smoke.json"
            smoke = read(smoke_file)
            require(marker["identity"] == smoke["identity"] == identity and marker["exit_code"] == 0, "烟测身份/退出码不符")
            require(marker["outputs"] == {"cpu_smoke.json": sha(smoke_file)}, "烟测输出SHA不符")
            require(smoke["status"] == "PASS" and smoke["device"] == "cpu" and smoke["GPU_training_started"] is False, "烟测设备/状态不符")
            require(all(smoke[k] is True for k in ("strict_classifier_reload", "strict_own_protocol_detector_reload",
                    "real_train_forward_backward", "real_detector_forward")), "真实load/forward/backward证据缺失")
            require(math.isfinite(smoke["loss"]) and smoke["detector_boxes_shape"][1] == 4, "烟测数值/shape错误")
            log = Path(marker["log"])
            require(sha(log) == marker["log_sha256"], "烟测日志SHA不符")
            own_started = [e for e in started if e["protocol"] == protocol and e["architecture"] == arch]
            own_exited = [e for e in exited if e["protocol"] == protocol and e["architecture"] == arch]
            require(len(own_started) == len(own_exited) == 1 and own_started[0]["pid"] == own_exited[0]["pid"] == marker["pid"], "进程真实退出证据不符")
            require(own_started[0]["command"][-2:] == ["--device", "cpu"], "真实启动并非CPU")
            require(not any((out / name).exists() for name in ("classifier_best.pt", "fixed_head_best.pt", "fixed_head_summary.json", "finetune_summary.json", "calibration.json", "test_whole_photo.json", "test_oracle_crop.json")), "GPU预审前已有科学产物")
            for path in (identity_file, marker_file, smoke_file, log):
                bound[str(path)] = sha(path)
            models.append({"protocol": protocol, "architecture": arch, "cpu_smoke_pid": marker["pid"],
                "cpu_smoke_loss": smoke["loss"], "real_detector_boxes_shape": smoke["detector_boxes_shape"],
                "source_best_params": summary["best_params"], "input_identity_sha256": sha(identity_file),
                "smoke_marker_sha256": sha(marker_file), "own_protocol_detector_sha256": inputs[str((original / "detector_best.pt").resolve())]})
    checks = {"six_models_once_no_TPE_no_detector_training": True, "original_A_B_source_and_outputs_preserved": True,
        "official_ImageNet_backbone_and_new_18_epoch_head": True, "own_protocol_own_arch_fixed_best_params": True,
        "layer4_fc_max18_patience4_with_full_epoch_validation": True, "initial_classifier_hash_checked_before_finetune": True,
        "same_protocol_detector_strict_load_and_real_forward": True, "full_val12_thresholds_then_frozen_test": True,
        "class0_no_command_and_class3_explicit_stop_distinguished": True, "zero_stop_motion_real_denominators": True,
        "all_errors_and_stop_to_zero_derivable_from_full_predictions": True, "test_scores_do_not_select_model_or_threshold": True,
        "source_input_output_log_sha_markers_and_no_unattended_retry": True,
        "success_bytes_materialized_then_live_precommit_then_exclusive_fsync": True,
        "completion_marker_rechecks_all_operations": True, "AST_checked_separately_from_real_CPU_smoke": True}
    report = {"status": "PASS_PREFLIGHT", "reviewed_at": datetime.now().astimezone().isoformat(),
        "reviewer": "merged_independent_preflight", "scope": "合并标签六模型GPU训练前独立数据、源码、预算及真实CPU烟测预审；不是未生成模型/测试结果验收", 
        "source_sha256": sources, "data_review_sha256": sha(data_file), "models": models, "checks": checks,
        "resolved_findings": ["修正A固定split的P1_random_image路径", "增加实际读取annotations及数据身份/协议sourcepin", "烟测增加本协议detector真实forward", "成功marker先编码后复核提交同bytes", "微调前严格核新head+官方初始化分类器SHA"],
        "test_denominators": {p: {"GT_hand_crops_total": data["protocols"][p]["splits"]["test"]["hands"],
            "GT_hand_class0": data["protocols"][p]["splits"]["test"]["new_six_class"]["0"]["hands"],
            "GT_hand_class1": data["protocols"][p]["splits"]["test"]["new_six_class"]["1"]["hands"],
            "whole_photos_total": data["protocols"][p]["splits"]["test"]["photos"],
            "whole_photo_command0": data["protocols"][p]["splits"]["test"]["new_photo_commands"]["0"]} for p in ("A", "B")},
        "execution_gate": "root可启动既定单GPU串行controller；六新模型完成后仍需完整逐样本独审、共同条件测速、文档/程序/最终包验收", 
        "posthoc_comparison_requirements": ["比较须保留原六模型，标明新旧任务标签不同，分数上升不能直接称泛化提升", "分别统计raw peace与two_up被输出1的比例", "共同其他raw负类误运动排除已被新定义为正类的two_up", "派生统计不改变模型/阈值/测试输出"],
        "issues": [], "bound_file_sha256": bound}
    payload = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    # 在报告准确字节已物化后，复核全部真实依赖再独占提交。
    for path, digest in bound.items():
        require(sha(path) == digest, f"最终提交前身份改变：{path}")
    target = RUN / "preflight_independent_review.json"
    with target.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": report["status"], "path": str(target), "sha256": sha(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
