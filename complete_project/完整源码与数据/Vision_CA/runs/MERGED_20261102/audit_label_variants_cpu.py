"""十二模型CPU结果独立数值审计；默认仅显示准备状态，--run不做模型推理。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import traceback

RUN = Path(__file__).resolve().parent
ROOT = RUN.parent.parent
DATA = RUN / "label_variants_cpu_benchmark.json"
JOURNAL = RUN / "label_variants_cpu_benchmark.attempt.jsonl"
OUTPUT = RUN / "label_variants_cpu_independent_review.json"
REFERENCE = ROOT / "runs/B_20261102_TPE/matched_cpu_benchmark.json"
REFERENCE_SHA = "23f8dd1e2b08ac7e9e2f4f1618534212a209f867410a54873663555a6e3fba68"
KEYS = [f"{v}_{p}_{a}" for a in ("resnet18", "resnet34", "resnet50")
        for p in ("A", "B") for v in ("unmerged", "merged")]
ORDERS = [KEYS, list(reversed(KEYS)), KEYS[6:] + KEYS[:6]]
SCOPES = ["classifier_only_preprocessed_224x224_tensor",
          "whole_photo_decode_detection_crop_classification_rules"]
PARAMETERS = {"resnet18": 11179590, "resnet34": 21287750, "resnet50": 23520326}
OLD_MAPPING = {"peace": 1, "peace_inverted": 1, "stop": 2, "stop_inverted": 2,
               "palm": 2, "fist": 3, "like": 4, "dislike": 5}
MAPPINGS = {"unmerged": OLD_MAPPING, "merged": {**OLD_MAPPING, "two_up": 1, "two_up_inverted": 1}}
LIMITATIONS = [
    "Independent JSON numerical/identity audit only; no inference or timing is executed.",
    "Tensor SHA is checked against the frozen old reference and producer evidence; tensors are not reconstructed here.",
    "Journal and pinned producer source establish recorded readiness/load order, not external attestation.",
    "Windows system silence, thermal state and scheduler isolation are not proved; all 72 background records are retained.",
    "Timing repetitions are 100 fixed inputs repeated three times, not 300 independent photos or a new accuracy test.",
    "Own frozen thresholds retain real workload differences; timing differences do not prove causal acceleration."
]


def sha(path):
    # 只读流式哈希，不反序列化权重。
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    # 允许numpy与stdlib求和的舍入差，不允许实际统计差异。
    return (type(a) in (int, float) and type(b) in (int, float)
            and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9))


def percentile(values, q):
    # 对应numpy默认linear分位数。
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(values):
    require(len(values) >= 2 and all(type(x) in (int, float) and math.isfinite(x) and x > 0
                                   for x in values), "invalid raw positive finite timings")
    return {"n": len(values), "mean_ms": statistics.mean(values), "p50_ms": percentile(values, .5),
            "p95_ms": percentile(values, .95), "sd_ms": statistics.stdev(values)}


def timestamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "timestamp lacks timezone")
    return result


class Audit:
    def __init__(self):
        self.bound = {}
        self.stats = []
        self.background = []

    def bind(self, path, expected=None):
        # 同一路径的多个声明必须一致，随后核实际字节。
        path = str(Path(path).resolve())
        digest = sha(path)
        require(expected is None or digest == expected, f"live SHA mismatch: {path}")
        require(path not in self.bound or self.bound[path] == digest, f"conflicting SHA: {path}")
        self.bound[path] = digest
        return digest

    def bind_map(self, values):
        require(isinstance(values, dict) and values, "empty binding map")
        for path, digest in values.items():
            self.bind(path, digest)

    def scientific(self, recorded):
        # 独立核真实逐模型证据，不能只相信汇总status。
        complete_path = RUN / "all_training_evaluation_complete.json"
        aggregate_path = RUN / "aggregate_review.json"
        self.bind(complete_path)
        self.bind(aggregate_path)
        complete, aggregate = read(complete_path), read(aggregate_path)
        require(complete.get("status") == "PASS", "scientific completion absent")
        require(aggregate.get("status") == "PASS_MERGED_SIX_MODEL_AUDIT_WITH_LIMITATIONS"
                and aggregate.get("issues") == [], "scientific aggregate failed")
        summaries = {(m["protocol"], m["architecture"]): m for m in aggregate["models"]}
        identities = {(p, a) for p in ("A", "B") for a in PARAMETERS}
        require(len(aggregate["models"]) == 6 and set(summaries) == identities, "scientific six identities")
        expected_reports = []
        self.bind_map(aggregate["bound_file_sha256"])
        for p, a in sorted(identities):
            folder = RUN / p / a
            path = folder / "model_review.json"
            report, cal = read(path), read(folder / "calibration.json")
            digest = aggregate["bound_file_sha256"][str(path)]
            self.bind(path, digest)
            require(report.get("status") == "PASS_MERGED_MODEL_AUDIT_WITH_LIMITATIONS"
                    and report.get("issues") == [] and report.get("protocol") == p
                    and report.get("architecture") == a, f"scientific report {p}/{a}")
            for name in ("classifier_best.pt", "detector_best.pt", "calibration.json", "val_whole_raw.json",
                         "test_oracle_crop.json", "test_whole_photo.json", "test_detector.json"):
                require(str(folder / name) in report["bound_file_sha256"], f"unbound scientific file: {name}")
            self.bind_map(report["bound_file_sha256"])
            require(report["weights"] == cal["weights"] == summaries[p, a]["weights"], "scientific weights")
            require(report["calibration_selected"] == cal["selected"] == summaries[p, a]["calibration_selected"],
                    "scientific frozen thresholds")
            for name, weight_sha in report["weights"].items():
                self.bind(folder / name, weight_sha)
            expected_reports.append({"protocol": p, "architecture": a, "path": str(path),
                                     "sha256": digest, "status": report["status"]})
        require(recorded.get("status") == "PASS" and recorded["model_reports"] == expected_reports,
                "benchmark recorded readiness identities")
        require(recorded["completion_path"] == str(complete_path)
                and recorded["aggregate_path"] == str(aggregate_path)
                and recorded["aggregate_status"] == aggregate["status"], "readiness source paths/status")
        self.bind_map(recorded["bound_file_sha256"])
        required = {str(complete_path), str(aggregate_path), *aggregate["bound_file_sha256"]}
        for p, a in identities:
            required.update(read(RUN / p / a / "model_review.json")["bound_file_sha256"])
        require(required <= set(recorded["bound_file_sha256"]), "readiness omitted scientific bindings")

    def background_block(self, key, scope, block):
        # 完整保留前后CPU原计数，重算估计值，不按背景负载丢弃块。
        bg = block["cpu_background"]
        before, after = bg["before"], bg["after"]
        for snap in (before, after):
            timestamp(snap["time"])
            require(snap["memory_available_bytes"] > 0 and snap["process_rss_bytes"] > 0
                    and snap["process_count"] > 0, "missing memory/process evidence")
        require(set(before["system_cpu_times"]) == set(after["system_cpu_times"]), "CPU counter schema")
        delta = {k: after["system_cpu_times"][k] - v for k, v in before["system_cpu_times"].items()}
        require(all(math.isfinite(v) and v >= -1e-8 for v in delta.values()), "CPU counter reversal")
        total = sum(delta.values())
        idle = delta.get("idle", 0) + delta.get("iowait", 0)
        own = sum(after["process_cpu_times"][k] - before["process_cpu_times"][k] for k in ("user", "system"))
        elapsed = after["monotonic"] - before["monotonic"]
        require(elapsed > 0 and own >= 0 and total > 0, "CPU elapsed/counter evidence")
        calculated = {"system_busy_percent": 100 * (total-idle) / max(total, 1e-9),
                      "process_cpu_seconds": own, "elapsed_seconds": elapsed,
                      "estimated_other_process_busy_percent": 100 * max(0, total-idle-own) / max(total, 1e-9)}
        require(all(close(bg[k], v) for k, v in calculated.items()), "background formula mismatch")
        require(sum(block["all_ms"]) / 1000 <= elapsed + .01, "timings exceed measured block elapsed")
        self.background.append({"model": key, "scope": scope, "repeat": block["repeat"],
                                "cpu_background": bg, "recomputed": calculated})

    def inspect(self):
        self.bind(Path(__file__))
        data_sha = self.bind(DATA)
        d = read(DATA)
        require(d.get("status") == "complete", "benchmark not complete")
        self.bind(REFERENCE, REFERENCE_SHA)
        reference = read(REFERENCE)
        old_review_path = REFERENCE.with_name("matched_cpu_independent_review.json")
        self.bind(old_review_path)
        old_review = read(old_review_path)
        require(old_review.get("status") == "PASS" and old_review.get("issues") == []
                and old_review["data_sha256"] == REFERENCE_SHA, "old benchmark independent audit")
        self.bind_map(old_review["sourceidentity"]["verified_file_sha256"])
        require(reference["status"] == "complete" and d["reference_benchmark"]["path"] == str(REFERENCE)
                and d["reference_benchmark"]["sha256"] == REFERENCE_SHA, "old reference identity")
        preflight_path = RUN / "label_variants_cpu_preflight_review.json"
        self.bind(preflight_path)
        preflight = read(preflight_path)
        require(preflight["status"] == "PASS_CPU_BENCHMARK_SOURCE_PREFLIGHT_WITH_LIMITATIONS"
                and preflight["issues"] == [], "producer source preflight")
        self.bind_map(preflight["bound_file_sha256"])
        self.bind(ROOT / "benchmark_label_variants_cpu.py", d["script_sha256"])
        self.bind(RUN / "LABEL_VARIANTS_CPU_PROTOCOL.md", d["protocol_sha256"])
        for name, digest in d["source_sha256"].items():
            self.bind(ROOT / name, digest)
        require(set(d["source_sha256"]) == {"train_A.py", "compare_resnets_A.py", "deployment_src/gesture_predict.py"},
                "producer source coverage")
        self.bind_map(d["input_index_sha256"])
        self.bind_map(reference["input_index_sha256"])
        require(set(reference["input_index_sha256"]) <= set(d["input_index_sha256"]), "input index coverage")
        for field, key_name in (("photos", "image_id"), ("classifier_inputs", "sample_key")):
            require(d[field] == reference[field] and len(d[field]) == 100
                    and len({x[key_name] for x in d[field]}) == 100, f"fixed input order/keys: {field}")
        for row in d["photos"]:
            self.bind(row["path"], row["source_sha256"])
        for row in d["classifier_inputs"]:
            self.bind(row["path"], row["crop_sha256"])
            self.bind(row["source_path"], row["source_sha256"])
            require(len(row["preprocessed_tensor_sha256"]) == 64
                    and all(c in "0123456789abcdef" for c in row["preprocessed_tensor_sha256"]), "tensor SHA format")
        fixed = {"repeat_orders": ORDERS, "scopes": SCOPES, "batch_size": 1, "dtype": "float32", "device": "cpu",
                 "threads": 8, "warmup_per_model_per_scope": 20, "repeats": 3, "unique_inputs_per_scope": 100,
                 "selection_split": "A_val", "accuracy_recomputed": False, "used_to_select_weights_or_thresholds": False,
                 "model_loading_included": False, "classifier_preprocessing_included": False,
                 "whole_photo_preprocessing_included": True, "label_mappings": MAPPINGS,
                 "whole_photo_call": "train_A.whole_photo_predictions(..., rows=[row], batch_size=1)"}
        require(all(d.get(k) == v for k, v in fixed.items()), "fixed execution protocol mismatch")
        self.scientific(d["scientific_readiness"])
        self.check_truth(d)
        require(len(d["models"]) == 12 and [m["key"] for m in d["models"]] == KEYS, "12 unique ordered identities")
        old_models = {m["key"]: m for m in reference["models"]}
        for m in d["models"]:
            variant, protocol, arch = m["key"].split("_", 2)
            old_folder = ROOT / f"runs/{protocol}_20261102_TPE"
            if arch != "resnet18":
                old_folder /= "compare_" + arch
            folder = old_folder if variant == "unmerged" else RUN / protocol / arch
            require(m["label_variant"] == variant and m["protocol"] == protocol and m["architecture"] == arch
                    and Path(m["run_folder"]).resolve() == folder.resolve(), "model folder/identity")
            require(m["output_classes"] == 6 and m["classifier_parameters"] == PARAMETERS[arch], "architecture parameter count")
            self.bind(folder / "calibration.json", m["calibration_sha256"])
            cal = read(folder / "calibration.json")
            require(cal["weights"] == m["weights"] and cal["selection_split"] == "val" and cal["test_accessed"] is False,
                    "calibration identity")
            require(cal["selected"]["detector_threshold"] == m["detector_threshold"]
                    and cal["selected"]["class_threshold"] == m["classifier_threshold"], "frozen thresholds")
            require(set(m["weights"]) == {"classifier_best.pt", "detector_best.pt"}, "two checkpoint identities")
            for name, digest in m["weights"].items():
                self.bind(folder / name, digest)
            old = old_models[f"{protocol}_{arch}"]
            require(m["weights"]["detector_best.pt"] == old["weights"]["detector_best.pt"], "detector reuse identity")
            require(variant != "unmerged" or m["weights"] == old["weights"], "old classifier changed")
            expected_parameters = {str(old_folder / "tpe_summary.json")}
            if variant == "merged":
                expected_parameters.update(str(folder / n) for n in ("merged_input_identity.json", "merged_data_identity.json",
                                                                    "fixed_head_summary.json", "finetune_summary.json"))
                identity = read(folder / "merged_data_identity.json")
                require(identity["mapping"] == MAPPINGS["merged"] and identity["old_mapping"] == OLD_MAPPING, "merged mapping")
            require(set(m["parameter_files_sha256"]) == expected_parameters, "parameter source coverage")
            self.bind_map(m["parameter_files_sha256"])
            require(isinstance(m["classifier_params"], dict) and m["classifier_params"]
                    == read(old_folder / "tpe_summary.json")["best_params"], "classifier params differ from fixed TPE selection")
            self.measurements(m, d["photos"])
        self.check_journal(d, data_sha)
        return d, data_sha

    def check_truth(self, d):
        # 从原始val标注独立重算元数据，只核语义，不计算准确率。
        audit_path = ROOT / "runs/A_20261102_TPE/data_audit.json"
        split = Path(read(audit_path)["split_root"]) / "val"
        self.bind(audit_path)
        self.bind(split / "manifest.csv")
        with (split / "manifest.csv").open(encoding="utf-8", newline="") as f:
            manifest = {r["image_id"]: r for r in csv.DictReader(f)}
        expected = {v: [] for v in MAPPINGS}
        annotations = {}
        for photo in d["photos"]:
            ident = photo["image_id"]
            entry = manifest[ident]
            path = split / "annotations" / (entry["class_name"] + ".json")
            if path not in annotations:
                self.bind(path)
                annotations[path] = read(path)
            raw = annotations[path][ident]["labels"]
            require(Path(photo["path"]).resolve() == (split / entry["image_path"]).resolve(), "manifest photo path")
            for variant, mapping in MAPPINGS.items():
                labels = [mapping.get(x, 0) for x in raw]
                active = set(labels) - {0}
                truth = 3 if 3 in labels else next(iter(active)) if len(active) == 1 else 0
                expected[variant].append({"image_id": ident, "raw_labels": raw, "labels": labels, "truth": truth})
        require(d["photo_truth_metadata"] == expected, "raw label/truth metadata")

    def measurements(self, m, photos):
        require(set(m["measurements"]) == set(SCOPES) and set(m["summary"]) == set(SCOPES), "scope coverage")
        for scope in SCOPES:
            blocks = m["measurements"][scope]
            require(len(blocks) == 3 and [b["repeat"] for b in blocks] == [1, 2, 3], "three complete repeats")
            values, means = [], []
            for b in blocks:
                require(len(b["all_ms"]) == 100, "100 raw timings per block")
                calculated = summarize(b["all_ms"])
                require(close(b["mean_ms"], calculated["mean_ms"]), "block mean")
                require(timestamp(b["started_at"]) <= timestamp(b["ended_at"]), "block timestamps")
                if scope == SCOPES[0]:
                    require(b["workloads"] == [], "classifier workload field")
                else:
                    require(len(b["workloads"]) == 100 and [x["image_id"] for x in b["workloads"]]
                            == [p["image_id"] for p in photos], "whole-photo input order")
                    for row in b["workloads"]:
                        boxes, accepted = row["detected_boxes_after_frozen_threshold"], row["accepted_hands"]
                        require(type(boxes) is int and type(accepted) is int and 0 <= accepted <= boxes, "whole workload counts")
                self.background_block(m["key"], scope, b)
                values.extend(b["all_ms"])
                means.append(calculated["mean_ms"])
            calculated = summarize(values)
            summary = m["summary"][scope]
            require(calculated["n"] == 300 and all(close(summary[k], v) for k, v in calculated.items()), "summary statistics")
            require(summary["sd_definition"] == "sample standard deviation ddof=1"
                    and len(summary["repeat_means_ms"]) == 3
                    and all(close(a, b) for a, b in zip(summary["repeat_means_ms"], means)), "summary repeat means/ddof")
            self.stats.append({"key": m["key"], "scope": scope, **calculated, "repeat_means_ms": means})

    def check_journal(self, d, data_sha):
        # 日志给出实际事件顺序，并与结果逐块精确对应。
        self.bind(JOURNAL)
        events = [json.loads(line) for line in JOURNAL.read_text(encoding="utf-8").splitlines() if line.strip()]
        require(events and events[0]["status"] == "started" and events[-1]["status"] == "complete"
                and not any(e["status"] == "failed" for e in events), "journal completion")
        require(events[0]["script_sha256"] == d["script_sha256"]
                and events[0]["protocol_sha256"] == d["protocol_sha256"], "journal source identity")
        require(events[-1]["output_sha256"] == data_sha and events[-1]["output"] == str(DATA), "journal output identity")
        require(all(timestamp(a["time"]) <= timestamp(b["time"]) for a, b in zip(events, events[1:])), "event time reversal")
        require(timestamp(d["started_at"]) <= timestamp(events[0]["time"])
                <= timestamp(d["ended_at"]) <= timestamp(events[-1]["time"]), "execution start/end chronology")
        expected_statuses = ["started", "scientific_readiness_pass", "dependency_preflight_pass"] + ["model_loaded"] * 12
        for scope in SCOPES:
            expected_statuses += ["warmup_complete"] * 12 + ["block_complete"] * 36
        expected_statuses += ["complete"]
        require([e["status"] for e in events] == expected_statuses, "readiness/dependency/load/warmup/block event sequence")
        require(events[1]["model_reports"] == d["scientific_readiness"]["model_reports"], "readiness event identity")
        dep = events[2]
        for name in ("python", "executable", "torch", "torchvision", "psutil"):
            require(dep[name] == d["environment"][name], "actual dependency event mismatch")
        require(dep["psutil"] == "7.0.0", "psutil pinned version")
        require(d["memory_budget"]["minimum_available_before_load_gib"] == 6
                and d["memory_budget"]["before_load"]["memory_available_bytes"] >= 6 * 1024**3,
                "preload memory gate")
        require(timestamp(d["memory_budget"]["before_load"]["time"]) <= timestamp(events[3]["time"])
                and timestamp(d["memory_budget"]["after_load"]["time"]) >= timestamp(events[14]["time"]), "load memory chronology")
        models = {m["key"]: m for m in d["models"]}
        for e, key in zip(events[3:15], KEYS):
            require(e["key"] == key and e["weights"] == models[key]["weights"] and e["memory"]["process_rss_bytes"] > 0,
                    "model load identity/memory")
        offset = 15
        last_end = None
        for scope in SCOPES:
            for e, key in zip(events[offset:offset+12], KEYS):
                require(e["key"] == key and e["scope"] == scope and e["count"] == 20, "20 warmup evidence")
            offset += 12
            for repeat, order in enumerate(ORDERS):
                for key in order:
                    event = events[offset]
                    block = models[key]["measurements"][scope][repeat]
                    require(event["key"] == key and event["scope"] == scope
                            and all(event[k] == value for k, value in block.items()), "block journal exact identity/order")
                    start, end = timestamp(block["started_at"]), timestamp(block["ended_at"])
                    require(last_end is None or start >= last_end, "block execution order")
                    last_end = end
                    offset += 1
        require(offset == len(events)-1 and len(self.background) == 72, "72 blocks / 7200 raw values")


def run():
    # 缺真实输入只等待，不占用正式审核文件，也不创建PASS。
    if not DATA.exists():
        print(json.dumps({"status": "WAITING", "missing": str(DATA), "inference_executed": False}))
        return 2
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {OUTPUT}")
    audit = Audit()
    report = {"status": "FAIL", "reviewed_at": datetime.now().astimezone().isoformat(),
              "scope": "independent numerical and live identity audit; no model loading/inference",
              "issues": [], "limitations": LIMITATIONS, "data_sha256": None,
              "audit_source_sha256": sha(__file__), "bound_file_sha256": audit.bound}
    try:
        d, digest = audit.inspect()
        report.update(status="PASS", data_sha256=digest, checks={"models": 12, "blocks": 72, "raw_timings": 7200,
                      "recomputed_summaries": 24, "no_block_dropped": True, "readiness_before_load": True},
                      recomputed=audit.stats, cpu_background_all_blocks=audit.background,
                      execution_started_at=d["started_at"], execution_ended_at=d["ended_at"], environment=d["environment"])
    except BaseException as error:
        report["issues"].append(repr(error))
        report["traceback"] = traceback.format_exc()
        report["data_sha256"] = audit.bound.get(str(DATA.resolve()))
    # 先物化字节，再复核每项活绑定；不重编码成功数据后绕过检查。
    payload = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if report["status"] == "PASS":
        try:
            for path, digest in audit.bound.items():
                require(sha(path) == digest, f"precommit dependency changed: {path}")
        except BaseException as error:
            report["status"] = "FAIL"
            report["issues"].append(repr(error))
            report["traceback"] = traceback.format_exc()
            payload = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with OUTPUT.open("xb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"status": report["status"], "issues": report["issues"], "output": str(OUTPUT), "sha256": sha(OUTPUT)}))
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="只审计唯一真实结果JSON及活身份，不加载模型，不测速，不覆盖已有审核")
    args = parser.parse_args()
    if args.run:
        return run()
    print(json.dumps({"status": "PREPARED" if DATA.exists() else "WAITING", "data": str(DATA), "output": str(OUTPUT),
                      "command": "python audit_label_variants_cpu.py --run", "inference_executed": False,
                      "note": "Preparation is not a PASS; --run audits numerical/identity evidence only."}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
