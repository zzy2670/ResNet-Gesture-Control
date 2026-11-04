"""合并数据独立核验：只读来源，逐行重算标签与照片命令，不启动GPU。"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
SPLITS = ("train", "val", "test")
ARCHES = ("resnet18", "resnet34", "resnet50")
OLD = {"peace": 1, "peace_inverted": 1, "stop": 2, "stop_inverted": 2,
       "palm": 2, "fist": 3, "like": 4, "dislike": 5}
NEW = {**OLD, "two_up": 1, "two_up_inverted": 1}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    # 本审只重核小清单与派生缓存，不重读全部原始照片。
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def aggregate(labels):
    # 独立实现停止优先及动作冲突拒识，逐图核真实分母。
    active = set(labels) - {0}
    return 3 if 3 in active else (next(iter(active)) if len(active) == 1 else 0)


def counts(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {label: {"hands": len(items), "photos": len({r["image_id"] for r in items}),
                    "users": len({r["user_id"] for r in items})}
            for label, items in sorted(grouped.items())}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    report = {"status": "PASS_DATA_PREFLIGHT", "reviewed_at": datetime.now().astimezone().isoformat(),
              "reviewer": "merged_independent_preflight", "gpu_used": False,
              "scope": "独立全量索引/冻结特征/标签/照片命令核验，非训练结果验收", "protocols": {}}
    bound = {str(Path(__file__).resolve()): sha(__file__)}
    for protocol in ("A", "B"):
        original = ROOT / "runs" / f"{protocol}_20261102_TPE"
        split_root = ROOT / "data" / f"{protocol}_split_verified_20261102"
        if protocol == "A":
            split_root /= "P1_random_image"
        old_audit = read(original / "data_audit.json")
        old_preflight = read(original / "preflight_independent_review.json")
        bound[str(original / "preflight_independent_review.json")] = sha(original / "preflight_independent_review.json")
        data, users, ids = {}, {}, {}
        protocol_report = {"splits": {}, "models": {}}
        for split in SPLITS:
            manifest = split_root / split / "manifest.csv"
            digest = sha(manifest)
            bound[str(manifest)] = digest
            require(digest == old_audit["splits"][split]["manifest_sha256"], f"旧{protocol}/{split}manifest已变")
            if protocol == "B":
                require(digest == old_preflight["manifests"][split]["sha256"], "B旧全raw审计manifest不符")
            with manifest.open(encoding="utf-8", newline="") as handle:
                photos = list(csv.DictReader(handle))
            annotations = {}
            for file in sorted((split_root / split / "annotations").glob("*.json")):
                bound[str(file)] = sha(file)
                annotations[file.stem] = read(file)
            old = read(original / f"crops_{split}.json")
            new, truths = [], []
            cursor = 0
            for photo in photos:
                ann = annotations[photo["class_name"]][photo["image_id"]]
                require(ann["user_id"] == photo["user_id"], "annotation user不符")
                require(len(ann["labels"]) == len(ann["bboxes"]), "annotation框标签不符")
                for hand_index, raw in enumerate(ann["labels"]):
                    row = old[cursor]
                    require((row["image_id"], row["hand_index"], row["user_id"], row["raw_label"]) ==
                            (photo["image_id"], hand_index, photo["user_id"], raw), "旧索引与原annotation次序不符")
                    require(row["source_path"] == str(split_root / split / photo["image_path"]), "旧图路径不符")
                    require(row["label"] == OLD.get(raw, 0) and Path(row["path"]).is_file(), "旧label/裁图缺失")
                    new.append({**row, "label": NEW.get(raw, 0)})
                    cursor += 1
                old_labels = [OLD.get(raw, 0) for raw in ann["labels"]]
                new_labels = [NEW.get(raw, 0) for raw in ann["labels"]]
                truths.append({"image_id": photo["image_id"], "user_id": photo["user_id"],
                    "raw_labels": ann["labels"], "old_labels": old_labels, "labels": new_labels,
                    "old_truth": aggregate(old_labels), "truth": aggregate(new_labels)})
            require(cursor == len(old), "原手索引未全覆盖")
            users[split] = {p["user_id"] for p in photos}
            ids[split] = {p["image_id"] for p in photos}
            require(len(ids[split]) == len(photos), "manifest重复photo")
            stat = {"photos": len(photos), "hands": len(old), "users": len(users[split]),
                "changed_hands": sum(a["label"] != b["label"] for a, b in zip(old, new)),
                "changed_photo_commands": sum(t["old_truth"] != t["truth"] for t in truths),
                "raw": counts(old, "raw_label"), "old_six_class": counts(old, "label"),
                "new_six_class": counts(new, "label"),
                "old_photo_commands": {str(k): v for k, v in Counter(t["old_truth"] for t in truths).items()},
                "new_photo_commands": {str(k): v for k, v in Counter(t["truth"] for t in truths).items()},
                "raw_photo_folder_classes": dict(Counter(p["class_name"] for p in photos))}
            require(stat["changed_hands"] == sum(r["raw_label"] in ("two_up", "two_up_inverted") for r in old), "改变raw范围不符")
            data[split] = {"old": old, "new": new, "truths": truths, "statistics": stat}
            protocol_report["splits"][split] = {"manifest_sha256": digest, **stat}
        protocol_report["overlaps"] = {}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            require(not ids[a] & ids[b], "照片跨组")
            if protocol == "B":
                require(not users[a] & users[b], "B user跨组")
            protocol_report["overlaps"][f"{a}_{b}"] = {"photo_ids": len(ids[a] & ids[b]), "users": len(users[a] & users[b])}
        for arch in ARCHES:
            source = original if arch == "resnet18" else original / f"compare_{arch}"
            target = RUN / protocol / arch
            identity = read(target / "merged_data_identity.json")
            require(identity["protocol"] == protocol and identity["architecture"] == arch, "模型身份错误")
            require(identity["mapping"] == NEW and identity["old_mapping"] == OLD, "映射错误")
            require(Path(identity["source_run"]) == source and Path(identity["source_split_root"]) == split_root, "来源路径错误")
            require(identity["split_reassignment"] is False and identity["new_feature_extraction"] is False, "禁止重新划分/提取")
            require(read(target / "label_statistics.json") == {s: data[s]["statistics"] for s in SPLITS}, "统计并非真实分母")
            model = {"source_run": str(source), "features": {}}
            for split in SPLITS:
                index = target / f"crops_{split}.json"
                require(read(index) == data[split]["new"], f"{protocol}/{arch}/{split}新索引非label字段或label错误")
                require(read(target / f"photo_truth_{split}.json") == data[split]["truths"], "照片真值错误")
                require(sha(source / f"crops_{split}.json") == sha(original / f"crops_{split}.json"), "比较架构来源次序错误")
                require(identity["splits"][split]["new_index_sha256"] == sha(index), "新索引身份错误")
                for path in (index, target / f"photo_truth_{split}.json"):
                    bound[str(path)] = sha(path)
            official = Path(identity["official_pretrained"]["path"])
            official_sha = sha(official)
            require(official_sha == identity["official_pretrained"]["sha256"], "官方weights改变")
            require(official_sha.startswith({"resnet18": "f37072fd", "resnet34": "b627a593", "resnet50": "0676ba61"}[arch]), "官方weights来源错误")
            bound[str(official)] = official_sha
            for split in ("train", "val"):
                old_path, new_path = source / f"features_{split}.pt", target / f"features_{split}.pt"
                old_cache = torch.load(old_path, map_location="cpu", weights_only=True)
                new_cache = torch.load(new_path, map_location="cpu", weights_only=True)
                require(set(old_cache) == set(new_cache) == {"x", "y", "pretrained_sha256"}, "cache键错误")
                require(old_cache["pretrained_sha256"] == new_cache["pretrained_sha256"] == official_sha, "cache不是该官方初始化")
                expected_shape = (len(data[split]["old"]), 2048 if arch == "resnet50" else 512)
                require(tuple(new_cache["x"].shape) == tuple(old_cache["x"].shape) == expected_shape, "cache shape错误")
                require(torch.equal(old_cache["x"], new_cache["x"]), "features像素表示发生改变")
                require(torch.isfinite(new_cache["x"]).all().item(), "特征非有限")
                for kind, cache in (("old", old_cache), ("new", new_cache)):
                    require(torch.equal(cache["y"], torch.tensor([r["label"] for r in data[split][kind]], dtype=cache["y"].dtype)), "cache labels次序错误")
                old_sha, new_sha = sha(old_path), sha(new_path)
                declared = identity["feature_caches"][split]
                require(declared["source_sha256"] == old_sha and declared["new_sha256"] == new_sha, "cache声明SHA错误")
                bound[str(old_path)], bound[str(new_path)] = old_sha, new_sha
                model["features"][split] = {"shape": list(expected_shape), "all_x_elements_equal": True,
                    "old_and_new_y_match_indexes_in_order": True, "source_sha256": old_sha, "new_sha256": new_sha}
                del old_cache, new_cache
            for name in ("merged_data_identity.json", "label_statistics.json", "prepare_merged_log.json"):
                bound[str(target / name)] = sha(target / name)
            require(not any((target / name).exists() for name in ("classifier_best.pt", "fixed_head_summary.json", "calibration.json", "test_whole_photo.json")), "预审前已训练/校准/test")
            model["all_nonlabel_fields_preserved"] = True
            model["new_training_or_test_absent"] = True
            protocol_report["models"][arch] = model
        report["protocols"][protocol] = protocol_report
    report["bound_file_sha256"] = bound
    report["old_full_raw_hash_audits_reused"] = True
    report["all_new_indexes_compared_and_all_feature_elements_compared"] = True
    # 写前复核本次实际审核的依赖，报告仅证明当前固定字节。
    payload = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    for path, digest in bound.items():
        require(sha(path) == digest, f"核验期间依赖改变：{path}")
    destination = RUN / "preflight_data_independent_review.json"
    with destination.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": report["status"], "path": str(destination), "sha256": sha(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
