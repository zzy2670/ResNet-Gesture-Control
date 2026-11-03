"""固定原A/B划分和像素，仅重建合并标签及冻结ImageNet特征缓存。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import torch

ROOT = Path(__file__).resolve().parent
OLD_MAPPING = {"peace": 1, "peace_inverted": 1, "stop": 2, "stop_inverted": 2,
               "palm": 2, "fist": 3, "like": 4, "dislike": 5}
NEW_MAPPING = {**OLD_MAPPING, "two_up": 1, "two_up_inverted": 1}
SPLITS = ("train", "val", "test")
ARCHES = ("resnet18", "resnet34", "resnet50")


def sha(path):
    # 分块计算，避免大缓存额外占用内存。
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def commit(path, data):
    # 仅允许创建新文件；重复执行时必须与已有字节一致。
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"拒绝覆盖不同内容：{path}")
        return
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path, value):
    commit(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def aggregate(labels):
    # 与旧训练程序一致：停止优先，多个不同非零动作冲突为0。
    if 3 in labels:
        return 3
    active = set(labels) - {0}
    return next(iter(active)) if len(active) == 1 else 0


def distribution(rows, label_key):
    # 同时保存手数、涉及照片数和涉及人数，不将这三种分母混用。
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[label_key])].append(row)
    return {label: {"hands": len(items), "photos": len({r["image_id"] for r in items}),
                    "users": len({r["user_id"] for r in items})}
            for label, items in sorted(groups.items())}


def prepare_protocol(protocol):
    source_root = ROOT / "runs" / f"{protocol}_20261102_TPE"
    audit = read_json(source_root / "data_audit.json")
    split_root = Path(audit["split_root"])
    result = {}
    seen_photos, user_sets = set(), {}
    for split in SPLITS:
        crop_file = source_root / f"crops_{split}.json"
        old_rows = read_json(crop_file)
        manifest = split_root / split / "manifest.csv"
        with manifest.open(encoding="utf-8", newline="") as stream:
            photos = list(csv.DictReader(stream))
        annotations = {p.stem: read_json(p) for p in (split_root / split / "annotations").glob("*.json")}
        photo_ids = {p["image_id"] for p in photos}
        assert len(photo_ids) == len(photos) and not seen_photos.intersection(photo_ids)
        seen_photos.update(photo_ids)
        user_sets[split] = {p["user_id"] for p in photos}
        expected, photo_truth = [], []
        for photo in photos:
            ann = annotations[photo["class_name"]][photo["image_id"]]
            assert len(ann["labels"]) == len(ann["bboxes"])
            assert ann["user_id"] == photo["user_id"]
            for index, raw in enumerate(ann["labels"]):
                expected.append((photo["image_id"], index, raw, photo["user_id"],
                                 str(split_root / split / photo["image_path"])))
            old_labels = [OLD_MAPPING.get(x, 0) for x in ann["labels"]]
            new_labels = [NEW_MAPPING.get(x, 0) for x in ann["labels"]]
            photo_truth.append({"image_id": photo["image_id"], "user_id": photo["user_id"],
                                "raw_labels": ann["labels"], "old_labels": old_labels,
                                "labels": new_labels, "old_truth": aggregate(old_labels),
                                "truth": aggregate(new_labels)})
        assert len(expected) == len(old_rows)
        new_rows = []
        for row, identity in zip(old_rows, expected):
            actual = (row["image_id"], row["hand_index"], row["raw_label"],
                      row["user_id"], row["source_path"])
            assert actual == identity, (actual, identity)
            assert row["label"] == OLD_MAPPING.get(row["raw_label"], 0)
            assert Path(row["path"]).is_file()
            new = {**row, "label": NEW_MAPPING.get(row["raw_label"], 0)}
            assert {k: v for k, v in row.items() if k != "label"} == {
                k: v for k, v in new.items() if k != "label"}
            new_rows.append(new)
        statistics = {"photos": len(photos), "hands": len(old_rows), "users": len(user_sets[split]),
                      "changed_hands": sum(a["label"] != b["label"] for a, b in zip(old_rows, new_rows)),
                      "changed_photo_commands": sum(p["old_truth"] != p["truth"] for p in photo_truth),
                      "raw": distribution(old_rows, "raw_label"),
                      "old_six_class": distribution(old_rows, "label"),
                      "new_six_class": distribution(new_rows, "label"),
                      "old_photo_commands": dict(Counter(p["old_truth"] for p in photo_truth)),
                      "new_photo_commands": dict(Counter(p["truth"] for p in photo_truth)),
                      "raw_photo_folder_classes": dict(Counter(p["class_name"] for p in photos))}
        result[split] = {"old": old_rows, "new": new_rows, "photo_truth": photo_truth,
                         "statistics": statistics, "source_crop_sha256": sha(crop_file),
                         "manifest_sha256": sha(manifest)}
        assert result[split]["manifest_sha256"] == audit["splits"][split]["manifest_sha256"]
    if protocol == "B":
        assert all(not user_sets[a].intersection(user_sets[b])
                   for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    return source_root, split_root, result


def prepare_arch(protocol, arch, source_root, split_root, data):
    # 18使用主目录，34/50使用各自独立架构目录。
    source = source_root if arch == "resnet18" else source_root / f"compare_{arch}"
    target = ROOT / "runs" / "MERGED_20261102" / protocol / arch
    identity = {"protocol": protocol, "architecture": arch, "source_run": str(source),
                "source_split_root": str(split_root), "target_run": str(target),
                "old_mapping": OLD_MAPPING, "mapping": NEW_MAPPING, "default_other": 0,
                "only_changed_raw_labels": ["two_up", "two_up_inverted"],
                "crop_pixels": "same read-only original crop paths; no pixel copying or re-encoding",
                "preprocessing": "original bbox; margin0.10 gives1.2square;224;ImageNet normalization",
                "split_reassignment": False, "new_feature_extraction": False,
                "test_predictions_computed": False, "splits": {}, "feature_caches": {}}
    for split, info in data.items():
        source_index = source / f"crops_{split}.json"
        # 比较架构的索引也必须与对应主协议索引逐字节一致。
        assert sha(source_index) == info["source_crop_sha256"]
        dest_index = target / f"crops_{split}.json"
        write_json(dest_index, info["new"])
        write_json(target / f"photo_truth_{split}.json", info["photo_truth"])
        identity["splits"][split] = {"source_index": str(source_index),
            "source_index_sha256": sha(source_index), "new_index_sha256": sha(dest_index),
            "manifest_sha256": info["manifest_sha256"], "all_nonlabel_fields_equal": True,
            "every_original_raw_label_verified": True, **info["statistics"]}
    weights_root = ROOT.parent / "ResNet项目完整交接_20261102" / "完整源码与数据" / "Vision_CA" / "pretrained"
    weight_name = {"resnet18": "resnet18-f37072fd.pth", "resnet34": "resnet34-b627a593.pth",
                   "resnet50": "resnet50-0676ba61.pth"}[arch]
    weight = weights_root / weight_name
    if not weight.exists():
        weight = weights_root / "checkpoints" / weight_name
    if not weight.exists():
        weight = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / weight_name
    official_sha = sha(weight)
    assert official_sha.startswith(weight_name.split("-")[1].split(".")[0])
    identity["official_pretrained"] = {"path": str(weight), "sha256": official_sha,
        "weights": f"{arch}:IMAGENET1K_V1", "hash_prefix_verified": True}
    for split in ("train", "val"):
        source_cache = source / f"features_{split}.pt"
        before_sha = sha(source_cache)
        cache = torch.load(source_cache, map_location="cpu", weights_only=True)
        assert set(cache) == {"x", "y", "pretrained_sha256"}
        x, old_y = cache["x"], cache["y"]
        assert x.device.type == old_y.device.type == "cpu"
        assert x.shape == (len(data[split]["old"]), 2048 if arch == "resnet50" else 512)
        assert torch.isfinite(x).all().item()
        expected_y = torch.tensor([r["label"] for r in data[split]["old"]], dtype=old_y.dtype)
        assert torch.equal(old_y, expected_y)
        assert cache["pretrained_sha256"] == official_sha
        new_y = torch.tensor([r["label"] for r in data[split]["new"]], dtype=old_y.dtype)
        new_cache = {**cache, "y": new_y}
        dest_cache = target / f"features_{split}.pt"
        buffer = io.BytesIO()
        torch.save(new_cache, buffer)
        commit(dest_cache, buffer.getvalue())
        saved = torch.load(dest_cache, map_location="cpu", weights_only=True)
        assert torch.equal(saved["x"], x) and torch.equal(saved["y"], new_y)
        assert sha(source_cache) == before_sha
        identity["feature_caches"][split] = {"source_path": str(source_cache),
            "source_sha256": before_sha, "new_sha256": sha(dest_cache),
            "feature_shape": list(x.shape), "old_labels_match_index_in_order": True,
            "feature_values_elementwise_equal": True, "new_labels_match_index_in_order": True,
            "changed_labels": int((old_y != new_y).sum()),
            "pretrained_sha256": official_sha,
            "feature_tensor_sha256": hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()}
        del saved, cache, new_cache, buffer, x
    # 最后提交身份文件，存在该文件表示本run全部缓存已通过核验。
    write_json(target / "label_statistics.json", {s: d["statistics"] for s, d in data.items()})
    write_json(target / "merged_data_identity.json", identity)
    write_json(target / "prepare_merged_log.json", {"status": "complete",
        "script_sha256": sha(Path(__file__)), "cpu_threads": torch.get_num_threads(),
        "gpu_used": False, "original_files_written": False,
        "identity_sha256": sha(target / "merged_data_identity.json")})
    print(json.dumps({"protocol": protocol, "architecture": arch,
          "status": "complete", "identity_sha256": sha(target / "merged_data_identity.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("A", "B", "all"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    for protocol in (("A", "B") if args.protocol == "all" else (args.protocol,)):
        source, split_root, data = prepare_protocol(protocol)
        for arch in ARCHES:
            prepare_arch(protocol, arch, source, split_root, data)


if __name__ == "__main__":
    main()
