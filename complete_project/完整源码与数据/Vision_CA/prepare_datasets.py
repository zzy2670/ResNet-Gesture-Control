"""准备HaGRID数据：A按照片划分三组，B当前只按人员整理照片。"""

# 阅读顺序：先看文件末尾的main()，再顺着它调用的函数向上读。
# P1就是A：读取照片和标注 → 按照片划分训练、验证、测试三组 → 保存文件。
# P2就是B的准备步骤：读取照片和标注 → 给每个人整理一个照片文件夹。
# 注意：B当前没有调用make_subject_split()，所以没有把人分到三组。
# “写有划分函数”和“运行时调用了该函数”是两回事。
# 本脚本只准备数据，不训练模型，也不把手势标签合并成数字0—5。

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")
# 比例和数量都按这个顺序排列：train训练、val验证、test测试。
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


# 通用辅助函数：读写数据，以及计算每组应分到多少个样本。
def write_json(path, data):
    # 把数据保存成JSON文件，保留原标注字段，并缩进排版以便阅读。
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_counts(total, ratios):
    # 先按比例计算并取整数，再优先给小数余量较大的组补齐，避免总数变少。
    # 例如11个样本按8∶1∶1分配，得到9、1、1，合计仍为11。
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    order = sorted(range(3), key=lambda i: (raw[i] - counts[i], -i), reverse=True)
    for index in order[:total - sum(counts)]:
        counts[index] += 1
    return counts


def load_records(source):
    # 只为本机实际存在的照片查找对应JSON标注，不要求下载全部标注中的照片。
    # 每条record对应一张完整照片，包含照片位置、人员编号和原始标注。
    # 同一张照片可能有多只手，因此可以对应多个手框和手势标签。
    image_root = source / "hagrid_30k"
    annotation_root = source / "ann_train_val"
    if not image_root.is_dir() or not annotation_root.is_dir():
        raise ValueError("The source must contain hagrid_30k and ann_train_val.")
    image_paths = sorted(path for path in image_root.rglob("*")
                         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    by_class = defaultdict(list)
    for path in image_paths:
        class_name = path.parent.name.removeprefix("train_val_")
        by_class[class_name].append(path)
    records = []
    seen_ids = set()
    annotated_total = 0
    for class_name, paths in sorted(by_class.items()):
        annotations = json.loads((annotation_root / f"{class_name}.json").read_text(encoding="utf-8"))
        # 标注文件可能覆盖更多照片，因此标注条目总数可能大于本地照片数。
        annotated_total += len(annotations)
        for path in paths:
            image_id = path.stem
            if image_id in seen_ids:
                raise ValueError(f"Duplicate image ID: {image_id}")
            if image_id not in annotations:
                raise ValueError(f"Missing annotation for {path}")
            annotation = annotations[image_id]
            user_id = annotation.get("user_id")
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError(f"Missing or invalid user_id: {image_id}")
            if len(annotation.get("bboxes", [])) != len(annotation.get("labels", [])):
                raise ValueError(f"Box and label counts differ: {image_id}")
            seen_ids.add(image_id)
            records.append({"image_id": image_id, "class_name": class_name,
                            "user_id": user_id, "source": path, "annotation": annotation})
    if not records:
        raise ValueError("No local images were found.")
    # 此处只返回内存中的照片记录列表，还没有分组、复制照片或训练模型。
    return records, annotated_total


# A的划分函数：决定每张照片进入哪一组；main()的P1分支会调用它。
def make_image_split(records, ratios, seed, existing_output=None):
    # 在每个原始类别内分别打乱、分组，让各类别都大致按相同比例进入三组。
    # 此处类别是原始照片文件夹的类别，不是项目合并后的0—5六类。
    # A按照片分，因此同一个人的不同照片允许进入不同组。
    rng = random.Random(seed)
    groups = defaultdict(list)
    for record in records:
        groups[record["class_name"]].append(record)
    result = {split: [] for split in SPLITS}
    assigned = set()
    if existing_output is not None:
        # 追加数据时保留原有分组，同时核对原清单、标注和照片是否一致。
        # 这需要原来的manifest.csv逐图清单和照片都存在。
        # summary.json只有数量，不能单靠它知道每张照片原来属于哪组。
        by_id = {record["image_id"]: record for record in records}
        allowed_files = {existing_output / "summary.json"}
        saved = json.loads((existing_output / "summary.json").read_text(encoding="utf-8"))
        if (saved["mode"] != "random_image" or saved["seed"] != seed
                or saved["requested_ratios"] != dict(zip(SPLITS, ratios))):
            raise ValueError("P1 settings must match the existing seed and ratios.")
        for split in SPLITS:
            split_dir = existing_output / split
            manifest = split_dir / "manifest.csv"
            allowed_files.add(manifest)
            expected_annotations = defaultdict(dict)
            with manifest.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    record = by_id[row["image_id"]]
                    relative = Path("images") / record["class_name"] / record["source"].name
                    if (record["image_id"] in assigned or row["image_path"] != relative.as_posix()
                            or row["class_name"] != record["class_name"]
                            or row["user_id"] != record["user_id"]):
                        raise ValueError(f"Conflicting P1 manifest entry: {row['image_id']}")
                    image = split_dir / relative
                    if image.read_bytes() != record["source"].read_bytes():
                        raise ValueError(f"Image differs from source: {image}")
                    allowed_files.add(image)
                    assigned.add(record["image_id"])
                    result[split].append(record)
                    expected_annotations[record["class_name"]][record["image_id"]] = record["annotation"]
            for path in (split_dir / "annotations").glob("*.json"):
                allowed_files.add(path)
                if json.loads(path.read_text(encoding="utf-8")) != expected_annotations.pop(path.stem, {}):
                    raise ValueError(f"Conflicting P1 annotations: {path}")
            if expected_annotations:
                raise ValueError(f"Missing P1 annotations: {split}")
        if any(path not in allowed_files for path in existing_output.rglob("*") if path.is_file()):
            raise ValueError("P1 contains unexpected files.")
    for class_name in sorted(groups):
        items = sorted((r for r in groups[class_name] if r["image_id"] not in assigned),
                       key=lambda record: record["image_id"])
        rng.shuffle(items)
        # 输入数据和设置相同时，固定随机种子可让打乱顺序重复得到相同结果。
        offset = 0
        for split, target in zip(SPLITS, split_counts(len(groups[class_name]), ratios)):
            # 补足该类别在本组中还缺的数量，不移动已经分好组的照片。
            count = target - sum(r["class_name"] == class_name for r in result[split])
            if count < 0:
                raise ValueError(f"Existing P1 split exceeds its target: {split}/{class_name}")
            result[split].extend(items[offset:offset + count])
            offset += count
    # result保存三组各有哪些照片；稍后由export_dataset()真正写到磁盘。
    return result


# 按人员划分三组的函数：已经写好，但当前main()没有调用它。
def make_subject_split(records, ratios, seed):
    # 先打乱人员名单并分组，再让每个人的全部照片跟随本人进入同一组。
    # 例如甲进训练组，甲的所有照片都进训练组，不再进入验证或测试组。
    # 比例按人数计算；每个人的照片数不同，所以照片比例可能有所偏差。
    users = sorted({record["user_id"] for record in records})
    random.Random(seed).shuffle(users)
    counts = split_counts(len(users), ratios)
    if any(count == 0 for count in counts):
        raise ValueError("There are too few users for three nonempty subsets.")
    user_split = {}
    offset = 0
    for split, count in zip(SPLITS, counts):
        for user_id in users[offset:offset + count]:
            user_split[user_id] = split
        offset += count
    result = {split: [] for split in SPLITS}
    for record in records:
        result[user_split[record["user_id"]]].append(record)
    # 定义函数并不会自动运行它；下面的P2入口当前没有使用这个划分结果。
    return result


# B当前真正执行的函数：按人整理照片，不分训练、验证、测试三组。
def export_user_dataset(output, records, source):
    # 将本地照片按人员编号归档，保留原有人员文件夹并补充缺少的照片。
    # 输出为“人员编号/images/类别/照片”，每个人另有一份annotations.json标注。
    # 相当于给甲、乙分别建照片文件夹，但没有决定谁用于训练、验证或测试。
    groups = defaultdict(list)
    expected_images = {}
    for record in records:
        groups[record["user_id"]].append(record)
        relative = (Path(record["user_id"]) / "images" / record["class_name"]
                    / record["source"].name)
        expected_images[relative] = record
    if output.exists():
        # 先检查已有文件是否属于这批数据，避免误改其他文件。
        for path in output.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(output)
            if relative in expected_images:
                continue
            if relative.as_posix() in {"manifest.csv", "summary.json"}:
                continue
            if (len(relative.parts) == 2 and relative.name == "annotations.json"
                    and relative.parts[0] in groups):
                annotations = json.loads(path.read_text(encoding="utf-8"))
                expected = {r["image_id"]: r["annotation"] for r in groups[relative.parts[0]]}
                if all(key in expected and value == expected[key]
                       for key, value in annotations.items()):
                    continue
            raise ValueError(f"Unexpected or conflicting P2 file: {path}")
    output.mkdir(parents=True, exist_ok=True)
    fields = ["image_id", "user_id", "class_name", "image_path",
              "annotation_path", "source_image"]
    rows = []
    label_counts = Counter()
    for relative, record in sorted(expected_images.items()):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(record["source"], destination)
        # 用哈希值核对已有照片和新复制照片，确保它们与原始照片内容相同。
        source_hash = hashlib.sha256(record["source"].read_bytes()).digest()
        if hashlib.sha256(destination.read_bytes()).digest() != source_hash:
            raise ValueError(f"Image differs from source: {destination}")
        label_counts.update(record["annotation"]["labels"])
        rows.append({"image_id": record["image_id"], "user_id": record["user_id"],
                     "class_name": record["class_name"], "image_path": relative.as_posix(),
                     "annotation_path": f'{record["user_id"]}/annotations.json',
                     "source_image": record["source"].relative_to(source).as_posix()})
    for user_id, items in sorted(groups.items()):
        write_json(output / user_id / "annotations.json",
                   {r["image_id"]: r["annotation"] for r in sorted(items, key=lambda r: r["image_id"])})
    # 全部照片检查完成后，再保存完整清单和统计文件。
    # 这份B清单记录照片属于谁、存在哪里，没有训练／验证／测试归属列。
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    # B的summary只统计人员、类别和照片数量，不记录三组划分结果。
    summary = {
        "mode": "group_by_user_id", "source": str(source),
        "total_images": len(records), "total_users": len(groups),
        "images_per_class": dict(sorted(Counter(r["class_name"] for r in records).items())),
        "images_per_user": {user: len(items) for user, items in sorted(groups.items())},
        "labels": sorted(label_counts), "annotations_per_label": dict(sorted(label_counts.items())),
        "layout": "<user_id>/images/<class>/<image>; <user_id>/annotations.json",
        "manifest_paths_relative_to": output.name, "image_storage": "independent copies",
        "validation": {"all_source_images_included_once": True,
                       "image_hashes_match_source": True, "original_annotations_preserved": True,
                       "each_user_has_exactly_one_folder": True}}
    write_json(output / "summary.json", summary)
    print(f"{output.name}: {len(records)} images, {len(groups)} users, "
          f"{len(summary['images_per_class'])} classes, {len(label_counts)} labels", flush=True)
    return summary


# 保存已经计算好的三组分配；当前main()把它用于A。
def export_dataset(output, partitions, records, source, ratios, seed, mode, annotated_total,
                   extend=False):
    # 默认不允许覆盖非空输出文件夹，防止重复运行时悄悄改变原有数据。
    # 本函数接收外面已经算好的分组，只负责保存，不重新打乱分配。
    if output.exists() and any(output.iterdir()) and not extend:
        raise ValueError(f"Output folder is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    classes = sorted({record["class_name"] for record in records})
    # 按手框统计标签（包括no_gesture），不要与照片文件夹的类别数量混淆。
    label_counts = Counter(label for record in records for label in record["annotation"]["labels"])
    user_sets = {}
    image_sets = {}
    summary = {"source": str(source), "seed": seed, "mode": mode,
               "requested_ratios": dict(zip(SPLITS, ratios)),
               "total_images": len(records), "total_users": len({r["user_id"] for r in records}),
               "annotation_entries_in_source_classes": annotated_total,
               "annotation_entries_without_local_images": annotated_total - len(records),
               "classes": classes, "labels": sorted(label_counts),
               "annotations_per_label": dict(sorted(label_counts.items())),
               "image_storage": "independent copies",
               "layout": "<split>/images/<class>/<image>; <split>/annotations/<class>.json",
               "ratio_unit": "users" if mode == "subject_independent" else "images_per_class",
               "splits": {}}
    for split in SPLITS:
        items = sorted(partitions[split], key=lambda r: (r["class_name"], r["image_id"]))
        if not items:
            raise ValueError(f"Empty subset: {split}")
        annotations = {class_name: {} for class_name in classes}
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        # manifest.csv是逐张照片的清单，比summary中的总数量更具体。
        # 清单所在的train、val或test文件夹，表明这些照片属于哪一组。
        with (split_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_id", "image_path", "class_name", "user_id", "source_image"])
            writer.writeheader()
            for record in items:
                relative = Path("images") / record["class_name"] / record["source"].name
                destination = split_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                # 追加模式中的已有照片前面已经核对过，只复制尚不存在的照片。
                if not destination.exists():
                    shutil.copy2(record["source"], destination)
                annotations[record["class_name"]][record["image_id"]] = record["annotation"]
                writer.writerow({"image_id": record["image_id"], "image_path": relative.as_posix(),
                                 "class_name": record["class_name"], "user_id": record["user_id"],
                                 "source_image": str(record["source"].relative_to(source))})
        for class_name, data in annotations.items():
            write_json(split_dir / "annotations" / f"{class_name}.json", data)
        user_sets[split] = {record["user_id"] for record in items}
        image_sets[split] = {record["image_id"] for record in items}
        summary["splits"][split] = {
            "images": len(items), "users": len(user_sets[split]),
            "image_fraction": len(items) / len(records),
            "images_per_class": dict(sorted(Counter(r["class_name"] for r in items).items()))}
        print(f"{output.name}/{split}: {len(items)} images, {len(user_sets[split])} users", flush=True)
    overlaps = {}
    # 三组不能出现同一张照片。A允许同一个人跨组；
    # 若用于人员互斥划分，则还必须检查三组人员没有重叠。
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            if image_sets[left] & image_sets[right]:
                raise ValueError(f"Images overlap: {left}, {right}")
            overlaps[f"{left}_{right}"] = len(user_sets[left] & user_sets[right])
    if mode == "subject_independent" and any(overlaps.values()):
        raise ValueError("Users overlap in the subject-independent dataset.")
    if set.union(*image_sets.values()) != {record["image_id"] for record in records}:
        raise ValueError("The dataset does not contain all source images.")
    summary["user_overlap_counts"] = overlaps
    # 三组通过完整性和重复检查后，再保存summary.json统计文件。
    write_json(output / "summary.json", summary)
    return summary


def main():
    # 这里是程序入口，读取数据路径、分组比例、随机种子等运行选项。
    # 实际顺序：读取选项 → 读取照片标注 → 按选择执行P1和／或P2分支。
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parent
    parser.add_argument("--source", type=Path, default=project / "data/raw_train_data/hagrid-sample-30k-384p")
    parser.add_argument("--output-root", type=Path, default=project / "data")
    # 三个比例依次对应训练、验证、测试；当前只有P1按这些比例分组。
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1))
    parser.add_argument("--seed", type=int, default=42)
    # both表示两个分支都运行，不表示P2也会自动进行三组划分。
    parser.add_argument("--protocol", choices=("p1", "p2", "both"), default="both")
    parser.add_argument("--extend-p1", action="store_true",
                        help="Add missing P1 images while preserving existing split assignments.")
    args = parser.parse_args()
    if any(value <= 0 for value in args.ratios) or abs(sum(args.ratios) - 1) > 1e-9:
        parser.error("Ratios must be positive and add up to 1.")
    source = args.source.resolve()
    output_root = args.output_root.resolve()
    p1_output = output_root / "P1_random_image"
    if (args.protocol in {"p1", "both"} and p1_output.exists()
            and any(p1_output.iterdir()) and not args.extend_p1):
        # 即使文件夹里只有summary.json，也属于非空文件夹，会受到保护。
        parser.error(f"Output folder is not empty: {p1_output}")
    records, annotated_total = load_records(source)
    print(f"Found {len(records)} local images from {len({r['user_id'] for r in records})} users.", flush=True)
    # 可单独运行P2的按人归档步骤，不必重建或改变P1。
    if args.protocol in {"p1", "both"}:
        # A第一步：在内存中决定每张照片进入训练、验证还是测试组。
        existing = p1_output if args.extend_p1 and p1_output.exists() else None
        partition = make_image_split(records, args.ratios, args.seed, existing)
        # A第二步：保存三组照片、标注、逐图清单，以及总体统计。
        export_dataset(p1_output, partition, records, source, args.ratios,
                       args.seed, "random_image", annotated_total, extend=args.extend_p1)
    if args.protocol in {"p2", "both"}:
        # B：只按人保存照片；这里没有调用make_subject_split()划分三组。
        # 文件夹名含subject_independent，不代表已实际完成人员互斥划分。
        export_user_dataset(output_root / "P2_subject_independent", records, source)


if __name__ == "__main__":
    # 直接运行本文件时才执行main()；被其他程序导入时不自动准备数据。
    main()
