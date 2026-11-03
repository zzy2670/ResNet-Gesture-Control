"""固定B协议人员互斥分组，核对原图内容，保存可供训练读取的三组副本。

先固定划分规则，再计算原图哈希。不会读取任何模型预测或测试分数，
不会修改A分组或原图；同内容跨人员时将相关人员连接为不可拆分组。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil

import prepare_datasets as original

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data/raw_train_data/hagrid-sample-30k-384p'
OUTPUT = ROOT / 'data/B_split_verified_20261102'
LABELS = {'peace': 1, 'peace_inverted': 1, 'stop': 2,
          'stop_inverted': 2, 'palm': 2, 'fist': 3, 'like': 4, 'dislike': 5}
SPLITS = ('train', 'val', 'test')
RATIOS = (0.8, 0.1, 0.1)
SEED = 42


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, data):
    # 证据文件只新建，避免误把旧分组悄悄替换成另一版。
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def components(users, duplicate_groups):
    parent = {user: user for user in users}

    def find(user):
        while parent[user] != user:
            parent[user] = parent[parent[user]]
            user = parent[user]
        return user

    for group in duplicate_groups:
        ordered = sorted({row['user_id'] for row in group})
        anchor = find(ordered[0])
        for user in ordered[1:]:
            other = find(user)
            if other != anchor:
                parent[other] = anchor
    result = defaultdict(list)
    for user in sorted(users):
        result[find(user)].append(user)
    return sorted(result.values(), key=lambda group: tuple(group))


def assign_connected(records, groups):
    # 只在存在跨人员精确重复时使用；相连人员永不拆开，以免内容跨组。
    random.Random(SEED).shuffle(groups)
    targets = dict(zip(SPLITS, original.split_counts(sum(map(len, groups)), RATIOS)))
    assigned = {split: [] for split in SPLITS}
    counts = Counter()
    for group in groups:
        # 选择加入该组后剩余人数缺口最大的分组，固定次序处理同分。
        split = max(SPLITS, key=lambda name: (targets[name] - counts[name], -SPLITS.index(name)))
        assigned[split].extend(group)
        counts[split] += len(group)
    owner = {user: split for split, members in assigned.items() for user in members}
    return {split: [row for row in records if owner[row['user_id']] == split] for split in SPLITS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f'输出已存在，拒绝覆盖既定划分：{output}')
    output.mkdir(parents=True)
    spec = {'protocol': 'B_subject_independent', 'seed': SEED,
            'requested_ratios': dict(zip(SPLITS, RATIOS)), 'ratio_unit': 'users',
            'normal_algorithm': 'prepare_datasets.make_subject_split: sorted users, Random(42).shuffle, largest-remainder user counts',
            'duplicate_policy': 'union users connected by identical source SHA256 before assignment; never split a connected component',
            'duplicate_component_algorithm': 'sorted components, Random(42).shuffle, greatest absolute user target deficit',
            'balance_policy': 'one fixed subject-random split; report all original/merged distributions and deviations; no candidate-seed or test-score selection',
            'source': str(source), 'label_mapping': LABELS, 'unlisted_labels_map_to': 0,
            'prediction_or_test_score_used': False, 'source_images_modified': False,
            'missing_class_policy': 'stop before export if any original source photo class absent in any split'}
    write_json(output / 'split_spec.json', spec)
    records, annotated_total = original.load_records(source)
    if len(records) != 31833:
        raise ValueError(f'原始照片数与既定样本不符：{len(records)}')
    print(json.dumps({'stage': 'loaded', 'images': len(records),
                      'users': len({row['user_id'] for row in records})}), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        hashes = dict(zip((row['image_id'] for row in records), executor.map(digest, (row['source'] for row in records))))
    by_hash = defaultdict(list)
    for row in records:
        by_hash[hashes[row['image_id']]].append(row)
    duplicates = [group for group in by_hash.values() if len(group) > 1]
    cross_user = [group for group in duplicates if len({row['user_id'] for row in group}) > 1]
    users = {row['user_id'] for row in records}
    connected = components(users, cross_user)
    partitions = (assign_connected(records, connected) if cross_user
                  else original.make_subject_split(records, RATIOS, SEED))
    classes = {row['class_name'] for row in records}
    for split, rows in partitions.items():
        missing = classes - {row['class_name'] for row in rows}
        if missing:
            raise ValueError(f'固定划分出现缺类，停止导出：{split} {sorted(missing)}')
    # 每组的人员、图片ID、文件内容都必须互斥，导出前先独立检查。
    image_sets = {split: {row['image_id'] for row in rows} for split, rows in partitions.items()}
    user_sets = {split: {row['user_id'] for row in rows} for split, rows in partitions.items()}
    hash_sets = {split: {hashes[row['image_id']] for row in rows} for split, rows in partitions.items()}
    overlap = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            value = {'users': len(user_sets[left] & user_sets[right]),
                     'image_ids': len(image_sets[left] & image_sets[right]),
                     'sha256': len(hash_sets[left] & hash_sets[right])}
            if any(value.values()):
                raise ValueError(f'跨组污染：{left}/{right} {value}')
            overlap[f'{left}_{right}'] = value
    # 复用原项目的保存格式，但不调用其会拒绝非空目录的高层导出函数。
    # split_spec必须先于分组输出固定，故此处逐组明确写出同格式文件。
    summary = {'protocol': 'B_subject_independent', 'mode': 'subject_independent', 'seed': SEED,
               'requested_ratios': dict(zip(SPLITS, RATIOS)), 'ratio_unit': 'users',
               'source': str(source), 'total_images': len(records), 'total_users': len(users),
               'total_hands': sum(len(row['annotation']['labels']) for row in records),
               'classes': sorted(classes), 'annotation_entries_in_source_classes': annotated_total,
               'image_storage': 'independent copies', 'splits': {}, 'cross_split_overlap': overlap}
    inventory = []
    assignments = []
    for split in SPLITS:
        rows = sorted(partitions[split], key=lambda row: (row['class_name'], row['image_id']))
        folder = output / split
        folder.mkdir()
        ann = {name: {} for name in sorted(classes)}
        with (folder / 'manifest.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['image_id', 'image_path', 'class_name', 'user_id', 'source_image'])
            writer.writeheader()
            for row in rows:
                relative = Path('images') / row['class_name'] / row['source'].name
                target = folder / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(row['source'], target)
                if digest(target) != hashes[row['image_id']]:
                    raise ValueError(f'复制内容与原始照片不符：{target}')
                writer.writerow({'image_id': row['image_id'], 'image_path': relative.as_posix(),
                                 'class_name': row['class_name'], 'user_id': row['user_id'],
                                 'source_image': row['source'].relative_to(source).as_posix()})
                ann[row['class_name']][row['image_id']] = row['annotation']
                inventory.append({'split': split, 'image_id': row['image_id'], 'user_id': row['user_id'],
                                  'path': f'{split}/{relative.as_posix()}', 'sha256': hashes[row['image_id']],
                                  'bytes': target.stat().st_size})
        (folder / 'annotations').mkdir()
        for name, values in ann.items():
            write_json(folder / 'annotations' / f'{name}.json', values)
        original_hands = Counter(label for row in rows for label in row['annotation']['labels'])
        merged_hands = Counter(LABELS.get(label, 0) for row in rows for label in row['annotation']['labels'])
        merged_photos = Counter(3 if 3 in [LABELS.get(x, 0) for x in row['annotation']['labels']]
                               else next(iter(active)) if len(active := {LABELS.get(x, 0) for x in row['annotation']['labels']} - {0}) == 1 else 0
                               for row in rows)
        summary['splits'][split] = {'images': len(rows), 'users': len(user_sets[split]),
                                   'hands': sum(original_hands.values()), 'image_fraction': len(rows) / len(records),
                                   'user_fraction': len(user_sets[split]) / len(users),
                                   'images_per_class': dict(sorted(Counter(row['class_name'] for row in rows).items())),
                                   'hands_per_original_label': dict(sorted(original_hands.items())),
                                   'hands_per_merged_label': {str(label): merged_hands[label] for label in range(6)},
                                   'photos_per_merged_command': {str(label): merged_photos[label] for label in range(6)}}
        write_json(output / f'{split}_users.json', sorted(user_sets[split]))
        assignments.extend({'split': split, 'user_id': user} for user in sorted(user_sets[split]))
        print(json.dumps({'stage': 'exported', 'split': split, **{k: summary['splits'][split][k] for k in ['images', 'users', 'hands']}}), flush=True)
    write_json(output / 'summary.json', summary)
    write_json(output / 'user_assignments.json', assignments)
    write_json(output / 'source_and_copy_hashes.json', inventory)
    write_json(output / 'duplicate_content_groups.json', {
        'duplicate_groups': [[{'image_id': row['image_id'], 'user_id': row['user_id'], 'sha256': hashes[row['image_id']]} for row in group]
                             for group in duplicates], 'cross_user_duplicate_groups': len(cross_user),
        'connected_user_component_count': len(connected),
        'component_users': connected if cross_user else [], 'normal_algorithm_used': not bool(cross_user)})
    manifests = {split: {'sha256': digest(output / split / 'manifest.csv'),
                         'bytes': (output / split / 'manifest.csv').stat().st_size} for split in SPLITS}
    write_json(output / 'integrity.json', {'status': 'PASS', 'all_source_images_accounted': len(inventory) == len(records),
                                         'source_image_count': len(records), 'copied_image_count': len(inventory),
                                         'source_unique_sha256_count': len(by_hash), 'cross_split_overlap': overlap,
                                         'every_exported_image_sha256_checked': True, 'manifests': manifests,
                                         'split_spec_sha256': digest(output / 'split_spec.json'),
                                         'source_and_copy_hashes_sha256': digest(output / 'source_and_copy_hashes.json')})
    print(json.dumps({'stage': 'complete', 'status': 'PASS', 'out': str(output), 'splits': summary['splits'],
                      'cross_split_overlap': overlap, 'manifests': manifests}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
