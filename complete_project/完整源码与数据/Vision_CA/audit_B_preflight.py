"""独立核验B人员互斥划分的原始来源、清单、实际图片及无泄漏边界。"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
from datetime import datetime

import prepare_datasets as source_reader

ROOT = Path(__file__).resolve().parent
DEFAULT_SPLIT = ROOT / 'data/B_split_verified_20261102'
OUT = ROOT / 'runs/B_20261102_TPE/preflight_independent_review.json'
MAPPING = {'peace': 1, 'peace_inverted': 1, 'stop': 2, 'stop_inverted': 2,
           'palm': 2, 'fist': 3, 'like': 4, 'dislike': 5}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-root', type=Path, default=DEFAULT_SPLIT)
    args = parser.parse_args()
    raw = ROOT / 'data/raw_train_data/hagrid-sample-30k-384p'
    original, annotated_total = source_reader.load_records(raw)
    by_id = {row['image_id']: row for row in original}
    assert len(by_id) == len(original) == 31833
    original_users = {row['user_id'] for row in original}
    records = {}
    identities, subjects, contents = {}, {}, {}
    manifests = {}
    # 独立逐行对照原始JSON，不能把导出脚本自己的summary当作真值。
    for split in ('train', 'val', 'test'):
        folder = args.split_root / split
        path = folder / 'manifest.csv'
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert rows, f'{split}为空'
        ann = {p.stem: read(p) for p in (folder / 'annotations').glob('*.json')}
        assert len(ann) == 18, f'{split}原类标注文件数不为18'
        seen, users, hashes = set(), set(), set()
        primary, hand_original, hand_merged = Counter(), Counter(), Counter()
        for row in rows:
            image_id = row['image_id']
            assert image_id not in seen, f'组内重复ID：{image_id}'
            seen.add(image_id)
            original_row = by_id[image_id]
            assert row['user_id'] == original_row['user_id']
            assert row['class_name'] == original_row['class_name']
            annotation = ann[row['class_name']][image_id]
            assert annotation == original_row['annotation'], f'标注已变：{image_id}'
            assert annotation['user_id'] == row['user_id']
            photo = folder / row['image_path']
            assert photo.is_file(), f'缺图片：{photo}'
            copied_hash = sha256(photo)
            assert copied_hash == sha256(original_row['source']), f'实际照片改变：{image_id}'
            hashes.add(copied_hash)
            users.add(row['user_id'])
            primary[row['class_name']] += 1
            hand_original.update(annotation['labels'])
            hand_merged.update(str(MAPPING.get(label, 0)) for label in annotation['labels'])
        assert set(primary) == set(ann), f'{split}未覆盖18个原类'
        assert set(hand_merged) == set('012345'), f'{split}未覆盖六类'
        assert sum(len(value) for value in ann.values()) == len(rows)
        identities[split], subjects[split], contents[split] = seen, users, hashes
        manifests[split] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
        records[split] = {'images': len(rows), 'users': len(users), 'hands': sum(hand_original.values()),
                          'primary_classes': dict(sorted(primary.items())),
                          'raw_hand_labels': dict(sorted(hand_original.items())),
                          'merged_hand_labels': dict(sorted(hand_merged.items())),
                          'unique_content_hashes': len(hashes)}
    overlaps = {}
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        item = {'image_id': len(identities[left] & identities[right]),
                'user_id': len(subjects[left] & subjects[right]),
                'sha256': len(contents[left] & contents[right])}
        assert not any(item.values()), f'跨组交叉：{left}/{right}: {item}'
        overlaps[f'{left}_{right}'] = item
    assert set.union(*identities.values()) == set(by_id)
    assert sum(len(values) for values in identities.values()) == len(by_id)
    assert set.union(*subjects.values()) == original_users
    result = read(OUT) if OUT.exists() else {}
    result.update({'updated_at': datetime.now().astimezone().isoformat(),
                   'status': 'PASS_DATA_SPLIT', 'scope': '独立全量原始来源与B划分核验；代码审核另列',
                   'source_images': len(original), 'source_users': len(original_users),
                   'raw_annotated_total': annotated_total, 'split_root': str(args.split_root.resolve()),
                   'splits': records, 'overlaps': overlaps, 'manifests': manifests,
                   'all_source_images_preserved': True, 'all_original_annotations_equal': True,
                   'all_copied_image_bytes_match_source': True, 'GPU_started_by_reviewer': False})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'splits': {key: {k: v for k, v in value.items() if k in ('users', 'images', 'hands')} for key, value in records.items()}, 'overlaps': overlaps}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
