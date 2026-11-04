"""迁移后只读核对A/B完整固定名单与原始字节；不划分、不训练、不改历史索引。"""
from pathlib import Path
import argparse
import csv
import hashlib
import itertools
import json

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data/raw_train_data/hagrid-sample-30k-384p'
SPLITS = {'A': ('A_split_verified_20261102/P1_random_image', [25468, 3185, 3180]),
          'B': ('B_split_verified_20261102', [25274, 3457, 3102])}
PRETRAINED = ['resnet18-f37072fd.pth', 'resnet34-b627a593.pth',
              'resnet50-0676ba61.pth', 'mobilenet_v3_large-8738ca79.pth']


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def local_path(root, value):
    # 历史A清单使用Windows分隔符；迁移后统一按包内相对路径读取。
    relative = Path(value.replace('\\', '/'))
    if relative.is_absolute() or '..' in relative.parts or ':' in str(relative):
        raise ValueError(f'非法相对路径：{value}')
    return root / relative


def check():
    raw_files = list(RAW.rglob('*.jpg'))
    if len(raw_files) != 31833:
        raise ValueError(f'raw照片应为31833，实际{len(raw_files)}')
    # 原图只散列一次，分组副本必须逐一与原字节一致。
    raw_hashes = {path.relative_to(RAW).as_posix(): sha(path) for path in raw_files}
    if len(set(raw_hashes.values())) != 31833:
        raise ValueError('raw存在精确重复照片或缺少唯一照片')
    result = {}
    for protocol, (folder, counts) in SPLITS.items():
        seen_ids, split_users, split_hashes, summaries = set(), {}, {}, {}
        for split, expected in zip(['train', 'val', 'test'], counts):
            directory = ROOT / 'data' / folder / split
            with (directory / 'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            ids = {row['image_id'] for row in rows}
            if len(rows) != expected or len(ids) != expected or ids & seen_ids:
                raise ValueError(f'{protocol}/{split}照片数或ID互斥失败')
            seen_ids.update(ids)
            if len(list((directory / 'annotations').glob('*.json'))) != 18:
                raise ValueError(f'{protocol}/{split}原18类标注不全')
            users, hashes = set(), set()
            for row in rows:
                users.add(row['user_id'])
                source_name = row['source_image'].replace('\\', '/')
                source_digest = raw_hashes.get(source_name)
                if source_digest is None:
                    raise FileNotFoundError(f'原始照片未包含：{source_name}')
                copy_path = local_path(directory, row['image_path'])
                digest = sha(copy_path)
                if digest != source_digest:
                    raise ValueError(f'固定分组副本与原图不一致：{copy_path}')
                hashes.add(digest)
            split_users[split], split_hashes[split] = users, hashes
            summaries[split] = {'photos': len(rows), 'users': len(users)}
        overlaps = {}
        for left, right in itertools.combinations(['train', 'val', 'test'], 2):
            overlap = {'users': len(split_users[left] & split_users[right]),
                       'sha256': len(split_hashes[left] & split_hashes[right])}
            # A只保证照片互斥；人员跨组是A协议本身的限制，必须如实保留。
            if overlap['sha256'] or (protocol == 'B' and overlap['users']):
                raise ValueError(f'{protocol}/{left}/{right}互斥要求失败：{overlap}')
            overlaps[left + '_' + right] = overlap
        result[protocol] = {'splits': summaries, 'cross_split_overlap': overlaps}
    pretrained = {}
    for name in PRETRAINED:
        path = ROOT / 'pretrained/checkpoints' / name
        digest = sha(path)
        # 官方文件名包含SHA256前八位，可直接核对离线依赖身份。
        if not digest.startswith(name.rsplit('-', 1)[1].split('.')[0]):
            raise ValueError(f'官方预训练身份不符：{path}')
        pretrained[name] = digest
    return {'status': 'PASS', 'protocols': result, 'raw_photos': len(raw_files),
            'raw_and_split_bytes_checked': True, 'pretrained_sha256': pretrained,
            'old_absolute_indices_reused': False, 'training_executed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check'])
    parser.parse_args()
    print(json.dumps(check(), ensure_ascii=False, indent=2))
