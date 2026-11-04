"""逐段校验并恢复完整项目ZIP，兼容更名前后两组固定分卷名称。"""
from pathlib import Path
import argparse
import hashlib
import json
import os

BASE_NAMES = ('slate-kit-full-20261103.zip', 'MA6216_ResNet_full_20261103.zip')
ARCHIVE_BYTES = 6188474282
ARCHIVE_SHA256 = '4448838cbc4e8a4a5cf4199a20ad593db996e1a5cf1d6208b9a24129d60a2a96'


def resolve_parts(folder, manifest):
    """验证固定项目身份及顺序；仅允许同编号的新旧文件名互为别名。"""
    folder = Path(folder).resolve()
    base = manifest.get('original_name')
    if base not in BASE_NAMES or manifest.get('archive_bytes') != ARCHIVE_BYTES or manifest.get('archive_sha256') != ARCHIVE_SHA256:
        raise ValueError('清单不属于本次已验收项目')
    rows = manifest['parts']
    if len(rows) != 4 or any(row['name'] != base + f'.{i:03d}' for i, row in enumerate(rows, 1)):
        raise ValueError('分块名称或顺序不符')
    resolved = []
    for i, row in enumerate(rows, 1):
        choices = [folder / row['name']] + [folder / (other + f'.{i:03d}') for other in BASE_NAMES if other != base]
        source = next((path for path in choices if path.is_file()), None)
        if source is None or source.stat().st_size != row['bytes']:
            raise ValueError('分块缺失或长度不符：' + row['name'])
        resolved.append((row, source))
    return resolved


def restore(folder, output=None):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / 'full_project_manifest.json').read_text(encoding='utf-8'))
    parts = resolve_parts(folder, manifest)
    output = Path(output).resolve() if output else folder / manifest['original_name']
    partial = output.with_suffix(output.suffix + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('输出或临时文件已存在，不覆盖；先核实已有文件')
    whole, total = hashlib.sha256(), 0
    with partial.open('xb') as target:
        for row, source_path in parts:
            digest = hashlib.sha256()
            with source_path.open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block)
                    whole.update(block)
                    target.write(block)
                    total += len(block)
            if digest.hexdigest() != row['sha256']:
                raise ValueError('分块SHA256不符，保留临时文件用于诊断')
            print('已核验：' + source_path.name, flush=True)
        if whole.hexdigest() != ARCHIVE_SHA256 or total != ARCHIVE_BYTES:
            raise ValueError('完整ZIP摘要或总长度不符')
        target.flush()
        os.fsync(target.fileno())
    if output.exists():
        raise FileExistsError('目标已被其他进程创建，拒绝覆盖')
    partial.rename(output)
    print('恢复成功：' + str(output), flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', nargs='?', default='.')
    parser.add_argument('--output')
    args = parser.parse_args()
    restore(args.folder, args.output)
