"""逐段校验后恢复完整项目ZIP；输入已下载的四段及清单，不联网。"""
from pathlib import Path
import argparse
import hashlib
import json
import os

def restore(folder, output=None):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / 'full_project_manifest.json').read_text(encoding='utf-8'))
    # 固定本次已验收项目身份，不让外来清单改变目标路径或项目内容。
    if (manifest.get('original_name') != 'MA6216_ResNet_full_20261103.zip'
            or manifest.get('archive_bytes') != 6188474282
            or manifest.get('archive_sha256') != '4448838cbc4e8a4a5cf4199a20ad593db996e1a5cf1d6208b9a24129d60a2a96'):
        raise ValueError('清单不属于本次已验收项目')
    output = Path(output).resolve() if output else folder / manifest['original_name']
    partial = output.with_suffix(output.suffix + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('输出或临时文件已存在，不覆盖；先核实已有文件')
    parts = manifest['parts']
    if len(parts) != 4 or any(row['name'] != manifest['original_name'] + f'.{i:03d}' for i, row in enumerate(parts, 1)):
        raise ValueError('分块名称或顺序不符')
    # 先确认所有分块存在且长度正确，避免下载不全时生成半成品。
    for row in parts:
        name = row['name']
        if Path(name).name != name or (folder / name).stat().st_size != row['bytes']:
            raise ValueError('分块路径或长度不符')
    whole, total = hashlib.sha256(), 0
    with partial.open('xb') as target:
        for row in parts:
            digest = hashlib.sha256()
            with (folder / row['name']).open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block)
                    whole.update(block)
                    target.write(block)
                    total += len(block)
            if digest.hexdigest() != row['sha256']:
                raise ValueError('分块SHA256不符，保留临时文件用于诊断')
            print('已核验：' + row['name'], flush=True)
        if whole.hexdigest() != manifest['archive_sha256'] or total != manifest['archive_bytes']:
            raise ValueError('完整ZIP摘要或总长度不符')
        target.flush()
        os.fsync(target.fileno())
    # 通过全部校验才提交正式文件；原分块仍保留。
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
