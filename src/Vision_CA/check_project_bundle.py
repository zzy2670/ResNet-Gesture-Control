"""只读核验完整交接目录及ZIP的逐文件字节；不执行训练或修改交付包。"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import stat
import zipfile


def digest(stream):
    value = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        value.update(block)
        size += len(block)
    return value.hexdigest(), size


def rows_from(data):
    # 兼容总包清单的常见命名，字段本身必须完整，不能跳过不认识的记录。
    rows = data if isinstance(data, list) else data.get('files')
    if not isinstance(rows, list) or not rows:
        raise ValueError('清单需要非空files数组')
    result = {}
    for row in rows:
        name = row.get('path', row.get('relative_path', row.get('destination_relative')))
        if not name or not row.get('sha256') or 'bytes' not in row:
            raise ValueError(f'清单条目缺少path/sha256/bytes：{row}')
        relative = PurePosixPath(name.replace('\\', '/'))
        if relative.is_absolute() or '..' in relative.parts or ':' in str(relative):
            raise ValueError(f'清单路径不是安全相对路径：{name}')
        if str(relative) in result:
            raise ValueError(f'清单有重复项：{relative}')
        result[str(relative)] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--manifest', default='SHA256SUMS.json')
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    root = args.bundle.resolve()
    manifest = root / args.manifest
    rows = rows_from(json.loads(manifest.read_text(encoding='utf-8-sig')))
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink() or path.is_junction():
            raise ValueError(f'包中仍有目录联接或符号链接：{path}')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    # 清单自身通常不自引用；另外的总包校验结果不应在归档后悄悄新增。
    allowed_extra = {manifest.relative_to(root).as_posix()}
    extra, missing = actual - set(rows) - allowed_extra, set(rows) - actual
    if extra or missing:
        raise ValueError(f'目录与清单集合不一致：extra={list(extra)[:5]},missing={list(missing)[:5]}')
    total = 0
    archive = zipfile.ZipFile(args.archive) if args.archive else None
    try:
        members = {} if archive is None else {item.filename: item for item in archive.infolist() if not item.is_dir()}
        if archive:
            # 支持压缩文件内有单一顶层目录，也支持直接装入根内容。
            prefix = root.name + '/' if any(name.startswith(root.name + '/') for name in members) else ''
            expected = {prefix + name for name in actual}
            if set(members) != expected:
                raise ValueError('ZIP成员集合与目录文件集合不一致')
            for info in members.values():
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f'ZIP内含符号链接：{info.filename}')
        for name, row in rows.items():
            path = root / name
            if archive:
                # 单次流式读取每个ZIP成员，同时触发ZIP CRC验证和SHA256核验。
                if path.stat().st_size != row['bytes']:
                    raise ValueError(f'目录文件长度不符：{name}')
                with archive.open(prefix + name) as stream:
                    value, size = digest(stream)
            else:
                with path.open('rb') as stream:
                    value, size = digest(stream)
            if value != row['sha256'] or size != row['bytes']:
                raise ValueError(f'文件SHA256或长度不符：{name}')
            total += size
        if archive:
            # 清单虽然不自引用，ZIP内清单也必须与已核目录清单字节一致。
            if archive.read(prefix + manifest.relative_to(root).as_posix()) != manifest.read_bytes():
                raise ValueError('ZIP内清单与目录清单不一致')
    finally:
        if archive:
            archive.close()
    print(json.dumps({'status': 'PASS', 'files': len(rows), 'bytes': total,
                      'archive_crc_sha256_verified': bool(args.archive),
                      'junction_or_symlink_count': 0, 'training_executed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
