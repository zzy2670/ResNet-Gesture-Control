"""把当前正式文档、双语图及全部相关中间材料封成可校验ZIP。"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import stat
import zipfile

ROOT = Path(__file__).resolve().parent
TARGET = Path('E:/ResNet_当前正式文档与双语图源_20261103.zip')
TOP = 'ResNet_当前正式文档与双语图源_20261103/'
GENERATED = {'PACKAGE_MANIFEST.json', 'PACKAGE_RECEIPT.json'}
REQUIRED_DOCS = [
    '基于ResNet18的手势识别项目方案_v13_中英文配图.docx',
    'A与B协议ResNet训练与对比实验报告_v4_中英文配图.docx',
    '交接说明_v2_中英文配图.docx',
]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def load(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def author_gates():
    # 文档审查为封包前真实结果，不将作者自检当作独审。
    review = load('reviews/documents_independent_review.json')
    if review.get('status') != 'PASS':
        raise ValueError('三份当前正式文档尚未独审通过')
    rows = review.get('documents', [])
    if {Path(row['path']).name for row in rows} != set(REQUIRED_DOCS):
        raise ValueError('正式文档审核范围不是三份最新文件')
    for row in rows:
        path = ROOT / 'documents' / Path(row['path']).name
        if row.get('status') != 'PASS' or sha(path) != row['sha256']:
            raise ValueError('正式文档身份偏离实际独审')
    for name in ['diagrams', 'results', 'reference']:
        result = load('reviews/' + name + '_independent_review.json')
        if result.get('status') != 'PASS':
            raise ValueError('双语图分支尚未独审：' + name)
        manifest_path = ROOT / ('assets_' + name) / 'manifest.json'
        if result.get('manifest_sha256') != sha(manifest_path):
            raise ValueError('图源清单已偏离独审：' + name)
        bindings = list(result.get('output_bindings', []))
        for item in result.get('items', []):
            bindings.extend(item.get('files', []))
        if not bindings:
            raise ValueError('图源独审没有文件身份绑定：' + name)
        for binding in bindings:
            candidate = (ROOT / binding['path']).resolve()
            if not candidate.is_relative_to(ROOT.resolve()) or sha(candidate) != binding['sha256']:
                raise ValueError('图源文件已偏离独审：' + binding['path'])
    return review


def main():
    partial = TARGET.with_suffix('.zip.partial')
    if TARGET.exists() or partial.exists() or (ROOT / 'PACKAGE_MANIFEST.json').exists():
        raise FileExistsError('本次包或冻结清单已存在，拒绝覆盖或盲目重跑')
    if not (ROOT / 'README.md').is_file():
        raise FileNotFoundError('没有当前交付入口README')
    review = author_gates()
    rows, excluded = [], []
    # 全部当前文档及其图源/中间资料均进入；只排除程序缓存和应用锁文件。
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if '__pycache__' in path.parts or path.suffix == '.pyc' or path.name.startswith('~$'):
            excluded.append(relative)
            continue
        if path.name in GENERATED:
            continue
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('交付树不能包含链接/重解析点：' + relative)
        rows.append({'path': relative, 'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns, 'sha256': sha(path)})
    manifest = {'scope': '当前正式三Word、中英全部图、可编辑图源、生成代码、必要数据与实际排版中间文件',
                'files': rows, 'excluded_only_program_cache_or_app_locks': excluded,
                'historical_original_documents_overwritten': False,
                'original_full_project_archive_sha256': '4448838cbc4e8a4a5cf4199a20ad593db996e1a5cf1d6208b9a24129d60a2a96',
                'review_sha256': sha(ROOT/'reviews/documents_independent_review.json')}
    raw = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    with (ROOT / 'PACKAGE_MANIFEST.json').open('xb') as stream:
        stream.write(raw)
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for number, row in enumerate(rows, 1):
            source = ROOT / row['path']
            archive.write(source, TOP + row['path'])
            after = source.stat()
            if after.st_size != row['bytes'] or after.st_mtime_ns != row['mtime_ns'] or sha(source) != row['sha256']:
                raise ValueError('封包时来源发生改变：' + row['path'])
            if number % 100 == 0:
                emit({'stage': 'bilingual_archive', 'written': number, 'total': len(rows)})
        archive.writestr(TOP + 'PACKAGE_MANIFEST.json', raw)
    # Windows必须使用可写句柄做落盘，不用只读rb句柄调用fsync。
    with partial.open('r+b') as stream:
        stream.flush()
        os.fsync(stream.fileno())
    with zipfile.ZipFile(partial) as archive:
        expected = {TOP + row['path'] for row in rows} | {TOP + 'PACKAGE_MANIFEST.json'}
        infos = archive.infolist()
        if len(infos) != len(expected) or {item.filename for item in infos} != expected:
            raise ValueError('ZIP成员集合不符或有重名')
        if archive.read(TOP + 'PACKAGE_MANIFEST.json') != raw:
            raise ValueError('ZIP冻结清单不符')
        for row in rows:
            digest = hashlib.sha256()
            size = 0
            with archive.open(TOP + row['path']) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
                    size += len(block)
            if digest.hexdigest() != row['sha256'] or size != row['bytes']:
                raise ValueError('ZIP成员SHA/CRC/长度不符：' + row['path'])
    author_gates()
    if partial.stat().st_size >= 2 * 1024**3:
        raise ValueError('完整材料超出单Release资产上限，保留临时包后再安排分块，不删材料凑体积')
    os.rename(partial, TARGET)
    result = {'status': 'PASS', 'archive': str(TARGET), 'archive_bytes': TARGET.stat().st_size,
              'archive_sha256': sha(TARGET), 'files': len(rows)+1, 'all_members_SHA_CRC_verified': True,
              'current_Word_documents': REQUIRED_DOCS, 'all_sources_and_intermediate_materials_included': True,
              'created_at': datetime.now().astimezone().isoformat()}
    with (ROOT / 'PACKAGE_RECEIPT.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    emit(result)


if __name__ == '__main__':
    main()
