"""只读汇集A/B项目来源，冻结唯一成员清单后直接流式压缩；不复制大型数据。"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import stat
import zipfile
import fnmatch

V = Path(__file__).resolve().parent
P = V.parent
DEFAULT_BUNDLE = P / 'ResNet_AB项目完整交接_20261102'
PREFIX = '完整源码与数据/Vision_CA/'
CACHE_DIRS = {'__pycache__', '.pytest_cache', 'crops'}
GENERATED = {'SHA256SUMS.json', '来源与整理说明.json', '冻结成员映射.json'}
PRETRAINED = ['resnet18-f37072fd.pth', 'resnet34-b627a593.pth',
              'resnet50-0676ba61.pth', 'mobilenet_v3_large-8738ca79.pth']
INTERNAL_DOCUMENT_FILES = {
    'document_update_materials.md', 'document_update_coverage.json',
    'project_plan_v11_working.docx', 'handover_working.md',
    'project_plan_revision_record.md', 'project_plan_source_contact_sheet.png',
    'report_interpretation_sections.md', '报告版式预览.docx',
    'audit_draft_storage.py', 'finalize_draft_storage_audit.py',
}
INTERNAL_DOCUMENT_PATTERNS = ['project_plan_v11_draft_render*',
                              'ab_report_A_preservation_preview*', '报告版式预览渲染*',
                              'project_plan_v11_final_render*', 'handover_AB_final_render*',
                              'formal_independent_document_*_text.txt',
                              'storage_*']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name:
        raise ValueError(f'不安全的成员路径：{name}')
    return path.as_posix()


def source_path(name):
    path = Path(name)
    return path if path.is_absolute() else P / path


def walk(root):
    # 仅raw入口允许已有目录联接；zip.write始终读取实际字节，归档不含联接。
    for path in sorted(root.iterdir()):
        if path.is_dir():
            yield from walk(path)
        elif path.is_file():
            yield path


def default_spec(bundle):
    documents = [{'source': str(bundle / name), 'destination': name}
                 for name in ['README.md', '交接说明.md', '交接说明.docx']]
    history = ['基于ResNet18的手势识别项目方案_v11.docx',
               'A与B协议ResNet训练与对比实验报告_v2.docx',
               'A协议ResNet训练与对比实验报告_v1.docx',
               '手势合并与数字编码.docx', '手势识别项目总体理解_v1.docx',
               'MA6216项目二_问题解答与实施建议_v4.docx',
               'Project 2-Hand-Gesture Recognition for Mobile Robot Control.pdf',
               'MA6216_CA3_final_project.pptx']
    history += [f'基于ResNet18的手势识别项目方案_v{n}.docx' for n in range(1, 11)]
    documents += [{'source': name, 'destination': '历史与正式文档/' + name} for name in history]
    inference = [{'protocol': protocol, 'architecture': f'resnet{size}',
                  'source': f'{folder}/ResNet{size}'}
                 for protocol, folder in [('A', 'ResNet手势识别交付'), ('B', 'ResNet_B手势识别交付')]
                 for size in [18, 34, 50]]
    return {'documents': documents, 'inference_packages': inference,
            'project_readme_source': 'Vision_CA/runs/B_20261102_TPE/packaged_project_README.md',
            'additional_files': [],
            'document_review': 'Vision_CA/runs/B_20261102_TPE/formal_documents_independent_review.json',
            'reviews': [{'source': 'Vision_CA/runs/B_20261102_TPE/final_independent_review.json',
                         'field': 'status', 'accepted': ['PASS']}]}


def gather(bundle, spec):
    members, excluded, folded = {}, [], set()
    def add(source, destination):
        destination = safe_name(destination)
        key = destination.casefold()
        if key in folded:
            raise ValueError(f'重复目标成员：{destination}')
        if destination in GENERATED:
            raise ValueError('保留的生成物名称不能用作来源目标')
        folded.add(key)
        members[destination] = Path(source)
    def tree(source, destination, filter_cache=False):
        for path in walk(source):
            relative = path.relative_to(source)
            if source == V / 'runs/B_20261102_TPE' and (
                    relative.parts[0] in INTERNAL_DOCUMENT_FILES or
                    any(fnmatch.fnmatchcase(relative.parts[0], pattern)
                        for pattern in INTERNAL_DOCUMENT_PATTERNS)):
                excluded.append({'source': str(path), 'reason': '明确排除的内部文档准备材料与排版草稿'})
                continue
            if filter_cache and (any(part in CACHE_DIRS for part in relative.parts)
                                 or path.suffix == '.pyc'
                                 or (path.name.startswith('features_') and path.suffix == '.pt')):
                excluded.append({'source': str(path), 'reason': '可由原图与代码重建的裁图/feature/字节码缓存'})
                continue
            add(path, destination + '/' + relative.as_posix())
    # 只收正式源码及两套权威结果，避免把旧划分、解压自测和个人笔记误带入。
    for path in sorted(V.iterdir()):
        if path.is_file() and (path.suffix in {'.py', '.md', '.json'}) and path.name not in {'make_resnet_note_route.py', 'build_full_resnet_bundle.py'}:
            # 源目录与完整ZIP的程序/文档位置不同，交付版说明必须使用包内真实链接。
            source = source_path(spec['project_readme_source']) if path.name == 'README.md' else path
            add(source, PREFIX + path.name)
    for name in ['raw_train_data', 'A_split_verified_20261102', 'B_split_verified_20261102']:
        tree(V / 'data' / name, PREFIX.rstrip('/') + '/data/' + name)
    for name in ['A_20261102_TPE', 'B_20261102_TPE']:
        tree(V / 'runs' / name, PREFIX.rstrip('/') + '/runs/' + name, True)
    tree(V / 'deployment_src', PREFIX.rstrip('/') + '/deployment_src', True)
    official = P / 'ResNet项目完整交接_20261102/完整源码与数据/Vision_CA/pretrained/checkpoints'
    for name in PRETRAINED:
        add(official / name, PREFIX + 'pretrained/checkpoints/' + name)
    identities = {(entry['protocol'], entry['architecture']) for entry in spec['inference_packages']}
    if identities != {(p, f'resnet{n}') for p in 'AB' for n in (18, 34, 50)} or len(spec['inference_packages']) != 6:
        raise ValueError('必须显式提供六个A/B独立推理目录')
    for entry in spec['inference_packages']:
        source = source_path(entry['source'])
        if source.is_dir():
            tree(source, '程序/' + entry['protocol'] + '/' + entry['architecture'], True)
        else:
            # list阶段可报告尚未生成的包；freeze仍会明确拒绝。
            add(source, '程序/' + entry['protocol'] + '/' + entry['architecture'] + '/__MISSING__')
    for entry in spec['documents'] + spec.get('additional_files', []):
        add(source_path(entry['source']), entry['destination'])
    required_roots = {'README.md', '交接说明.md', '交接说明.docx'}
    if not required_roots <= set(members):
        raise ValueError('根README/交接MD/Word必须全部映射')
    excluded += [{'source': str(P / name), 'reason': reason} for name, reason in [
        ('ResNet双视频学习', '用户要求不发个人全部图文学习笔记及其媒体/关键帧/转写'),
        ('视频笔记_QQ20261101-180117/素材', '会议录屏/音频、视频讲解素材、关键帧与转写不交付'),
        ('HaGRID/hagrid-sample-30k-384p.zip', 'raw已有真实文件，下载ZIP不重复收录'),
        ('Vision_CA/runs/A_deployment_validation_20261102', '临时解压核验副本'),
        ('ResNet项目完整交接_20261102', '旧草稿不交付；仅只读提取四官方预训练依赖')]]
    return members, excluded


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def check_ready(spec, frozen_rows=None):
    marker = V / 'runs/B_20261102_TPE/all_stages_complete.json'
    if json.loads(marker.read_text(encoding='utf-8'))['status'] != 'PASS':
        raise ValueError('B全部训练/测试/CPU测速尚未成功')
    if not spec.get('reviews'):
        raise ValueError('必须提供正式独立审查门禁')
    for review in spec['reviews']:
        value = json.loads(source_path(review['source']).read_text(encoding='utf-8'))
        for key in review.get('field', 'status').split('.'):
            value = value[key]
        if value not in review.get('accepted', ['PASS']):
            raise ValueError(f'独审未通过：{review["source"]}')
    check_document_review(spec, frozen_rows)


def check_document_review(spec, frozen_rows=None):
    # 正式Word必须由独审绑定实际字节；任意实验PASS不能替代文档审核。
    review = json.loads(source_path(spec['document_review']).read_text(encoding='utf-8'))
    if review.get('status') != 'PASS':
        raise ValueError('正式文档独审尚未PASS')
    records = {}
    for row in review.get('documents', []):
        key = str(source_path(row['source']).resolve()).casefold()
        if key in records:
            raise ValueError('正式文档审核中存在重复来源')
        records[key] = row
    required = {'交接说明.docx', '基于ResNet18的手势识别项目方案_v11.docx',
                'A与B协议ResNet训练与对比实验报告_v2.docx'}
    formal = [entry for entry in spec['documents']
              if Path(entry['destination']).name in required or entry.get('formal', False)]
    if not required <= {Path(entry['destination']).name for entry in formal}:
        raise ValueError('spec必须包含三份正式Word')
    for entry in formal:
        path = source_path(entry['source']).resolve()
        record = records.get(str(path).casefold())
        if not record or record.get('status') != 'PASS' or record.get('sha256') != sha(path):
            raise ValueError(f'正式Word未通过独审或审核后字节已改变：{path}')
        if frozen_rows is not None:
            frozen = [row for row in frozen_rows if str(Path(row['source']).resolve()).casefold() == str(path).casefold()]
            if len(frozen) != 1 or frozen[0]['sha256'] != record['sha256']:
                raise ValueError(f'归档中的正式Word字节与独审身份不一致：{path}')


def freeze(bundle, spec, members, excluded):
    check_ready(spec)
    missing = [str(path) for path in members.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'缺少{len(missing)}个来源：{missing[:6]}')
    if any((bundle / name).exists() for name in GENERATED):
        raise FileExistsError('已冻结清单不可覆盖；修改后使用新的--bundle目录')
    rows = []
    for index, (name, path) in enumerate(sorted(members.items()), 1):
        before = path.stat()
        digest = sha(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'读取时来源仍在变化：{path}')
        rows.append({'path': name, 'source': str(path.resolve()), 'bytes': before.st_size,
                     'mtime_ns': before.st_mtime_ns, 'sha256': digest})
        if index % 5000 == 0:
            emit({'stage': 'freeze', 'hashed': index, 'total': len(members)})
    provenance = {'scope': '原始数据+A/B固定名单、全部真实训练证据、六个独立推理包及正式文档',
                  'original_files_modified': False, 'excluded': excluded, 'specification': spec,
                  'storage': '直接归档源文件；ZIP成员均为真实字节，无junction；裁图/feature可重建缓存不重复收录'}
    check_ready(spec, rows)
    bundle.mkdir(parents=True, exist_ok=True)
    write_new(bundle / '来源与整理说明.json', provenance)
    path = bundle / '来源与整理说明.json'
    rows.append({'path': path.name, 'source': str(path.resolve()), 'bytes': path.stat().st_size,
                 'mtime_ns': path.stat().st_mtime_ns, 'sha256': sha(path)})
    # 内部来源映射与SHA清单分开；兼容原check_project_bundle.py的files字段。
    write_new(bundle / '冻结成员映射.json', {'files': rows, 'bundle_name': bundle.name})
    path = bundle / '冻结成员映射.json'
    manifest_rows = [{k: row[k] for k in ('path', 'bytes', 'sha256')} for row in rows]
    manifest_rows.append({'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)})
    write_new(bundle / 'SHA256SUMS.json', {'files': manifest_rows, 'count': len(manifest_rows),
                                        'bytes': sum(row['bytes'] for row in manifest_rows)})
    emit({'stage': 'frozen', 'files': len(manifest_rows), 'bytes': sum(row['bytes'] for row in manifest_rows)})


def archive(bundle, output):
    if output.exists() or output.with_suffix(output.suffix + '.partial').exists():
        raise FileExistsError('最终ZIP或partial已存在，不覆盖也不盲目重试')
    mapping = json.loads((bundle / '冻结成员映射.json').read_text(encoding='utf-8'))
    provenance = json.loads((bundle / '来源与整理说明.json').read_text(encoding='utf-8'))
    check_ready(provenance['specification'], mapping['files'])
    rows = {row['path']: row for row in mapping['files']}
    manifest = json.loads((bundle / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    for name in ['冻结成员映射.json', 'SHA256SUMS.json']:
        path = bundle / name
        rows[name] = {'source': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
    expected = {row['path']: row for row in manifest['files']}
    if set(expected) != set(rows) - {'SHA256SUMS.json'}:
        raise ValueError('冻结来源映射与SHA清单集合不一致')
    for name, row in expected.items():
        if any(rows[name][field] != row[field] for field in ['bytes', 'sha256']):
            raise ValueError('来源映射与SHA清单摘要不一致')
    partial = output.with_suffix(output.suffix + '.partial')
    # 用x模式创建独占输出；每个来源在写入时核验冻结SHA，变化即保留partial并失败。
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as target:
        for index, (name, row) in enumerate(sorted(rows.items()), 1):
            digest, count = hashlib.sha256(), 0
            info = zipfile.ZipInfo(bundle.name + '/' + safe_name(name))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            with Path(row['source']).open('rb') as source, target.open(info, 'w', force_zip64=True) as sink:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    sink.write(block)
                    digest.update(block)
                    count += len(block)
            if digest.hexdigest() != row['sha256'] or count != row['bytes']:
                raise ValueError(f'来源已偏离冻结字节：{name}')
            if index % 5000 == 0:
                emit({'stage': 'archive', 'written': index, 'total': len(rows)})
    with partial.open('rb') as committed_bytes:
        os.fsync(committed_bytes.fileno())
    result = verify(bundle, partial)
    check_ready(provenance['specification'], mapping['files'])
    os.rename(partial, output)
    result.update(archive=str(output), archive_sha256=sha(output), archive_bytes=output.stat().st_size)
    write_new(output.with_suffix('.verification.json'), result)
    emit(result)


def verify(bundle, archive_path):
    # 无需解压全部数据；完整读到成员尾会同时触发zipfile CRC核验。
    with zipfile.ZipFile(archive_path) as archive_file:
        infos = archive_file.infolist()
        prefix = bundle.name + '/'
        names = [item.filename for item in infos]
        if len(names) != len(set(name.casefold() for name in names)):
            raise ValueError('ZIP存在重复成员')
        if any(not name.startswith(prefix) or stat.S_ISLNK(item.external_attr >> 16)
               for name, item in zip(names, infos)):
            raise ValueError('ZIP含错误顶层或符号链接')
        manifest_bytes = archive_file.read(prefix + 'SHA256SUMS.json')
        if manifest_bytes != (bundle / 'SHA256SUMS.json').read_bytes():
            raise ValueError('ZIP清单偏离已冻结清单')
        manifest = json.loads(manifest_bytes)
        manifest_names = [row['path'] for row in manifest['files']]
        if len(manifest_names) != len(set(name.casefold() for name in manifest_names)):
            raise ValueError('SHA清单有重复成员')
        if set(names) != {prefix + row['path'] for row in manifest['files']} | {prefix + 'SHA256SUMS.json'}:
            raise ValueError('ZIP成员集合与SHA清单不一致')
        for row in manifest['files']:
            safe_name(row['path'])
            digest, count = hashlib.sha256(), 0
            with archive_file.open(prefix + row['path']) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
                    count += len(block)
            if digest.hexdigest() != row['sha256'] or count != row['bytes']:
                raise ValueError(f'ZIP内SHA或长度不符：{row["path"]}')
    return {'status': 'PASS', 'files': len(infos), 'archive_crc_sha256_verified': True,
            'junction_or_symlink_count': 0, 'training_executed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['spec', 'list', 'freeze', 'archive', 'verify'])
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--spec', type=Path, help='显式来源白名单；不提供时使用默认名单')
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    spec = json.loads(args.spec.read_text(encoding='utf-8')) if args.spec else default_spec(bundle)
    if args.stage == 'spec':
        emit(spec)
    elif args.stage in ['archive', 'verify']:
        output = args.archive or bundle.with_suffix('.zip')
        if args.stage == 'archive':
            archive(bundle, output)
        else:
            emit(verify(bundle, output))
    else:
        members, excluded = gather(bundle, spec)
        if args.stage == 'list':
            missing = [str(path) for path in members.values() if not path.is_file()]
            emit({'stage': 'list', 'files': len(members), 'source_bytes': sum(path.stat().st_size for path in members.values() if path.is_file()),
                  'missing': missing, 'excluded_entries': len(excluded), 'archive_created': False})
        else:
            freeze(bundle, spec, members, excluded)


if __name__ == '__main__':
    main()
