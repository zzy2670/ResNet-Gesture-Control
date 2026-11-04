"""为合并/不合并两标签方案建立十二程序完整交接包；固定来源后直接流式ZIP。"""
from pathlib import Path
import argparse
import fnmatch
import json

import build_ab_project_bundle as transport
import build_merged_gesture_delivery as merged_delivery

V, P = transport.V, transport.P
RUN = V / 'runs/MERGED_20261102'
DEFAULT_BUNDLE = P / 'ResNet_合并与不合并完整交接_20261103'
VARIANTS = ['不合并', '合并']
FORMAL_NAMES = {'交接说明.docx', '基于ResNet18的手势识别项目方案_v12.docx',
                'A与B协议ResNet训练与对比实验报告_v3.docx'}
# 只排除明确的文档编辑材料；报告/CAM科研图片和真实失败尝试日志保留。
DRAFT_NAMES = transport.INTERNAL_DOCUMENT_FILES | {
    'project_plan_v12_working.docx', 'label_comparison_handover_working.md',
    'merged_report_working.docx', 'merged_document_update_materials.md',
    'merged_document_update_coverage.json'}
DRAFT_PATTERNS = transport.INTERNAL_DOCUMENT_PATTERNS + [
    'project_plan_v12_draft_render*', 'project_plan_v12_final_render*',
    'merged_report_draft_render*', 'merged_report_final_render*',
    'handover_label_comparison_render*', 'formal_merged_document_*_text.txt']


def default_spec(bundle):
    # 旧文档作为历史保存，新正式入口指向v12/v3和新交接，绝不覆盖旧冻结清单。
    spec = transport.default_spec(bundle)
    spec['documents'] += [
        {'source': '基于ResNet18的手势识别项目方案_v12.docx',
         'destination': '历史与正式文档/基于ResNet18的手势识别项目方案_v12.docx'},
        {'source': 'A与B协议ResNet训练与对比实验报告_v3.docx',
         'destination': '历史与正式文档/A与B协议ResNet训练与对比实验报告_v3.docx'}]
    spec['project_readme_source'] = 'Vision_CA/runs/MERGED_20261102/packaged_project_README.md'
    spec['document_review'] = 'Vision_CA/runs/MERGED_20261102/formal_documents_independent_review.json'
    spec['package_review'] = 'Vision_CA/runs/MERGED_20261102/twelve_packages_independent_review.json'
    spec['reviews'] = [
        {'source': 'Vision_CA/runs/B_20261102_TPE/ab_final_independent_review.json',
         'field': 'status', 'accepted': ['PASS_AB_RESULT_AUDIT_WITH_LIMITATIONS']},
        {'source': 'Vision_CA/runs/MERGED_20261102/aggregate_review.json',
         'field': 'status', 'accepted': [merged_delivery.AGGREGATE_REVIEW_STATUS]}]
    packages = []
    for variant in VARIANTS:
        for protocol in 'AB':
            for size in [18, 34, 50]:
                if variant == '不合并':
                    folder = 'ResNet手势识别交付' if protocol == 'A' else 'ResNet_B手势识别交付'
                    archive = f'ResNet{size}_图片识别程序.zip' if protocol == 'A' else f'ResNet{size}_B协议图片识别程序.zip'
                else:
                    folder = f'ResNet_合并手势识别交付/{protocol}'
                    archive = f'ResNet{size}_合并{protocol}协议图片识别程序.zip'
                packages.append({'variant': variant, 'protocol': protocol, 'architecture': f'resnet{size}',
                                 'source': folder + f'/ResNet{size}', 'archive': folder + '/' + archive})
    spec['inference_packages'] = packages
    return spec


def internal(path):
    return (path.parts[0] in DRAFT_NAMES or
            any(fnmatch.fnmatchcase(path.parts[0], pattern) for pattern in DRAFT_PATTERNS))


def gather(bundle, spec):
    expected = {(variant, protocol, f'resnet{size}') for variant in VARIANTS for protocol in 'AB' for size in [18, 34, 50]}
    identities = [(row['variant'], row['protocol'], row['architecture']) for row in spec['inference_packages']]
    if len(identities) != 12 or set(identities) != expected:
        raise ValueError('必须明确列出两标签方案×A/B×三个架构，共十二程序')
    legacy_spec = dict(spec)
    legacy_spec['inference_packages'] = [row for row in spec['inference_packages'] if row['variant'] == '不合并']
    members, excluded = transport.gather(bundle, legacy_spec)
    members = {('程序/不合并/' + name[len('程序/'):]) if name.startswith('程序/') else name: source
               for name, source in members.items()}
    folded = {name.casefold() for name in members}
    def add(source, name):
        name = transport.safe_name(name)
        if name.casefold() in folded or name in transport.GENERATED | {'十二程序与ZIP清单.json'}:
            raise ValueError(f'成员重复或占用保留名称：{name}')
        folded.add(name.casefold())
        members[name] = source
    for row in spec['inference_packages']:
        source = transport.source_path(row['source'])
        if row['variant'] == '合并':
            destination = '程序/合并/' + row['protocol'] + '/' + row['architecture']
            if source.is_dir():
                for path in transport.walk(source):
                    relative = path.relative_to(source)
                    if '__pycache__' not in relative.parts and path.suffix != '.pyc':
                        add(path, destination + '/' + relative.as_posix())
            else:
                add(source, destination + '/__MISSING__')
        archive = transport.source_path(row['archive'])
        add(archive, '独立程序ZIP/' + row['variant'] + '/' + row['protocol'] + '/' + archive.name)
    for path in transport.walk(RUN):
        relative = path.relative_to(RUN)
        if internal(relative) or any(part in transport.CACHE_DIRS for part in relative.parts) or path.suffix == '.pyc' or (path.name.startswith('features_') and path.suffix == '.pt'):
            excluded.append({'source': str(path), 'reason': '明确编辑草稿、临时排版渲染或可重建裁图/特征缓存'})
            continue
        add(path, transport.PREFIX + 'runs/MERGED_20261102/' + relative.as_posix())
    return members, excluded


def checked_reviews(spec, frozen_rows=None):
    # 训练完成标记只证明执行结束，正式实验独审与程序独审仍分别必须通过。
    markers = [V / 'runs/B_20261102_TPE/all_stages_complete.json', RUN / 'all_training_evaluation_complete.json']
    for path in markers:
        if json.loads(path.read_text(encoding='utf-8')).get('status') != 'PASS':
            raise ValueError(f'科学执行尚未成功：{path}')
    # 两份真实权威审核始终必需；spec中的附加门禁不能把它们替换成任意PASS。
    verify_unmerged_review(frozen_rows)
    merged_delivery.ready(RUN / 'aggregate_review.json', frozen_rows)
    if len(spec.get('reviews', [])) < 2:
        raise ValueError('必须保留原A/B与合并实验的正式独审')
    for row in spec['reviews']:
        value = json.loads(transport.source_path(row['source']).read_text(encoding='utf-8'))
        for key in row.get('field', 'status').split('.'):
            value = value[key]
        if value not in row.get('accepted', ['PASS']):
            raise ValueError(f'实验独审未通过：{row["source"]}')


def verify_unmerged_review(frozen_rows=None):
    frozen = {} if frozen_rows is None else {str(Path(row['source']).resolve()).casefold(): row['sha256'] for row in frozen_rows}
    def science_sha(path):
        digest = transport.sha(path)
        key = str(Path(path).resolve()).casefold()
        if key in frozen and frozen[key] != digest:
            raise ValueError(f'原科学文件与冻结归档不一致：{path}')
        return digest
    aroot, broot = V / 'runs/A_20261102_TPE', V / 'runs/B_20261102_TPE'
    science_sha(broot / 'ab_final_independent_review.json')
    review = merged_delivery.read(broot / 'ab_final_independent_review.json')
    if review.get('status') != 'PASS_AB_RESULT_AUDIT_WITH_LIMITATIONS' or review.get('issues'):
        raise ValueError('原A/B正式科学独审未通过')
    preflight = merged_delivery.read(broot / 'preflight_independent_review.json')
    for name, digest in preflight['source_sha256'].items():
        if science_sha(V / name) != digest:
            raise ValueError(f'原B独审固定源码改变：{name}')
    arches = {'resnet18', 'resnet34', 'resnet50'}
    identities = review.get('B_model_identity_sha256', {})
    if set(identities) != arches or set(review.get('B_report_statuses', {})) != arches:
        raise ValueError('原B科学审核未完整覆盖三个架构')
    for arch, files in identities.items():
        if review['B_report_statuses'][arch] != 'PASS_FROZEN_B_MODEL_AUDIT_WITH_LIMITATIONS' or not files:
            raise ValueError('原B单模型审核未通过或缺少身份绑定')
        out = broot if arch == 'resnet18' else broot / ('compare_' + arch)
        for name, digest in files.items():
            if science_sha(out / name) != digest:
                raise ValueError(f'原B审核绑定的科学文件改变：{out / name}')
    for split, digest in review.get('manifest_sha256', {}).items():
        if science_sha(V / 'data/B_split_verified_20261102' / split / 'manifest.csv') != digest:
            raise ValueError('原B固定清单偏离独审身份')
    afile = aroot / 'final_independent_review.json'
    if science_sha(afile) != review.get('A_independent_review_sha256'):
        raise ValueError('原A独审本身偏离A/B汇总锚定身份')
    original = merged_delivery.read(afile)
    if original.get('status') != 'pass' or original.get('issues'):
        raise ValueError('原A真实科学独审未通过')
    for name, digest in {**original['weights'], **original['artifact_sha256']}.items():
        if science_sha(aroot / name) != digest:
            raise ValueError(f'原A18科学文件改变：{name}')
    for arch, comparison in original['comparison_reviews'].items():
        out = aroot / ('compare_' + arch)
        science_sha(out / 'independent_review.json')
        subordinate = merged_delivery.read(out / 'independent_review.json')
        if comparison.get('status') != 'pass' or subordinate.get('status') != 'pass' or subordinate.get('issues'):
            raise ValueError('原A比较架构独审未通过')
        for name, digest in comparison['weights'].items():
            if science_sha(out / name) != digest or subordinate.get('weights', {}).get(name) != digest:
                raise ValueError('原A比较架构权重偏离独审')
        for name, digest in subordinate.get('artifact_sha256', {}).items():
            if science_sha(out / name) != digest:
                raise ValueError('原A比较架构科学文件改变')
    speed = original.get('unified_CPU_speed_review', {})
    for group, filename in [('classifier_only', 'classifier_cpu_latency_unified.json'), ('whole_photo', 'cpu_latency.json')]:
        for arch, row in speed.get(group, {}).items():
            out = aroot if arch == 'resnet18' else aroot / ('compare_' + arch)
            if science_sha(out / filename) != row['artifact_sha256']:
                raise ValueError('原A统一速度数据改变')
    if science_sha(aroot / 'classifier_speed_comparison.json') != speed.get('aggregate_speed_artifact_sha256'):
        raise ValueError('原A统一速度汇总改变')


def check_ready(spec, frozen_rows=None):
    checked_reviews(spec, frozen_rows)
    review = json.loads(transport.source_path(spec['document_review']).read_text(encoding='utf-8'))
    if review.get('status') != 'PASS':
        raise ValueError('新正式文档独审尚未PASS')
    records = {}
    for row in review.get('documents', []):
        key = str(transport.source_path(row['source']).resolve()).casefold()
        if key in records:
            raise ValueError('正式文档审核有重复来源')
        records[key] = row
    formal = [row for row in spec['documents'] if Path(row['destination']).name in FORMAL_NAMES or row.get('formal', False)]
    if not FORMAL_NAMES <= {Path(row['destination']).name for row in formal}:
        raise ValueError('缺少新v12方案/v3报告/新交接三份正式Word')
    frozen = {} if frozen_rows is None else {str(Path(row['source']).resolve()).casefold(): row['sha256'] for row in frozen_rows}
    for row in formal:
        source = transport.source_path(row['source']).resolve()
        key = str(source).casefold()
        actual = transport.sha(source)
        record = records.get(key, {})
        if record.get('status') != 'PASS' or record.get('sha256') != actual or (frozen_rows is not None and frozen.get(key) != actual):
            raise ValueError(f'正式Word未审核或审核字节与冻结包不一致：{source}')
    package_review = json.loads(transport.source_path(spec['package_review']).read_text(encoding='utf-8'))
    if package_review.get('status') != 'PASS':
        raise ValueError('十二独立程序的离线接口/ZIP独审尚未PASS')
    archives = {}
    for row in package_review.get('archives', []):
        key = str(transport.source_path(row['source']).resolve()).casefold()
        if key in archives:
            raise ValueError('独立ZIP审核有重复来源')
        archives[key] = row
    if len(archives) != 12:
        raise ValueError('独立审核必须有十二个唯一ZIP')
    for row in spec['inference_packages']:
        source = transport.source_path(row['archive']).resolve()
        key = str(source).casefold()
        record = archives.get(key, {})
        actual = transport.sha(source)
        if record.get('status') != 'PASS' or not record.get('interface_verified') or not record.get('member_crc_sha256_verified') or record.get('sha256') != actual or (frozen_rows is not None and frozen.get(key) != actual):
            raise ValueError(f'独立ZIP尚未实际核SHA/CRC和离线接口，或字节改变：{source}')
        # 当前目录、目录清单和ZIP三方必须同一文件集及字节，避免只审了旧ZIP。
        merged_delivery.verify_package_archive(transport.source_path(row['source']), source)


def freeze(bundle, spec, members, excluded):
    check_ready(spec)
    missing = [str(path) for path in members.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'缺少实际来源：{missing[:8]}')
    if any((bundle / name).exists() for name in transport.GENERATED | {'十二程序与ZIP清单.json'}):
        raise FileExistsError('新总包已有冻结生成物，拒绝覆盖；选择全新--bundle目录')
    bundle.mkdir(parents=True, exist_ok=True)
    index = [{**row, 'config_sha256': transport.sha(transport.source_path(row['source']) / 'config.json'),
              'zip_sha256': transport.sha(transport.source_path(row['archive']))} for row in spec['inference_packages']]
    transport.write_new(bundle / '十二程序与ZIP清单.json', {'packages': index, 'count': 12})
    members['十二程序与ZIP清单.json'] = bundle / '十二程序与ZIP清单.json'
    rows = []
    for number, (name, path) in enumerate(sorted(members.items()), 1):
        before = path.stat()
        digest = transport.sha(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'来源仍在改变：{path}')
        rows.append({'path': name, 'source': str(path.resolve()), 'bytes': before.st_size,
                     'mtime_ns': before.st_mtime_ns, 'sha256': digest})
        if number % 5000 == 0:
            transport.emit({'stage': 'freeze', 'hashed': number, 'total': len(members)})
    check_ready(spec, rows)
    provenance = {'scope': '合并/不合并两方案、A/B协议、十二独立程序及ZIP、完整科学证据和正式文档',
                  'original_files_modified': False, 'excluded': excluded, 'specification': spec,
                  'data_copies': 'raw、A固定分组、B固定分组各一次；不为合并方案另复制原图',
                  'retraining_boundary': '合并历史裁图索引保留证据原路径；缓存已排除。跨机新训练须从包内原图和固定清单重建裁图、索引与ImageNet特征，并重新记录新运行身份；原历史controller不是迁移入口。',
                  'storage': '直接流式ZIP，成员为实际文件字节；无junction'}
    transport.write_new(bundle / '来源与整理说明.json', provenance)
    path = bundle / '来源与整理说明.json'
    rows.append({'path': path.name, 'source': str(path.resolve()), 'bytes': path.stat().st_size,
                 'sha256': transport.sha(path), 'mtime_ns': path.stat().st_mtime_ns})
    transport.write_new(bundle / '冻结成员映射.json', {'files': rows, 'bundle_name': bundle.name})
    path = bundle / '冻结成员映射.json'
    manifest = [{key: row[key] for key in ['path', 'bytes', 'sha256']} for row in rows]
    manifest.append({'path': path.name, 'bytes': path.stat().st_size, 'sha256': transport.sha(path)})
    transport.write_new(bundle / 'SHA256SUMS.json', {'files': manifest, 'count': len(manifest),
                                                  'bytes': sum(row['bytes'] for row in manifest)})
    transport.emit({'stage': 'frozen', 'files': len(manifest)})


def archive(bundle, output):
    # 只复用已有的逐字节传输实现；进程内门禁指向本模块，不编辑旧builder文件。
    original_gate = transport.check_ready
    transport.check_ready = check_ready
    try:
        transport.archive(bundle, output)
    finally:
        transport.check_ready = original_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['spec', 'list', 'freeze', 'archive', 'verify'])
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if bundle == transport.DEFAULT_BUNDLE or bundle == P / 'ResNet项目完整交接_20261102':
        raise ValueError('不得复用原交接目录或旧冻结清单')
    spec = json.loads(args.spec.read_text(encoding='utf-8')) if args.spec else default_spec(bundle)
    if args.stage == 'spec':
        transport.emit(spec)
    elif args.stage in ['archive', 'verify']:
        target = args.archive or bundle.with_suffix('.zip')
        if args.stage == 'archive':
            archive(bundle, target)
        else:
            transport.emit(transport.verify(bundle, target))
    else:
        members, excluded = gather(bundle, spec)
        if args.stage == 'list':
            transport.emit({'stage': 'list', 'files': len(members),
                            'source_bytes': sum(path.stat().st_size for path in members.values() if path.is_file()),
                            'missing': [str(path) for path in members.values() if not path.is_file()],
                            'excluded_entries': len(excluded), 'archive_created': False})
        else:
            freeze(bundle, spec, members, excluded)


if __name__ == '__main__':
    main()
