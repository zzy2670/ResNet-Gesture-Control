"""复用原B离线核验工具，适配六个新合并程序；默认只打印计划。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import traceback

sys.dont_write_bytecode = True
RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PROJECT = ROOT.parent
DELIVERY = PROJECT / 'ResNet_合并手势识别交付'
HELPER = ROOT / 'audit_B_gesture_delivery.py'
ARCHES = ('resnet18', 'resnet34', 'resnet50')
MODEL_STATUS = 'PASS_MERGED_MODEL_AUDIT_WITH_LIMITATIONS'
AGGREGATE_STATUS = 'PASS_MERGED_SIX_MODEL_AUDIT_WITH_LIMITATIONS'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(bound, paths):
    for path in paths:
        path = Path(path)
        digest = sha(path)
        key = str(path.resolve())
        if key in bound and bound[key] != digest:
            raise ValueError('审核期间文件身份变化：' + key)
        bound[key] = digest


def commit(path, report, bound):
    # 报告先编码，再复核所有活身份，最后独占写同一字节并落盘。
    payload = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    for file, digest in bound.items():
        if sha(file) != digest:
            raise ValueError('提交前文件变化：' + file)
    with Path(path).open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def evidence(path, value):
    # 中间诊断独占落盘，不把失败诊断伪装为成功结果。
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    with Path(path).open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def plain_contained(path, boundary):
    # Windows仅处理本次新临时目录下的普通文件，拒绝联接、软链及越界。
    path, boundary = Path(path).absolute(), Path(boundary).absolute()
    if path == boundary or not path.is_relative_to(boundary):
        raise ValueError('临时清理目标不在允许目录内部')
    if boundary.is_symlink() or boundary.is_junction():
        raise ValueError('临时根目录不是普通目录')
    current = path
    while current != boundary:
        if current.is_symlink() or current.is_junction():
            raise ValueError('临时路径含软链或联接')
        current = current.parent
    if not path.resolve(strict=True).is_relative_to(boundary.resolve(strict=True)):
        raise ValueError('临时目标最终解析越界')
    return path


def cleanup_extracted_weights(temporary, package, archive, builder, source_snapshot):
    # 离线全部断言成功后，只删除两份可重解压的临时权重；不递归删除任何目录。
    parent = (RUN / 'merged_deployment_validation').absolute()
    temporary = plain_contained(temporary, parent)
    if temporary.parent != parent:
        raise ValueError('只允许当前mkdtemp生成的一层临时目录')
    receipt_path = temporary / 'extracted_weight_copy_cleanup.json'
    removed = []
    try:
        builder.verify_package_archive(package, archive)
        for file, digest in source_snapshot.items():
            if sha(file) != digest:
                raise ValueError('原ZIP或程序manifest在离线测试中变化，保留解压证据')
        extracted = temporary / '中文 空格路径' / package.name
        manifest_path = plain_contained(extracted / 'SHA256SUMS.json', temporary)
        if sha(manifest_path) != sha(package / 'SHA256SUMS.json'):
            raise ValueError('解压manifest和完整源程序不一致')
        manifest = {r['path']: r for r in read(manifest_path)['files']}
        plan = []
        for relative in ('weights/classifier.pt', 'weights/detector.pt'):
            path = plain_contained(extracted / relative, temporary)
            before = path.stat()
            row = manifest[relative]
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError('解压权重不是独占普通文件，保留证据')
            if before.st_size != row['bytes'] or sha(path) != row['sha256']:
                raise ValueError('解压权重字节与已核manifest不符')
            plan.append({'path': str(path), 'bytes': row['bytes'], 'sha256': row['sha256']})
        evidence(temporary / 'extracted_weight_copy_cleanup_plan.json', {'status': 'VERIFIED_PLAN',
            'only_two_new_extracted_weight_copies': True, 'source_snapshot': source_snapshot, 'files': plan})
        for row in plan:
            path = plain_contained(row['path'], temporary)
            before = path.stat()
            if before.st_nlink != 1 or before.st_size != row['bytes'] or sha(path) != row['sha256']:
                raise ValueError('清理前瞬时身份变化，停止后续删除')
            path.unlink()
            removed.append(row)
        evidence(receipt_path, {'status': 'REMOVED_VERIFIED_EXTRACTED_WEIGHT_COPIES',
            'offline_helper_completed_without_exception': True, 'source_package_and_ZIP_reverified': True,
            'source_snapshot': source_snapshot, 'removed': removed, 'removed_bytes': sum(r['bytes'] for r in removed),
            'retained': 'manifest/config/samples/examples/predict/wrapper/详细结果/CLI stdout和stderr记录/清理计划及本回执；原ZIP和完整程序未改。'})
    except BaseException as error:
        if not receipt_path.exists():
            evidence(receipt_path, {'status': 'CLEANUP_FAILED_EVIDENCE_RETAINED', 'removed': removed,
                'error': repr(error), 'remaining_extracted_files_retained': True})
        raise


def readiness(cpu_review_path):
    # 在torch导入或任何权重加载前要求科学与公平测速均已完成并真实独审。
    sys.path.insert(0, str(ROOT))
    builder = load(ROOT / 'build_merged_gesture_delivery.py', 'merged_package_readiness')
    builder.ready(RUN / 'aggregate_review.json')
    aggregate = read(RUN / 'aggregate_review.json')
    bound = dict(aggregate['bound_file_sha256'])
    capture(bound, [RUN / 'aggregate_review.json', RUN / 'all_training_evaluation_complete.json'])
    cpu_result_path = RUN / 'label_variants_cpu_benchmark.json'
    cpu_result, cpu_review = read(cpu_result_path), read(cpu_review_path)
    if cpu_result.get('status') != 'complete' or cpu_review.get('status') != 'PASS' or cpu_review.get('issues'):
        raise ValueError('十二模型公平CPU测速及其独审尚未通过')
    if cpu_review.get('data_sha256') != sha(cpu_result_path):
        raise ValueError('CPU独审未绑定当前实际测速结果data_sha256')
    if len(cpu_result.get('models', [])) != 12:
        raise ValueError('公平测速没有十二模型')
    capture(bound, [cpu_result_path, cpu_review_path, ROOT / 'build_merged_gesture_delivery.py',
                    ROOT / 'deployment_src/gesture_predict.py', HELPER, Path(__file__).resolve(),
                    RUN / 'MERGED_DELIVERY_VALIDATION_PROTOCOL.md'])
    for file, digest in bound.items():
        if sha(file) != digest:
            raise ValueError('依赖已偏离独审：' + file)
    return builder, bound


@contextmanager
def isolated_workspace(name, package, archive, builder):
    # 所有解压和CLI中间证据留在新run下，不写旧B核验目录。
    parent = RUN / 'merged_deployment_validation'
    parent.mkdir(exist_ok=True)
    if parent.is_symlink() or parent.is_junction():
        raise ValueError('新临时验证根目录不允许链接')
    path = Path(tempfile.mkdtemp(prefix=name + '_', dir=parent)).absolute()
    plain_contained(path, parent)
    source_snapshot = {str(archive): sha(archive), str(package / 'SHA256SUMS.json'): sha(package / 'SHA256SUMS.json')}
    try:
        yield path
    except BaseException as error:
        evidence(path / 'extracted_weight_copy_cleanup.json', {'status': 'RETAINED_AFTER_INTERFACE_FAILURE',
            'error': repr(error), 'deleted_files': [], 'all_failure_evidence_retained': True})
        raise
    else:
        cleanup_extracted_weights(path, package, archive, builder, source_snapshot)


def offline_with_transcripts(legacy, package, archive, samples, first_expected, builder):
    # 只在当前函数调用期间增加CLI原始输出持久化，原helper文件不改。
    original_workspace, original_run = legacy.audit_workspace, legacy.subprocess.run
    legacy.audit_workspace = lambda name: isolated_workspace(name, package, archive, builder)
    calls = []
    def logged_run(*args, **kwargs):
        temporary = Path(kwargs['cwd']).parent
        plain_contained(temporary, RUN / 'merged_deployment_validation')
        number = len(calls) + 1
        transcript = temporary / f'cli_call_{number}.json'
        try:
            result = original_run(*args, **kwargs)
        except BaseException as error:
            def text(value):
                return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value
            evidence(transcript, {'command': args[0], 'cwd': str(kwargs['cwd']), 'error': repr(error),
                'stdout': text(getattr(error, 'stdout', None)), 'stderr': text(getattr(error, 'stderr', None))})
            calls.append(transcript)
            raise
        evidence(transcript, {'command': args[0], 'cwd': str(kwargs['cwd']), 'returncode': result.returncode,
                             'stdout': result.stdout, 'stderr': result.stderr})
        calls.append(transcript)
        return result
    legacy.subprocess.run = logged_run
    try:
        result = legacy.offline_cli_check(package, archive, samples, first_expected)
        result['CLI_transcripts'] = [str(path) for path in calls]
        result['extracted_weight_copy_cleanup'] = str(Path(result['validation_directory']) / 'extracted_weight_copy_cleanup.json')
        return result
    finally:
        legacy.audit_workspace = original_workspace
        legacy.subprocess.run = original_run


def run_cpu(cpu_review_path):
    final_path = RUN / 'merged_packages_independent_review.json'
    attempt_path = RUN / 'merged_packages_independent_review.attempt.jsonl'
    if final_path.exists() or attempt_path.exists():
        raise FileExistsError('审核结果或尝试已存在；保留所有证据，禁止无痕重跑')
    builder, bound = readiness(cpu_review_path)
    report = {'status': 'RUNNING', 'models': [], 'archives': [], 'bound_file_sha256': bound}
    with attempt_path.open('x', encoding='utf-8', buffering=1) as journal:
        def event(**fields):
            journal.write(json.dumps({'time': datetime.now().astimezone().isoformat(), **fields}, ensure_ascii=False) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        try:
            event(status='started', script_sha256=sha(__file__))
            # 原helper只定义数学和子进程核验方法，本适配不调用其B专用main。
            legacy = load(HELPER, 'merged_reused_offline_checks')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            import torch
            import socket
            import urllib.request
            from PIL import Image
            import train_A as baseline
            import compare_resnets_A as comparison
            torch.set_num_threads(8)
            base_resnet = baseline.resnet
            baseline.LABELS.update({'two_up': 1, 'two_up_inverted': 1})
            baseline.CLASS_NAMES[:] = ['other', 'peace_or_two_up_forward', 'palm', 'fist_stop', 'like', 'dislike']
            def deny(*args, **kwargs):
                raise RuntimeError('合并包独审禁止网络及权重下载')
            socket.create_connection = deny
            socket.socket.connect = deny
            urllib.request.urlopen = deny
            torch.hub.download_url_to_file = deny
            registry_path = DELIVERY / '程序包清单.json'
            registry = read(registry_path)['packages']
            expected_keys = {(p, a) for p in 'AB' for a in ARCHES}
            assert len(registry) == 6 and {(r['protocol'], r['architecture']) for r in registry} == expected_keys
            capture(bound, [registry_path])
            by_key = {(r['protocol'], r['architecture']): r for r in registry}
            for protocol in 'AB':
                split_root = ROOT / 'data' / f'{protocol}_split_verified_20261102'
                if protocol == 'A':
                    split_root /= 'P1_random_image'
                rows = sorted(baseline.load_split(split_root, 'val'), key=lambda row: row['image_id'])
                by_id = {r['image_id']: r for r in rows}
                selected = [(str(i), next(r for r in rows if r['labels'] == [i]), 'single') for i in range(6)]
                selected += [('multi', next(r for r in rows if len(r['labels']) > 1), 'multi')]
                selected += [(raw, next(r for r in rows if r['raw_labels'] == [raw]), 'raw_forward')
                             for raw in ('peace', 'peace_inverted', 'two_up', 'two_up_inverted')]
                capture(bound, [split_root / 'val/manifest.csv', *sorted((split_root / 'val/annotations').glob('*.json'))])
                for arch in ARCHES:
                    source = RUN / protocol / arch
                    title = arch.replace('resnet', 'ResNet')
                    package = DELIVERY / protocol / title
                    item = by_key[(protocol, arch)]
                    assert item['variant'] == builder.VARIANT and Path(item['directory']) == package
                    archive = Path(item['zip'])
                    assert archive == DELIVERY / protocol / f'{title}_合并{protocol}协议图片识别程序.zip'
                    assert archive.stat().st_size == item['bytes'] and sha(archive) == item['sha256']
                    # 两个既有实现分别核目录manifest和目录/ZIP三方字节一致，完整读成员触发CRC。
                    file_count = legacy.check_package_files(package)
                    builder.verify_package_archive(package, archive)
                    config = read(package / 'config.json')
                    calibration = read(source / 'calibration.json')
                    provenance = config['provenance']
                    assert config['architecture'] == arch and provenance['variant'] == builder.VARIANT
                    assert provenance['protocol'] == protocol and provenance['selection'] == 'merged_validation_only'
                    assert provenance['classifier_retrained'] is True and provenance['postprocessing_zero_to_one'] is False
                    assert set(provenance['class1_raw']) == {'peace', 'peace_inverted', 'two_up', 'two_up_inverted'}
                    assert provenance['run'] == source.relative_to(ROOT).as_posix()
                    assert calibration['selection_split'] == 'val' and calibration['test_accessed'] is False
                    assert config['thresholds'] == {'detector': calibration['selected']['detector_threshold'],
                        'classifier': calibration['selected']['class_threshold']}
                    assert sha(package / 'predict.py') == sha(ROOT / 'deployment_src/gesture_predict.py')
                    for role, name in (('classifier', 'classifier_best.pt'), ('detector', 'detector_best.pt')):
                        weight = config['weights'][role]
                        assert sha(package / weight['path']) == weight['sha256'] == calibration['weights'][name] == sha(source / name)
                    for label, command in enumerate(['无有效动作指令', '前进', '后退', '停止', '左转', '右转']):
                        assert config['mapping'][str(label)]['command'] == command
                    samples_doc = read(package / 'samples.json')
                    samples = samples_doc['samples']
                    assert samples_doc['variant'] == builder.VARIANT and 'no prediction filtering' in samples_doc['selection']
                    assert len(samples) == 12
                    for sample, (short, row, kind) in zip(samples[:11], selected):
                        assert sample['file'] == 'examples/' + short + Path(row['path']).suffix.lower()
                        assert sample['image_id'] == row['image_id'] and sample['raw_labels'] == row['raw_labels']
                        assert sample['labels'] == row['labels'] and sample['truth'] == baseline.aggregate_commands(row['labels'])
                        assert sample['split'] == 'val' and sha(package / sample['file']) == sha(row['path'])
                        capture(bound, [Path(row['path'])])
                    assert samples[-1] == {'file': 'examples/empty.png', 'truth': 0, 'split': 'synthetic'}
                    with Image.open(package / samples[-1]['file']) as blank:
                        assert blank.mode == 'RGB' and blank.size == (384, 384)
                        assert blank.getextrema() == ((255, 255), (255, 255), (255, 255))
                    capture(bound, [p for p in package.rglob('*') if p.is_file()])
                    capture(bound, [archive, source / 'calibration.json', source / 'classifier_best.pt', source / 'detector_best.pt'])
                    module = legacy.load_module(package / 'predict.py', f'checked_merged_{protocol}_{arch}')
                    for labels, expected in (([], 0), ([0, 1], 1), ([1, 2], 0), ([3, 1, 2], 3), ([4, 4], 4)):
                        assert module.aggregate_commands(labels) == expected
                    # 源工厂需要本地官方MobileNet缓存，包内工厂则完全不依赖该缓存。
                    identities = read(source / 'merged_input_identity.json')['inputs_sha256']
                    mobilenet = next(Path(p) for p in identities if Path(p).name == 'mobilenet_v3_large-8738ca79.pth')
                    torch.hub.set_dir(str(mobilenet.parent.parent))
                    if arch == 'resnet18':
                        baseline.resnet = base_resnet
                    else:
                        comparison.bind_architecture(argparse.Namespace(arch=arch))
                    detector, classifier = baseline.frozen_models(source, torch.device('cpu'))
                    predictor = module.GesturePredictor(device='cpu', threads=8)
                    cases, first_expected = [], None
                    for index, sample in enumerate(samples):
                        path = package / sample['file']
                        if index == 11:
                            row = {'path': str(path), 'image_id': 'synthetic_blank_interface_check', 'labels': []}
                            kind = 'empty'
                        else:
                            row = {**by_id[sample['image_id']], 'path': str(path)}
                            kind = selected[index][2]
                        expected = baseline.whole_photo_predictions(detector, classifier, [row], torch.device('cpu'),
                            config['thresholds']['detector'], config['thresholds']['classifier'], 1)[0]
                        actual = predictor.predict_details(path)
                        event(status='case_observed', protocol=protocol, architecture=arch, sample=sample,
                              source_cpu=expected, package_cpu=actual)
                        difference = legacy.check_hands(expected, actual)
                        assert actual['command'] == config['mapping'][str(actual['prediction'])]['command']
                        assert actual['class_name'] == config['mapping'][str(actual['prediction'])]['name']
                        if first_expected is None:
                            first_expected = expected
                        # 自然识别错例保留，接口一致性不要求真实图片都被分类正确。
                        cases.append({'file': sample['file'], 'image_id': row['image_id'], 'kind': kind,
                            'truth': sample['truth'], 'source_cpu': expected['prediction'], 'package_cpu': actual['prediction'],
                            'matches_truth': sample['truth'] == actual['prediction'], 'hand_count': len(actual['hands']),
                            'maximum_numeric_difference': difference})
                        if kind == 'empty':
                            assert actual['prediction'] == 0, '合成白图必须保留为0；失败证据已写journal，不能换图'
                    del detector, classifier, predictor
                    offline = offline_with_transcripts(legacy, package, archive, samples, first_expected, builder)
                    capture(bound, [p for p in Path(offline['validation_directory']).rglob('*') if p.is_file()])
                    result = {'protocol': protocol, 'architecture': arch, 'cases': cases,
                        'file_count': file_count, 'offline_zip_run': offline, 'strict_model_load': True}
                    report['models'].append(result)
                    report['archives'].append({'source': str(archive), 'protocol': protocol, 'architecture': arch,
                        'variant': '合并', 'status': 'PASS', 'interface_verified': True,
                        'member_crc_sha256_verified': True, 'sha256': sha(archive),
                        'directory': str(package), 'directory_manifest_sha256': sha(package / 'SHA256SUMS.json')})
                    event(status='model_complete', **result)
            assert len(report['models']) == 6 and sum(len(m['cases']) for m in report['models']) == 72
            report.update(status='PASS', reviewed_at=datetime.now().astimezone().isoformat(),
                purpose='合并六程序的实现/可移植性一致性，不是新增准确率或阈值选择',
                samples_per_model=12, same_CPU_comparison_cases=72, GPU_executed=False,
                original_scientific_artifacts_modified=False, thresholds_changed=False,
                helper_reused=str(HELPER), identity_sha256=bound)
            commit(final_path, report, bound)
            event(status='complete', report=str(final_path), sha256=sha(final_path))
        except BaseException as error:
            event(status='failed', error=repr(error), traceback=traceback.format_exc(), partial_result=report)
            raise
    print(json.dumps({'status': 'PASS', 'models': 6, 'CPU_comparison_cases': 72, 'report': str(final_path)}, ensure_ascii=False))


def legacy_identity_bindings(protocol, directory, readiness_path):
    # A没有顶层identity_sha256：用原六程序readiness锚定旧review、manifest及原离线ZIP记录。
    prior_path = directory / '交付独立检查.json'
    prior, ready = read(prior_path), read(readiness_path)
    assert prior['status'] == 'PASS' and ready['status'] == 'PASS'
    programs = [r for r in ready['programs'] if r['protocol'] == protocol]
    models = {r['model']: r for r in prior['models']}
    expected = {a.replace('resnet', 'ResNet') for a in ARCHES}
    assert len(programs) == 3 and {r['model'] for r in programs} == set(models) == expected
    bound = {str(prior_path.resolve()): sha(prior_path), str(readiness_path.resolve()): sha(readiness_path)}
    if protocol == 'B':
        assert isinstance(prior['identity_sha256'], dict) and prior['identity_sha256']
        bound.update(prior['identity_sha256'])
    else:
        # 原A额外root执行证据独立保留，不把它冒充文件身份或全新pip环境。
        extra = prior['additional_root_validation']
        assert extra['exit_code'] == 0 and extra['stdout'] in ('0', '1', '2', '3', '4', '5')
        assert extra['clean_environment_pip_install_executed'] is False
    for row in programs:
        title = row['model']
        package = directory / title
        archive = directory / (f'{title}_图片识别程序.zip' if protocol == 'A' else f'{title}_B协议图片识别程序.zip')
        model = models[title]
        assert Path(row['directory']) == package and Path(row['archive']) == archive
        assert row['independent_review_sha256'] == sha(prior_path)
        manifest_path = package / 'SHA256SUMS.json'
        assert sha(manifest_path) == row['package_manifest_sha256']
        assert sha(package / 'predict.py') == model['code_sha256']
        assert sha(package / 'config.json') == model['config_sha256']
        offline = model['offline_zip_run']
        assert offline['zip_sha256'] == row['archive_sha256']
        assert all(offline[k] is True for k in ('relocated_chinese_space_path', 'cwd_outside_package',
            'torch_home_initially_empty', 'network_calls_denied', 'missing_image_nonzero_exit', 'missing_image_stdout_empty'))
        bound[str(archive.resolve())] = row['archive_sha256']
        bound[str(manifest_path.resolve())] = row['package_manifest_sha256']
        manifest = read(manifest_path)['files']
        assert len(manifest) == row['files_verified'] == offline['extracted_manifest_files']
        for item in manifest:
            path = package / item['path']
            assert path.resolve().is_relative_to(package.resolve())
            key = str(path.resolve())
            assert key not in bound or bound[key] == item['sha256']
            bound[key] = item['sha256']
    return prior, bound


def combine_twelve():
    # 复用旧六程序真实接口审核，只读核其当前字节；本步骤不重新模型推理。
    sys.path.insert(0, str(ROOT))
    builder = load(ROOT / 'build_merged_gesture_delivery.py', 'merged_package_archive_checks')
    bound, archives = {}, []
    new_report_path = RUN / 'merged_packages_independent_review.json'
    new = read(new_report_path)
    assert new['status'] == 'PASS' and len(new['archives']) == 6
    for file, digest in new['bound_file_sha256'].items():
        assert sha(file) == digest
        bound[file] = digest
    capture(bound, [new_report_path, Path(__file__).resolve(), ROOT / 'build_merged_gesture_delivery.py',
                    RUN / 'MERGED_DELIVERY_VALIDATION_PROTOCOL.md'])
    readiness_path = ROOT / 'runs/B_20261102_TPE/six_program_readiness.json'
    for protocol, directory in (('A', PROJECT / 'ResNet手势识别交付'), ('B', PROJECT / 'ResNet_B手势识别交付')):
        prior_path = directory / '交付独立检查.json'
        prior, prior_bindings = legacy_identity_bindings(protocol, directory, readiness_path)
        for file, digest in prior_bindings.items():
            assert sha(file) == digest
            bound[file] = digest
        for arch in ARCHES:
            title = arch.replace('resnet', 'ResNet')
            package = directory / title
            filename = f'{title}_图片识别程序.zip' if protocol == 'A' else f'{title}_B协议图片识别程序.zip'
            archive = directory / filename
            assert str(archive.resolve()) in prior_bindings
            builder.verify_package_archive(package, archive)
            capture(bound, [archive, *[p for p in package.rglob('*') if p.is_file()]])
            archives.append({'source': str(archive), 'protocol': protocol, 'architecture': arch,
                'variant': '不合并', 'status': 'PASS', 'interface_verified': True, 'member_crc_sha256_verified': True,
                'sha256': sha(archive), 'directory': str(package),
                'directory_manifest_sha256': sha(package / 'SHA256SUMS.json'), 'interface_review_source': str(prior_path)})
    for item in new['archives']:
        builder.verify_package_archive(Path(item['directory']), Path(item['source']))
        archives.append(item)
    assert len(archives) == 12 and len({r['source'] for r in archives}) == 12
    assert {(r['variant'], r['protocol'], r['architecture']) for r in archives} == {
        (v, p, a) for v in ('合并', '不合并') for p in 'AB' for a in ARCHES}
    report = {'status': 'PASS', 'reviewed_at': datetime.now().astimezone().isoformat(), 'archives': archives,
        'bound_file_sha256': bound, 'scope': '旧六程序沿用真实接口审核且当前字节全核；新六程序72同CPU例及离线审核；12ZIP当前CRC/SHA和目录清单一致。',
        'new_model_inference_in_combine_step': False, 'original_packages_modified': False}
    commit(RUN / 'twelve_packages_independent_review.json', report, bound)
    print(json.dumps({'status': 'PASS', 'archives': 12}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-cpu', action='store_true')
    parser.add_argument('--combine-twelve', action='store_true')
    parser.add_argument('--check-legacy-schema', action='store_true', help='仅小文件/schema校验；不读权重/ZIP内容、不推理')
    parser.add_argument('--cpu-review', type=Path, default=RUN / 'label_variants_cpu_independent_review.json')
    args = parser.parse_args()
    if sum((args.run_cpu, args.combine_twelve, args.check_legacy_schema)) > 1:
        parser.error('模型CPU检查、12ZIP汇总与schema检查分别显式执行')
    if args.run_cpu:
        run_cpu(args.cpu_review)
    elif args.combine_twelve:
        combine_twelve()
    elif args.check_legacy_schema:
        rows = []
        for protocol, directory in (('A', PROJECT / 'ResNet手势识别交付'), ('B', PROJECT / 'ResNet_B手势识别交付')):
            prior, bindings = legacy_identity_bindings(protocol, directory, ROOT / 'runs/B_20261102_TPE/six_program_readiness.json')
            rows.append({'protocol': protocol, 'models': len(prior['models']), 'bound_files': len(bindings),
                         'has_top_level_identity': 'identity_sha256' in prior})
        print(json.dumps({'status': 'PASS_ACTUAL_LEGACY_SCHEMA_ONLY', 'protocols': rows,
            'weight_or_ZIP_content_read': False, 'models_loaded': False}, ensure_ascii=False))
    else:
        print(json.dumps({'status': 'PREPARED_NOT_EXECUTED', 'helper_reused': str(HELPER),
            'new_models': 6, 'fixed_samples_each': 12, 'CPU_comparison_cases': 72,
            'science_gate': str(RUN / 'aggregate_review.json'), 'cpu_review_gate': str(args.cpu_review),
            'new_report': str(RUN / 'merged_packages_independent_review.json'),
            'twelve_report': str(RUN / 'twelve_packages_independent_review.json'), 'models_loaded': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
