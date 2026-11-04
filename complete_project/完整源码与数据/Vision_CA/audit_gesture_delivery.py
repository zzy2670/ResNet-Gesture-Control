"""独立核查可移植手势推理包的冻结权重、CPU一致性、离线运行及ZIP解压。

这里只用少量确定样例检查打包实现；不产生新的准确率或重新选阈值。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DELIVERY = PROJECT / 'ResNet手势识别交付'
RUN = HERE / 'runs' / 'A_20261102_TPE'
MODEL_DIRS = {'ResNet18': RUN, 'ResNet34': RUN / 'compare_resnet34',
              'ResNet50': RUN / 'compare_resnet50'}
EXPECTED_ARCH = {'ResNet18': 'resnet18', 'ResNet34': 'resnet34', 'ResNet50': 'resnet50'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_module(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def check_package_files(package):
    manifest = read_json(package / 'SHA256SUMS.json')
    listed = set()
    for item in manifest['files']:
        path = package / item['path']
        assert path.resolve().is_relative_to(package.resolve())
        assert path.stat().st_size == item['bytes']
        assert sha256(path) == item['sha256'], path
        listed.add(item['path'])
    actual = {path.relative_to(package).as_posix() for path in package.rglob('*') if path.is_file()}
    assert actual == listed | {'SHA256SUMS.json'}, actual ^ listed
    return len(listed)


def check_hands(expected, actual):
    # 同CPU、同batch=1入口下比较检测框、概率和接受决定，而不是旧GPU保存值。
    assert expected['prediction'] == actual['prediction']
    assert expected['image_size'] == actual['image_size']
    assert len(expected['hands']) == len(actual['hands'])
    max_difference = 0.0
    for baseline, packaged in zip(expected['hands'], actual['hands']):
        for key in ['label', 'accepted']:
            assert baseline[key] == packaged[key], (key, baseline[key], packaged[key])
        for key in ['detector_score', 'confidence']:
            difference = abs(baseline[key] - packaged[key])
            max_difference = max(max_difference, difference)
            assert difference <= 1e-5, (key, difference)
        for key in ['bbox', 'probabilities']:
            assert len(baseline[key]) == len(packaged[key])
            for left, right in zip(baseline[key], packaged[key]):
                difference = abs(left - right)
                max_difference = max(max_difference, difference)
                assert difference <= (1e-4 if key == 'bbox' else 1e-5), (key, difference)
    return max_difference


@contextmanager
def audit_workspace(model_name):
    # 留存独审中间结果给root统一处理；只创建本任务专属的隔离子目录。
    parent = HERE / 'runs' / 'A_deployment_validation_20261102'
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=model_name + '_', dir=parent)).resolve()
    assert folder.is_relative_to(parent.resolve())
    yield folder


def offline_cli_check(package, archive, samples):
    with audit_workspace(package.name) as temporary:
        unpacked = temporary / '中文 空格路径'
        unpacked.mkdir()
        with zipfile.ZipFile(archive) as zf:
            for entry in zf.infolist():
                target = (unpacked / entry.filename).resolve()
                assert target.is_relative_to(unpacked.resolve()), entry.filename
            zf.extractall(unpacked)
        entries = list(unpacked.rglob('predict.py'))
        assert len(entries) == 1, entries
        entry = entries[0]
        extracted_root = entry.parent
        extracted_file_count = check_package_files(extracted_root)
        assert sha256(extracted_root / 'SHA256SUMS.json') == sha256(package / 'SHA256SUMS.json')
        empty_home = temporary / '空的 torch 缓存'
        empty_home.mkdir()
        work = temporary / '另一个 中文工作目录'
        work.mkdir()
        chosen = samples[0]
        sample_path = extracted_root / chosen['file']
        assert sample_path.is_file(), sample_path
        wrapper = temporary / 'run_offline.py'
        wrapper.write_text('''import importlib.util, socket, sys, urllib.request
from pathlib import Path
import torch
def deny(*args, **kwargs):
    raise RuntimeError("独立检查禁止网络和预训练下载")
socket.create_connection=deny
socket.socket.connect=deny
urllib.request.urlopen=deny
torch.hub.load_state_dict_from_url=deny
torch.hub.download_url_to_file=deny
entry=Path(sys.argv[1])
spec=importlib.util.spec_from_file_location("standalone_delivery",entry)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.argv=[str(entry),*sys.argv[2:]]
raise SystemExit(module.main())
''', encoding='utf-8')
        environment = dict(os.environ, TORCH_HOME=str(empty_home),
                           PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
        details_path = temporary / '详细结果.json'
        command = [sys.executable, str(wrapper), str(entry), str(sample_path),
                   '--device', 'cpu', '--threads', '8', '--json', str(details_path)]
        result = subprocess.run(command, cwd=work, env=environment, capture_output=True,
                                text=True, encoding='utf-8', timeout=120)
        assert result.returncode == 0, result.stderr
        assert re.fullmatch(r'[0-5]\r?\n', result.stdout), repr(result.stdout)
        detail = read_json(details_path)
        assert int(result.stdout) == detail['prediction']
        assert not list(empty_home.rglob('*.pth')), '离线入口产生了预训练缓存'
        # 读不到文件必须报错退出，不能把I/O错误伪装为“没有手”。
        invalid = subprocess.run([sys.executable, str(wrapper), str(entry),
                                  str(work / '不存在的图片.jpg'), '--device', 'cpu'],
                                 cwd=work, env=environment, capture_output=True,
                                 text=True, encoding='utf-8', timeout=120)
        assert invalid.returncode != 0
        assert invalid.stdout == '', repr(invalid.stdout)
        assert invalid.stderr.strip()
        return {'zip_sha256': sha256(archive), 'validation_directory': str(temporary),
                'extracted_manifest_files': extracted_file_count,
                'relocated_chinese_space_path': True,
                'cwd_outside_package': True, 'torch_home_initially_empty': True,
                'network_calls_denied': True, 'stdout': result.stdout.strip(),
                'json_prediction': detail['prediction'], 'missing_image_nonzero_exit': True,
                'missing_image_stdout_empty': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-cpu', action='store_true', help='root确认集成完成后才执行CPU模型检查')
    args = parser.parse_args()
    if not args.run_cpu:
        print('独审脚本已准备；确认程序包与ZIP完成后使用 --run-cpu。')
        return
    import torch
    import train_A as baseline
    import compare_resnets_A as comparison
    torch.set_num_threads(8)
    original_resnet = baseline.resnet
    original_rows = {row['image_id']: row for row in baseline.load_split(
        HERE / 'data/A_split_verified_20261102/P1_random_image', 'val')}
    models = []
    archive_manifest = {item['model']: item for item in read_json(DELIVERY / '程序包清单.json')['packages']}
    for name, original_run in MODEL_DIRS.items():
        package = DELIVERY / name
        packaged_file_count = check_package_files(package)
        entry = package / 'predict.py'
        config = read_json(package / 'config.json')
        calibration = read_json(original_run / 'calibration.json')
        assert config['architecture'] == EXPECTED_ARCH[name]
        assert config['thresholds'] == {
            'detector': calibration['selected']['detector_threshold'],
            'classifier': calibration['selected']['class_threshold']}
        assert calibration['selection_split'] == 'val' and calibration['test_accessed'] is False
        for role, original_name in [('classifier', 'classifier_best.pt'), ('detector', 'detector_best.pt')]:
            specification = config['weights'][role]
            assert not Path(specification['path']).is_absolute()
            assert sha256(package / specification['path']) == calibration['weights'][original_name]
            assert specification['sha256'] == calibration['weights'][original_name]
        assert str(HERE) not in entry.read_text(encoding='utf-8')
        module = load_module(entry, 'audit_' + name)
        for label, command in enumerate(['无有效动作指令', '前进', '后退', '停止', '左转', '右转']):
            assert config['mapping'][str(label)]['command'] == command
        for labels, expected in [([], 0), ([0], 0), ([0, 1], 1), ([1, 1], 1),
                                 ([1, 2], 0), ([3, 1, 2], 3), ([0, 4, 4], 4), ([5], 5)]:
            assert module.aggregate_commands(labels) == expected
        sample_manifest = read_json(package / 'samples.json')
        samples = sample_manifest['samples'] if isinstance(sample_manifest, dict) else sample_manifest
        assert len(samples) == 8
        assert {row['truth'] for row in samples if row['kind'] == 'single'} == set(range(6))
        predictor = module.GesturePredictor(device='cpu', threads=8)
        if name == 'ResNet18':
            baseline.resnet = original_resnet
        else:
            comparison.bind_architecture(argparse.Namespace(arch=EXPECTED_ARCH[name]))
        detector, classifier = baseline.frozen_models(original_run, torch.device('cpu'))
        cases = []
        for sample in samples:
            path = package / sample['file']
            if sample['kind'] == 'empty':
                row = {'path': str(path), 'image_id': sample['image_id'], 'labels': []}
            else:
                source = original_rows[sample['image_id']]
                assert sha256(source['path']) == sha256(path)
                assert baseline.aggregate_commands(source['labels']) == sample['truth']
                row = {**source, 'path': str(path)}
            expected = baseline.whole_photo_predictions(
                detector, classifier, [row], torch.device('cpu'),
                detector_threshold=config['thresholds']['detector'],
                class_threshold=config['thresholds']['classifier'], batch_size=1)[0]
            actual = predictor.predict_details(path)
            assert actual['command'] == config['mapping'][str(actual['prediction'])]['command']
            assert actual['class_name'] == config['mapping'][str(actual['prediction'])]['name']
            difference = check_hands(expected, actual)
            if sample['kind'] == 'empty':
                assert actual['prediction'] == 0
            cases.append({'image_id': sample['image_id'], 'kind': sample['kind'],
                          'truth': sample['truth'], 'original_cpu': expected['prediction'],
                          'packaged_cpu': actual['prediction'], 'hand_count': len(actual['hands']),
                          'maximum_numeric_difference': difference})
        del detector, classifier, predictor
        archive_item = archive_manifest[name]
        archive = DELIVERY / archive_item['zip']
        assert archive.stat().st_size == archive_item['bytes']
        assert sha256(archive) == archive_item['sha256']
        offline = offline_cli_check(package, archive, samples)
        models.append({'model': name, 'code_sha256': sha256(entry),
                       'config_sha256': sha256(package / 'config.json'),
                       'manifest_files_verified': packaged_file_count,
                       'cases': cases, 'offline_zip_run': offline})
    report = {'status': 'PASS', 'purpose': '推理实现与打包一致性检查，不是新增准确率评价',
              'models': models, 'thresholds_changed': False, 'training_executed': False,
              'GPU_executed': False, 'original_scientific_artifacts_modified': False}
    (DELIVERY / '交付独立检查.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'models': len(models), 'cases_per_model': 8}, ensure_ascii=False))


if __name__ == '__main__':
    main()
