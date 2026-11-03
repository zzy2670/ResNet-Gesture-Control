"""独立核查B推理包的冻结权重、24例CPU一致性、离线运行及ZIP解压。

这里只用少量确定样例检查打包实现；不产生新的准确率或重新选阈值。
"""
from __future__ import annotations

import argparse
import datetime
import stat
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
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DELIVERY = PROJECT / 'ResNet_B手势识别交付'
RUN = HERE / 'runs' / 'B_20261102_TPE'
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
    folded = set()
    for item in manifest['files']:
        assert item['path'].casefold() not in folded, '清单成员重复'
        folded.add(item['path'].casefold())
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
    parent = HERE / 'runs' / 'B_deployment_validation_20261102'
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=model_name + '_', dir=parent)).resolve()
    assert folder.is_relative_to(parent.resolve())
    yield folder


def offline_cli_check(package, archive, samples, first_expected):
    with audit_workspace(package.name) as temporary:
        unpacked = temporary / '中文 空格路径'
        unpacked.mkdir()
        with zipfile.ZipFile(archive) as zf:
            entries = zf.infolist()
            assert len(entries) == len({x.filename.casefold() for x in entries}), 'ZIP重复成员'
            expected_names = {package.name + '/' + r['path']
                              for r in read_json(package / 'SHA256SUMS.json')['files']}
            expected_names.add(package.name + '/SHA256SUMS.json')
            assert {x.filename for x in entries} == expected_names, 'ZIP有漏件或额外成员'
            for entry in entries:
                assert not stat.S_ISLNK(entry.external_attr >> 16), 'ZIP不允许符号链接'
                assert '\\' not in entry.filename and ':' not in entry.filename, 'ZIP非法成员'
                assert not Path(entry.filename).is_absolute() and '..' not in Path(entry.filename).parts
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
        offline_max_difference = check_hands(first_expected, detail)
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
                'network_calls_denied': True, 'offline_details_max_difference': offline_max_difference,
                'stdout': result.stdout.strip(),
                'json_prediction': detail['prediction'], 'missing_image_nonzero_exit': True,
                'missing_image_stdout_empty': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-cpu', action='store_true', help='root确认集成完成后才执行CPU模型检查')
    args = parser.parse_args()
    if not args.run_cpu:
        print('独审脚本已准备；确认程序包与ZIP完成后使用 --run-cpu。')
        return
    # 控制器完成统一CPU测速以后才允许运行；整个进程隐藏CUDA设备。
    completion = read_json(RUN / 'all_stages_complete.json')
    assert completion['status'] == 'PASS', '等待B全部训练/测试/CPU测速结束'
    final_report = DELIVERY / '交付独立检查.json'
    assert not final_report.exists(), '已有正式审核文件，拒绝无痕覆盖或重复运行'
    for arch in ('resnet18', 'resnet34', 'resnet50'):
        review = read_json(RUN / (arch + '_final_independent_review.json'))
        assert review['status'].startswith('PASS'), 'B模型独审尚未通过：' + arch
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    import socket
    import urllib.request
    import train_A as baseline
    import compare_resnets_A as comparison
    torch.set_num_threads(8)
    # 原算法检测工厂只允许读已有官方缓存；禁止任何网络传输或下载。
    pretrained = read_json(RUN / 'pretrained_identity.json')['detector_initialization']['path']
    torch.hub.set_dir(str(Path(pretrained).parent.parent))
    def deny(*args, **kwargs):
        raise RuntimeError('B独立审核禁止联网和下载')
    socket.create_connection = deny
    socket.socket.connect = deny
    urllib.request.urlopen = deny
    torch.hub.download_url_to_file = deny
    original_resnet = baseline.resnet
    original_rows = {row['image_id']: row for row in baseline.load_split(
        HERE / 'data/B_split_verified_20261102', 'val')}
    # 八例严格按B验证集ID选取，允许预测错误，不以预测表现重选。
    ordered_rows = sorted(original_rows.values(), key=lambda row: row['image_id'])
    prescribed = [(str(label), next(row for row in ordered_rows if row['labels'] == [label]), 'single')
                  for label in range(6)]
    prescribed.append(('multi', next(row for row in ordered_rows if len(row['labels']) > 1), 'multi'))
    models = []
    identity_files = {RUN / 'all_stages_complete.json', DELIVERY / '程序包清单.json',
                      HERE / 'deployment_src/gesture_predict.py'}
    identities = {}
    # 首次使用前记录身份，后续遇到同一文件必须仍为同一批字节。
    def capture(paths):
        for path in paths:
            key, value = str(path), sha256(path)
            if key in identities:
                assert identities[key] == value, '审核期间文件身份改变：' + key
            else:
                identities[key] = value
    capture(identity_files)
    archive_manifest = {item['model']: item for item in read_json(DELIVERY / '程序包清单.json')['packages']}
    assert set(archive_manifest) == set(MODEL_DIRS), '程序包清单必须有B三个模型'
    for name, original_run in MODEL_DIRS.items():
        package = DELIVERY / name
        packaged_file_count = check_package_files(package)
        entry = package / 'predict.py'
        config = read_json(package / 'config.json')
        calibration = read_json(original_run / 'calibration.json')
        assert config['architecture'] == EXPECTED_ARCH[name]
        assert config['provenance']['protocol'] == 'B_fixed_subject_independent_split'
        assert config['provenance']['A_finetuned_weights_used'] is False
        assert config['provenance']['selection'] == 'validation_only'
        assert config['provenance']['run'] == original_run.relative_to(HERE / 'runs').as_posix()
        assert sha256(entry) == sha256(HERE / 'deployment_src/gesture_predict.py')
        assert archive_manifest[name]['protocol'] == 'B'
        assert config['thresholds'] == {
            'detector': calibration['selected']['detector_threshold'],
            'classifier': calibration['selected']['class_threshold']}
        assert calibration['selection_split'] == 'val' and calibration['test_accessed'] is False
        for role, original_name in [('classifier', 'classifier_best.pt'), ('detector', 'detector_best.pt')]:
            specification = config['weights'][role]
            assert not Path(specification['path']).is_absolute()
            assert sha256(package / specification['path']) == calibration['weights'][original_name]
            assert specification['sha256'] == calibration['weights'][original_name]
            assert sha256(original_run / original_name) == calibration['weights'][original_name]
            identity_files.add(original_run / original_name)
            identity_files.add(package / specification['path'])
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
        assert sample_manifest['selection_rule'] == 'B validation first image_id per single class, then first multi-hand; no prediction filtering'
        for sample, (short, row, kind) in zip(samples[:7], prescribed):
            assert sample['image_id'] == row['image_id'] and sample['kind'] == kind
            assert sample['split'] == 'val' and sample['raw_labels'] == row['raw_labels']
            assert sample['truth'] == baseline.aggregate_commands(row['labels'])
            assert sample['file'] == 'examples/' + short + Path(row['path']).suffix.lower()
        assert samples[7] == {'file': 'examples/empty.png', 'image_id': 'synthetic_blank_interface_check',
                              'truth': 0, 'kind': 'empty', 'split': 'synthetic'}
        from PIL import Image
        with Image.open(package / samples[7]['file']) as blank:
            assert blank.mode == 'RGB' and blank.size == (384, 384)
            assert blank.getextrema() == ((255, 255), (255, 255), (255, 255))
        identity_files.update([package / 'config.json', package / 'samples.json', entry,
                               package / 'SHA256SUMS.json', original_run / 'calibration.json'])
        identity_files.update(path for path in package.rglob('*') if path.is_file())
        capture(identity_files)
        assert {row['truth'] for row in samples if row['kind'] == 'single'} == set(range(6))
        predictor = module.GesturePredictor(device='cpu', threads=8)
        if name == 'ResNet18':
            baseline.resnet = original_resnet
        else:
            comparison.bind_architecture(argparse.Namespace(arch=EXPECTED_ARCH[name]))
        detector, classifier = baseline.frozen_models(original_run, torch.device('cpu'))
        cases = []
        first_expected = None
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
            if first_expected is None:
                first_expected = expected
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
        identity_files.add(archive)
        capture(identity_files)
        offline = offline_cli_check(package, archive, samples, first_expected)
        models.append({'model': name, 'code_sha256': sha256(entry),
                       'config_sha256': sha256(package / 'config.json'),
                       'manifest_files_verified': packaged_file_count,
                       'cases': cases, 'offline_zip_run': offline})
    # 冻结审核所依赖的实际文件身份；序列化后再核同一批字节并独占提交PASS。
    capture(identity_files)
    report = {'status': 'PASS', 'protocol': 'B_subject_independent',
              'reviewed_at': datetime.datetime.now().astimezone().isoformat(),
              'purpose': 'B推理实现与打包一致性检查，不是新增准确率评价',
              'identity_sha256': identities, 'validation_split': 'data/B_split_verified_20261102/val',
              'samples_per_model': 8, 'same_CPU_comparison_cases': 24,
              'models': models, 'thresholds_changed': False, 'training_executed': False,
              'GPU_executed': False, 'original_scientific_artifacts_modified': False}
    payload = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    assert read_json(RUN / 'all_stages_complete.json')['status'] == 'PASS'
    for path, value in identities.items():
        assert sha256(Path(path)) == value, '审核期间文件身份改变：' + path
    with final_report.open('xb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({'status': report['status'], 'models': len(models), 'cases_per_model': 8}, ensure_ascii=False))


if __name__ == '__main__':
    main()
