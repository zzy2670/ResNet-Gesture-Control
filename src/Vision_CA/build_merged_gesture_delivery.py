"""整理真实重新训练的合并标签六模型；只在正式训练和独审完成后复制与压缩。"""
from pathlib import Path, PurePosixPath
import argparse
import csv
import hashlib
import json
import shutil
import zipfile
import stat

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/MERGED_20261102'
DELIVERY = ROOT.parent / 'ResNet_合并手势识别交付'
VARIANT = 'merged_peace_two_up_6class'
MODEL_REVIEW_STATUS = 'PASS_MERGED_MODEL_AUDIT_WITH_LIMITATIONS'
AGGREGATE_REVIEW_STATUS = 'PASS_MERGED_SIX_MODEL_AUDIT_WITH_LIMITATIONS'
LABELS = {'peace': 1, 'peace_inverted': 1, 'two_up': 1, 'two_up_inverted': 1,
          'stop': 2, 'stop_inverted': 2, 'palm': 2, 'fist': 3, 'like': 4, 'dislike': 5}
MAPPING = {str(key): {'name': name, 'command': command} for key, name, command in [
    (0, '其他或无有效指令', '无有效动作指令'), (1, 'V形或并拢两指前进', '前进'),
    (2, '张开手掌', '后退'), (3, '握拳', '停止'), (4, '拇指向上', '左转'), (5, '拇指向下', '右转')]}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2))


def aggregate(labels):
    if 3 in labels:
        return 3
    active = set(labels) - {0}
    return next(iter(active)) if len(active) == 1 else 0


def split_root(protocol):
    root = ROOT / f'data/{protocol}_split_verified_20261102'
    return root / 'P1_random_image' if protocol == 'A' else root


def examples(protocol):
    # 新照片真值来自合并raw映射，绝不借用旧标签或按模型正确率筛样。
    rows = sorted(read(RUN / protocol / 'resnet18/photo_truth_val.json'), key=lambda row: row['image_id'])
    for row in rows:
        expected = [LABELS.get(raw, 0) for raw in row['raw_labels']]
        if row['labels'] != expected or row['truth'] != aggregate(expected):
            raise ValueError('合并验证真值与新raw映射不一致')
    with (split_root(protocol) / 'val/manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        photos = {row['image_id']: row for row in csv.DictReader(stream)}
    chosen = [(str(label), next(row for row in rows if row['labels'] == [label])) for label in range(6)]
    chosen += [('multi', next(row for row in rows if len(row['labels']) > 1))]
    chosen += [(raw, next(row for row in rows if row['raw_labels'] == [raw]))
               for raw in ['peace', 'peace_inverted', 'two_up', 'two_up_inverted']]
    result = []
    for name, row in chosen:
        relative = Path(photos[row['image_id']]['image_path'].replace('\\', '/'))
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('验证照片路径必须位于包内固定分组')
        result.append((name, row, split_root(protocol) / 'val' / relative))
    return result


def ready(review_path, frozen_rows=None):
    if read(RUN / 'all_training_evaluation_complete.json').get('status') != 'PASS':
        raise ValueError('合并六模型尚未全部完成训练与评价')
    review = verify_bound_review(review_path, AGGREGATE_REVIEW_STATUS, frozen_rows)
    models = review.get('models', [])
    expected = {(protocol, f'resnet{size}') for protocol in 'AB' for size in [18, 34, 50]}
    if len(models) != 6 or {(row['protocol'], row['architecture']) for row in models} != expected:
        raise ValueError('合并汇总独审必须覆盖六个唯一模型')
    for protocol, arch in sorted(expected):
        path = RUN / protocol / arch / 'model_review.json'
        # 汇总必须锚定当前单模型审核本身，不能悄悄替换下级审核。
        if review['bound_file_sha256'].get(str(path.resolve())) != sha(path):
            raise ValueError(f'汇总未绑定当前单模型审核：{path}')
        model_review = read(path)
        if model_review.get('status') != MODEL_REVIEW_STATUS or model_review.get('issues'):
            raise ValueError('合并单模型真实独审未通过')
        # 汇总已逐一核过所有bound字节；这里只核下级集合纳入同一汇总，避免重复读大特征。
        if not model_review.get('bound_file_sha256') or any(
                review['bound_file_sha256'].get(source) != digest
                for source, digest in model_review['bound_file_sha256'].items()):
            raise ValueError('单模型审核依赖未全部纳入汇总绑定')
        if (model_review.get('protocol'), model_review.get('architecture')) != (protocol, arch):
            raise ValueError('合并单模型审核身份不符')
        for name, digest in model_review.get('weights', {}).items():
            if sha(RUN / protocol / arch / name) != digest:
                raise ValueError('当前模型权重偏离单模型独审')


def verify_bound_review(path, accepted_status, frozen_rows=None):
    # 审核结论保留WITH_LIMITATIONS原文，并逐项核实际科学依赖字节。
    report = read(path)
    if report.get('status') != accepted_status or report.get('issues'):
        raise ValueError(f'真实科学审核未通过：{path}')
    bound = report.get('bound_file_sha256')
    if not isinstance(bound, dict) or not bound:
        raise ValueError(f'审核缺少非空bound_file_sha256：{path}')
    frozen = {} if frozen_rows is None else {str(Path(row['source']).resolve()).casefold(): row['sha256'] for row in frozen_rows}
    review_key = str(Path(path).resolve()).casefold()
    if review_key in frozen and frozen[review_key] != sha(path):
        raise ValueError('科学审核本身偏离冻结归档身份')
    for source, digest in bound.items():
        if sha(Path(source)) != digest:
            raise ValueError(f'审核绑定的科学文件发生变化：{source}')
        key = str(Path(source).resolve()).casefold()
        if key in frozen and frozen[key] != digest:
            raise ValueError(f'科学审核字节与冻结归档不一致：{source}')
    return report


def verify_package_archive(package, archive):
    # 逐成员完整读取同时核SHA、字节数和CRC；这不替代离线真实图片接口测试。
    manifest = read(package / 'SHA256SUMS.json')['files']
    manifest_names = []
    for row in manifest:
        name = row['path']
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or ':' in name or '\\' in name:
            raise ValueError('独立程序清单包含不安全路径')
        manifest_names.append(name)
    if not manifest_names or len(manifest_names) != len(set(name.casefold() for name in manifest_names)):
        raise ValueError('独立程序清单为空或存在重复成员')
    actual = set()
    for path in package.rglob('*'):
        if path.is_symlink() or path.is_junction():
            raise ValueError('独立程序目录含链接')
        relative = path.relative_to(package)
        if path.is_file() and '__pycache__' not in relative.parts and path.suffix != '.pyc':
            actual.add(relative.as_posix())
    if actual != set(manifest_names) | {'SHA256SUMS.json'}:
        raise ValueError('当前独立程序目录文件集与清单不一致')
    for row in manifest:
        path = package / row['path']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError(f'当前程序文件偏离清单：{path}')
    with zipfile.ZipFile(archive) as stream:
        infos = stream.infolist()
        names = [item.filename for item in infos]
        if any(stat.S_ISLNK(item.external_attr >> 16) for item in infos):
            raise ValueError('独立ZIP内含符号链接')
        expected = {package.name + '/' + row['path'] for row in manifest} | {package.name + '/SHA256SUMS.json'}
        if set(names) != expected or len(names) != len(expected):
            raise ValueError('独立ZIP成员集合不一致')
        if stream.read(package.name + '/SHA256SUMS.json') != (package / 'SHA256SUMS.json').read_bytes():
            raise ValueError('独立ZIP内部清单偏离实际清单')
        for row in manifest:
            digest, size = hashlib.sha256(), 0
            with stream.open(package.name + '/' + row['path']) as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block)
                    size += len(block)
            if digest.hexdigest() != row['sha256'] or size != row['bytes']:
                raise ValueError('独立ZIP成员SHA或长度不符')


def build(review_path):
    ready(review_path)
    if DELIVERY.exists():
        raise FileExistsError('新合并交付目录已存在，拒绝覆盖或无痕重建')
    from PIL import Image
    packages = []
    for protocol in ['A', 'B']:
        selected = examples(protocol)
        for size in [18, 34, 50]:
            arch, title = f'resnet{size}', f'ResNet{size}'
            source = RUN / protocol / arch
            identity = read(source / 'merged_input_identity.json')
            if identity['label_version'] != VARIANT or identity['unmerged_classifier_weights_used']:
                raise ValueError('不是独立合并训练模型')
            calibration = read(source / 'calibration.json')
            package = DELIVERY / protocol / title
            package.mkdir(parents=True)
            weights = {}
            for key, filename in [('classifier', 'classifier_best.pt'), ('detector', 'detector_best.pt')]:
                digest = sha(source / filename)
                if digest != calibration['weights'][filename]:
                    raise ValueError('合并权重与新val冻结身份不一致')
                target = package / 'weights' / (key + '.pt')
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(source / filename, target)
                if sha(target) != digest:
                    raise ValueError('权重复制字节不一致')
                weights[key] = {'path': 'weights/' + key + '.pt', 'sha256': digest}
            original = ROOT / f'runs/{protocol}_20261102_TPE'
            original = original if size == 18 else original / ('compare_' + arch)
            if weights['classifier']['sha256'] == sha(original / 'classifier_best.pt'):
                raise ValueError('合并模型不能冒用旧分类器权重')
            config = {'architecture': arch,
                      'classifier_dropout': read(source / 'fixed_head_summary.json')['best_params']['dropout'],
                      'thresholds': {'detector': calibration['selected']['detector_threshold'],
                                     'classifier': calibration['selected']['class_threshold']},
                      'weights': weights, 'mapping': MAPPING,
                      'preprocessing': {'input': 'full_RGB_photo', 'crop_side_factor': 1.2,
                                        'crop_size': [224, 224], 'resize': 'PIL_BILINEAR',
                                        'mean': [.485, .456, .406], 'std': [.229, .224, .225]},
                      'provenance': {'variant': VARIANT, 'protocol': protocol,
                                     'class1_raw': [raw for raw, label in LABELS.items() if label == 1],
                                     'manifest_sha256': {split: sha(split_root(protocol) / split / 'manifest.csv')
                                                         for split in ['train', 'val', 'test']},
                                     'run': source.relative_to(ROOT).as_posix(),
                                     'selection': 'merged_validation_only',
                                     'classifier_retrained': True, 'postprocessing_zero_to_one': False,
                                     'detector_reused_within_same_protocol': True,
                                     'training_required': False}}
            write(package / 'config.json', config)
            shutil.copy2(ROOT / 'deployment_src/gesture_predict.py', package / 'predict.py')
            samples = []
            for name, row, photo in selected:
                relative = 'examples/' + name + photo.suffix.lower()
                (package / 'examples').mkdir(exist_ok=True)
                shutil.copy2(photo, package / relative)
                samples.append({'file': relative, 'image_id': row['image_id'], 'truth': row['truth'],
                                'raw_labels': row['raw_labels'], 'labels': row['labels'], 'split': 'val'})
            Image.new('RGB', (384, 384), 'white').save(package / 'examples/empty.png')
            samples.append({'file': 'examples/empty.png', 'truth': 0, 'split': 'synthetic'})
            write(package / 'samples.json', {'variant': VARIANT, 'selection': 'first validation image_id per single six-class, multi-hand and four raw forward gestures; no prediction filtering', 'samples': samples})
            write(package / 'requirements.txt', 'torch==2.11.0\ntorchvision==0.26.0\nPillow==12.1.1\n')
            write(package / 'README.md', f'''# 合并标签 {protocol}协议 {title}

输入完整照片，输出0—5。1包含peace、peace_inverted、two_up、two_up_inverted，表示V形或并拢两指前进。此包使用重新训练的合并分类器，阈值由相同协议的新验证结果固定。

安装`requirements.txt`后执行`python predict.py "examples/two_up.jpg"`，或换成自己的照片。默认CPU；需要框与概率可加`--json result.json`。模型已包含，推理不下载权重。标准输出只有整数；读图或加载失败写stderr并非零退出。

0为无有效动作，1前进，2张掌后退，3握拳停止，4拇指向上左转，5拇指向下右转。多手握拳优先；其余不同非零动作冲突输出0。程序只识别图片，不连接机器人。A允许同人跨组；B人员互斥，两者的分数不可直接混为部署保证。

`config.json`保存协议、合并版本、固定分组SHA、双权重SHA和新val阈值。示例按验证ID固定选择，可能识别错误；它们不构成新增benchmark。合成空白图仅用于接口测试。

连续图片可`from predict import GesturePredictor`，加载一次`model=GesturePredictor()`后调用`model.predict("图片.jpg")`。计时应区分加载和稳定推理。
''')
            files = [{'path': file.relative_to(package).as_posix(), 'bytes': file.stat().st_size, 'sha256': sha(file)}
                     for file in sorted(package.rglob('*')) if file.is_file()]
            write(package / 'SHA256SUMS.json', {'files': files})
            archive = DELIVERY / protocol / f'{title}_合并{protocol}协议图片识别程序.zip'
            with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as stream:
                for file in sorted(package.rglob('*')):
                    if file.is_file():
                        stream.write(file, title + '/' + file.relative_to(package).as_posix())
            verify_package_archive(package, archive)
            packages.append({'variant': VARIANT, 'protocol': protocol, 'architecture': arch,
                             'directory': str(package), 'zip': str(archive), 'sha256': sha(archive),
                             'bytes': archive.stat().st_size, 'member_crc_sha_verified': True,
                             'independent_offline_interface_review_pending': True})
    ready(review_path)
    write(DELIVERY / '程序包清单.json', {'packages': packages})
    print(json.dumps({'packages_built': 6, 'manifest': str(DELIVERY / '程序包清单.json'),
                      'offline_interface_review_pending': True}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['list', 'build'])
    parser.add_argument('--review', type=Path, default=RUN / 'aggregate_review.json')
    args = parser.parse_args()
    if args.stage == 'build':
        build(args.review)
    else:
        print(json.dumps({'variant': VARIANT, 'packages': [str(DELIVERY / p / f'ResNet{n}') for p in 'AB' for n in [18, 34, 50]],
                          'completion_required': str(RUN / 'all_training_evaluation_complete.json'),
                          'independent_review_required': str(args.review), 'build_executed': False}, ensure_ascii=False))
