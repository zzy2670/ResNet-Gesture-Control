"""将B人员互斥协议的冻结模型整理成三个独立图片识别包。

复用已核验的离线推理实现；只复制权重，不训练、不调整验证阈值。
"""
from pathlib import Path
import json
import zipfile

import torch
from PIL import Image

import train_A as experiment
from build_gesture_delivery import copy_exact, sha, json_new, write_new, MAPPING

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RUN = ROOT / 'runs/B_20261102_TPE'
DELIVERY = PROJECT / 'ResNet_B手势识别交付'


def examples():
    # 示例仅按验证集ID选取，不按测试表现或预测对错筛选。
    rows = sorted(experiment.load_split(ROOT / 'data/B_split_verified_20261102', 'val'),
                  key=lambda item: item['image_id'])
    chosen = [(str(label), next(row for row in rows if row['labels'] == [label]), 'single')
              for label in range(6)]
    chosen.append(('multi', next(row for row in rows if len(row['labels']) > 1), 'multi'))
    return chosen


def main():
    completion = json.loads((RUN / 'all_stages_complete.json').read_text(encoding='utf-8'))
    if completion['status'] != 'PASS':
        raise RuntimeError('B固定预算训练、测试和测速未全部结束，不能生成最终推理包')
    selected_examples = examples()
    packages = []
    for title, arch in [('ResNet18', 'resnet18'), ('ResNet34', 'resnet34'), ('ResNet50', 'resnet50')]:
        source = RUN if arch == 'resnet18' else RUN / ('compare_' + arch)
        package = DELIVERY / title
        calibration = json.loads((source / 'calibration.json').read_text(encoding='utf-8'))
        checkpoint = torch.load(source / 'classifier_best.pt', map_location='cpu', weights_only=True)
        weights = {}
        for name, filename in [('classifier', 'classifier_best.pt'), ('detector', 'detector_best.pt')]:
            digest = sha(source / filename)
            if digest != calibration['weights'][filename]:
                raise RuntimeError(f'{title}权重与验证冻结身份不一致')
            copy_exact(source / filename, package / 'weights' / (name + '.pt'))
            weights[name] = {'path': 'weights/' + name + '.pt', 'sha256': digest}
        config = {
            'architecture': arch, 'classifier_dropout': checkpoint['params']['dropout'],
            'thresholds': {'detector': calibration['selected']['detector_threshold'],
                           'classifier': calibration['selected']['class_threshold']},
            'weights': weights, 'mapping': MAPPING,
            'preprocessing': {'input': 'full_RGB_photo', 'crop_side_factor': 1.2,
                              'crop_size': [224, 224], 'resize': 'PIL_BILINEAR',
                              'mean': [.485, .456, .406], 'std': [.229, .224, .225]},
            'provenance': {'protocol': 'B_fixed_subject_independent_split',
                           'run': source.relative_to(ROOT / 'runs').as_posix(),
                           'selection': 'validation_only', 'training_required': False,
                           'A_finetuned_weights_used': False}}
        json_new(package / 'config.json', config)
        copy_exact(ROOT / 'deployment_src/gesture_predict.py', package / 'predict.py')
        samples = []
        for short, row, kind in selected_examples:
            relative = 'examples/' + short + Path(row['path']).suffix.lower()
            copy_exact(row['path'], package / relative)
            samples.append({'file': relative, 'image_id': row['image_id'],
                            'truth': experiment.aggregate_commands(row['labels']),
                            'kind': kind, 'raw_labels': row['raw_labels'], 'split': 'val'})
        blank = package / 'examples/empty.png'
        if not blank.exists():
            Image.new('RGB', (384, 384), 'white').save(blank)
        samples.append({'file': 'examples/empty.png', 'image_id': 'synthetic_blank_interface_check',
                        'truth': 0, 'kind': 'empty', 'split': 'synthetic'})
        json_new(package / 'samples.json', {
            'selection_rule': 'B validation first image_id per single class, then first multi-hand; no prediction filtering',
            'purpose': '接口示例与原程序CPU一致性检查，不是新增benchmark', 'samples': samples})
        write_new(package / 'requirements.txt', 'torch==2.11.0\ntorchvision==0.26.0\nPillow==12.1.1\n')
        write_new(package / 'README.md', f'''# B协议{title}：输入完整照片，输出0—5

本包使用按人员互斥划分的B训练模型，与原A程序分开保存。三种架构的B检测器相同，分类器分别训练；检测器和分类器均没有继承A的微调权重。

## 使用步骤

1. 保留整个文件夹，在Python 3.12环境安装`requirements.txt`。
2. 执行`python predict.py "examples/1.jpg"`或将路径换成自己的完整照片。
3. 标准输出为一个整数。需保留框和概率时加`--json result.json`。

依赖安装可能需要网络；模型文件已经包含，推理不会联网下载权重。默认CPU，可显式加`--device cuda`。读图或加载失败写标准错误并返回非零退出码，不伪装成0。

| 数字 | 手势 | 控制约定 |
|---|---|---|
| 0 | 其他、无有效动作或动作冲突 | 不发起或续期运动 |
| 1 | peace、peace_inverted | 前进 |
| 2 | stop、stop_inverted、palm | 后退 |
| 3 | fist | 明确停止当前运动 |
| 4 | like | 左转 |
| 5 | dislike | 右转 |

已接受的握拳优先输出3；否则非零有效动作只有一种就输出该数字，不同动作冲突输出0。0类手不抵消另一只手的单一有效指令。程序仅识别图片，不会控制机器人。

## 多张照片

```python
from predict import GesturePredictor

# 模型只加载一次，后续重复调用。
model = GesturePredictor()
number = model.predict("自己的照片.jpg")
print(number)
```

`config.json`保存架构、类别映射、预处理、检测阈值{config['thresholds']['detector']}、分类阈值{config['thresholds']['classifier']}和权重SHA256。阈值仅由B验证集确定，不能按演示图片修改后仍引用原测试分数。

`weights`是两个冻结权重；`predict.py`是离线架构与CLI/API；`samples.json`记录验证样例及真实标签；`SHA256SUMS.json`用于完整性核查。示例按固定ID选取，可能预测错误；空白图只是合成接口检查，不等于真实无手场景准确率。

报告毫秒值是模型加载后的耗时，不包含每次启动Python和读权重。B测试覆盖本地HaGRID样本中的未见人员，不等于实机机器人、所有背景或新设备已通过验收。
''')
        manifest = [{'path': f.relative_to(package).as_posix(), 'bytes': f.stat().st_size,
                     'sha256': sha(f)} for f in sorted(package.rglob('*'))
                    if f.is_file() and f.name != 'SHA256SUMS.json' and '__pycache__' not in f.parts]
        json_new(package / 'SHA256SUMS.json', {'files': manifest})
        archive = DELIVERY / (title + '_B协议图片识别程序.zip')
        if archive.exists():
            raise FileExistsError(f'不覆盖已有压缩包：{archive}')
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as handle:
            for item in manifest:
                handle.write(package / item['path'], title + '/' + item['path'])
            handle.write(package / 'SHA256SUMS.json', title + '/SHA256SUMS.json')
        with zipfile.ZipFile(archive) as handle:
            if handle.testzip() is not None:
                raise RuntimeError('压缩包CRC检查失败')
        packages.append({'model': title, 'protocol': 'B', 'directory': title, 'zip': archive.name,
                         'bytes': archive.stat().st_size, 'sha256': sha(archive),
                         'thresholds': config['thresholds']})
        print(json.dumps(packages[-1], ensure_ascii=False), flush=True)
    json_new(DELIVERY / '程序包清单.json', {'packages': packages})
    write_new(DELIVERY / 'README.md', '''# B人员互斥协议：三个独立识别程序

三个同名文件夹分别包含ResNet18、34、50的B冻结模型，可单独运行。也可以解压对应`*_B协议图片识别程序.zip`。原A程序保存在相邻的`ResNet手势识别交付`目录，不被本目录替代。

输入完整RGB照片，按包内README执行`python predict.py "照片路径"`，得到0—5中的一个整数。程序不连接机器人。A/B实测准确率和速度请读新版《A与B协议ResNet训练与对比实验报告_v2.docx》；独立打包一致性检查不计为新的测试成绩。
''')


if __name__ == '__main__':
    main()
