"""复制冻结模型，生成三个独立图片识别包及统一交付说明；不改训练产物。"""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
import torch
from PIL import Image
import train_A as experiment

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RUN = ROOT / 'runs/A_20261102_TPE'
DELIVERY = PROJECT / 'ResNet手势识别交付'
MAPPING = {
    '0': {'name': '其他或无有效指令', 'command': '无有效动作指令'},
    '1': {'name': 'V形手势', 'command': '前进'},
    '2': {'name': '张开手掌', 'command': '后退'},
    '3': {'name': '握拳', 'command': '停止'},
    '4': {'name': '拇指向上', 'command': '左转'},
    '5': {'name': '拇指向下', 'command': '右转'},
}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def copy_exact(source, destination):
    # 同一目标只能复用字节相同的文件，不能覆盖用户后来修改的版本。
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(source) != sha(destination):
            raise FileExistsError(f'目标已有不同内容：{destination}')
    else:
        shutil.copy2(source, destination)
    if sha(source) != sha(destination):
        raise ValueError(f'复制校验失败：{destination}')


def write_new(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode('utf-8')
    if path.exists() and path.read_bytes() != data:
        raise FileExistsError(f'目标已有不同内容：{path}')
    path.write_bytes(data)


def json_new(path, value):
    write_new(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def example_rows():
    # 固定按ID选择验证集样例，不根据模型是否预测正确挑图。
    rows = sorted(experiment.load_split(ROOT / 'data/A_split_verified_20261102/P1_random_image', 'val'),
                  key=lambda row: row['image_id'])
    selected = []
    for label in range(6):
        row = next(item for item in rows if item['labels'] == [label])
        selected.append((str(label), row, 'single'))
    selected.append(('multi', next(item for item in rows if len(item['labels']) > 1), 'multi'))
    return selected


def main():
    DELIVERY.mkdir(exist_ok=True)
    examples = example_rows()
    summary = []
    for name, arch, run in [('ResNet18', 'resnet18', RUN),
                            ('ResNet34', 'resnet34', RUN / 'compare_resnet34'),
                            ('ResNet50', 'resnet50', RUN / 'compare_resnet50')]:
        package = DELIVERY / name
        package.mkdir(exist_ok=True)
        calibration = json.loads((run / 'calibration.json').read_text(encoding='utf-8'))
        checkpoint = torch.load(run / 'classifier_best.pt', map_location='cpu', weights_only=True)
        weights = {}
        for key, filename in [('classifier', 'classifier_best.pt'), ('detector', 'detector_best.pt')]:
            source = run / filename
            digest = sha(source)
            assert digest == calibration['weights'][filename]
            target = package / 'weights' / (key + '.pt')
            copy_exact(source, target)
            weights[key] = {'path': 'weights/' + key + '.pt', 'sha256': digest}
        config = {'architecture': arch, 'classifier_dropout': checkpoint['params']['dropout'],
                  'thresholds': {'detector': calibration['selected']['detector_threshold'],
                                 'classifier': calibration['selected']['class_threshold']},
                  'weights': weights, 'mapping': MAPPING,
                  'preprocessing': {'input': 'full_RGB_photo', 'crop_side_factor': 1.2,
                                    'crop_size': [224, 224], 'resize': 'PIL_BILINEAR',
                                    'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
                  'provenance': {'protocol': 'A_fixed_random_image_split',
                                 'run': 'A_20261102_TPE' if arch == 'resnet18' else 'A_20261102_TPE/compare_' + arch,
                                 'selection': 'validation_only', 'training_required': False}}
        json_new(package / 'config.json', config)
        copy_exact(ROOT / 'deployment_src/gesture_predict.py', package / 'predict.py')
        samples = []
        for short, row, kind in examples:
            relative = 'examples/' + short + Path(row['path']).suffix.lower()
            copy_exact(row['path'], package / relative)
            samples.append({'file': relative, 'image_id': row['image_id'], 'truth': experiment.aggregate_commands(row['labels']),
                            'kind': kind, 'raw_labels': row['raw_labels'], 'split': 'val'})
        blank = package / 'examples/empty.png'
        if not blank.exists():
            Image.new('RGB', (384, 384), 'white').save(blank)
        samples.append({'file': 'examples/empty.png', 'image_id': 'synthetic_blank_interface_check',
                        'truth': 0, 'kind': 'empty', 'split': 'synthetic'})
        json_new(package / 'samples.json', {'selection_rule': 'validation single-hand first image_id per class, then first multi-hand; no prediction filtering',
                                           'purpose': '接口示例与原程序CPU一致性检查，不是新benchmark', 'samples': samples})
        write_new(package / 'requirements.txt', 'torch==2.11.0\ntorchvision==0.26.0\nPillow==12.1.1\n')
        write_new(package / 'README.md', f'''# {name}：输入照片，输出0—5

本文件夹包含已训练的{name}分类器、共用的手部检测器和冻结阈值。直接输入完整照片，无需先手动裁出手，也不需要原训练数据或原项目目录。

## 快速使用

解压整个文件夹，保留`predict.py`、`config.json`和`weights`的相对位置。使用Python 3.12，在该文件夹打开终端。

```powershell
python -m pip install -r requirements.txt
python predict.py "examples/1.jpg"
python predict.py "你的照片.jpg"
```

终端标准输出只有一个整数，例如`1`；错误写入标准错误并返回非零退出码。权重已放在包中，预测不会联网下载模型。依赖第一次安装需要网络；本机已有相同依赖时无需再次安装。程序默认使用CPU，CUDA环境可显式加`--device cuda`。

## 输出对应关系

| 数字 | 手势 | 约定的控制含义 |
|---|---|---|
| 0 | 其他手势、无被接受的有效手势或有效动作冲突 | 无有效动作指令 |
| 1 | peace、peace_inverted：V形手势 | 前进 |
| 2 | stop、stop_inverted、palm：张开手掌 | 后退 |
| 3 | fist：握拳 | 停止 |
| 4 | like：拇指向上 | 左转 |
| 5 | dislike：拇指向下 | 右转 |

程序仅输出识别结果，不连接机器人。多手时，已接受的握拳优先输出3；否则仅有一种有效动作就输出该数字，不同有效动作冲突则输出0。0类手不会抵消另一只手的单一有效指令。没有通过检测与分类门槛的手也输出0。

## 查看框与概率

```powershell
python predict.py "你的照片.jpg" --json "result.json"
```

JSON保留各手检测框、检测分数、六类概率、接受状态和最终整数。`config.json`中的检测阈值为{config['thresholds']['detector']}、分类阈值为{config['thresholds']['classifier']}，均来自验证集，未按新图片调整。

## 在自己的Python程序里使用

```python
from predict import GesturePredictor

# 加载一次模型，多张图片重复调用，避免每次重新读权重。
model = GesturePredictor()
number = model.predict("你的照片.jpg")
print(number)
```

首次启动包含导入依赖、校验和加载模型，需额外时间；报告中的毫秒耗时指模型已加载后的照片处理时间。样例来自验证集固定规则选取，可能识别错误；`samples.json`记录真实类别，不是程序写死的返回值。空白图只是接口检查，不代表已完成真实无手场景的性能测试。

## 文件结构

- `predict.py`：命令行入口和可复用的`GesturePredictor`。
- `config.json`：当前模型架构、固定阈值、类别映射与权重SHA256。
- `weights/classifier.pt`：本架构的六类分类权重。
- `weights/detector.pt`：已训练的离线手部检测权重。
- `examples`、`samples.json`：完整照片示例和来源说明。
- `requirements.txt`：本轮验证过的核心依赖版本。
- `SHA256SUMS.json`：各文件的完整性清单。

原始训练、验证、逐样本测试、故障修复与独立核查记录见统一交付文件夹的`训练记录`。本包只用于推理；A协议允许同一人员出现在不同分组，不能将该成绩当成人员互斥泛化或机器人实际控制效果。
''')
        # 校验清单不包含自身，避免自引用哈希。
        manifest = [{'path': file.relative_to(package).as_posix(), 'bytes': file.stat().st_size, 'sha256': sha(file)}
                    for file in sorted(package.rglob('*')) if file.is_file() and file.name != 'SHA256SUMS.json' and '__pycache__' not in file.parts]
        json_new(package / 'SHA256SUMS.json', {'files': manifest})
        archive = DELIVERY / (name + '_图片识别程序.zip')
        if archive.exists():
            raise FileExistsError(f'压缩包已存在：{archive}')
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for file in sorted(package.rglob('*')):
                if file.is_file() and '__pycache__' not in file.parts:
                    z.write(file, file.relative_to(DELIVERY).as_posix())
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
        summary.append({'model': name, 'directory': name, 'zip': archive.name, 'bytes': archive.stat().st_size,
                        'sha256': sha(archive), 'thresholds': config['thresholds']})
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
    copy_exact(PROJECT / 'A协议ResNet训练与对比实验报告_v1.docx', DELIVERY / 'A协议ResNet训练与对比实验报告_v1.docx')
    json_new(DELIVERY / '程序包清单.json', {'packages': summary})
    write_new(DELIVERY / 'README.md', '''# ResNet手势识别交付

本目录包含三个可分别使用的已训练图片识别程序，以及本轮训练记录。

## 先用哪一个

| 程序 | 人工框手图准确率 | 完整照片准确率 | 本机CPU整图平均耗时 |
|---|---:|---:|---:|
| ResNet18 | 99.34% | 93.62% | 32.25 ms |
| ResNet34 | 99.18% | 93.14% | 45.88 ms |
| ResNet50 | 99.44% | 94.34% | 54.76 ms |

以上为固定A协议结果；CPU条件为8线程、batch=1，模型已加载。本轮ResNet18较快，ResNet50总体准确率较高。完整照片成绩包含自动手检测、裁图、分类和多手汇总，与人工框分类成绩分别列出。

## 使用步骤

1. 选择`ResNet18_图片识别程序.zip`、`ResNet34_图片识别程序.zip`或`ResNet50_图片识别程序.zip`，解压到任意位置。也可直接使用本目录同名的已解压文件夹。
2. 按包内`README.md`安装依赖。
3. 在该程序文件夹执行`python predict.py "你的照片.jpg"`，得到0—5中的一个数字。

数字含义：0无有效动作、1前进（V形）、2后退（张掌）、3停止（握拳）、4左转（拇指向上）、5右转（拇指向下）。程序不会发送机器人控制命令。

## 目录

- `ResNet18`、`ResNet34`、`ResNet50`：相互独立的推理程序、权重、配置及示例。
- 三个`*_图片识别程序.zip`：对应程序的完整压缩包，不需要把另外两个一起解压。
- `训练记录/README.md`：训练记录目录、保留项、缓存排除项及如何追溯结果。
- `A协议ResNet训练与对比实验报告_v1.docx`：完整实验报告、对比大表、错误案例及Grad-CAM。
- `程序包清单.json`：三个压缩包的大小、SHA256和冻结阈值。
- `交付独立检查.json`：解压、离线运行、输出契约与原程序CPU预测一致性的检查结果。

图片程序运行不依赖原训练项目或数据路径。训练记录包含原始证据，其中的历史索引可能仍记录原机器的数据位置；它们不是推理程序的运行依赖。若需重新训练，应先恢复对应原始数据与固定分组，不能仅凭本次记录目录声称已具备全部原图。
''')


if __name__ == '__main__':
    main()
