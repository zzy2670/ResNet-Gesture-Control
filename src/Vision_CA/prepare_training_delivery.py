"""整理A协议三模型的训练记录；复制并校验原文件，不改训练结果。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
PARENT = PROJECT.parent
RUN = PROJECT / 'runs' / 'A_20261102_TPE'
DESTINATION = PARENT / 'ResNet手势识别交付' / '训练记录'
SPLIT = PROJECT / 'data' / 'A_split_verified_20261102' / 'P1_random_image'
EXCLUDED_DIRECTORIES = {
    'crops': '人工框派生JPEG可由固定A名单和train_A.py的prepare阶段重建，不重复复制',
    '报告含曲线预览渲染': '排版预览及中间渲染，不属于正式训练证据',
    '报告预览渲染': '排版预览及中间渲染，不属于正式训练证据',
    '最终报告渲染': '已被修字体后的正式Word替代的渲染版本',
    '最终报告渲染_修分页': '已被修字体后的正式Word替代的渲染版本',
    '最终报告渲染_修字体': '正式Word已另存报告目录；逐页渲染可从Word重建，审查JSON保留',
}
CODE_FILES = [
    'AGENTS.md', 'README.md', 'prepare_datasets.py', 'train_A.py',
    'compare_resnets_A.py', 'run_A_budget.py', 'run_A_comparisons.py',
    'benchmark_A_classifiers.py', 'postprocess_A_results.py',
    'make_A_trained_gradcam.py', 'make_A_error_examples.py',
    'save_A_environment.py', 'prepare_training_delivery.py',
    'build_gesture_delivery.py', 'audit_gesture_delivery.py',
    'deployment_src/gesture_predict.py',
]


def digest(path: Path) -> str:
    # 分块读取权重，避免一次占用大量内存。
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_once(path: Path, data: bytes) -> None:
    # 已存在且内容相同可复核；内容不同则停下，绝不覆盖既有交付。
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f'已有文件与本次内容不同，未覆盖：{path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def copy_verified(source: Path, destination: Path) -> dict:
    before = source.stat()
    source_hash = digest(source)
    if destination.exists():
        if destination.stat().st_size != before.st_size or digest(destination) != source_hash:
            raise FileExistsError(f'已有文件与源文件不同，未覆盖：{destination}')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'复制期间源文件发生变化：{source}')
    if digest(source) != source_hash or digest(destination) != source_hash:
        raise RuntimeError(f'SHA256校验未通过：{source}')
    return {'source_relative': source.relative_to(PARENT).as_posix(),
            'destination_relative': destination.relative_to(DESTINATION).as_posix(),
            'sha256': source_hash, 'bytes': before.st_size}


def exclusion(path: Path) -> str | None:
    relative = path.relative_to(RUN)
    if relative.parts[0] in EXCLUDED_DIRECTORIES:
        return EXCLUDED_DIRECTORIES[relative.parts[0]]
    if path.name.startswith('features_') and path.suffix == '.pt':
        return '冻结主干特征缓存可由features阶段重建，不重复复制'
    if path.name in {'报告含曲线预览.docx', '报告排版预览.docx'}:
        return '中间排版Word，最终报告已另存'
    if '__pycache__' in relative.parts or path.suffix == '.pyc':
        return 'Python自动字节码，不是训练证据'
    return None


def readme_text() -> str:
    return '''# A协议三模型训练记录

这里集中保存2026年11月2日本机ResNet18、ResNet34、ResNet50实验的正式结果和原始训练证据。原训练目录保持不变；本目录是按用途整理、逐文件SHA256校验的交付副本。输入整张照片输出0—5的独立程序在上一级三个模型程序包中，本目录用于查看实验过程和核对结果。

## 先看什么

1. `报告/A协议ResNet训练与对比实验报告_v1.docx`：完整实验报告，含三模型大表、错误照片和Grad-CAM。
2. `实验输出/comparison_results.csv`：三模型同口径汇总；相同内容也保存在`comparison_results.json`。
3. `实验输出/实验报告正文.md`：报告文字源稿。
4. `实验输出/report_content_independent_review.json`：最终报告独立核查；`final_independent_review.json`记录逐样本与速度核查。
5. `文件清单.json`：每个复制文件的来源相对路径、交付相对路径、字节数和SHA256。
6. `整理记录.json`：复制和排除的文件数量、总字节数及逐项理由。

## 目录结构

```text
训练记录/
├─ README.md
├─ 文件清单.json
├─ 整理记录.json
├─ 报告/                         正式Word报告
├─ 代码/                         训练、数据准备、汇总和可解释性脚本
├─ 数据划分/                     固定A三组manifest与原summary
└─ 实验输出/
   ├─ classifier_best.pt         ResNet18最终分类权重
   ├─ detector_best.pt           三模型共用的手检测权重
   ├─ calibration.json          ResNet18验证集冻结阈值与权重身份
   ├─ tpe.sqlite3、trials/       ResNet18全部24次TPE及分类头试验权重
   ├─ compare_resnet34/          ResNet34权重、TPE、阈值、完整预测和日志
   ├─ compare_resnet50/          ResNet50权重、TPE、阈值、完整预测和日志
   ├─ code_snapshot/             原训练时及推理工厂修复后的代码快照
   ├─ report_figures/            从真实结果生成的混淆图、曲线及错误案例
   └─ trained_gradcam/           已训练ResNet18热图、激活/梯度张量和元信息
```

根部实验输出对应ResNet18；34和50分别保存在各自子目录，不能把某模型的分类权重与另一模型的阈值混用。各模型均完成24次TPE和18轮微调；三模型共用的检测器完成16轮训练。分类权重按完整验证集宏平均F1选取，微调最佳轮次依次为15、15、16；检测权重选第10轮。

## 三模型正式结果

| 模型 | 人工框手图准确率（3,917只手） | 完整照片准确率（3,180张） | CPU整图平均耗时 | 检测/分类阈值 |
|---|---:|---:|---:|---:|
| ResNet18 | 99.34% | 93.62% | 32.25 ms | 0.3/0.3 |
| ResNet34 | 99.18% | 93.14% | 45.88 ms | 0.3/0.5 |
| ResNet50 | 99.44% | 94.34% | 54.76 ms | 0.3/0.3 |

CPU整图耗时使用8线程、batch=1，计入打开照片、检测、裁图、分类和数字汇总；模型加载不计入。人工框成绩不等于输入完整照片时的成绩。完整测试照片均带有手标注，未评价完全无手的空场景。A按照片随机分组，同一个人可跨组；结果不代表人员互斥的B协议，B未划分或训练。

## 核对训练过程和预测

- `events.jsonl`：每个完整训练轮的样本数、损失、验证结果和耗时。三个模型分别有自己的文件。
- `tpe.sqlite3`、`tpe_summary.json`和`trials/`：24次完整分类头搜索及其参数、分数和权重；TPE只作用于冻结特征上的分类头。
- `finetune_console.log`、`detector_console.log`、`budget_events.jsonl`、`comparison_budget_events.jsonl`：原始控制台和进程起止/退出记录。日志中的阶段截止时间记录的是当时配置，原记录不改写；三模型均已完成全部18轮微调。
- `calibration.json`与`val_whole_raw.json`：模型冻结后完整验证照片上的原始预测、12种阈值候选及最终选择，不使用测试集调阈值。
- `test_oracle_crop.json`：全部3,917只人工框手图的真实标签、概率和分类输出。
- `test_whole_photo.json`：全部3,180张完整照片的手框、概率、接受判断和最终0—5输出。
- `test_detector.json`：同批检测框在明确阈值/IoU规则下的匹配统计，不是COCO AP。
- `classifier_cpu_latency_unified.json`：每模型同100个预处理输入、20次预热和3轮共300次纯分类CPU前向计时；`cpu_latency.json`为完整照片计时。
- `compare_resnet34/independent_review.json`、`compare_resnet50/independent_review.json`：额外两模型的逐样本独审。

## 故障与修复证据

`calibrate_first_failure_full_tail.log`保留首次验证校准的真实失败；`detector_factory_fix_20261102.json`记录推理工厂尾部宽度不一致的原因和修复。分类及检测训练权重未因该故障重训，修复后的校准、测试和测速均已成功结束。失败日志不能当作最终实验仍失败，成功结果也不能用来删除历史异常。

`code_snapshot/train_A.py`是原训练代码；`train_A_after_inference_factory_fix.py`是修复推理构建后的快照，身份与原因在修复JSON中记录。`代码/`是整理时对应的项目脚本，便于阅读；不覆盖原代码快照。

## 哪些文件没有重复复制

- 38,884张派生手部JPEG：由固定A照片清单、人工框和`train_A.py prepare`可重建；原训练目录中的裁图仍保留。
- 三模型`features_train.pt`与`features_val.pt`：冻结主干的特征缓存，`features`阶段可重建；原文件仍保留。
- 中间报告Word与各版逐页PNG/PDF：保留正式Word和独立审查记录，排版可从正式Word重新渲染。
- 原始31,833张照片及标注没有放入本训练记录副本；仍使用原项目接入的数据。完整训练复现须先恢复该数据，再按`代码/README.md`和固定manifest准备，不应把本目录误当成包含原始图像的完整数据包。

精确排除项目和字节数见`整理记录.json`。`crops_*.json`、数据manifest、原日志和环境快照中的旧绝对路径按原样保留，它们用于追溯历史，不是独立预测程序的运行依赖。预测程序使用各自包内相对路径。

## 校验副本

`文件清单.json`的`source_relative`相对原项目父目录，`destination_relative`相对本目录。对文件计算SHA256并与清单比较即可校验。整理脚本`代码/prepare_training_delivery.py`的职责是从原项目复制和复核；它不是独立推理入口，只有在原项目完整结构中才用于再次整理。
'''


def main() -> None:
    selected: list[tuple[Path, Path]] = []
    excluded = []
    for source in sorted(RUN.rglob('*')):
        if not source.is_file():
            continue
        relative = source.relative_to(RUN)
        reason = exclusion(source)
        if reason:
            excluded.append({'source_relative': source.relative_to(PARENT).as_posix(),
                             'bytes': source.stat().st_size, 'reason': reason})
        else:
            selected.append((source, DESTINATION / '实验输出' / relative))
    for name in CODE_FILES:
        selected.append((PROJECT / name, DESTINATION / '代码' / name))
    for split in ('train', 'val', 'test'):
        selected.append((SPLIT / split / 'manifest.csv', DESTINATION / '数据划分' / split / 'manifest.csv'))
    selected.append((SPLIT / 'summary.json', DESTINATION / '数据划分' / 'summary.json'))
    report = PARENT / 'A协议ResNet训练与对比实验报告_v1.docx'
    selected.append((report, DESTINATION / '报告' / report.name))
    records = [copy_verified(source, destination) for source, destination in selected]
    source_count = sum(1 for source in RUN.rglob('*') if source.is_file())
    run_copy_count = sum(record['destination_relative'].startswith('实验输出/') for record in records)
    assert run_copy_count + len(excluded) == source_count
    manifest = {'format_version': 1, 'source_base': 'MA6216_CA3_final_project',
                'destination_base': '本文件所在目录', 'algorithm': 'SHA256',
                'files': records}
    summary = {'status': 'PASS', 'original_run_unchanged': True,
               'copied_source_files': len(records), 'copied_source_bytes': sum(r['bytes'] for r in records),
               'run_source_files': source_count, 'run_copied_files': run_copy_count,
               'run_excluded_files': len(excluded), 'run_excluded_bytes': sum(r['bytes'] for r in excluded),
               'excluded_files': excluded,
               'verification': '每个复制源和目标在复制后SHA256一致；原文件大小及修改时间未变化',
               'generated_files': ['README.md', '文件清单.json', '整理记录.json']}
    write_once(DESTINATION / 'README.md', readme_text().encode('utf-8'))
    write_once(DESTINATION / '文件清单.json', (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    write_once(DESTINATION / '整理记录.json', (json.dumps(summary, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({key: value for key, value in summary.items() if key != 'excluded_files'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
