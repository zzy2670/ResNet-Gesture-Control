"""迁移后显式运行B的一个新实验阶段；重建当前路径索引，保护全部历史A/B结果。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
STAGES = ['prepare', 'smoke', 'features', 'search', 'finetune', 'detector',
          'calibrate', 'evaluate', 'latency', 'classifier-latency']
WEIGHTS = ['resnet18-f37072fd.pth', 'resnet34-b627a593.pth',
           'resnet50-0676ba61.pth', 'mobilenet_v3_large-8738ca79.pth']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def stage_marker(out, arch, stage):
    return out / 'portable_stage_records' / f'{arch}_{stage}_success.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES, nargs='?')
    parser.add_argument('--list', action='store_true', help='只列出阶段，不导入模型依赖、不创建目录')
    parser.add_argument('--out', help='本项目runs内尚不存在的新目录；首阶段必须18/prepare')
    parser.add_argument('--arch', choices=['resnet18', 'resnet34', 'resnet50'], default='resnet18')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--workers', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--trials', type=int, default=24)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--patience', type=int, default=4)
    args = parser.parse_args()
    if args.list:
        print(json.dumps({'stages': STAGES, 'first': 'prepare --arch resnet18',
                          'shared': '先完成18的prepare；34/50复用本次新生成的裁图索引和18检测器',
                          'execution': '每条命令只执行一个阶段；不调用历史controller'}, ensure_ascii=False))
        return
    if not args.stage or not args.out:
        parser.error('执行阶段必须同时提供stage和--out')
    # 默认参数与正式B监督器一致；显式参数仅作用于本次新实验。
    defaults = {'device': 'cuda', 'workers': 4, 'batch_size': 128, 'epochs': 18, 'lr': 1e-4}
    defaults.update({'smoke': {'device': 'cpu', 'workers': 0},
                     'search': {'batch_size': 512},
                     'detector': {'batch_size': 24, 'epochs': 16, 'lr': 0.003},
                     'calibrate': {'batch_size': 32}, 'evaluate': {'batch_size': 32},
                     'latency': {'device': 'cpu', 'workers': 0, 'batch_size': 1},
                     'classifier-latency': {'device': 'cpu', 'workers': 0, 'batch_size': 1}}.get(args.stage, {}))
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    candidate = Path(args.out)
    out = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    runs = (ROOT / 'runs').resolve()
    if not out.is_relative_to(runs) or out == runs:
        raise ValueError('--out必须为当前项目runs内的独立新目录')
    for historical in ['A_20261102_TPE', 'B_20261102_TPE']:
        protected = runs / historical
        if out == protected or out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError('拒绝写入或包含原A/B权威结果目录')
    source_files = ['portable_B_training.py', 'run_B_pipeline.py', 'train_A.py', 'compare_resnets_A.py']
    identity = {'project_root': str(ROOT), 'new_output': str(out),
                'source_sha256': {name: sha(ROOT / name) for name in source_files},
                'manifest_sha256': {split: sha(ROOT / 'data/B_split_verified_20261102' / split / 'manifest.csv')
                                    for split in ['train', 'val', 'test']},
                'pretrained_sha256': {name: sha(ROOT / 'pretrained/checkpoints' / name) for name in WEIGHTS},
                'purpose': 'new_B_experiment_only', 'historical_crops_reused': False}
    for name, digest in identity['pretrained_sha256'].items():
        if not digest.startswith(name.rsplit('-', 1)[1].split('.')[0]):
            raise ValueError(f'官方预训练身份不符：{name}')
    owner = out / 'portable_B_experiment.json'
    if not out.exists():
        if (args.arch, args.stage) != ('resnet18', 'prepare'):
            raise ValueError('新实验首阶段必须prepare --arch resnet18，以当前包内照片重建裁图')
        out.mkdir(parents=True)
        write_new(owner, identity)
    elif not owner.is_file() or json.loads(owner.read_text(encoding='utf-8')) != identity:
        raise ValueError('已有目录不是本入口的同身份新实验；移动项目或源码变化后需另选新目录')
    prerequisites = {'prepare': [], 'smoke': ['prepare'], 'features': ['prepare', 'smoke'],
                     'search': ['features'], 'finetune': ['search'], 'detector': ['prepare'],
                     'calibrate': ['finetune'], 'evaluate': ['calibrate'],
                     'latency': ['evaluate'], 'classifier-latency': ['evaluate']}
    needed = [(args.arch, stage) for stage in prerequisites[args.stage]]
    if args.arch != 'resnet18':
        needed.append(('resnet18', 'prepare'))
    if args.stage in ['calibrate', 'evaluate', 'latency', 'classifier-latency']:
        needed.append(('resnet18', 'detector'))
    if args.stage == 'detector' and args.arch != 'resnet18':
        raise ValueError('检测器仅由新实验resnet18目录训练一次')
    for arch, stage in needed:
        if not stage_marker(out, arch, stage).is_file():
            raise ValueError(f'前置阶段尚未成功：{arch}/{stage}')
    records = out / 'portable_stage_records'
    records.mkdir(exist_ok=True)
    attempt = records / f'{args.arch}_{args.stage}_attempt.json'
    if attempt.exists():
        raise FileExistsError('该阶段已有尝试记录，拒绝重复执行或覆盖')
    active = out / 'portable_active_stage.json'
    # 单卡新实验也只允许一个显式阶段运行；异常保留锁和尝试证据供审查。
    write_new(active, {'pid': os.getpid(), 'architecture': args.arch, 'stage': args.stage})
    # 不覆盖任何成功或失败尝试；意外中断需保留证据并另行审查恢复。
    write_new(attempt, {'arguments': vars(args), 'identity': identity})
    import run_B_pipeline as pipeline
    pipeline.RUN = out
    pipeline.SPLIT = ROOT / 'data/B_split_verified_20261102'
    pipeline.PRETRAINED = ROOT / 'pretrained'
    original_argv = sys.argv
    forwarded = ['run_B_pipeline.py', args.stage, '--arch', args.arch]
    for key in ['device', 'workers', 'batch_size', 'epochs', 'trials', 'lr', 'patience']:
        forwarded += ['--' + key.replace('_', '-'), str(getattr(args, key))]
    try:
        sys.argv = forwarded
        pipeline.main()
    finally:
        sys.argv = original_argv
    write_new(stage_marker(out, args.arch, args.stage),
              {'status': 'PASS', 'architecture': args.arch, 'stage': args.stage,
               'purpose': 'new_B_experiment_only', 'identity': identity})
    active.unlink()


if __name__ == '__main__':
    main()
