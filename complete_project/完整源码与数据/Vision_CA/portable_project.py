"""移动项目后的只读检查与可选新实验入口；不会恢复或覆盖历史实验。

check不加载模型；prepare在全新输出目录重建当前机器的裁图索引。
run每次仅执行明确指定的一个阶段，不自动训练、评价或划分B。
"""
from pathlib import Path
import argparse
import csv
import json
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
SPLIT = ROOT / 'data/A_split_verified_20261102/P1_random_image'
WEIGHTS = {'resnet18': 'resnet18-f37072fd.pth', 'resnet34': 'resnet34-b627a593.pth',
           'resnet50': 'resnet50-0676ba61.pth', 'detector': 'mobilenet_v3_large-8738ca79.pth'}
COUNTS = {'train': 25468, 'val': 3185, 'test': 3180}


def check():
    """核验清单、实际照片和本地预训练文件，不修改任何数据。"""
    result = {}
    seen = set()
    for split, count in COUNTS.items():
        folder = SPLIT / split
        with (folder / 'manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != count:
            raise ValueError(f'{split}照片数应为{count}，实际{len(rows)}')
        ids = {row['image_id'] for row in rows}
        if len(ids) != count or ids & seen:
            raise ValueError(f'{split}存在重复或跨组照片ID')
        seen |= ids
        missing = [row['image_path'] for row in rows if not (folder / row['image_path']).is_file()]
        if missing:
            raise FileNotFoundError(f'{split}缺少照片，例如{missing[0]}')
        if len(list((folder / 'annotations').glob('*.json'))) != 18:
            raise ValueError(f'{split}应有18份原始类别标注')
        result[split] = len(rows)
    raw = ROOT / 'data/raw_train_data/hagrid-sample-30k-384p'
    if not raw.is_dir():
        raise FileNotFoundError(f'缺少原始数据目录：{raw}')
    raw_photos = sum(1 for path in raw.rglob('*') if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg'})
    if raw_photos != 31833:
        raise ValueError(f'原始样本照片应有31833张，实际{raw_photos}')
    cache = ROOT / 'pretrained/checkpoints'
    for filename in WEIGHTS.values():
        if not (cache / filename).is_file():
            raise FileNotFoundError(f'缺少本地预训练权重：{cache / filename}')
    return {'status': 'PASS', 'split_photos': result, 'raw_photos': raw_photos, 'raw_data': str(raw),
            'pretrained_files': list(WEIGHTS.values()),
            'old_indices_reused': False, 'training_executed': False}


def output_path(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    runs = (ROOT / 'runs').resolve()
    if not path.is_relative_to(runs) or path == runs:
        raise ValueError('新实验输出必须为本项目runs内的独立子目录')
    historical = runs / 'A_20261102_TPE'
    if path == historical or path.is_relative_to(historical):
        raise ValueError('不能写入已完成的历史A实验目录')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('check', help='只读核验本包数据与离线预训练文件')
    prepare = sub.add_parser('prepare', help='仅为新实验重建裁图和当前路径索引')
    run = sub.add_parser('run', help='显式执行一个新实验阶段')
    for part in (prepare, run):
        part.add_argument('--out', default='runs/new_experiment')
        part.add_argument('--arch', choices=['resnet18', 'resnet34', 'resnet50'], default='resnet18')
        part.add_argument('--workers', type=int, default=4)
    run.add_argument('--stage', required=True, choices=['features', 'search', 'finetune', 'detector', 'calibrate', 'evaluate', 'latency'])
    run.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    run.add_argument('--batch-size', type=int, default=128)
    run.add_argument('--epochs', type=int, default=18)
    run.add_argument('--trials', type=int, default=24)
    run.add_argument('--lr', type=float, default=1e-4)
    run.add_argument('--patience', type=int, default=4)
    args = parser.parse_args()
    checked = check()
    if args.action == 'check':
        print(json.dumps(checked, ensure_ascii=False, indent=2))
        return
    out = output_path(args.out)
    marker = out / 'portable_experiment.json'
    if args.action == 'prepare':
        if out.exists():
            raise FileExistsError('prepare只接受尚不存在的新输出目录，请更换--out')
    else:
        if not marker.is_file():
            raise ValueError('该目录不是由本入口建立的新实验，请先执行prepare')
        owner = json.loads(marker.read_text(encoding='utf-8'))
        if owner['architecture'] != args.arch or owner['project_root'] != str(ROOT):
            raise ValueError('实验架构或当前路径不一致；移动后应在另一个新目录prepare')
    # 只在用户明确prepare/run后导入训练依赖，check保持轻量且只读。
    import torch
    import train_A as base
    import compare_resnets_A as compare
    torch.hub.set_dir(str(ROOT / 'pretrained'))
    base.seed_all(42)
    torch.set_num_threads(8)
    if args.arch != 'resnet18':
        compare.bind_architecture(SimpleNamespace(arch=args.arch))
    common = dict(split_root=str(SPLIT), out=str(out), workers=args.workers,
                  pretrained=str(ROOT / 'pretrained/checkpoints' / WEIGHTS[args.arch]),
                  deadline=None, timeout=0, arch=args.arch)
    if args.action == 'prepare':
        base.prepare_crops(SimpleNamespace(**common))
        marker.write_text(json.dumps({'architecture': args.arch, 'project_root': str(ROOT),
                                     'purpose': 'new_experiment_only'}, ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        common.update(device=args.device, batch_size=args.batch_size, epochs=args.epochs,
                      trials=args.trials, lr=args.lr, patience=args.patience)
        parameters = SimpleNamespace(**common)
        # 不设置旧日期截止；evaluate仍使用原函数的冻结权重和验证记录门禁。
        stages = {'features': base.feature_cache, 'search': base.search if args.arch == 'resnet18' else compare.search,
                  'finetune': base.fine_tune, 'detector': base.train_detector,
                  'calibrate': base.calibrate, 'evaluate': base.final_evaluate, 'latency': base.cpu_latency}
        stages[args.stage](parameters)


if __name__ == '__main__':
    main()
