"""合并peace与two_up后的独立阶段入口；原A/B源码及结果保持只读。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/MERGED_20261102'
DEFAULT_PRETRAINED = ROOT.parent / 'ResNet项目完整交接_20261102/完整源码与数据/Vision_CA/pretrained'
WEIGHTS = {'resnet18': 'resnet18-f37072fd.pth', 'resnet34': 'resnet34-b627a593.pth',
           'resnet50': 'resnet50-0676ba61.pth', 'detector': 'mobilenet_v3_large-8738ca79.pth'}
SOURCES = ('run_merged_pipeline.py', 'run_merged_training.py', 'train_A.py', 'compare_resnets_A.py',
           'prepare_merged_labels.py', 'runs/MERGED_20261102/MERGED_PROTOCOL.md')
STAGES = ('smoke', 'head', 'finetune', 'reuse-detector', 'calibrate', 'evaluate')


def sha(path):
    # 分块计算身份，避免一次读取大缓存。
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def original_run(protocol, arch='resnet18'):
    root = ROOT / f'runs/{protocol}_20261102_TPE'
    return root if arch == 'resnet18' else root / ('compare_' + arch)


def out_for(protocol, arch):
    return RUN / protocol / arch


def split_for(protocol):
    # A清单保留原P1子目录，B清单位于协议根目录。
    root = ROOT / f'data/{protocol}_split_verified_20261102'
    return root / 'P1_random_image' if protocol == 'A' else root


def make_identity(protocol, arch, pretrained_root):
    # 清单、标签、特征、原超参、官方初始化及协议检测器都锁定到实际字节。
    out = out_for(protocol, arch)
    split = split_for(protocol)
    paths = [out / f'crops_{s}.json' for s in ('train', 'val', 'test')]
    paths += [out / f'features_{s}.pt' for s in ('train', 'val')]
    paths += [split / s / 'manifest.csv' for s in ('train', 'val', 'test')]
    paths += [p for s in ('train', 'val', 'test') for p in sorted((split / s / 'annotations').glob('*.json'))]
    paths += [out / 'merged_data_identity.json']
    paths += [out / f'photo_truth_{s}.json' for s in ('train', 'val', 'test')]
    paths += [original_run(protocol, arch) / 'tpe_summary.json', original_run(protocol) / 'detector_best.pt']
    paths += [Path(pretrained_root) / 'checkpoints' / WEIGHTS[k] for k in (arch, 'detector')]
    inputs = {str(p.resolve()): sha(p) for p in paths}
    for key in (arch, 'detector'):
        file = Path(pretrained_root) / 'checkpoints' / WEIGHTS[key]
        if not inputs[str(file.resolve())].startswith(WEIGHTS[key].rsplit('-', 1)[1].split('.')[0]):
            raise ValueError(f'官方预训练哈希不匹配：{file}')
    return {'protocol': protocol, 'architecture': arch, 'label_version': 'merged_peace_two_up_6class',
            'class1_raw': ['peace', 'peace_inverted', 'two_up', 'two_up_inverted'],
            'inputs_sha256': inputs, 'source_sha256': {n: sha(ROOT / n) for n in SOURCES},
            'initialization': 'official_ImageNet1K_V1_backbone_new_fixed_head',
            'unmerged_classifier_weights_used': False, 'new_TPE_trials': 0}


def pin_identity(path, identity):
    # 身份已有时必须完全相同，不覆盖历史协议。
    payload = json.dumps(identity, ensure_ascii=False, indent=2)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != identity:
            raise ValueError(f'身份发生变化：{path}')
    else:
        with path.open('x', encoding='utf-8') as handle:
            handle.write(payload)


def fixed_head(base, args, identity):
    import torch
    import torch.nn as nn
    out = Path(args.out)
    if (out / 'fixed_head_summary.json').exists() or (out / 'classifier_best.pt').exists():
        raise FileExistsError('已有分类训练产物，禁止无痕重训')
    params = base.read_json(original_run(args.protocol, args.arch) / 'tpe_summary.json')['best_params']
    train = torch.load(out / 'features_train.pt', map_location='cpu', weights_only=True)
    val = torch.load(out / 'features_val.pt', map_location='cpu', weights_only=True)
    official_sha = sha(args.pretrained)
    dim = 2048 if args.arch == 'resnet50' else 512
    for split, cache in (('train', train), ('val', val)):
        rows = base.read_json(out / f'crops_{split}.json')
        if cache['pretrained_sha256'] != official_sha or cache['x'].shape != (len(rows), dim):
            raise ValueError('特征来源或维数不匹配')
        expected = torch.tensor([r['label'] for r in rows], dtype=torch.long)
        if not torch.equal(cache['y'], expected):
            raise ValueError('特征标签和新索引不一致')
    device = torch.device(args.device)
    xtrain, ytrain = train['x'].to(device), train['y'].to(device)
    xval, yval = val['x'].to(device), val['y'].to(device)
    counts = torch.bincount(ytrain, minlength=6).float()
    if (counts == 0).any():
        raise ValueError('训练集缺少输出类')
    base.seed_all(42)
    head = nn.Sequential(nn.Dropout(params['dropout']), nn.Linear(dim, 6)).to(device)
    weights = (counts.sum() / (6 * counts)).sqrt() if params['class_weight'] else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=params['label_smoothing'])
    optimizer = base.optimizer_for(head.parameters(), params['optimizer'], params['lr'], params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18)
    best, history = -1.0, []
    for epoch in range(18):
        head.train()
        total, samples = 0.0, 0
        for indices in torch.randperm(len(ytrain), device=device).split(512):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(xtrain[indices]), ytrain[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError('分类头loss非有限值')
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            samples += len(indices)
        head.eval()
        with torch.inference_mode():
            pred = torch.cat([head(x).argmax(1).cpu() for x in xval.split(1024)])
        metrics = base.confusion_metrics(yval.cpu().numpy(), pred.numpy())
        scheduler.step()
        record = {'epoch': epoch + 1, 'samples': samples, 'loss': total / samples, 'validation': metrics}
        history.append(record)
        base.log_event(out, stage='fixed_head', **record)
        if metrics['macro_f1'] > best:
            best = metrics['macro_f1']
            base.atomic_torch(out / 'fixed_head_best.pt', {'head': head.state_dict(), 'params': params,
                'epoch': epoch + 1, 'validation': metrics, 'architecture': args.arch})
    selected = torch.load(out / 'fixed_head_best.pt', map_location='cpu', weights_only=True)
    model = base.resnet(args.pretrained, params['dropout'])
    model.fc.load_state_dict(selected['head'], strict=True)
    checkpoint = {'model': model.state_dict(), 'params': params, 'stage': 'merged_fixed_head',
                  'validation': selected['validation'], 'epoch': selected['epoch'], 'architecture': args.arch}
    base.atomic_torch(out / 'classifier_best.pt', checkpoint)
    base.atomic_json(out / 'fixed_head_summary.json', {'method': 'single_fixed_configuration_no_search',
        'initial_classifier_sha256': sha(out / 'classifier_best.pt'),
        'best_params': params, 'source_tpe_summary': str(original_run(args.protocol, args.arch) / 'tpe_summary.json'),
        'epochs_completed': 18, 'batch_size': 512, 'selected_epoch': selected['epoch'],
        'validation': selected['validation'], 'history': history, 'identity': identity, 'test_accessed': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--protocol', choices=('A', 'B'), required=True)
    parser.add_argument('--arch', choices=('resnet18', 'resnet34', 'resnet50'), required=True)
    parser.add_argument('--pretrained-root', type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    import torch
    import train_A as base
    import compare_resnets_A as compare
    # 只修改独立子进程内映射，旧代码和旧实验不变。
    base.LABELS.update({'two_up': 1, 'two_up_inverted': 1})
    base.CLASS_NAMES[:] = ['other', 'peace_or_two_up_forward', 'palm', 'fist_stop', 'like', 'dislike']
    socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('仅允许本地预训练文件'))
    torch.hub.set_dir(str(args.pretrained_root))
    torch.set_num_threads(8)
    base.seed_all(42)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('请求CUDA但CUDA不可用，不静默切换')
    if args.arch != 'resnet18':
        compare.bind_architecture(args)
    out = out_for(args.protocol, args.arch)
    identity = make_identity(args.protocol, args.arch, args.pretrained_root)
    pin_identity(out / 'merged_input_identity.json', identity)
    pretrained = args.pretrained_root / 'checkpoints' / WEIGHTS[args.arch]
    params = SimpleNamespace(**vars(args), out=str(out), pretrained=str(pretrained),
        split_root=str(split_for(args.protocol)),
        epochs=18, batch_size=128, lr=1e-4, patience=4, deadline=None)
    base.log_event(out, stage='merged_stage', operation=args.stage, status='start', identity=identity)
    if args.stage == 'smoke':
        # CPU使用真实训练裁图，严格重载工厂并验证一次真实前向和反向。
        if args.device != 'cpu':
            raise ValueError('烟测必须CPU执行')
        model = base.resnet(str(pretrained), fine_tune=True).eval()
        rebuilt = base.resnet(None, fine_tune=True).eval()
        rebuilt.load_state_dict(model.state_dict(), strict=True)
        sample, label, index = base.CropDataset(out, 'train')[0]
        result = rebuilt(sample[None])
        loss = torch.nn.functional.cross_entropy(result, torch.tensor([label]))
        loss.backward()
        if result.shape != (1, 6) or not torch.isfinite(result).all() or not torch.isfinite(loss):
            raise RuntimeError('CPU分类器真实样本烟测失败')
        detector = base.detector_model(False).eval()
        state = torch.load(original_run(args.protocol) / 'detector_best.pt', map_location='cpu', weights_only=True)
        detector.load_state_dict(state['model'], strict=True)
        from torchvision.transforms import functional as TF
        from PIL import Image
        with Image.open(base.CropDataset(out, 'train').rows[0]['path']) as image:
            detector_input = TF.to_tensor(image.convert('RGB'))
        with torch.inference_mode():
            detected = detector([detector_input])[0]
        if detected['boxes'].ndim != 2 or detected['boxes'].shape[1] != 4 or not all(
                torch.isfinite(detected[k]).all() for k in ('boxes', 'scores')):
            raise RuntimeError('CPU协议检测器真实样本前向失败')
        base.atomic_json(out / 'cpu_smoke.json', {'status': 'PASS', 'device': 'cpu', 'train_index': index,
            'loss': float(loss.detach()), 'strict_classifier_reload': True, 'strict_own_protocol_detector_reload': True,
            'real_train_forward_backward': True, 'real_detector_forward': True,
            'detector_boxes_shape': list(detected['boxes'].shape), 'GPU_training_started': False, 'identity': identity})
    elif args.stage == 'head':
        fixed_head(base, params, identity)
    elif args.stage == 'finetune':
        # 原函数允许冻结头保留最优，微调最多18轮、patience4，完整日志原样保留。
        if (out / 'finetune_summary.json').exists():
            raise FileExistsError('微调已完成')
        before = torch.load(out / 'classifier_best.pt', map_location='cpu', weights_only=True)
        summary = base.read_json(out / 'fixed_head_summary.json')
        if before['stage'] != 'merged_fixed_head' or sha(out / 'classifier_best.pt') != summary['initial_classifier_sha256']:
            raise ValueError('微调起点必须是本次固定头，禁止旧classifier或未复盘的重复微调')
        base.fine_tune(params)
        events = [json.loads(line) for line in (out / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        epochs = [e for e in events if e.get('stage') == 'finetune' and e.get('status') == 'complete_epoch']
        checkpoint = torch.load(out / 'classifier_best.pt', map_location='cpu', weights_only=True)
        base.atomic_json(out / 'finetune_summary.json', {'epochs_completed': len(epochs), 'max_epochs': 18,
            'patience': 4, 'selected_stage': checkpoint['stage'], 'selected_epoch': checkpoint['epoch'],
            'weights_sha256': sha(out / 'classifier_best.pt'), 'history': epochs})
    elif args.stage == 'reuse-detector':
        source, target = original_run(args.protocol) / 'detector_best.pt', out / 'detector_best.pt'
        if target.exists() and sha(target) != sha(source):
            raise ValueError('目标检测器身份不同')
        if not target.exists():
            shutil.copy2(source, target)
        base.atomic_json(out / 'shared_detector_identity.json', {'protocol': args.protocol, 'source': str(source),
            'sha256': sha(source), 'retrained': False, 'hand_background_labels_unchanged': True})
    else:
        if sha(out / 'detector_best.pt') != sha(original_run(args.protocol) / 'detector_best.pt'):
            raise ValueError('评估检测器必须是本协议原检测器')
        if args.stage == 'evaluate' and any((out / name).exists() for name in
                ('test_oracle_crop.json', 'test_whole_photo.json', 'test_detector.json')):
            raise FileExistsError('已有测试产物，不覆盖不重复评价')
        params.batch_size = 32
        {'calibrate': base.calibrate, 'evaluate': base.final_evaluate}[args.stage](params)
    if make_identity(args.protocol, args.arch, args.pretrained_root) != identity:
        raise RuntimeError('阶段运行中输入或源码变化，禁止成功')
    base.log_event(out, stage='merged_stage', operation=args.stage, status='exit0')


if __name__ == '__main__':
    main()
