"""B协议单阶段入口：复用A算法，只使用B人员划分与独立输出。"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/B_20261102_TPE'
SPLIT = ROOT / 'data/B_split_verified_20261102'
PRETRAINED = ROOT.parent / 'ResNet项目完整交接_20261102/完整源码与数据/Vision_CA/pretrained'
WEIGHTS = {'resnet18': 'resnet18-f37072fd.pth', 'resnet34': 'resnet34-b627a593.pth',
           'resnet50': 'resnet50-0676ba61.pth', 'detector': 'mobilenet_v3_large-8738ca79.pth'}
STAGES = ('prepare', 'features', 'search', 'finetune', 'detector', 'calibrate', 'evaluate', 'latency', 'classifier-latency', 'smoke')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def out_for(arch):
    return RUN if arch == 'resnet18' else RUN / ('compare_' + arch)


def copy_same(source, target):
    source, target = Path(source), Path(target)
    if target.exists():
        if sha(source) != sha(target):
            raise ValueError(f'已有文件身份不匹配，拒绝覆盖：{target}')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def manifest_identity():
    return {split: sha(SPLIT / split / 'manifest.csv') for split in ('train', 'val', 'test')}


@contextmanager
def b_study_names():
    # 只改变本进程的Optuna研究名，不改原A源代码或TPE算法。
    import optuna
    original = optuna.create_study
    def create(*args, **kwargs):
        name = kwargs.get('study_name', '')
        if name.startswith('A_'):
            kwargs['study_name'] = 'B_' + name[2:]
        return original(*args, **kwargs)
    optuna.create_study = create
    try:
        yield
    finally:
        optuna.create_study = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--arch', choices=['resnet18', 'resnet34', 'resnet50'], default='resnet18')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=18)
    parser.add_argument('--trials', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=4)
    args = parser.parse_args()
    if args.stage == 'detector' and args.arch != 'resnet18':
        raise ValueError('B手检测器只训练一次，由主目录持有')
    import torch
    import torch.nn as nn
    import train_A as base
    import compare_resnets_A as compare
    import numpy as np
    from torchvision import models
    # 禁止本阶段意外联网，所有官方初始化都已提供完整本地参数。
    socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('B训练入口只允许本地预训练权重'))
    torch.hub.set_dir(str(PRETRAINED))
    base.seed_all(42)
    torch.set_num_threads(8)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    out = out_for(args.arch)
    out.mkdir(parents=True, exist_ok=True)
    identities = {}
    for key, filename in WEIGHTS.items():
        path = PRETRAINED / 'checkpoints' / filename
        digest = sha(path)
        # 官方文件名中的哈希前缀是来源完整性约束，不接受A微调权重。
        if not digest.startswith(filename.rsplit('-', 1)[1].split('.')[0]):
            raise ValueError(f'官方预训练权重哈希前缀不符：{filename}')
        identities[key] = {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}
    source_hashes = {name: sha(ROOT / name) for name in ('train_A.py', 'compare_resnets_A.py', 'run_B_pipeline.py')}
    spec = {'protocol': 'B_subject_independent', 'architecture': args.arch,
            'pretraining': 'ImageNet1K_V1_only', 'pretrained': identities[args.arch],
            'detector_initialization': identities['detector'], 'manifest_sha256': manifest_identity(),
            'A_finetuned_weights_used': False, 'source_sha256': source_hashes}
    identity_file = out / 'pretrained_identity.json'
    if identity_file.exists() and base.read_json(identity_file) != spec:
        raise ValueError('B输出身份或源码已变化，停止以待审查，不覆盖原记录')
    base.atomic_json(identity_file, spec)
    if args.arch != 'resnet18':
        compare.bind_architecture(SimpleNamespace(arch=args.arch))
    params = SimpleNamespace(**vars(args), split_root=str(SPLIT), out=str(out),
                             shared_run=str(RUN), pretrained=identities[args.arch]['path'],
                             deadline=None, timeout=0)
    base.log_event(out, stage='B_stage', status='start', operation=args.stage, architecture=args.arch,
                   parameters=vars(params), identity=spec)
    if args.stage == 'prepare':
        if args.arch == 'resnet18':
            base.prepare_crops(params)
        else:
            for name in ('crops_train.json', 'crops_val.json', 'crops_test.json', 'data_audit.json'):
                copy_same(RUN / name, out / name)
        model = base.resnet(None, fine_tune=True)
        base.atomic_json(out / 'model_info.json', {'protocol': 'B_subject_independent', 'architecture': args.arch,
            'parameters': sum(p.numel() for p in model.parameters()),
            'finetune_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'feature_dim': model.fc[1].in_features, 'pretraining': 'ImageNet1K_V1',
            'fine_tune': 'layer4_and_fc_only', 'layer4_lr': 1e-4, 'fc_lr': 5e-4,
            'manifest_sha256': spec['manifest_sha256'], 'source_sha256': source_hashes})
        del model
    elif args.stage == 'smoke':
        # 在长期GPU任务前，CPU检查真实推理工厂完整保存/重载和前向。
        model = base.resnet(identities[args.arch]['path']).eval()
        with torch.inference_mode():
            result = model(torch.zeros(1, 3, 224, 224))
        if result.shape != (1, 6) or not torch.isfinite(result).all():
            raise RuntimeError('分类器CPU烟测失败')
        if args.arch == 'resnet18':
            detector = base.detector_model().eval()
            state = detector.state_dict()
            rebuilt = base.detector_model(False).eval()
            rebuilt.load_state_dict(state, strict=True)
            with torch.inference_mode():
                pred = rebuilt([torch.zeros(3, 384, 384)])[0]
            if pred['boxes'].shape[1] != 4 or not torch.isfinite(pred['scores']).all():
                raise RuntimeError('检测器CPU烟测失败')
        base.atomic_json(out / 'cpu_smoke.json', {'status': 'PASS', 'protocol': 'B_subject_independent',
            'architecture': args.arch, 'initialization': 'official_ImageNet_only', 'GPU_training_started': False,
            'strict_detector_factory_reload': args.arch == 'resnet18'})
    elif args.stage == 'search':
        if (out / 'tpe_summary.json').exists():
            raise FileExistsError('TPE已完成，禁止重新搜索增加试验数')
        if (out / 'tpe.sqlite3').exists():
            raise RuntimeError('存在未完成TPE数据库，需要记录失败原因后制定明确恢复方案')
        with b_study_names():
            (base.search if args.arch == 'resnet18' else compare.search)(params)
        summary = base.read_json(out / 'tpe_summary.json')
        if len(summary['trials']) != args.trials or any(t['state'] != 'COMPLETE' for t in summary['trials']):
            raise RuntimeError('TPE没有完成全部预定试验，保留账本与摘要，不当成完整成功')
    elif args.stage == 'classifier-latency':
        config = base.read_json(out / 'calibration.json')
        if config['weights'] != base.checkpoint_hashes(out):
            raise ValueError('测速权重不是冻结模型')
        checkpoint = torch.load(out / 'classifier_best.pt', map_location='cpu', weights_only=True)
        model = base.resnet(None, checkpoint['params']['dropout']).eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        data = base.CropDataset(out, 'test')
        inputs = [data[i][0][None] for i in range(min(100, len(data)))]
        import time
        times = []
        with torch.inference_mode():
            for i in range(20): model(inputs[i % len(inputs)])
            for repeat in range(3):
                for image in inputs:
                    start = time.perf_counter(); model(image)
                    times.append((time.perf_counter() - start) * 1000)
        base.atomic_json(out / 'classifier_cpu_latency_unified.json', {
            'architecture': args.arch, 'scope': 'classifier_only_preprocessed_224x224_tensor',
            'device': 'cpu', 'dtype': 'float32', 'threads': 8, 'batch_size': 1, 'warmup': 20,
            'unique_inputs': len(inputs), 'repeats': 3, 'n_forward': len(times), 'all_ms': times,
            'mean_ms': float(np.mean(times)), 'p50_ms': float(np.percentile(times, 50)),
            'p95_ms': float(np.percentile(times, 95)), 'parameters': sum(p.numel() for p in model.parameters()),
            'sample_keys': [f"{r['image_id']}__hand{r['hand_index']}" for r in data.rows[:len(inputs)]],
            'weights': config['weights'], 'model_loading_included': False, 'preprocessing_included': False})
    else:
        if args.stage in ('calibrate', 'evaluate', 'latency') and args.arch != 'resnet18':
            copy_same(RUN / 'detector_best.pt', out / 'detector_best.pt')
            base.atomic_json(out / 'shared_detector_identity.json', {
                'protocol': 'B_subject_independent', 'source': str(RUN / 'detector_best.pt'),
                'sha256': sha(RUN / 'detector_best.pt'), 'trained_only_on_B_train': True})
        if args.stage == 'evaluate' and any((out / name).exists() for name in ('test_oracle_crop.json', 'test_whole_photo.json', 'test_detector.json')):
            raise FileExistsError('已有最终测试输出，禁止无痕覆盖或重复评价')
        {'features': base.feature_cache, 'finetune': base.fine_tune, 'detector': base.train_detector,
         'calibrate': base.calibrate, 'evaluate': base.final_evaluate, 'latency': base.cpu_latency}[args.stage](params)
    base.log_event(out, stage='B_stage', status='exit0', operation=args.stage, architecture=args.arch)


if __name__ == '__main__':
    main()
