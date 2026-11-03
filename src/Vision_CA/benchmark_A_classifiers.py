"""统一测量三个已冻结六分类模型的CPU前向速度，不用于选择权重。"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import train_A as base

ROOT = Path(__file__).resolve().parent
SHARED = ROOT / 'runs/A_20261102_TPE'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architectures', nargs='+', default=['resnet18', 'resnet34', 'resnet50'])
    args = parser.parse_args()
    torch.set_num_threads(8)
    dataset = base.CropDataset(SHARED, 'test')
    inputs = [dataset[index][0].unsqueeze(0) for index in range(100)]
    results = []
    # 同样的100个已预处理输入、batch1、float32、同线程数，单独比较网络计算。
    for arch in args.architectures:
        out = SHARED if arch == 'resnet18' else SHARED / ('compare_' + arch)
        frozen = base.read_json(out / 'calibration.json')
        identities = base.checkpoint_hashes(out)
        assert frozen['weights'] == identities
        checkpoint = torch.load(out / 'classifier_best.pt', map_location='cpu', weights_only=True)
        net = getattr(models, arch)(weights=None)
        dim = net.fc.in_features
        net.fc = nn.Sequential(nn.Dropout(checkpoint['params']['dropout']), nn.Linear(dim, 6))
        net.load_state_dict(checkpoint['model'], strict=True)
        net.eval()
        durations = []
        with torch.inference_mode():
            for index in range(20):
                net(inputs[index % len(inputs)])
            # 三轮计时不重新评价准确率，只记录前向调用耗时。
            for repeat in range(3):
                for tensor in inputs:
                    begin = time.perf_counter()
                    net(tensor)
                    durations.append((time.perf_counter() - begin) * 1000)
        item = {'architecture': arch, 'scope': 'classifier_only_preprocessed_224x224_tensor',
                'dtype': 'float32', 'device': 'cpu', 'threads': 8, 'batch_size': 1,
                'warmup': 20, 'unique_inputs': 100, 'repeats': 3, 'n_forward': len(durations),
                'parameters': sum(p.numel() for p in net.parameters()),
                'checkpoint_bytes': (out / 'classifier_best.pt').stat().st_size,
                'weights': identities, 'mean_ms': float(np.mean(durations)),
                'p50_ms': float(np.percentile(durations, 50)),
                'p95_ms': float(np.percentile(durations, 95)), 'all_ms': durations,
                'sample_keys': [f"{row['image_id']}__hand{row['hand_index']}" for row in dataset.rows[:100]],
                'cpu': platform.processor(), 'model_loading_included': False,
                'preprocessing_included': False, 'used_to_select_weights': False}
        base.atomic_json(out / 'classifier_cpu_latency_unified.json', item)
        results.append(item)
        print(json.dumps({k: item[k] for k in ['architecture', 'parameters', 'mean_ms', 'p95_ms']}, ensure_ascii=False), flush=True)
        del net
    base.atomic_json(SHARED / 'classifier_speed_comparison.json', {'models': results})


if __name__ == '__main__':
    main()
