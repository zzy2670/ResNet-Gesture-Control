"""B完整测试保存后，用B冻结ResNet18生成六类分层Grad-CAM；不选模型。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_A as base

ROOT = Path(__file__).resolve().parent / 'runs/B_20261102_TPE'


def main():
    # 解释阶段不训练、不选模型；必须等完整测试产物已保存。
    assert (ROOT / 'test_whole_photo.json').is_file()
    test = base.read_json(ROOT / 'test_oracle_crop.json')
    checkpoint = ROOT / 'classifier_best.pt'
    identity = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert identity == test['checkpoint_sha256']
    torch.set_num_threads(2)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    net = base.resnet(None, state['params']['dropout']).eval()
    net.load_state_dict(state['model'], strict=True)
    dataset = base.CropDataset(ROOT, 'test')
    # 每类按不可变样本编号取第一例，不按预测是否成功挑图。
    selected = []
    for label in range(6):
        selected.append(min((i for i, r in enumerate(dataset.rows) if r['label'] == label),
                            key=lambda i: (dataset.rows[i]['image_id'], dataset.rows[i]['hand_index'])))
    dest = ROOT / 'trained_gradcam'
    dest.mkdir(exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    figure, axes = plt.subplots(6, 5, figsize=(12.5, 15))
    titles = ['分类输入', 'layer4 主解释 7×7', 'layer1 56×56', 'layer2 28×28', 'layer3 14×14']
    metadata = []
    for row_number, index in enumerate(selected):
        record = dataset.rows[index]
        captured = {}
        handles = []
        def hook(name):
            def save(module, args, output):
                output.retain_grad()
                captured[name] = output
            return save
        for name in ['layer1', 'layer2', 'layer3', 'layer4']:
            handles.append(getattr(net, name)[-1].register_forward_hook(hook(name)))
        tensor, truth, _ = dataset[index]
        tensor = tensor.unsqueeze(0).requires_grad_(True)
        net.zero_grad(set_to_none=True)
        scores = net(tensor)
        probabilities = scores.detach().softmax(1)[0].numpy()
        prediction = int(probabilities.argmax())
        scores[0, truth].backward()
        image = np.asarray(Image.open(record['path']).convert('RGB'))
        axes[row_number, 0].imshow(image)
        arrays = {}
        for column, name in enumerate(['layer4', 'layer1', 'layer2', 'layer3'], start=1):
            activation = captured[name].detach()[0]
            gradient = captured[name].grad.detach()[0]
            weight = gradient.mean(dim=(1, 2))
            native = torch.relu((weight[:, None, None] * activation).sum(0))
            up = torch.nn.functional.interpolate(native[None, None], size=(224, 224), mode='bilinear', align_corners=False)[0, 0]
            lo, hi = up.min(), up.max()
            scaled = ((up - lo) / (hi - lo)).numpy() if hi > lo else np.zeros((224, 224))
            colored = plt.colormaps['turbo'](scaled)[..., :3]
            overlay = (0.58 * image + 0.42 * 255 * colored).clip(0, 255).astype(np.uint8)
            axes[row_number, column].imshow(overlay)
            arrays[name + '_activation'] = activation.numpy()
            arrays[name + '_gradient'] = gradient.numpy()
            arrays[name + '_native_cam'] = native.numpy()
            arrays[name + '_normalized_224_cam'] = scaled
        for handle in handles:
            handle.remove()
        np.savez_compressed(dest / f'row_{truth}_tensors.npz', **arrays)
        axes[row_number, 0].set_ylabel(f'真实{truth} / 预测{prediction}\n{record["raw_label"]}\n目标为真实类{truth}',
                                     fontsize=10, rotation=0, ha='right', va='center', labelpad=15)
        if row_number == 0:
            for ax, title in zip(axes[0], titles):
                ax.set_title(title, fontsize=11)
        for ax in axes[row_number]:
            ax.set_xticks([])
            ax.set_yticks([])
        metadata.append({'source': record, 'truth': truth, 'prediction': prediction,
                         'target': truth, 'probabilities': probabilities.tolist(),
                         'selection': 'first_sample_key_within_each_ground_truth_class'})
    figure.suptitle('B人员互斥：已训练ResNet18的六类分层Grad-CAM', fontsize=15)
    figure.text(.5, .015, '每图独立归一化；解释真实类别分数；颜色不代表准确率，跨层颜色深浅不作绝对强度比较', ha='center', fontsize=10)
    figure.subplots_adjust(top=.945, bottom=.045, left=.16, right=.99, hspace=.1, wspace=.03)
    figure.savefig(dest / 'trained_six_class_matrix.png', dpi=200)
    plt.close(figure)
    base.atomic_json(dest / 'metadata.json', {'checkpoint_sha256': identity, 'device': 'cpu',
        'trained_project_model': True, 'used_for_selection': False, 'rows': metadata,
        'formula': 'ReLU(sum_k(mean_spatial(gradient_k)*activation_k)); bilinear224; per_map_minmax',
        'overlay': '0.58*RGB + 0.42*turbo'})
    print('真实已训练模型六类分层Grad-CAM完成', flush=True)


if __name__ == '__main__':
    main()
