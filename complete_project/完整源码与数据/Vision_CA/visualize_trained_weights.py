"""读取已训练的 ResNet18 权重，生成卷积核与参数分布示意图；不训练、不修改权重。"""

from pathlib import Path
import json
import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    # 固定使用合并手势版本 B 协议的现有权重，计算全部在 CPU 上完成。
    root = Path(__file__).resolve().parent
    folder = root / 'runs/MERGED_20261102/B/resnet18'
    source = folder / 'classifier_best.pt'
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    state = checkpoint['model']
    kernels = state['conv1.weight'].float().numpy()
    conv_weights = [(key, value.float().numpy()) for key, value in state.items()
                    if key.endswith('weight') and value.ndim == 4]
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['svg.fonttype'] = 'none'

    # 64 个卷积核共用同一色标：RGB 三通道分别映射到屏幕颜色，不逐核增强。
    bound = float(np.abs(kernels).max())
    tiles = np.clip(0.5 + kernels.transpose(0, 2, 3, 1) / (2 * bound), 0, 1)
    canvas = np.ones((8 * 9 - 2, 8 * 9 - 2, 3))
    for index, tile in enumerate(tiles):
        row, col = divmod(index, 8)
        canvas[row * 9:row * 9 + 7, col * 9:col * 9 + 7] = tile

    for lang in ['zh', 'en']:
        zh = lang == 'zh'
        fig = plt.figure(figsize=(15, 8.5), facecolor='#f4f7fb')
        grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], hspace=0.48, wspace=0.27)
        left = fig.add_subplot(grid[:, 0])
        left.imshow(canvas, interpolation='nearest')
        left.set_axis_off()
        left.set_title('第一层：64 个已学习卷积核，每个 7×7×3\nFirst convolution: 64 learned RGB kernels'
                       if zh else 'First convolution: 64 learned RGB kernels\nEach kernel: 7 × 7 × 3',
                       loc='left', fontsize=14, fontweight='bold', pad=18)
        top = fig.add_subplot(grid[0, 1])
        for prefix, label, color in [('conv1.', 'conv1', '#2675c9'),
                                     ('layer2.', 'layer2', '#23a58b'),
                                     ('layer4.', 'layer4', '#bd60ac')]:
            values = np.concatenate([value.ravel() for key, value in conv_weights if key.startswith(prefix)])
            top.hist(values, bins=110, range=(-0.3, 0.3), density=True,
                     histtype='step', linewidth=1.6, label=label, color=color)
        top.set_title('不同层的卷积权重分布 / Weight distributions' if zh else 'Convolution-weight distributions', loc='left')
        top.set_xlabel('权重值 / Weight value' if zh else 'Weight value')
        top.set_ylabel('范围内密度 / Density in range' if zh else 'Density normalized within range')
        top.legend(frameon=False)
        top.text(0.02, 0.93, '仅显示 −0.3 到 0.3；范围外权重未在此图展示。' if zh
                 else 'Shown range: −0.3 to 0.3; values outside this range are not displayed.',
                 transform=top.transAxes, fontsize=9, color='#536174')
        bottom = fig.add_subplot(grid[1, 1])
        groups = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
        means = [float(np.mean(np.abs(np.concatenate([value.ravel() for key, value in conv_weights
                                                      if key.startswith(group + '.')])))) for group in groups]
        bottom.bar(groups, means, color=['#2675c9', '#428bcb', '#23a58b', '#f1ae47', '#bd60ac'], width=0.6)
        bottom.set_title('各阶段卷积权重的平均绝对值 / Mean |weight|' if zh else 'Mean absolute convolution weight by stage', loc='left')
        bottom.set_ylabel('平均绝对值 / Mean |weight|' if zh else 'Mean |weight|')
        for index, value in enumerate(means):
            bottom.text(index, value, f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        bottom.set_ylim(0, max(means) * 1.2)
        for axis in [top, bottom]:
            axis.spines[['top', 'right']].set_visible(False)
            axis.grid(axis='y', alpha=0.12)
        fig.suptitle('训练后的权重长什么样？ / Inside the trained ResNet18' if zh
                     else 'Inside the trained ResNet18', x=0.06, ha='left', fontsize=22, fontweight='bold', color='#18354b')
        note = ('合并手势 · B 人员互斥协议 · 实际 checkpoint；左图是参数，不是输入照片或 Grad-CAM。\n'
                '统一映射：−最大绝对权重→黑，0→灰，+最大绝对权重→亮；参数大小不能直接说明准确率。') if zh else (
                'Merged gestures · B subject-independent protocol · Actual checkpoint. Kernels are parameters, not photos or Grad-CAM.\n'
                'Shared mapping: negative maximum→black, zero→gray, positive maximum→bright. Weight magnitude does not measure accuracy.')
        fig.text(0.06, 0.025, note, fontsize=10, color='#536174', linespacing=1.7)
        fig.subplots_adjust(left=0.06, right=0.97, top=0.83, bottom=0.17)
        for extension in ['png', 'svg']:
            fig.savefig(folder / f'trained_weights_{lang}.{extension}', dpi=180, facecolor=fig.get_facecolor())
        plt.close(fig)
    # 记录来源和绘图口径，避免把权重图误认为样本解释图。
    metadata = {'source': str(source), 'epoch': checkpoint['epoch'], 'stage': checkpoint['stage'],
                'conv1_shape': list(kernels.shape), 'shared_kernel_abs_max': bound,
                'convolution_stage_mean_abs_weight': dict(zip(groups, means)),
                'note': 'Actual learned weights; not activations or Grad-CAM; no training performed.'}
    (folder / 'trained_weights_visualization.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(folder / 'trained_weights_zh.png')


if __name__ == '__main__':
    main()
