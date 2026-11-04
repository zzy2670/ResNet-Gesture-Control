"""从真实预测和日志生成报告图表；不重新训练、不重新运行准确率测试。"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / 'runs/A_20261102_TPE'
ASSETS = ROOT / 'report_figures'
NAMES = ['0 其他', '1 V形', '2 张掌', '3 握拳/停止', '4 拇指上', '5 拇指下']
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'font.size': 10,
                     'savefig.dpi': 180, 'figure.facecolor': 'white'})


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def confusion(data, title, path):
    matrix = np.asarray(data['metrics']['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(7.1, 6))
    im = ax.imshow(matrix / np.maximum(matrix.sum(1, keepdims=True), 1), vmin=0, vmax=1, cmap='Blues')
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    color='white' if matrix[i, j] / max(1, matrix[i].sum()) > .55 else 'black')
    ax.set(xticks=range(6), yticks=range(6), xticklabels=NAMES, yticklabels=NAMES,
           xlabel='预测标签', ylabel='真实标签', title=title)
    plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
    fig.colorbar(im, ax=ax, label='真实类别内的比例')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ASSETS.mkdir(exist_ok=True)
    directories = [('ResNet18', ROOT), ('ResNet34', ROOT / 'compare_resnet34'),
                   ('ResNet50', ROOT / 'compare_resnet50')]
    rows = []
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    for name, out in directories:
        if not (out / 'test_whole_photo.json').exists():
            continue
        whole = read(out / 'test_whole_photo.json')
        crop = read(out / 'test_oracle_crop.json')
        # 逐样本已有预测重算指标，而非相信报告摘要或重新推理。
        for data, keytype in [(whole, 'image'), (crop, 'hand')]:
            records = data['predictions']
            keys = [r['image_id'] if keytype == 'image' else (r['image_id'], r['hand_index']) for r in records]
            assert len(keys) == len(set(keys)) == data['metrics']['n']
            matrix = np.zeros((6, 6), dtype=int)
            for record in records:
                matrix[record['truth'], record['prediction']] += 1
            assert matrix.tolist() == data['metrics']['confusion_matrix']
        confusion(whole, f'{name} 整张照片测试混淆矩阵（3180张）', ASSETS / f'{name}_whole_confusion.png')
        confusion(crop, f'{name} 人工框手图测试混淆矩阵（3917只手）', ASSETS / f'{name}_crop_confusion.png')
        events = [json.loads(line) for line in (out / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        history = [r for r in events if r.get('stage') == 'finetune' and r.get('status') == 'complete_epoch']
        if history:
            ax.plot([r['epoch'] for r in history], [100 * r['validation']['macro_f1'] for r in history],
                    marker='o', markersize=3, label=name)
        speed_path = out / 'classifier_cpu_latency_unified.json'
        speed = read(speed_path) if speed_path.exists() else {}
        full_speed = read(out / 'cpu_latency.json') if (out / 'cpu_latency.json').exists() else {}
        rows.append({'model': name, 'test_crop_n': crop['metrics']['n'],
                     'test_crop_accuracy': crop['metrics']['accuracy'], 'test_crop_macro_f1': crop['metrics']['macro_f1'],
                     'test_photo_n': whole['metrics']['n'], 'test_photo_accuracy': whole['metrics']['accuracy'],
                     'test_photo_macro_f1': whole['metrics']['macro_f1'],
                     'classifier_cpu_mean_ms': speed.get('mean_ms'), 'classifier_cpu_p95_ms': speed.get('p95_ms'),
                     'whole_cpu_mean_ms': full_speed.get('mean_ms'), 'whole_cpu_p95_ms': full_speed.get('p95_ms'),
                     'classifier_parameters': speed.get('parameters'),
                     'complete_finetune_epochs': len(history), 'finetune_seconds': sum(r['seconds'] for r in history),
                     'best_validation_macro_f1': max((r['validation']['macro_f1'] for r in history), default=None),
                     'zero_to_motion': whole['safety']['zero_to_motion'], 'stop_to_motion': whole['safety']['stop_to_motion']})
    ax.set(xlabel='完整微调轮数', ylabel='验证集宏平均F1（%）', title='三种分类网络的验证学习曲线')
    if rows:
        ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(ASSETS / 'classifier_learning_curves.png')
    plt.close(fig)
    with (ROOT / 'comparison_results.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (ROOT / 'comparison_results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
