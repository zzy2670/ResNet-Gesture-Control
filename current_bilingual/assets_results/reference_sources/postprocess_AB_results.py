"""从A、B冻结预测生成六模型对照表与报告图，不重新计算测试预测。"""
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from postprocess_A_results import confusion, read

ROOT = Path(__file__).resolve().parent
B = ROOT / 'runs/B_20261102_TPE'
FIGURES = B / 'report_figures'


def validate_predictions(data, unit):
    # 逐条重建矩阵，防止把手图分母与完整照片分母混用。
    keys = [(r['image_id'], r['hand_index']) if unit == 'hand' else r['image_id']
            for r in data['predictions']]
    if len(set(keys)) != len(keys) or len(keys) != data['metrics']['n']:
        raise ValueError('预测样本存在重复或数量不一致')
    matrix = np.zeros((6, 6), dtype=int)
    for row in data['predictions']:
        matrix[row['truth'], row['prediction']] += 1
    if matrix.tolist() != data['metrics']['confusion_matrix']:
        raise ValueError('逐样本结果与保存混淆矩阵不一致')
    return matrix


def main():
    complete = read(B / 'all_stages_complete.json')
    if complete['status'] != 'PASS':
        raise RuntimeError('等待B全部执行阶段正常完成')
    FIGURES.mkdir(exist_ok=True)
    rows = []
    class_rows = []
    curves, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, protocol in zip(axes, ['A', 'B']):
        run = ROOT / ('runs/' + protocol + '_20261102_TPE')
        for model in ['ResNet18', 'ResNet34', 'ResNet50']:
            out = run if model == 'ResNet18' else run / ('compare_' + model.lower())
            whole = read(out / 'test_whole_photo.json')
            crop = read(out / 'test_oracle_crop.json')
            cm = validate_predictions(whole, 'image')
            validate_predictions(crop, 'hand')
            detector = read(out / 'test_detector.json')
            speed = read(out / 'cpu_latency.json')
            cls_speed = read(out / 'classifier_cpu_latency_unified.json')
            selection = read(out / 'calibration.json')
            trial = read(out / 'tpe_summary.json')
            events = [json.loads(line) for line in (out / 'events.jsonl').read_text(encoding='utf-8').splitlines() if line]
            history = [row for row in events if row.get('stage') == 'finetune' and row.get('status') == 'complete_epoch']
            if len({row['epoch'] for row in history}) != len(history):
                raise ValueError('微调日志有重复轮次，需要审查后再汇总')
            row = {'protocol': protocol, 'model': model,
                   'crop_n': crop['metrics']['n'], 'crop_correct': sum(r['truth'] == r['prediction'] for r in crop['predictions']),
                   'crop_accuracy': crop['metrics']['accuracy'], 'crop_macro_f1': crop['metrics']['macro_f1'],
                   'photo_n': whole['metrics']['n'], 'photo_correct': int(np.trace(cm)),
                   'photo_errors': int(cm.sum() - np.trace(cm)),
                   'photo_accuracy': whole['metrics']['accuracy'], 'photo_macro_f1': whole['metrics']['macro_f1'],
                   'whole_cpu_mean_ms': speed['mean_ms'], 'whole_cpu_p95_ms': speed['p95_ms'],
                   'classifier_cpu_mean_ms': cls_speed['mean_ms'], 'classifier_cpu_p95_ms': cls_speed['p95_ms'],
                   'classifier_parameters': cls_speed['parameters'],
                   'tpe_complete_trials': sum(t['state'] == 'COMPLETE' for t in trial['trials']),
                   'complete_finetune_epochs': len(history), 'finetune_seconds': sum(r['seconds'] for r in history),
                   'best_validation_macro_f1': max(r['validation']['macro_f1'] for r in history),
                   'thresholds': {'detector': selection['selected']['detector_threshold'],
                                  'classifier': selection['selected']['class_threshold']},
                   'zero_samples': whole['safety']['zero_samples'], 'zero_to_motion': whole['safety']['zero_to_motion'],
                   'stop_samples': whole['safety']['stop_samples'], 'stop_to_motion': whole['safety']['stop_to_motion'],
                   'stop_to_zero': int(cm[3, 0]), 'detector_metrics': detector['metrics'],
                   'weights': whole['weights']}
            rows.append(row)
            for label in range(6):
                class_rows.append({'protocol': protocol, 'model': model, 'class': label,
                                   'support': whole['metrics']['support'][label],
                                   **{key: whole['metrics'][key][label] for key in ['precision', 'recall', 'f1']}})
            axis.plot([r['epoch'] for r in history], [100 * r['validation']['macro_f1'] for r in history],
                      marker='o', markersize=3, label=model)
            if protocol == 'B':
                confusion(whole, f'B {model} 整张照片（{whole["metrics"]["n"]}张）', FIGURES / (model + '_whole_confusion.png'))
                confusion(crop, f'B {model} 人工框手图（{crop["metrics"]["n"]}只手）', FIGURES / (model + '_crop_confusion.png'))
        axis.set(title=protocol + '协议验证学习曲线', xlabel='完整微调轮次')
        axis.grid(alpha=.2)
        axis.legend()
    axes[0].set_ylabel('验证宏平均F1（%）')
    curves.tight_layout()
    curves.savefig(FIGURES / 'AB_classifier_learning_curves.png')
    plt.close(curves)
    for name, values in [('ab_comparison_results', rows), ('ab_per_class_results', class_rows)]:
        (B / (name + '.json')).write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')
        # 嵌套身份与检测指标保留JSON；CSV的大表只保留可直接比较的标量。
        fields = [key for key, value in values[0].items() if not isinstance(value, dict)]
        with (B / (name + '.csv')).open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(values)
    print(json.dumps({'status': 'derived_from_saved_predictions', 'rows': len(rows), 'per_class_rows': len(class_rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
