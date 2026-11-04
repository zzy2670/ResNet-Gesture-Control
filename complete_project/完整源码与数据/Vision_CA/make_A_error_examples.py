"""从已保存整图错误中取固定代表例，展示真实标注与自动框，不改变任何模型。"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    root = Path(__file__).resolve().parent
    run = root / 'runs/A_20261102_TPE'
    predictions = json.loads((run / 'test_whole_photo.json').read_text(encoding='utf-8'))['predictions']
    crops = json.loads((run / 'crops_test.json').read_text(encoding='utf-8'))
    sources = {}
    for crop in crops:
        sources.setdefault(crop['image_id'], []).append(crop)
    chosen = [min((row for row in predictions if row['truth'] == label and row['prediction'] != label),
                  key=lambda row: row['image_id']) for label in [0, 3, 4, 5]]
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    figure, axes = plt.subplots(2, 2, figsize=(11, 8.6))
    output = []
    for axis, row in zip(axes.flat, chosen):
        truth = sources[row['image_id']]
        image = Image.open(truth[0]['source_path']).convert('RGB')
        axis.imshow(image)
        for crop in truth:
            x1, y1, x2, y2 = crop['bbox_xyxy']
            axis.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='#00C764', linewidth=1.8))
            axis.text(x1, max(10, y1-3), f"GT {crop['label']}", color='#005222', fontsize=8,
                      bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=.4))
        for hand in row['hands']:
            x1, y1, x2, y2 = hand['bbox']
            color = '#C00000' if hand['accepted'] else '#666666'
            axis.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=1.3, linestyle='--'))
            axis.text(x1, min(image.height-4, y2+9), f"P {hand['label']} / {hand['confidence']:.2f}", color=color, fontsize=8,
                      bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=.4))
        axis.set_title(f"真实命令{row['truth']} → 输出{row['prediction']}\n{row['image_id'][:12]}", fontsize=11)
        axis.axis('off')
        output.append({'image_id': row['image_id'], 'truth': row['truth'], 'prediction': row['prediction'],
                       'gt_hands': truth, 'predicted_hands': row['hands'],
                       'selection': 'first_wrong_image_key_in_preselected_truth_classes_0_3_4_5'})
    figure.suptitle('整图错误的固定代表例：绿色实线为人工标注，虚线为自动手框', fontsize=13)
    figure.tight_layout(rect=[0, 0, 1, .95])
    dest = run / 'report_figures'
    dest.mkdir(exist_ok=True)
    figure.savefig(dest / 'whole_photo_error_examples.png', dpi=200)
    plt.close(figure)
    (run / 'error_examples.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    # 对同一照片比较两种实际裁图，帮助区分定位输入变化和分类输出变化。
    oracle = json.loads((run / 'test_oracle_crop.json').read_text(encoding='utf-8'))['predictions']
    by_key = {(row['image_id'], row['hand_index']): row for row in oracle}
    fig, grid = plt.subplots(2, 2, figsize=(8.5, 8.3))
    comparisons = []
    for row_number, example in enumerate(output[:2]):
        hand = next(row for row in example['gt_hands'] if row['label'] == example['truth'])
        detected = max(example['predicted_hands'], key=lambda row: row['detector_score'])
        source = Image.open(hand['source_path']).convert('RGB')
        x1, y1, x2, y2 = detected['bbox']
        side = max(x2-x1, y2-y1) * 1.2
        cx, cy = (x1+x2)/2, (y1+y2)/2
        rectangle = (math.floor(cx-side/2), math.floor(cy-side/2),
                     math.ceil(cx+side/2), math.ceil(cy+side/2))
        automatic = source.crop(rectangle).resize((224, 224), Image.Resampling.BILINEAR)
        known = by_key[(example['image_id'], hand['hand_index'])]
        grid[row_number, 0].imshow(Image.open(hand['path']).convert('RGB'))
        grid[row_number, 1].imshow(automatic)
        grid[row_number, 0].set_title(f"人工框输入 → 预测{known['prediction']}\n真实类{hand['label']}（{hand['raw_label']}）", fontsize=11)
        grid[row_number, 1].set_title(f"自动框输入 → 预测{detected['label']}\n分类置信度{detected['confidence']:.3f}", fontsize=11)
        for axis in grid[row_number]:
            axis.axis('off')
        comparisons.append({'image_id': example['image_id'], 'hand_index': hand['hand_index'],
                            'truth': hand['label'], 'oracle_prediction': known['prediction'],
                            'automatic_prediction': detected['label'], 'automatic_crop_rectangle': rectangle,
                            'oracle_crop_rectangle': hand['crop_rectangle'], 'network_rerun': False})
    fig.suptitle('同一照片的人工框与自动框裁图对照', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(dest / 'actual_crop_comparison.png', dpi=200)
    plt.close(fig)
    (run / 'actual_crop_comparison.json').write_text(json.dumps(comparisons, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps([{'truth': r['truth'], 'prediction': r['prediction'], 'hands': len(r['predicted_hands'])} for r in output], ensure_ascii=False))


if __name__ == '__main__':
    main()
