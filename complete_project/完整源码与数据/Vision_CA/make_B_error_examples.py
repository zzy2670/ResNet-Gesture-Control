"""用B冻结预测和原图展示整图错误与真实裁图差异，不重新推理。"""
from pathlib import Path
import json
import math

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent / 'runs/B_20261102_TPE'


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def main():
    predictions = read('test_whole_photo.json')['predictions']
    crops = read('crops_test.json')
    oracle = {(r['image_id'], r['hand_index']): r for r in read('test_oracle_crop.json')['predictions']}
    sources = {}
    for row in crops:
        sources.setdefault(row['image_id'], []).append(row)
    # 固定类别后按ID取首个错误，保留没检出手的情况，不能只挑有框的错误。
    chosen = [min((r for r in predictions if r['truth'] == label and r['prediction'] != label),
                  key=lambda r: r['image_id']) for label in [0, 3, 4, 5]]
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'], 'axes.unicode_minus': False})
    dest = ROOT / 'report_figures'
    dest.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    records = []
    for ax, row in zip(axes.flat, chosen):
        truth = sources[row['image_id']]
        image = Image.open(truth[0]['source_path']).convert('RGB')
        ax.imshow(image)
        for hand in truth:
            x1, y1, x2, y2 = hand['bbox_xyxy']
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='#008B57', linewidth=1.7))
            ax.text(x1, max(9, y1-3), f"GT {hand['label']}", fontsize=9, color='#005A35',
                    bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': 'none', 'pad': .4})
        for hand in row['hands']:
            x1, y1, x2, y2 = hand['bbox']
            color = '#B72B2B' if hand['accepted'] else '#666666'
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=1.4, linestyle='--'))
            ax.text(x1, min(image.height-4, y2+10), f"P {hand['label']} / {hand['confidence']:.2f}", fontsize=9, color=color,
                    bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': 'none', 'pad': .4})
        ax.set_title(f"真实命令{row['truth']} → 输出{row['prediction']}\n{row['image_id'][:14]}")
        ax.axis('off')
        records.append({'image_id': row['image_id'], 'truth': row['truth'], 'prediction': row['prediction'],
                        'gt_hands': truth, 'predicted_hands': row['hands'],
                        'selection': 'first_wrong_image_id_in_prespecified_classes_0_3_4_5'})
    fig.suptitle('B整图错误：绿色实线为人工框，虚线为自动检测框', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(dest / 'whole_photo_error_examples.png', dpi=180)
    plt.close(fig)
    (ROOT / 'error_examples.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    fig, grid = plt.subplots(2, 2, figsize=(8.5, 8.5))
    comparisons = []
    for line, example in enumerate(records[:2]):
        candidates = [r for r in example['gt_hands'] if r['label'] == example['truth']]
        hand = min(candidates or example['gt_hands'], key=lambda r: r['hand_index'])
        known = oracle[(example['image_id'], hand['hand_index'])]
        grid[line, 0].imshow(Image.open(hand['path']).convert('RGB'))
        grid[line, 0].set_title(f"人工框输入：真实{hand['label']}，预测{known['prediction']}")
        entry = {'image_id': example['image_id'], 'hand_index': hand['hand_index'],
                 'truth': hand['label'], 'oracle_prediction': known['prediction'],
                 'oracle_crop_rectangle': hand['crop_rectangle'], 'network_rerun': False}
        if example['predicted_hands']:
            detected = max(example['predicted_hands'], key=lambda r: r['detector_score'])
            image = Image.open(hand['source_path']).convert('RGB')
            x1, y1, x2, y2 = detected['bbox']
            side = max(x2-x1, y2-y1) * 1.2
            cx, cy = (x1+x2)/2, (y1+y2)/2
            rectangle = (math.floor(cx-side/2), math.floor(cy-side/2), math.ceil(cx+side/2), math.ceil(cy+side/2))
            grid[line, 1].imshow(image.crop(rectangle).resize((224, 224), Image.Resampling.BILINEAR))
            grid[line, 1].set_title(f"最高检测分数框：预测{detected['label']}，置信度{detected['confidence']:.3f}")
            entry.update(automatic_prediction=detected['label'], automatic_crop_rectangle=rectangle,
                         comparison_scope='highest_score_detected_hand_not_guaranteed_GT_matching')
        else:
            grid[line, 1].text(.5, .5, '没有通过门槛的手框\n没有进入分类器的自动裁图', ha='center', va='center', fontsize=12)
            grid[line, 1].set_title('自动定位阶段没有产生分类输入')
            entry.update(automatic_prediction=None, automatic_crop_rectangle=None)
        for ax in grid[line]:
            ax.axis('off')
        comparisons.append(entry)
    fig.suptitle('B同一照片的人工框输入与自动定位输入', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(dest / 'actual_crop_comparison.png', dpi=180)
    plt.close(fig)
    (ROOT / 'actual_crop_comparison.json').write_text(json.dumps(comparisons, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps([{'truth': r['truth'], 'prediction': r['prediction'], 'hands': len(r['predicted_hands'])} for r in records], ensure_ascii=False))


if __name__ == '__main__':
    main()
