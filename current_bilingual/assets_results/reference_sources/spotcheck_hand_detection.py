"""仅展示四张指定照片的冻结检测结果，不加载模型、不遍历图片、不改变阈值。"""
from pathlib import Path
import csv
import hashlib
import itertools
import json
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/B_20261102_TPE'
DATA = ROOT / 'data/B_split_verified_20261102/test'
OUT = Path(__file__).parent / 'detection_spotcheck_20261103'
IDS = [
    ('01aa0cd7-798e-4e2b-b595-39cc2376a259', '两只手均正确定位'),
    ('0d72367c-e360-43b1-9c08-5ddea11c3b8c', '两只手均未检出'),
    ('163e7b4f-00ef-436e-9cfc-e693bedce0a0', '漏掉举起的拳头，只检出垂下的手'),
    ('0fd29f45-cad4-4c0f-8ea0-0f98e81fb17c', '检测框偏移，手势也分错'),
]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iou(a, b):
    # IoU只衡量框的重合程度，不等同于手势识别是否正确。
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = w * h
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0


def main():
    OUT.mkdir(exist_ok=True)
    detector_path = RUN / 'test_detector.json'
    whole_path = RUN / 'test_whole_photo.json'
    detector = read(detector_path)
    whole_doc = read(whole_path)
    # 读取已有JSON/清单定位指定记录，仅打开IDS列出的四张照片。
    detections = {r['image_id']: r for r in detector['predictions']}
    whole = {r['image_id']: r for r in whole_doc['predictions']}
    manifest = {r['image_id']: r for r in csv.DictReader((DATA/'manifest.csv').open(encoding='utf-8'))}
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 22)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    canvas = Image.new('RGB', (1120, 2050), '#f5f7fa')
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 12), 'B协议 ResNet18：四张指定双手照片抽查（冻结测试结果）', fill='#172b4d', font=font)
    draw.text((24, 47), '左：人工标注（绿色）    右：实际检测（橙色）；未重新推理、未改阈值', fill='#344563', font=small)
    records = []
    for index, (rid, title) in enumerate(IDS):
        reference = manifest[rid]
        image_path = DATA / reference['image_path']
        image = Image.open(image_path).convert('RGB')
        row, pred = detections[rid], whole[rid]
        ann_path = DATA / 'annotations' / (reference['class_name']+'.json')
        annotation = read(ann_path)[rid]
        y = 88 + index * 485
        draw.text((24, y), f'{index+1}. {title}；真值{pred["truth"]} → 输出{pred["prediction"]}', fill='#172b4d', font=font)
        scale = min(490/image.width, 410/image.height)
        size = (round(image.width*scale), round(image.height*scale))
        for column in (0, 1):
            x = 24 + column*555
            xp = x + (510-size[0])//2
            yp = y + 40
            canvas.paste(image.resize(size, Image.Resampling.LANCZOS), (xp, yp))
            boxes = row['gt_boxes'] if column == 0 else row['boxes']
            for number, box in enumerate(boxes):
                rectangle = [xp+box[0]*scale, yp+box[1]*scale, xp+box[2]*scale, yp+box[3]*scale]
                color = '#08c77b' if column == 0 else '#ff8a00'
                draw.rectangle(rectangle, outline=color, width=3)
                label = annotation['labels'][number] if column == 0 else f'D{number+1} p={row["scores"][number]:.2f}'
                tx, ty = rectangle[0], max(yp, rectangle[1]-22)
                bounds = draw.textbbox((tx, ty), label, font=small)
                draw.rectangle(bounds, fill='#142238')
                draw.text((tx, ty), label, font=small, fill=color)
            if not boxes:
                draw.text((x+20, yp+12), '无检测框', fill='#e64a19', font=font)
        matrix = [[iou(a, b) for b in row['boxes']] for a in row['gt_boxes']]
        # 两个GT最多逐一配对到不同检测框，防止同一框被重复当成两只手。
        pairs = []
        candidates = sorted(((value, gi, di) for gi, values in enumerate(matrix) for di, value in enumerate(values)), reverse=True)
        used_g, used_d = set(), set()
        for value, gi, di in candidates:
            if value >= .5 and gi not in used_g and di not in used_d:
                pairs.append({'GT_index': gi, 'detection_index': di, 'IoU': value})
                used_g.add(gi)
                used_d.add(di)
        records.append({'image_id': rid, 'description': title, 'source_image': str(image_path),
            'source_image_sha256': digest(image_path), 'raw_GT_labels': annotation['labels'],
            'GT_boxes': row['gt_boxes'], 'predicted_boxes': row['boxes'], 'detector_scores': row['scores'],
            'IoU_matrix_GT_by_detection': matrix, 'one_to_one_matches_IoU05': pairs,
            'missed_GT_indices_at_IoU05': sorted(set(range(len(row['gt_boxes'])))-used_g),
            'whole_prediction_record': pred})
    output = OUT / '四张双手照片_人工标注与检测对照.png'
    canvas.save(output)
    report = {'scope': '四张固定B-test照片的定向错误抽查，非随机样本，不估计总体发生率；未重新推理。',
        'model': 'unmerged_B_resnet18', 'detector_threshold': whole_doc['detector_threshold'],
        'classifier_threshold': whole_doc['class_threshold'], 'detector_metrics_original': detector['metrics'],
        'source_sha256': {str(detector_path): digest(detector_path), str(whole_path): digest(whole_path),
                          str(Path(__file__)): digest(Path(__file__))},
        'records': records, 'figure': str(output), 'figure_sha256': digest(output)}
    (OUT/'spotcheck.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'figure': str(output), 'samples': [{k:r[k] for k in ('image_id','raw_GT_labels','IoU_matrix_GT_by_detection','missed_GT_indices_at_IoU05')} for r in records]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
