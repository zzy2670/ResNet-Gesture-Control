"""只核四个指定ID的冻结记录与框，不遍历照片、不加载模型。"""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = RUN / 'detection_spotcheck_20261103'
IDS = ('01aa0cd7-798e-4e2b-b595-39cc2376a259', '0d72367c-e360-43b1-9c08-5ddea11c3b8c',
       '163e7b4f-00ef-436e-9cfc-e693bedce0a0', '0fd29f45-cad4-4c0f-8ea0-0f98e81fb17c')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlap(a, b):
    # 独立由交并面积重算，不调用作者的IoU函数。
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return ix * iy / (area_a + area_b - ix * iy)


def main():
    from PIL import Image
    source = OUT / 'spotcheck.json'
    spot = read(source)
    assert [r['image_id'] for r in spot['records']] == list(IDS)
    bound = {str(source): sha(source), str(Path(__file__).resolve()): sha(__file__)}
    for path, digest in spot['source_sha256'].items():
        assert sha(path) == digest
        bound[path] = digest
    det_doc = read(ROOT / 'runs/B_20261102_TPE/test_detector.json')
    whole_doc = read(ROOT / 'runs/B_20261102_TPE/test_whole_photo.json')
    det = {r['image_id']: r for r in det_doc['predictions'] if r['image_id'] in IDS}
    whole = {r['image_id']: r for r in whole_doc['predictions'] if r['image_id'] in IDS}
    assert len(det) == len(whole) == 4
    assert spot['detector_threshold'] == whole_doc['detector_threshold'] == .3
    assert spot['classifier_threshold'] == whole_doc['class_threshold'] == .3
    dataset = ROOT / 'data/B_split_verified_20261102/test'
    with (dataset / 'manifest.csv').open(encoding='utf-8', newline='') as stream:
        manifest = {r['image_id']: r for r in csv.DictReader(stream) if r['image_id'] in IDS}
    result = []
    for row in spot['records']:
        rid = row['image_id']
        frozen = det[rid]
        assert frozen['gt_boxes'] == row['GT_boxes'] and frozen['boxes'] == row['predicted_boxes']
        assert frozen['scores'] == row['detector_scores'] and whole[rid] == row['whole_prediction_record']
        picture = Path(row['source_image'])
        assert picture == dataset / manifest[rid]['image_path'] and sha(picture) == row['source_image_sha256']
        bound[str(picture)] = sha(picture)
        # 仅打开这四张指定照片的文件头，不枚举或扫描照片目录。
        with Image.open(picture) as image:
            width, height = image.size
        assert [width, height] == whole[rid]['image_size']
        annotation_file = dataset / 'annotations' / (manifest[rid]['class_name'] + '.json')
        annotation = read(annotation_file)[rid]
        bound[str(annotation_file)] = sha(annotation_file)
        assert annotation['labels'] == row['raw_GT_labels']
        expected = [[max(0, x * width), max(0, y * height), min(width, (x + w) * width),
                     min(height, (y + h) * height)] for x, y, w, h in annotation['bboxes']]
        assert len(expected) == len(row['GT_boxes']) == 2
        assert max(abs(a - b) for p, q in zip(expected, row['GT_boxes']) for a, b in zip(p, q)) < 1e-4
        matrix = [[overlap(a, b) for b in row['predicted_boxes']] for a in row['GT_boxes']]
        assert all(abs(a - b) < 1e-12 for p, q in zip(matrix, row['IoU_matrix_GT_by_detection']) for a, b in zip(p, q))
        used_g, used_d = set(), set()
        matches = []
        for value, gi, di in sorted(((value, gi, di) for gi, values in enumerate(matrix) for di, value in enumerate(values)), reverse=True):
            if value >= .5 and gi not in used_g and di not in used_d:
                matches.append({'GT_index': gi, 'detection_index': di, 'IoU': value})
                used_g.add(gi)
                used_d.add(di)
        assert matches == row['one_to_one_matches_IoU05']
        assert sorted({0, 1} - used_g) == row['missed_GT_indices_at_IoU05']
        assert row['detector_scores'] == [h['detector_score'] for h in whole[rid]['hands']]
        assert row['predicted_boxes'] == [h['bbox'] for h in whole[rid]['hands']]
        result.append({'image_id': rid, 'matched_GT_hands_at_IoU05': len(matches), 'GT_hands': 2,
                       'missed_GT_indices': sorted({0, 1} - used_g), 'recomputed_IoU': matrix,
                       'truth': whole[rid]['truth'], 'prediction': whole[rid]['prediction']})
    figure = Path(spot['figure'])
    assert sha(figure) == spot['figure_sha256']
    bound[str(figure)] = sha(figure)
    fourth = whole[IDS[3]]['hands'][1]
    assert fourth['label'] == 0 and fourth['accepted'] is True and fourth['probabilities'][0] > fourth['probabilities'][1]
    report = {'status': 'PASS_FOUR_SPECIFIED_RECORDS_WITH_WORDING_LIMITATIONS',
        'reviewed_at': datetime.now().astimezone().isoformat(), 'reviewer': 'merged_independent_preflight',
        'scope': '只核四个指定B-test ID、源标注/冻结输出/独立IoU及已实际查看的对照图；未遍历照片、未推理、未调参数。',
        'records': result, 'issues': [], 'visual_review': {'figure_actually_viewed': True,
            'left_GT_and_right_detection_consistent': True, 'hand_boxes_and_missed_hands_visible': True,
            'no_material_label_occlusion_or_misplaced_annotations_observed': True},
        'wording': ['不能说双手全部找出：指定例2两手均无框，例3漏fist，例4另一手框IoU约0.264未达0.5。',
            '例2真值0输出0只说明最终指令碰巧一致，不说明检测成功。',
            '例4peace手有IoU约0.673的合格匹配框，但这个检测裁图被分类为0且接受；不能说peace手完全漏检，也不能说是低置信拒识。',
            '例4可称定位不充分与裁图分类错误并存；未做对照推理，不能证明框偏移是分类错误的唯一原因。',
            '图中D1/D2的p是检测分数，不是ResNet手势概率；建议在说明中明确。',
            '四张为定向抽查，不能给出或外推总体漏检比例。'],
        'bound_file_sha256': bound}
    payload = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    for path, digest in bound.items():
        assert sha(path) == digest
    destination = OUT / 'spotcheck_independent_review.json'
    with destination.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'records': len(result), 'output': str(destination)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
