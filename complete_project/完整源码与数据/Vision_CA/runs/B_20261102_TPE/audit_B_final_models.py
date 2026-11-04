"""只读复核冻结B模型；不训练、不推理、不调用CUDA、不修改研究源文件。

用法：python audit_B_final_models.py --arch resnet18 --dry-run
      python audit_B_final_models.py --arch resnet34
      python audit_B_final_models.py --arch all --require-latency
不完整模型只返回等待状态，不写PASS。默认拒绝覆盖既有审核；重审须明确给--replace-review。
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sqlite3
import sys

# 本机依赖仅用于CPU读取张量、数组统计和图像头；不导入训练入口。
sys.path.insert(0, 'E:/python/Lib/site-packages')
import numpy as np
import torch
from PIL import Image

# 审核张量不需要多核竞争，训练控制器继续持有GPU及主要计算资源。
torch.set_num_threads(1)

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
DATA = ROOT / 'data/B_split_verified_20261102'
ARCHES = ('resnet18', 'resnet34', 'resnet50')
MAP = {'peace': 1, 'peace_inverted': 1, 'stop': 2,
       'stop_inverted': 2, 'palm': 2, 'fist': 3, 'like': 4, 'dislike': 5}
OFFICIAL = {'resnet18': 'f37072fd', 'resnet34': 'b627a593', 'resnet50': '0676ba61'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def events(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.startswith('{')]


def folder(arch, run=RUN):
    return run if arch == 'resnet18' else run / ('compare_' + arch)


def command(labels):
    if 3 in labels:
        return 3
    active = set(labels) - {0}
    return next(iter(active)) if len(active) == 1 else 0


def from_matrix(matrix):
    cm = np.asarray(matrix, dtype=np.int64)
    tp = cm.diagonal().astype(float)
    p = np.divide(tp, cm.sum(0), out=np.zeros(6), where=cm.sum(0) != 0)
    r = np.divide(tp, cm.sum(1), out=np.zeros(6), where=cm.sum(1) != 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros(6), where=p + r != 0)
    return {'n': int(cm.sum()), 'accuracy': float(tp.sum() / max(1, cm.sum())),
            'macro_f1': float(f.mean()), 'confusion_matrix': cm.tolist(),
            'precision': p.tolist(), 'recall': r.tolist(), 'f1': f.tolist(),
            'support': cm.sum(1).tolist()}


def metrics(rows):
    cm = [[0] * 6 for _ in range(6)]
    for row in rows:
        cm[row['truth']][row['prediction']] += 1
    return from_matrix(cm)


def safety(rows):
    zero = [r for r in rows if r['truth'] == 0]
    stop = [r for r in rows if r['truth'] == 3]
    zm = sum(r['prediction'] in {1, 2, 4, 5} for r in zero)
    sm = sum(r['prediction'] in {1, 2, 4, 5} for r in stop)
    return {'zero_samples': len(zero), 'zero_to_motion': zm,
            'stop_samples': len(stop), 'stop_to_motion': sm,
            'zero_to_motion_rate': zm / max(1, len(zero)),
            'stop_to_motion_rate': sm / max(1, len(stop))}


def errors_and_safety(rows):
    cm = metrics(rows)['confusion_matrix']
    return {**safety(rows), 'stop_to_zero': cm[3][0],
            'stop_to_zero_rate': cm[3][0] / max(1, sum(cm[3])),
            'stop_correct': cm[3][3], 'zero_to_stop': cm[0][3],
            'error_directions': [{'truth': i, 'prediction': j, 'count': cm[i][j]}
                                 for i in range(6) for j in range(6) if i != j and cm[i][j]],
            'errors': [{'image_id': r['image_id'], **({'hand_index': r['hand_index']}
                       if 'hand_index' in r else {}), 'truth': r['truth'],
                       'prediction': r['prediction']} for r in rows
                       if r['truth'] != r['prediction']]}


class Audit:
    def __init__(self):
        self.issues = []
        self.counts = collections.Counter()

    def check(self, ok, label, detail=None):
        self.counts[label.split(':')[0].split('.')[0].split('[')[0]] += 1
        if not ok:
            self.issues.append({'check': label, 'detail': detail})

    def same(self, left, right, label, tol=1e-11):
        if isinstance(left, dict) and isinstance(right, dict):
            self.check(set(left) == set(right), label + ':keys', [list(left), list(right)])
            for key in left.keys() & right.keys():
                self.same(left[key], right[key], label + '.' + key, tol)
        elif isinstance(left, list) and isinstance(right, list):
            self.check(len(left) == len(right), label + ':length', [len(left), len(right)])
            for i, (a, b) in enumerate(zip(left, right)):
                self.same(a, b, label + f'[{i}]', tol)
        elif isinstance(left, (float, int)) and isinstance(right, (float, int)):
            self.check(math.isfinite(left) and math.isfinite(right)
                       and abs(left - right) <= tol, label, [left, right])
        else:
            self.check(left == right, label, [left, right])


def dataset(audit):
    # 复用已完成全量图片SHA证据，本轮重新核清单、标注和身份交叉。
    pre = read(RUN / 'preflight_independent_review.json')
    truth, counts, hashes = {}, {}, {}
    registry = read(DATA / 'source_and_copy_hashes.json')
    for split in ('train', 'val', 'test'):
        with (DATA / split / 'manifest.csv').open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        hashes[split] = digest(DATA / split / 'manifest.csv')
        audit.check(hashes[split] == pre['manifests'][split]['sha256'], 'manifest:' + split)
        ann = {p.stem: read(p) for p in (DATA / split / 'annotations').glob('*.json')}
        truth[split] = {}
        hands = collections.Counter()
        for row in rows:
            item = ann[row['class_name']][row['image_id']]
            audit.check(item['user_id'] == row['user_id'], 'annotation_user:' + row['image_id'])
            audit.check(len(item['labels']) == len(item['bboxes']), 'annotation_shape:' + row['image_id'])
            labels = [MAP.get(label, 0) for label in item['labels']]
            hands.update(labels)
            truth[split][row['image_id']] = {**row, 'annotation': item,
                                           'labels': labels, 'truth': command(labels)}
        ids = set(truth[split])
        users = {row['user_id'] for row in rows}
        reg = [r for r in registry if r['split'] == split]
        audit.check(len(ids) == len(rows), 'manifest_unique:' + split)
        audit.check(users == set(read(DATA / (split + '_users.json'))), 'user_list:' + split)
        audit.check({r['image_id'] for r in reg} == ids, 'hash_registry_coverage:' + split)
        for row in reg:
            audit.check(row['user_id'] == truth[split][row['image_id']]['user_id'],
                        'hash_registry_user:' + row['image_id'])
        counts[split] = {'photos': len(rows), 'users': len(users), 'hands': sum(hands.values()),
                         'hand_class_support': [hands[i] for i in range(6)]}
        audit.check(len(rows) == pre['splits'][split]['images'] and
                    len(users) == pre['splits'][split]['users'] and
                    sum(hands.values()) == pre['splits'][split]['hands'], 'preflight_counts:' + split)
    overlap = {}
    for left, right in itertools.combinations(truth, 2):
        overlap[left + '_' + right] = {
            'image_id': len(set(truth[left]) & set(truth[right])),
            'user_id': len({x['user_id'] for x in truth[left].values()} &
                           {x['user_id'] for x in truth[right].values()}),
            'recorded_sha256': len({r['sha256'] for r in registry if r['split'] == left} &
                                   {r['sha256'] for r in registry if r['split'] == right})}
        audit.check(not any(overlap[left + '_' + right].values()), 'split_overlap:' + left + right)
    return truth, counts, hashes, overlap


def tpe_and_training(audit, out, arch, counts, identity):
    # SQLite以只读模式打开，逐trial还原参数、18轮中间值与原日志。
    summary = read(out / 'tpe_summary.json')
    with sqlite3.connect((out / 'tpe.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as con:
        studies = con.execute('SELECT study_id,study_name FROM studies').fetchall()
        audit.check(len(studies) == 1 and studies[0][1] == f'B_{arch}_frozen_features_TPE', 'tpe_study')
        trials = con.execute('SELECT trial_id,number,state FROM trials ORDER BY number').fetchall()
        audit.check(len(trials) == 24 and [x[1] for x in trials] == list(range(24))
                    and all(x[2] == 'COMPLETE' for x in trials), 'tpe_24_complete')
        logrows = events(out / 'stage_logs/search_attempt1.log')
        trial_logs = [r for r in logrows if r.get('stage') == 'tpe']
        audit.check(len(trial_logs) == 24 * 18, 'tpe_log_total_epochs')
        reconstructed = []
        for tid, number, state in trials:
            steps = con.execute('SELECT step,intermediate_value FROM trial_intermediate_values '
                                'WHERE trial_id=? ORDER BY step', (tid,)).fetchall()
            audit.check([x[0] for x in steps] == list(range(18)), f'tpe_epochs:{number}')
            params = {}
            for name, value, distribution in con.execute(
                    'SELECT param_name,param_value,distribution_json FROM trial_params WHERE trial_id=?', (tid,)):
                dist = json.loads(distribution)
                params[name] = dist['attributes']['choices'][int(value)] if dist['name'] == 'CategoricalDistribution' else value
            values = con.execute('SELECT value FROM trial_values WHERE trial_id=?', (tid,)).fetchall()
            audit.check(len(values) == 1, f'tpe_objective_count:{number}')
            value = values[0][0]
            audit.check(math.isfinite(value) and abs(value - max(x[1] for x in steps)) < 1e-12,
                        f'tpe_objective_max:{number}')
            logs = [r for r in trial_logs if r['trial'] == number]
            audit.check([r['epoch'] for r in logs] == list(range(1, 19)), f'tpe_log_epochs:{number}')
            audit.same([x[1] for x in steps], [r['validation_macro_f1'] for r in logs], f'tpe_sqlite_log:{number}')
            head = torch.load(out / 'trials' / f'trial_{number}_best.pt', map_location='cpu', weights_only=True)
            audit.same(head['params'], params, f'tpe_checkpoint_params:{number}')
            audit.same(from_matrix(head['validation']['confusion_matrix']), head['validation'], f'tpe_checkpoint_metrics:{number}')
            audit.check(abs(head['validation']['macro_f1'] - value) < 1e-12
                        and head['epoch'] == next(step + 1 for step, score in steps if score == value),
                        f'tpe_checkpoint_best:{number}')
            reconstructed.append({'number': number, 'state': state, 'value': value, 'params': params})
        audit.same(reconstructed, summary['trials'], 'tpe_summary')
    best = max(reconstructed, key=lambda x: x['value'])
    audit.check(summary['best_trial'] == best['number'] and summary['test_accessed'] is False, 'tpe_selection')
    audit.same(summary['best_params'], best['params'], 'tpe_best_params')
    audit.same(summary['best_value'], best['value'], 'tpe_best_value')
    logs = events(out / 'stage_logs/finetune_attempt1.log')
    epochs = [r for r in logs if r.get('stage') == 'finetune' and r.get('status') == 'complete_epoch']
    audit.check([r['epoch'] for r in epochs] == list(range(1, len(epochs) + 1))
                and 1 <= len(epochs) <= 18, 'finetune_contiguous_epochs')
    bad = 0
    score = best['value']
    selected = None
    for row in epochs:
        audit.check(row['samples'] == counts['train']['hands']
                    and row['validation']['n'] == counts['val']['hands'], 'finetune_denominators')
        audit.same(from_matrix(row['validation']['confusion_matrix']), row['validation'], 'finetune_metrics')
        if row['validation']['macro_f1'] > score:
            score, bad, selected = row['validation']['macro_f1'], 0, row
        else:
            bad += 1
        audit.check(abs(row['best'] - score) < 1e-12, 'finetune_running_best')
    audit.check(len(epochs) == 18 or bad >= 4, 'finetune_natural_budget_or_patience')
    audit.check(not any(r.get('status') in ('partial_epoch_not_selected', 'deadline_before_epoch') for r in logs),
                'finetune_no_partial_deadline')
    ck = torch.load(out / 'classifier_best.pt', map_location='cpu', weights_only=True)
    audit.same(ck['params'], best['params'], 'classifier_tpe_params')
    audit.check(selected is not None and ck['stage'] == 'layer4_finetune'
                and ck['epoch'] == selected['epoch'], 'classifier_best_epoch')
    if selected:
        audit.same(ck['validation'], selected['validation'], 'classifier_best_validation')
    official = Path(identity['pretrained']['path'])
    official_hash = digest(official)
    audit.check(official_hash == identity['pretrained']['sha256']
                and official_hash.startswith(OFFICIAL[arch]), 'official_imagenet_sha')
    original = torch.load(official, map_location='cpu', weights_only=True)
    frozen = [key for key in ck['model'] if not key.startswith(('layer4.', 'fc.'))]
    inserted_counters = []
    for key in frozen:
        # 老版官方权重省略BN计数器，TorchVision严格加载时补零；单独核零不能冒充官方已有张量。
        if key not in original and key.endswith('.num_batches_tracked'):
            inserted_counters.append(key)
            audit.check(ck['model'][key].item() == 0, 'official_inserted_zero_counter:' + key)
        else:
            audit.check(key in original and torch.equal(ck['model'][key], original[key]), 'official_frozen_tensor:' + key)
    dim = 2048 if arch == 'resnet50' else 512
    audit.check(list(ck['model']['fc.1.weight'].shape) == [6, dim], 'classifier_head_shape')
    for split in ('train', 'val'):
        cached = torch.load(out / f'features_{split}.pt', map_location='cpu', weights_only=True)
        crops = read(out / f'crops_{split}.json')
        audit.check(list(cached['x'].shape) == [counts[split]['hands'], dim]
                    and torch.isfinite(cached['x']).all().item(), 'feature_shape_finite:' + split)
        audit.check(cached['y'].tolist() == [r['label'] for r in crops]
                    and cached['pretrained_sha256'] == official_hash, 'feature_identity:' + split)
    return {'complete_trials': len(reconstructed), 'epochs_each_trial': 18,
            'best_trial': best['number'], 'best_params': best['params'], 'best_value': best['value'],
            'finetune_complete_epochs': len(epochs), 'finetune_best_epoch': ck['epoch'],
            'finetune_best_validation': ck['validation'], 'official_sha256': official_hash,
            'frozen_official_tensors_equal': len(frozen) - len(inserted_counters),
            'official_missing_BN_counters_inserted_zero': inserted_counters,
            'checkpoint_sha256': digest(out / 'classifier_best.pt')}


def raw_predictions(audit, doc, truth, split, detector_threshold, class_threshold):
    rows = doc['predictions']
    ids = [r['image_id'] for r in rows]
    audit.check(len(ids) == len(set(ids)) and set(ids) == set(truth[split]), 'photo_coverage:' + split)
    for row in rows:
        ref = truth[split][row['image_id']]
        audit.check(row['truth'] == ref['truth'], 'photo_truth:' + split)
        with Image.open(DATA / split / ref['image_path']) as image:
            wh = list(image.size)
        audit.check(row['image_size'] == wh, 'photo_dimensions:' + split)
        accepted = []
        for hand in row['hands']:
            p = hand['probabilities']
            audit.check(len(p) == 6 and all(math.isfinite(x) and 0 <= x <= 1 for x in p)
                        and abs(sum(p) - 1) < 1e-5, 'hand_probabilities:' + split)
            audit.check(hand['label'] == int(np.argmax(p)) and abs(hand['confidence'] - max(p)) < 1e-7,
                        'hand_argmax_confidence:' + split)
            audit.check(hand['accepted'] == (hand['confidence'] >= class_threshold)
                        and math.isfinite(hand['detector_score']) and
                        hand['detector_score'] >= detector_threshold, 'hand_threshold:' + split)
            box = hand['bbox']
            audit.check(len(box) == 4 and all(math.isfinite(x) for x in box)
                        and box[2] > box[0] and box[3] > box[1], 'hand_box:' + split)
            if hand['confidence'] >= class_threshold:
                accepted.append(hand['label'])
        audit.check(row['prediction'] == command(accepted), 'photo_aggregation:' + split)
    return rows


def detector_metrics(audit, doc, whole, truth, threshold):
    # 用NumPy float32复现几何计算，保持逐框原顺序，不调用模型。
    tp = fp = fn = 0
    by_id = {r['image_id']: r for r in whole}
    ids = []
    details = []
    for row in doc['predictions']:
        rid = row['image_id']
        ids.append(rid)
        photo = by_id[rid]
        w, h = photo['image_size']
        gt = [[max(0., x * w), max(0., y * h), min(float(w), (x + bw) * w),
               min(float(h), (y + bh) * h)] for x, y, bw, bh in truth['test'][rid]['annotation']['bboxes']]
        audit.same(np.asarray(gt, dtype=np.float32).tolist(), row['gt_boxes'], 'detector_gt:' + rid)
        audit.same(row['boxes'], [r['bbox'] for r in photo['hands']], 'detector_shared_boxes:' + rid)
        audit.same(row['scores'], [r['detector_score'] for r in photo['hands']], 'detector_shared_scores:' + rid)
        audit.check(len(row['boxes']) == len(row['scores']), 'detector_shapes:' + rid)
        used = set()
        pt = pf = 0
        gt_array = np.asarray(row['gt_boxes'], dtype=np.float32).reshape(-1, 4)
        for box, score in zip(row['boxes'], row['scores']):
            if score < threshold:
                continue
            box = np.asarray(box, dtype=np.float32)
            area = np.prod(np.maximum(0., gt_array[:, 2:] - gt_array[:, :2]), axis=1)
            inter = np.prod(np.maximum(0., np.minimum(box[2:], gt_array[:, 2:])
                                      - np.maximum(box[:2], gt_array[:, :2])), axis=1)
            iou = inter / (area + np.prod(box[2:] - box[:2]) - inter)
            order = sorted(range(len(iou)), key=lambda i: float(iou[i]), reverse=True)
            match = next((i for i in order if i not in used and iou[i] >= .5), None)
            if match is None:
                pf += 1
            else:
                used.add(match)
                pt += 1
        nf = len(gt) - len(used)
        tp, fp, fn = tp + pt, fp + pf, fn + nf
        details.append({'image_id': rid, 'tp': pt, 'fp': pf, 'fn': nf})
    audit.check(len(ids) == len(set(ids)) and set(ids) == set(truth['test']), 'detector_coverage')
    result = {'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp / max(1, tp + fp),
              'recall': tp / max(1, tp + fn), 'score_threshold': threshold, 'iou_threshold': .5}
    audit.same(result, doc['metrics'], 'detector_metrics')
    return {**result, 'f1': 2 * tp / max(1, 2 * tp + fp + fn), 'photos': len(ids),
            'gt_hand_denominator': tp + fn, 'predicted_hand_denominator': tp + fp,
            'per_photo': details}


def latency(audit, out, weights, truth, require):
    results = {}
    for name in ('cpu_latency.json', 'classifier_cpu_latency_unified.json'):
        path = out / name
        if not path.exists():
            audit.check(not require, 'latency_missing:' + name)
            results[name] = {'state': '尚未完成'}
            continue
        doc = read(path)
        vals = doc['all_ms']
        audit.check(all(math.isfinite(x) and x > 0 for x in vals), 'latency_positive:' + name)
        audit.same(doc['weights'], weights, 'latency_weights:' + name)
        for key, value in [('mean_ms', float(np.mean(vals))), ('p50_ms', float(np.percentile(vals, 50))),
                           ('p95_ms', float(np.percentile(vals, 95)))]:
            audit.same(value, doc[key], 'latency_statistics:' + name + ':' + key)
        audit.check(doc['threads'] == 8 and doc['batch_size'] == 1
                    and doc['model_loading_included'] is False, 'latency_protocol:' + name)
        if name == 'cpu_latency.json':
            audit.check(doc['n'] == len(vals) == 100 and doc['warmup'] == 10
                        and doc['image_ids'] == list(truth['test'])[:100], 'latency_photo_inputs')
        else:
            crops = read(out / 'crops_test.json')[:100]
            audit.check(doc['n_forward'] == len(vals) == 300 and doc['unique_inputs'] == 100
                        and doc['repeats'] == 3 and doc['warmup'] == 20
                        and doc['preprocessing_included'] is False
                        and doc['sample_keys'] == [f"{r['image_id']}__hand{r['hand_index']}" for r in crops],
                        'latency_classifier_inputs')
        results[name] = {k: v for k, v in doc.items() if k != 'all_ms'}
    return results


def review(arch, require_latency=False):
    out = folder(arch)
    # 完成门禁在读取模型与最终预测之前，未完成时不得制造审核PASS。
    required = ['stage_markers/evaluate.json', 'stage_markers/finetune.json', 'test_oracle_crop.json',
                'test_whole_photo.json', 'test_detector.json', 'val_whole_raw.json', 'calibration.json']
    missing = [name for name in required if not (out / name).is_file()]
    if require_latency:
        missing += [name for name in ('cpu_latency.json', 'classifier_cpu_latency_unified.json')
                    if not (out / name).is_file()]
    if missing:
        return {'architecture': arch, 'status': 'WAITING_FOR_COMPLETION', 'missing': missing}
    audit = Audit()
    truth, counts, manifests, overlap = dataset(audit)
    pre = read(RUN / 'preflight_independent_review.json')
    names = ['classifier_best.pt', 'detector_best.pt', 'calibration.json', 'val_whole_raw.json',
             'test_oracle_crop.json', 'test_whole_photo.json', 'test_detector.json', 'tpe.sqlite3',
             'tpe_summary.json', 'pretrained_identity.json']
    names += [f'crops_{s}.json' for s in ('train', 'val', 'test')]
    if require_latency:
        names += ['cpu_latency.json', 'classifier_cpu_latency_unified.json']
    start_hash = {name: digest(out / name) for name in names}
    for name, expected in pre['source_sha256'].items():
        audit.check(digest(ROOT / name) == expected, 'source_frozen:' + name)
    identity = read(out / 'pretrained_identity.json')
    audit.check(identity['protocol'] == 'B_subject_independent' and identity['architecture'] == arch
                and identity['pretraining'] == 'ImageNet1K_V1_only'
                and identity['A_finetuned_weights_used'] is False, 'initialization_protocol')
    audit.same(identity['manifest_sha256'], manifests, 'initialization_manifests')
    training = tpe_and_training(audit, out, arch, counts, identity)
    # 每个完成marker与输出文件和冻结源代码一致。
    stages = ['prepare', 'smoke', 'features', 'search', 'finetune', 'calibrate', 'evaluate']
    if arch == 'resnet18':
        stages.append('detector')
    if require_latency:
        stages += ['latency', 'classifier-latency']
    for stage in stages:
        marker = read(out / 'stage_markers' / (stage + '.json'))
        audit.check(marker['status'] == 'exit0' and marker['architecture'] == arch, 'stage_exit:' + stage)
        for name, expected in marker['source_sha256'].items():
            audit.check(digest(ROOT / name) == expected, 'marker_source:' + stage + ':' + name)
        for name, expected in marker['outputs'].items():
            audit.check(digest(out / name) == expected, 'marker_output:' + stage + ':' + name)
    for split in ('train', 'val', 'test'):
        crops = read(out / f'crops_{split}.json')
        if arch != 'resnet18':
            audit.check(digest(out / f'crops_{split}.json') == digest(RUN / f'crops_{split}.json'), 'shared_crop_hash:' + split)
        seen = set()
        for row in crops:
            key = (row['image_id'], row['hand_index'])
            audit.check(key not in seen, 'crop_unique:' + split)
            seen.add(key)
            ref = truth[split][row['image_id']]
            i = row['hand_index']
            audit.check(row['raw_label'] == ref['annotation']['labels'][i]
                        and row['label'] == ref['labels'][i]
                        and row['user_id'] == ref['user_id'], 'crop_truth:' + split)
        expected = {(rid, i) for rid, row in truth[split].items() for i in range(len(row['labels']))}
        audit.check(seen == expected, 'crop_coverage:' + split)
    calibration = read(out / 'calibration.json')
    weights = calibration['weights']
    for name, value in weights.items():
        audit.check(start_hash[name] == value, 'frozen_weight:' + name)
    audit.check(start_hash['detector_best.pt'] == digest(RUN / 'detector_best.pt'), 'shared_B_detector')
    if arch != 'resnet18':
        shared = read(out / 'shared_detector_identity.json')
        audit.check(shared['trained_only_on_B_train'] is True and shared['sha256'] == weights['detector_best.pt']
                    and Path(shared['source']).resolve() == (RUN / 'detector_best.pt').resolve(), 'shared_detector_provenance')
    audit.check(calibration['selection_split'] == 'val' and calibration['test_accessed'] is False, 'calibration_scope')
    val = read(out / 'val_whole_raw.json')
    audit.same(val['weights'], weights, 'val_weights')
    raw = raw_predictions(audit, val, truth, 'val', .3, .0)
    candidates = []
    for dt, ct in itertools.product((.3, .5, .7), (.3, .5, .7, .9)):
        adjusted = [{'truth': row['truth'], 'prediction': command([h['label'] for h in row['hands']
                     if h['detector_score'] >= dt and h['confidence'] >= ct])} for row in raw]
        candidates.append({'detector_threshold': dt, 'class_threshold': ct,
                           'metrics': metrics(adjusted), 'safety': safety(adjusted)})
    audit.same(candidates, calibration['candidates'], 'calibration_12_candidates')
    best = max(candidates, key=lambda c: (c['metrics']['macro_f1'], -c['safety']['stop_to_motion'],
               -c['safety']['zero_to_motion'], c['class_threshold'], c['detector_threshold']))
    audit.same(best, calibration['selected'], 'calibration_selected')
    dt, ct = best['detector_threshold'], best['class_threshold']
    oracle = read(out / 'test_oracle_crop.json')
    seen = set()
    for row in oracle['predictions']:
        key = (row['image_id'], row['hand_index'])
        audit.check(key not in seen, 'oracle_unique')
        seen.add(key)
        ref, index = truth['test'][row['image_id']], row['hand_index']
        p = row['probabilities']
        audit.check(row['truth'] == ref['labels'][index]
                    and row['raw_label'] == ref['annotation']['labels'][index], 'oracle_truth')
        audit.check(len(p) == 6 and all(math.isfinite(v) and 0 <= v <= 1 for v in p)
                    and abs(sum(p) - 1) < 1e-5 and row['prediction'] == int(np.argmax(p)), 'oracle_probability')
    expected = {(rid, i) for rid, row in truth['test'].items() for i in range(len(row['labels']))}
    audit.check(seen == expected, 'oracle_coverage')
    audit.check(oracle['checkpoint_sha256'] == weights['classifier_best.pt'], 'oracle_weight')
    audit.same(metrics(oracle['predictions']), oracle['metrics'], 'oracle_metrics')
    whole = read(out / 'test_whole_photo.json')
    audit.check(whole['detector_threshold'] == dt and whole['class_threshold'] == ct, 'whole_frozen_thresholds')
    audit.same(whole['weights'], weights, 'whole_weights')
    photos = raw_predictions(audit, whole, truth, 'test', dt, ct)
    audit.same(metrics(photos), whole['metrics'], 'whole_metrics')
    audit.same(safety(photos), whole['safety'], 'whole_safety')
    det = read(out / 'test_detector.json')
    audit.same(det['weights'], weights, 'detector_weights')
    detection = detector_metrics(audit, det, photos, truth, dt)
    history = events(RUN / 'controller_events.jsonl')
    starts = [x for x in history if x.get('architecture') == arch and x.get('stage') == 'evaluate' and x.get('status') == 'started']
    exits = [x for x in history if x.get('architecture') == arch and x.get('stage') == 'evaluate' and x.get('status') == 'exited']
    successes = [x for x in exits if x['exit_code'] == 0]
    audit.check(bool(starts) and len(successes) == 1
                and successes[0]['pid'] == starts[-1]['pid']
                and calibration['frozen_at'] < starts[0]['time'], 'evaluation_frozen_history')
    observations = []
    recovery_comparison = None
    if arch in ('resnet18', 'resnet50'):
        archive = RUN / ('recovery_20261102_2202' if arch == 'resnet18' else 'recovery_resnet50_20261102_2256')
        archived_path = archive / 'test_oracle_crop.json'
        archived = read(archived_path)
        audit.check(len(starts) == 2, arch + '_interruption_history')
        audit.check(len(exits) == (1 if arch == 'resnet18' else 2), arch + '_interruption_exit_count')
        if arch == 'resnet50':
            audit.check(exits[0]['exit_code'] == 1073807364 and exits[0]['pid'] == starts[0]['pid'],
                        'resnet50_system_restart_exit')
            evidence = read(archive / 'recovery_resnet50_system_events.json')
            audit.check(evidence['event_query_returncode'] == 0 and evidence['initiator_program_or_human_unknown'],
                        'resnet50_system_restart_evidence')
        audit.check(archived['checkpoint_sha256'] == weights['classifier_best.pt'], 'recovery_same_classifier')
        audit.same(metrics(archived['predictions']), archived['metrics'], 'recovery_archived_metrics')
        old = {(r['image_id'], r['hand_index']): r for r in archived['predictions']}
        current = {(r['image_id'], r['hand_index']): r for r in oracle['predictions']}
        audit.check(len(old) == len(archived['predictions']) == len(current) == 3759
                    and set(old) == set(current), 'recovery_same_complete_sample_keys')
        differences, changed_labels = [], []
        for key in sorted(set(old) & set(current)):
            a, b = old[key], current[key]
            audit.check((a['truth'], a['raw_label']) == (b['truth'], b['raw_label']), 'recovery_same_truth')
            delta = float(np.max(np.abs(np.asarray(a['probabilities']) - np.asarray(b['probabilities']))))
            audit.check(math.isfinite(delta), 'recovery_finite_probability_difference')
            if delta:
                differences.append({'image_id': key[0], 'hand_index': key[1], 'maximum_absolute_probability_difference': delta})
            if a['prediction'] != b['prediction']:
                changed_labels.append({'image_id': key[0], 'hand_index': key[1], 'attempt1': a['prediction'], 'attempt2': b['prediction']})
        recovery_comparison = {'archived_oracle_sha256': digest(archived_path),
                               'current_oracle_sha256': digest(out / 'test_oracle_crop.json'),
                               'sample_count': len(current), 'changed_prediction_count': len(changed_labels),
                               'changed_predictions': changed_labels, 'probability_changed_sample_count': len(differences),
                               'maximum_absolute_probability_difference': max((x['maximum_absolute_probability_difference'] for x in differences), default=0.0),
                               'probability_differences': differences,
                               'metrics_identical': archived['metrics'] == oracle['metrics'],
                               'interpretation': '保留两次测试访问及概率差异；当前结果来自恢复后完整attempt2，未按测试成绩择优，也不计作独立重复试验。'}
        observations.append('本模型evaluate曾中断并恢复，已比较两次完整人工框子产物；不能称test从未访问或逐位复现。')
    else:
        audit.check(len(starts) == 1, 'single_final_evaluation')
    speeds = latency(audit, out, weights, truth, require_latency)
    for name, value in start_hash.items():
        audit.check(digest(out / name) == value, 'unchanged_before_commit:' + name)
    return {'architecture': arch,
            'status': 'PASS_FROZEN_B_MODEL_AUDIT_WITH_LIMITATIONS' if not audit.issues else 'FAIL_REVIEW_REQUIRED',
            'reviewed_at': datetime.datetime.now().astimezone().isoformat(),
            'scope': '独立逐样本与训练账本复核；CPU只读张量，不推理、不访问CUDA、不改训练源码/数据/A',
            'issues': audit.issues, 'observations': observations,
            'check_counts': dict(audit.counts), 'total_checks': sum(audit.counts.values()),
            'identity_sha256': start_hash, 'manifest_sha256': manifests,
            'split_counts': counts, 'split_overlaps': overlap, 'training': training,
            'all_12_validation_candidates_recomputed': True, 'calibration_selected': best,
            'oracle_metrics': metrics(oracle['predictions']), 'oracle_safety_and_errors': errors_and_safety(oracle['predictions']),
            'whole_metrics': metrics(photos), 'whole_safety_and_errors': errors_and_safety(photos),
            'detector_metrics': detection, 'latency': speeds,
            'evaluation_history': {'starts': starts, 'exits': exits},
            'recovery_comparison': recovery_comparison,
            'reused_preflight_sha256': digest(RUN / 'preflight_independent_review.json'),
            'source_image_hashes_rerun': False,
            'limitations': ['沿用preflight完整31833源图及副本字节SHA；本轮重核manifest和记录hash交叉，未做近重复审计。',
                            'B为同HaGRID来源的user_id互斥留出，不代表真实机器人或独立设备外部泛化。',
                            '未做多seed、标签打乱与背景shortcut控制；高oracle性能不证明全部shortcut排除。',
                            '人工真框分类与整图端到端分母不同；检测指标为固定阈值precision/recall，不是mAP。',
                            '0无新指令与3明确停止不同；误运动为0仍须披露stop→0。',
                            'B18与B50曾测试中断恢复；概率差异逐模型保留，不能称test从未访问或重复。']}


def aggregate(reports, require_latency):
    # 原A只读既有完整预测与既有独审；A/B不是相同测试人员的配对试验。
    audit = Audit()
    rows = []
    aroot = ROOT / 'runs/A_20261102_TPE'
    for report in reports:
        arch = report['architecture']
        audit.check(report['status'].startswith('PASS_'), 'B_model_pass:' + arch)
        out = folder(arch, aroot)
        oracle, whole = read(out / 'test_oracle_crop.json'), read(out / 'test_whole_photo.json')
        audit.same(metrics(oracle['predictions']), oracle['metrics'], 'A_oracle_raw:' + arch)
        audit.same(metrics(whole['predictions']), whole['metrics'], 'A_whole_raw:' + arch)
        audit.same(safety(whole['predictions']), whole['safety'], 'A_safety_raw:' + arch)
        rows.append({'architecture': arch, 'A_oracle': oracle['metrics'], 'A_whole': whole['metrics'],
                     'A_whole_safety': errors_and_safety(whole['predictions']),
                     'B_oracle': report['oracle_metrics'], 'B_whole': report['whole_metrics'],
                     'B_whole_safety': report['whole_safety_and_errors'],
                     'B_minus_A_oracle_accuracy_percentage_points': 100 * (report['oracle_metrics']['accuracy'] - oracle['metrics']['accuracy']),
                     'B_minus_A_whole_accuracy_percentage_points': 100 * (report['whole_metrics']['accuracy'] - whole['metrics']['accuracy']),
                     'B_latency': report['latency']})
    audit.check(len({r['identity_sha256']['detector_best.pt'] for r in reports}) == 1, 'same_B_detector_weights')
    audit.check(all(r['manifest_sha256'] == reports[0]['manifest_sha256'] for r in reports), 'same_B_manifests')
    for key in ('cpu_latency.json', 'classifier_cpu_latency_unified.json'):
        speeds = [r['latency'][key] for r in reports]
        if all('mean_ms' in x for x in speeds):
            index = 'image_ids' if key == 'cpu_latency.json' else 'sample_keys'
            audit.check(all(x[index] == speeds[0][index] for x in speeds), 'same_latency_inputs:' + key)
    return {'status': 'PASS_AB_RESULT_AUDIT_WITH_LIMITATIONS' if not audit.issues else 'FAIL_REVIEW_REQUIRED',
            'reviewed_at': datetime.datetime.now().astimezone().isoformat(), 'issues': audit.issues,
            'B_all_three_training_and_test_completed': True, 'CPU_latency_required_for_this_audit': require_latency,
            'comparison': rows, 'A_independent_review_sha256': digest(aroot / 'final_independent_review.json'),
            'B_report_statuses': {r['architecture']: r['status'] for r in reports},
            'B_model_identity_sha256': {r['architecture']: r['identity_sha256'] for r in reports},
            'manifest_sha256': reports[0]['manifest_sha256'],
            'limitations': ['A既有训练/冻结/数据核验沿用既有独审，本轮另用A原始逐样本预测重算分类与安全指标。',
                            'A与B数据划分不同，分母不同，差值是描述性协议比较，不是同人员配对因果效应。',
                            '原A及其数据、权重、指标未修改；B从官方ImageNet独立训练并共享B检测器。',
                            'B18与B50中断恢复已实际重复访问test，数值差异已披露；未以测试成绩选择参数。',
                            '该审核不代表正式文档、程序交付包与最终ZIP已完成。']}


def commit(path, report, replace):
    # 先生成确切字节，再核冻结身份，最后提交同一批字节；不在核验前截断已有文件。
    payload = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    sources = read(RUN / 'preflight_independent_review.json')['source_sha256']
    for name, expected in sources.items():
        if digest(ROOT / name) != expected:
            raise RuntimeError('审核提交前源码身份改变：' + name)
    identities = report.get('B_model_identity_sha256', {})
    if 'architecture' in report:
        identities = {report['architecture']: report['identity_sha256']}
    for arch, files in identities.items():
        for name, expected in files.items():
            if digest(folder(arch) / name) != expected:
                raise RuntimeError('审核提交前产物身份改变：' + arch + '/' + name)
    for split, expected in report.get('manifest_sha256', {}).items():
        if digest(DATA / split / 'manifest.csv') != expected:
            raise RuntimeError('审核提交前分组身份改变：' + split)
    mode = 'wb' if replace else 'xb'
    if replace and path.exists():
        history = RUN / 'review_history'
        history.mkdir(exist_ok=True)
        previous = path.read_bytes()
        previous_digest = hashlib.sha256(previous).hexdigest()
        archived = history / (path.name + '.' + previous_digest + '.json')
        if not archived.exists():
            with archived.open('xb') as handle:
                handle.write(previous)
                handle.flush()
                os.fsync(handle.fileno())
        if digest(archived) != previous_digest:
            raise RuntimeError('旧审核归档身份不符：' + str(archived))
    with path.open(mode) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', choices=(*ARCHES, 'all'), default='all')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--require-latency', action='store_true')
    parser.add_argument('--replace-review', action='store_true')
    args = parser.parse_args()
    arches = ARCHES if args.arch == 'all' else (args.arch,)
    reports = []
    for arch in arches:
        report = review(arch, args.require_latency)
        reports.append(report)
        # 当前审者已获三模型新复审的独占写权；--replace-review显式覆盖旧复审。
        if not args.dry_run and report['status'] != 'WAITING_FOR_COMPLETION':
            commit(RUN / (arch + '_final_independent_review.json'), report, args.replace_review)
        print(json.dumps({'architecture': arch, 'status': report['status'],
                          'issue_count': len(report.get('issues', [])),
                          'first_issues': report.get('issues', [])[:10], 'missing': report.get('missing', []),
                          'checks': report.get('total_checks')}, ensure_ascii=False), flush=True)
    if args.arch == 'all' and all(r['status'] != 'WAITING_FOR_COMPLETION' for r in reports):
        final = aggregate(reports, args.require_latency)
        if not args.dry_run:
            commit(RUN / 'ab_final_independent_review.json', final, args.replace_review)
        print(json.dumps({'aggregate_status': final['status'], 'issues': final['issues']}, ensure_ascii=False))
    return 0 if all(r['status'].startswith('PASS_') for r in reports) else 2


if __name__ == '__main__':
    raise SystemExit(main())
