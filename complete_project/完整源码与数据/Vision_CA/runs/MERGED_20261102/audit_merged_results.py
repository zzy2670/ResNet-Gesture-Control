"""只读复核合并模型账本与全部已保存预测，不训练、不推理、不调用CUDA。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, 'E:/python/Lib/site-packages')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import torch

torch.set_num_threads(1)
RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
ARCHES = ('resnet18', 'resnet34', 'resnet50')
STAGES = ('smoke', 'head', 'finetune', 'reuse-detector', 'calibrate', 'evaluate')
OUTPUTS = {'smoke': {'cpu_smoke.json'}, 'head': {'fixed_head_best.pt', 'fixed_head_summary.json'},
    'finetune': {'classifier_best.pt', 'finetune_summary.json'},
    'reuse-detector': {'detector_best.pt', 'shared_detector_identity.json'},
    'calibrate': {'calibration.json', 'val_whole_raw.json'},
    'evaluate': {'test_oracle_crop.json', 'test_whole_photo.json', 'test_detector.json'}}
OLD = {'peace': 1, 'peace_inverted': 1, 'stop': 2, 'stop_inverted': 2,
       'palm': 2, 'fist': 3, 'like': 4, 'dislike': 5}
NEW = {**OLD, 'two_up': 1, 'two_up_inverted': 1}
MOVING = {1, 2, 4, 5}
LEGACY_FILE = ROOT / 'runs/B_20261102_TPE/audit_B_final_models.py'
# 仅复用已独审的数学、逐框匹配及已保存概率核验，绝不调用其训练/数据入口。
_spec = importlib.util.spec_from_file_location('merged_readonly_math_helpers', LEGACY_FILE)
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)
read, sha, Audit = helpers.read, helpers.digest, helpers.Audit


def folder(protocol, arch):
    return RUN / protocol / arch


def original_folder(protocol, arch):
    base = ROOT / 'runs' / f'{protocol}_20261102_TPE'
    return base if arch == 'resnet18' else base / f'compare_{arch}'


def event_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.startswith('{')]


def model_events(protocol, arch):
    return [r for r in event_rows(RUN / 'controller_events.jsonl')
            if r.get('protocol') == protocol and r.get('architecture') == arch]


def bind(bound, path, expected=None):
    # 审核过程中只绑定本模型已完成的不可变文件，不绑定仍在追加的全局事件日志。
    path = Path(path)
    value = sha(path)
    if expected is not None and value != expected:
        raise ValueError(f'实际SHA与冻结记录不符：{path}')
    if str(path) in bound and bound[str(path)] != value:
        raise ValueError(f'同一审核读取期间身份变化：{path}')
    bound[str(path)] = value
    return value


def validate_probability(audit, row, label):
    p = row['probabilities']
    audit.check(len(p) == 6 and all(math.isfinite(v) and 0 <= v <= 1 for v in p)
                and abs(sum(p) - 1) < 1e-5 and row['prediction'] == int(np.argmax(p)), label)


def dataset(audit, out, protocol, bound):
    import csv
    split_root = ROOT / 'data' / f'{protocol}_split_verified_20261102'
    if protocol == 'A':
        split_root /= 'P1_random_image'
    data_identity = read(out / 'merged_data_identity.json')
    truth, counts = {}, {}
    for split in ('train', 'val', 'test'):
        path = split_root / split / 'manifest.csv'
        bind(bound, path, data_identity['splits'][split]['manifest_sha256'])
        with path.open(encoding='utf-8', newline='') as stream:
            photos = list(csv.DictReader(stream))
        annotation = {}
        for path in (split_root / split / 'annotations').glob('*.json'):
            bind(bound, path)
            annotation[path.stem] = read(path)
        truth[split], raw_counts = {}, Counter()
        for photo in photos:
            ann = annotation[photo['class_name']][photo['image_id']]
            audit.check(ann['user_id'] == photo['user_id'] and len(ann['labels']) == len(ann['bboxes']), 'annotation_identity:' + split)
            labels = [NEW.get(raw, 0) for raw in ann['labels']]
            raw_counts.update(labels)
            truth[split][photo['image_id']] = {**photo, 'annotation': ann, 'labels': labels,
                'truth': helpers.command(labels), 'old_truth': helpers.command([OLD.get(raw, 0) for raw in ann['labels']])}
        audit.check(len(truth[split]) == len(photos), 'unique_photos:' + split)
        crops = read(out / f'crops_{split}.json')
        expected_keys = {(rid, index) for rid, row in truth[split].items() for index in range(len(row['labels']))}
        seen = set()
        for row in crops:
            key = row['image_id'], row['hand_index']
            audit.check(key not in seen, 'crop_unique:' + split)
            seen.add(key)
            photo, index = truth[split][row['image_id']], row['hand_index']
            audit.check(row['raw_label'] == photo['annotation']['labels'][index]
                and row['label'] == photo['labels'][index] and row['user_id'] == photo['user_id'], 'crop_truth:' + split)
        audit.check(seen == expected_keys, 'crop_full_coverage:' + split)
        counts[split] = {'photos': len(photos), 'hands': len(crops), 'users': len({p['user_id'] for p in photos}),
                         'hand_support': [raw_counts[i] for i in range(6)],
                         'photo_support': [sum(p['truth'] == i for p in truth[split].values()) for i in range(6)]}
        audit.same(counts[split]['hand_support'], [data_identity['splits'][split]['new_six_class'][str(i)]['hands'] for i in range(6)], 'preflight_hand_support:' + split)
        for name in (f'crops_{split}.json', f'photo_truth_{split}.json'):
            bind(bound, out / name)
    helpers.DATA = split_root
    return truth, counts


def training(audit, out, protocol, arch, counts, identity, bound):
    head = read(out / 'fixed_head_summary.json')
    source_summary = original_folder(protocol, arch) / 'tpe_summary.json'
    source = read(source_summary)
    audit.same(head['best_params'], source['best_params'], 'own_old_fixed_parameters')
    audit.check(head['method'] == 'single_fixed_configuration_no_search' and head['test_accessed'] is False
        and head['epochs_completed'] == 18 and head['batch_size'] == 512
        and Path(head['source_tpe_summary']) == source_summary, 'fixed_head_protocol')
    logs = event_rows(out / 'stage_logs/head_attempt1.log')
    epochs = [row for row in logs if row.get('stage') == 'fixed_head']
    audit.same(epochs, [{'time': e['time'], 'stage': 'fixed_head', **h} for e, h in zip(epochs, head['history'])], 'head_log_and_history')
    audit.check([r['epoch'] for r in head['history']] == list(range(1, 19)) and len(epochs) == 18, 'head_exact18_complete')
    for row in head['history']:
        audit.check(row['samples'] == counts['train']['hands'] and row['validation']['n'] == counts['val']['hands']
                    and math.isfinite(row['loss']), 'head_full_train_val_denominators')
        audit.same(helpers.from_matrix(row['validation']['confusion_matrix']), row['validation'], 'head_logged_metrics')
    best_head = max(head['history'], key=lambda row: row['validation']['macro_f1'])
    audit.check(head['selected_epoch'] == best_head['epoch'], 'head_val_best_first_max')
    audit.same(head['validation'], best_head['validation'], 'head_selected_validation')
    selected = torch.load(out / 'fixed_head_best.pt', map_location='cpu', weights_only=True)
    audit.check(selected['architecture'] == arch and selected['epoch'] == head['selected_epoch'], 'selected_head_identity')
    audit.same(selected['params'], source['best_params'], 'selected_head_params')
    audit.same(selected['validation'], head['validation'], 'selected_head_val')
    dim = 2048 if arch == 'resnet50' else 512
    audit.check(tuple(selected['head']['1.weight'].shape) == (6, dim) and
                all(torch.isfinite(v).all().item() for v in selected['head'].values()), 'new_head_tensor_shape_finite')
    fine = read(out / 'finetune_summary.json')
    epochs = [r for r in event_rows(out / 'stage_logs/finetune_attempt1.log')
              if r.get('stage') == 'finetune' and r.get('status') == 'complete_epoch']
    audit.same(epochs, fine['history'], 'finetune_history_log')
    audit.check(fine['max_epochs'] == 18 and fine['patience'] == 4 and fine['epochs_completed'] == len(epochs)
                and 1 <= len(epochs) <= 18 and [r['epoch'] for r in epochs] == list(range(1, len(epochs) + 1)), 'finetune_budget_contiguous')
    score, bad, best_fine = head['validation']['macro_f1'], 0, None
    for index, row in enumerate(epochs):
        audit.check(row['samples'] == counts['train']['hands'] and row['validation']['n'] == counts['val']['hands']
                    and math.isfinite(row['loss']), 'finetune_full_denominators')
        audit.same(helpers.from_matrix(row['validation']['confusion_matrix']), row['validation'], 'finetune_logged_metrics')
        if row['validation']['macro_f1'] > score:
            score, bad, best_fine = row['validation']['macro_f1'], 0, row
        else:
            bad += 1
        audit.same(row['best'], score, 'finetune_running_val_best')
        audit.check(bad < 4 or index == len(epochs) - 1, 'no_epochs_after_patience4')
    audit.check(len(epochs) == 18 or bad == 4, 'natural_max18_or_patience4')
    logs = event_rows(out / 'stage_logs/finetune_attempt1.log')
    audit.check(not any(r.get('status') in ('partial_epoch_not_selected', 'deadline_before_epoch') for r in logs), 'no_partial_epochs_or_deadline')
    ck = torch.load(out / 'classifier_best.pt', map_location='cpu', weights_only=True)
    audit.same(ck['params'], source['best_params'], 'final_fixed_params')
    chosen_stage = 'merged_fixed_head' if best_fine is None else 'layer4_finetune'
    chosen_epoch = best_head['epoch'] if best_fine is None else best_fine['epoch']
    chosen_validation = best_head['validation'] if best_fine is None else best_fine['validation']
    audit.check(ck['stage'] == fine['selected_stage'] == chosen_stage and ck['epoch'] == fine['selected_epoch'] == chosen_epoch, 'final_head_or_finetune_selection')
    audit.same(ck['validation'], chosen_validation, 'final_selected_validation')
    audit.check(fine['weights_sha256'] == sha(out / 'classifier_best.pt'), 'fine_summary_weights')
    official = Path(read(out / 'merged_data_identity.json')['official_pretrained']['path'])
    original = torch.load(official, map_location='cpu', weights_only=True)
    inserted, frozen_count = [], 0
    for key, value in ck['model'].items():
        if key.startswith('fc.') or (best_fine is not None and key.startswith('layer4.')):
            continue
        if key not in original and key.endswith('.num_batches_tracked'):
            inserted.append(key)
            audit.check(value.item() == 0, 'official_inserted_zero_BN:' + key)
        else:
            audit.check(key in original and torch.equal(value, original[key]), 'frozen_official_tensor:' + key)
            frozen_count += 1
    audit.check(tuple(ck['model']['fc.1.weight'].shape) == (6, dim), 'final_head_shape')
    audit.check(all(torch.isfinite(v).all().item() for v in ck['model'].values()), 'final_all_tensor_finite')
    if best_fine is None:
        for key, value in selected['head'].items():
            audit.check(torch.equal(value, ck['model']['fc.' + key]), 'kept_head_exact_tensor:' + key)
        audit.check(head['initial_classifier_sha256'] == sha(out / 'classifier_best.pt'), 'kept_initial_classifier_exact_bytes')
    else:
        audit.check(ck.get('optimizer') == 'AdamW' and ck.get('fine_tune_lr') == 1e-4, 'finetune_fixed_AdamW_lr')
        audit.check(any(not torch.equal(value, original[key]) for key, value in ck['model'].items()
                        if key.startswith('layer4.') and key in original), 'selected_layer4_actually_changed')
    # preflight已全量逐元素核旧/新特征，这里核同SHA与新标签，不重复任何模型forward。
    cache_report = {}
    for split in ('train', 'val'):
        path = out / f'features_{split}.pt'
        cached = torch.load(path, map_location='cpu', weights_only=True)
        audit.check(cached['x'].shape == (counts[split]['hands'], dim)
                    and cached['pretrained_sha256'] == sha(official), 'cache_official_shape:' + split)
        audit.check(cached['y'].tolist() == [r['label'] for r in read(out / f'crops_{split}.json')], 'cache_new_label_order:' + split)
        cache_report[split] = {'sha256': sha(path), 'shape': list(cached['x'].shape)}
        del cached
    bind(bound, source_summary)
    bind(bound, official)
    return {'method': head['method'], 'new_TPE_trials': 0, 'fixed_params': source['best_params'],
        'head_epochs': 18, 'head_selected_epoch': head['selected_epoch'], 'head_validation': head['validation'],
        'head_initial_classifier_sha256': head['initial_classifier_sha256'], 'finetune_epochs': len(epochs),
        'finetune_patience': 4, 'selected_stage': chosen_stage, 'selected_epoch': chosen_epoch,
        'selected_validation': chosen_validation, 'official_sha256': sha(official),
        'frozen_official_tensors_equal': frozen_count, 'official_missing_BN_inserted_zero': inserted,
        'cache_identity': cache_report,
        'validation_evidence_limit': '按完整日志和混淆矩阵重算指标并核选型；未重新运行模型/线性头，未独立生成未保存的训练验证逐样本概率。'}


def rate(rows, predicate):
    n = len(rows)
    numerator = sum(predicate(r) for r in rows)
    return {'n': n, 'count': numerator, 'rate': numerator / n if n else None}


def comparison(audit, protocol, arch, truth, oracle, photos, bound):
    # 原输出与新输出按稳定照片/手索引配对；分母变化与共同负类单独保留。
    original = original_folder(protocol, arch)
    old_oracle_file, old_whole_file = original / 'test_oracle_crop.json', original / 'test_whole_photo.json'
    old_oracle, old_whole = read(old_oracle_file), read(old_whole_file)
    old_by_key = {(r['image_id'], r['hand_index']): r for r in old_oracle['predictions']}
    new_by_key = {(r['image_id'], r['hand_index']): r for r in oracle}
    audit.check(len(old_by_key) == len(old_oracle['predictions']) and set(old_by_key) == set(new_by_key), 'paired_old_new_hand_keys')
    for key, old in old_by_key.items():
        new = new_by_key[key]
        audit.check(old['raw_label'] == new['raw_label'] and old['truth'] == OLD.get(old['raw_label'], 0), 'paired_old_raw_truth')
        validate_probability(audit, old, 'old_saved_probability')
    audit.same(helpers.metrics(old_oracle['predictions']), old_oracle['metrics'], 'old_original_oracle_metrics')
    raw_table = []
    for raw in sorted({r['raw_label'] for r in oracle}):
        new_rows = [r for r in oracle if r['raw_label'] == raw]
        old_rows = [old_by_key[(r['image_id'], r['hand_index'])] for r in new_rows]
        raw_table.append({'raw_label': raw, 'old_target': OLD.get(raw, 0), 'merged_target': NEW.get(raw, 0),
            'old_output1': rate(old_rows, lambda r: r['prediction'] == 1),
            'merged_output1': rate(new_rows, lambda r: r['prediction'] == 1),
            'old_motion': rate(old_rows, lambda r: r['prediction'] in MOVING),
            'merged_motion': rate(new_rows, lambda r: r['prediction'] in MOVING)})
    common = [r for r in oracle if OLD.get(r['raw_label'], 0) == NEW.get(r['raw_label'], 0) == 0]
    common_old = [old_by_key[(r['image_id'], r['hand_index'])] for r in common]
    old_photo = {r['image_id']: r for r in old_whole['predictions']}
    new_photo = {r['image_id']: r for r in photos}
    audit.check(set(old_photo) == set(new_photo) and len(old_photo) == len(old_whole['predictions']), 'paired_photo_keys')
    for rid, row in old_photo.items():
        audit.check(row['truth'] == truth['test'][rid]['old_truth'], 'old_photo_truth')
    audit.same(helpers.metrics(old_whole['predictions']), old_whole['metrics'], 'old_original_whole_metrics')
    common_ids = [rid for rid, ref in truth['test'].items() if
        all(OLD.get(raw, 0) == NEW.get(raw, 0) == 0 for raw in ref['annotation']['labels'])]
    intersect_zero = [rid for rid, ref in truth['test'].items() if ref['old_truth'] == ref['truth'] == 0]
    photo_raw = []
    for raw in ('peace', 'peace_inverted', 'two_up', 'two_up_inverted'):
        keys = [rid for rid, ref in truth['test'].items() if ref['class_name'] == raw]
        photo_raw.append({'raw_photo_folder': raw, 'old_command1': rate([old_photo[k] for k in keys], lambda r: r['prediction'] == 1),
            'merged_command1': rate([new_photo[k] for k in keys], lambda r: r['prediction'] == 1)})
    bind(bound, old_oracle_file)
    bind(bound, old_whole_file)
    return {'raw_GT_hand_output_rates': raw_table, 'raw_photo_folder_output1': photo_raw,
        'common_other_raw_GT_hands': {'definition': 'old与new均映射0的原始手标签，明确排除two_up及two_up_inverted',
            'raw_labels': sorted({r['raw_label'] for r in common}), 'old_false_motion': rate(common_old, lambda r: r['prediction'] in MOVING),
            'merged_false_motion': rate(common, lambda r: r['prediction'] in MOVING)},
        'common_other_raw_photos': {'definition': '照片每一只标注手的raw标签在old/new均为0，明确排除新增正类',
            'old_false_motion': rate([old_photo[k] for k in common_ids], lambda r: r['prediction'] in MOVING),
            'merged_false_motion': rate([new_photo[k] for k in common_ids], lambda r: r['prediction'] in MOVING)},
        'common_photo_command0': {'definition': '同照片旧/新聚合真值都为0的交集；可含动作冲突，区别于纯other原始负类',
            'old_false_motion': rate([old_photo[k] for k in intersect_zero], lambda r: r['prediction'] in MOVING),
            'merged_false_motion': rate([new_photo[k] for k in intersect_zero], lambda r: r['prediction'] in MOVING)},
        'original_oracle_metrics': old_oracle['metrics'], 'original_whole_metrics': old_whole['metrics'],
        'interpretation': '同一固定协议/架构的描述性配对，class1定义和class0分母已改变；合并分数提高本身不能证明泛化提升。'}


def review(protocol, arch):
    out = folder(protocol, arch)
    required = [out / 'stage_markers' / f'{stage}.json' for stage in STAGES]
    missing = [str(p.relative_to(out)) for p in required if not p.is_file()]
    if missing:
        return {'status': 'WAITING_FOR_COMPLETION', 'protocol': protocol, 'architecture': arch, 'missing': missing}
    audit, bound = Audit(), {str(Path(__file__).resolve()): sha(__file__), str(LEGACY_FILE): sha(LEGACY_FILE)}
    preflight = read(RUN / 'preflight_independent_review.json')
    audit.check(preflight['status'] == 'PASS_PREFLIGHT', 'independent_preflight_pass')
    bind(bound, RUN / 'preflight_independent_review.json')
    data_review = read(RUN / 'preflight_data_independent_review.json')
    bind(bound, RUN / 'preflight_data_independent_review.json', preflight['data_review_sha256'])
    identity = read(out / 'merged_input_identity.json')
    audit.check(identity['protocol'] == protocol and identity['architecture'] == arch
                and identity['new_TPE_trials'] == 0 and identity['unmerged_classifier_weights_used'] is False, 'merged_protocol_identity')
    for name, digest in preflight['source_sha256'].items():
        bind(bound, ROOT / name, digest)
        bind(bound, RUN / 'code_snapshot' / name, digest)
    audit.same(identity['source_sha256'], preflight['source_sha256'], 'source_pin_all6')
    for path, digest in identity['inputs_sha256'].items():
        bind(bound, path, digest)
        if path in data_review['bound_file_sha256']:
            audit.check(data_review['bound_file_sha256'][path] == digest, 'preflight_data_same_bytes')
    for path in (out / 'merged_input_identity.json', out / 'merged_data_identity.json'):
        bind(bound, path)
    for stage in STAGES:
        file = out / 'stage_markers' / f'{stage}.json'
        marker = read(file)
        audit.check(marker['protocol'] == protocol and marker['architecture'] == arch and marker['stage'] == stage
                    and marker['exit_code'] == 0, 'stage_exit:' + stage)
        audit.check(set(marker['outputs']) == OUTPUTS[stage], 'stage_required_output_set:' + stage)
        audit.same(marker['identity'], identity, 'stage_identity:' + stage)
        bind(bound, file)
        bind(bound, marker['log'], marker['log_sha256'])
        for name, digest in marker['outputs'].items():
            bind(bound, out / name, digest)
    truth, counts = dataset(audit, out, protocol, bound)
    train_report = training(audit, out, protocol, arch, counts, identity, bound)
    own_detector = ROOT / 'runs' / f'{protocol}_20261102_TPE' / 'detector_best.pt'
    shared = read(out / 'shared_detector_identity.json')
    audit.check(shared['protocol'] == protocol and shared['retrained'] is False and shared['hand_background_labels_unchanged'] is True
        and Path(shared['source']) == own_detector and sha(out / 'detector_best.pt') == shared['sha256'] == sha(own_detector), 'reuse_own_protocol_detector')
    calibration = read(out / 'calibration.json')
    weights = calibration['weights']
    for name, digest in weights.items():
        bind(bound, out / name, digest)
    audit.check(calibration['selection_split'] == 'val' and calibration['test_accessed'] is False, 'val_only_freeze')
    val = read(out / 'val_whole_raw.json')
    audit.same(val['weights'], weights, 'val_frozen_weights')
    raw = helpers.raw_predictions(audit, val, truth, 'val', .3, .0)
    candidates = []
    for dt, ct in itertools.product((.3, .5, .7), (.3, .5, .7, .9)):
        adjusted = [{'truth': r['truth'], 'prediction': helpers.command([h['label'] for h in r['hands']
                    if h['detector_score'] >= dt and h['confidence'] >= ct])} for r in raw]
        candidates.append({'detector_threshold': dt, 'class_threshold': ct,
                           'metrics': helpers.metrics(adjusted), 'safety': helpers.safety(adjusted)})
    audit.same(candidates, calibration['candidates'], 'all12_validation_candidates')
    best = max(candidates, key=lambda c: (c['metrics']['macro_f1'], -c['safety']['stop_to_motion'],
        -c['safety']['zero_to_motion'], c['class_threshold'], c['detector_threshold']))
    audit.same(best, calibration['selected'], 'val_threshold_selection')
    dt, ct = best['detector_threshold'], best['class_threshold']
    oracle_doc = read(out / 'test_oracle_crop.json')
    oracle = oracle_doc['predictions']
    seen = set()
    for row in oracle:
        key = row['image_id'], row['hand_index']
        audit.check(key not in seen, 'oracle_unique')
        seen.add(key)
        photo = truth['test'][row['image_id']]
        audit.check(row['truth'] == photo['labels'][row['hand_index']] and row['raw_label'] == photo['annotation']['labels'][row['hand_index']], 'oracle_truth_raw')
        validate_probability(audit, row, 'oracle_probability')
    audit.check(seen == {(rid, i) for rid, row in truth['test'].items() for i in range(len(row['labels']))}, 'oracle_all_GT_hands')
    audit.check(oracle_doc['checkpoint_sha256'] == weights['classifier_best.pt'], 'oracle_frozen_weight')
    audit.same(helpers.metrics(oracle), oracle_doc['metrics'], 'oracle_all_metrics')
    whole_doc = read(out / 'test_whole_photo.json')
    audit.same(whole_doc['weights'], weights, 'whole_weights')
    audit.check(whole_doc['detector_threshold'] == dt and whole_doc['class_threshold'] == ct, 'whole_frozen_thresholds')
    photos = helpers.raw_predictions(audit, whole_doc, truth, 'test', dt, ct)
    audit.same(helpers.metrics(photos), whole_doc['metrics'], 'whole_all_metrics')
    audit.same(helpers.safety(photos), whole_doc['safety'], 'whole_real_safety_denominators')
    detector_doc = read(out / 'test_detector.json')
    audit.same(detector_doc['weights'], weights, 'detector_frozen_weights')
    detection = helpers.detector_metrics(audit, detector_doc, photos, truth, dt)
    history = model_events(protocol, arch)
    for stage in STAGES:
        starts = [r for r in history if r.get('stage') == stage and r.get('status') == 'started']
        exits = [r for r in history if r.get('stage') == stage and r.get('status') == 'exited']
        audit.check(len(starts) == len(exits) == 1 and exits[0]['exit_code'] == 0 and starts[0]['pid'] == exits[0]['pid'], 'single_success_attempt:' + stage)
        if stage == 'evaluate' and starts:
            audit.check(calibration['frozen_at'] < starts[0]['time'], 'threshold_frozen_before_test')
    audit.check(not any(r.get('stage') in ('search', 'detector', 'features') for r in history), 'no_search_detector_or_feature_training')
    paired = comparison(audit, protocol, arch, truth, oracle, photos, bound)
    return {'status': 'PASS_MERGED_MODEL_AUDIT_WITH_LIMITATIONS' if not audit.issues else 'FAIL_REVIEW_REQUIRED',
        'protocol': protocol, 'architecture': arch, 'reviewed_at': datetime.now().astimezone().isoformat(),
        'scope': 'CPU单线程只读账本/张量/全部保存概率，不推理、不访问CUDA、不写原A/B；不包含尚未安排的公平CPU测速或最终文档/打包验收',
        'issues': audit.issues, 'check_counts': dict(audit.counts), 'total_checks': sum(audit.counts.values()),
        'training': train_report, 'split_counts': counts, 'weights': weights, 'calibration_selected': best,
        'all12_val_candidates_recomputed': True, 'oracle_metrics': helpers.metrics(oracle),
        'oracle_safety_and_errors': helpers.errors_and_safety(oracle), 'whole_metrics': helpers.metrics(photos),
        'whole_safety_and_errors': helpers.errors_and_safety(photos), 'detector_metrics': detection,
        'paired_original_vs_merged': paired, 'model_controller_events': history,
        'bound_file_sha256': bound,
        'limitations': ['新旧任务class1定义不同、class0分母不同；描述性分数变化不能直接证明泛化改善。',
            'A按原图片划分，允许人员跨组；B沿用原人员互斥；两协议不构成相同测试人员配对比较。',
            '只沿用已完整审计的照片像素身份；本审不重新hash全部raw，不做近重复或背景shortcut控制。',
            '人工框手图与完整照片命令分母分别统计；0没有新指令和3明确停止不同，完整披露stop到0。',
            '训练验证分数按原完整日志/混淆矩阵核算；未重新推理验证训练模型，未补造其未保存逐样本概率。']}


def commit(path, report):
    # 物化准确报告字节后复核所有静态依赖及本模型事件，再独占写入同一字节。
    payload = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    for file, digest in report['bound_file_sha256'].items():
        if sha(file) != digest:
            raise RuntimeError(f'提交前依赖改变：{file}')
    if 'protocol' in report and model_events(report['protocol'], report['architecture']) != report['model_controller_events']:
        raise RuntimeError('提交前本模型事件变化')
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def aggregate():
    audit, bound, models = Audit(), {str(Path(__file__).resolve()): sha(__file__)}, []
    for protocol in ('A', 'B'):
        for arch in ARCHES:
            path = folder(protocol, arch) / 'model_review.json'
            if not path.exists():
                return {'status': 'WAITING_FOR_COMPLETION', 'missing': str(path)}
            report = read(path)
            audit.check(report['status'] == 'PASS_MERGED_MODEL_AUDIT_WITH_LIMITATIONS'
                        and report['protocol'] == protocol and report['architecture'] == arch, 'model_independent_pass')
            bind(bound, path)
            for file, digest in report['bound_file_sha256'].items():
                bind(bound, file, digest)
            models.append({k: report[k] for k in ('protocol', 'architecture', 'training', 'split_counts', 'weights',
                'calibration_selected', 'oracle_metrics', 'oracle_safety_and_errors', 'whole_metrics',
                'whole_safety_and_errors', 'paired_original_vs_merged')})
    for protocol in ('A', 'B'):
        same = [m for m in models if m['protocol'] == protocol]
        audit.check(len({m['weights']['detector_best.pt'] for m in same}) == 1, 'one_own_protocol_detector:' + protocol)
        audit.check(all(m['split_counts'] == same[0]['split_counts'] for m in same), 'same_protocol_denominators:' + protocol)
    complete_file = RUN / 'all_training_evaluation_complete.json'
    if not complete_file.exists():
        return {'status': 'WAITING_FOR_COMPLETION', 'missing': str(complete_file)}
    audit.check(read(complete_file)['status'] == 'PASS', 'all_training_evaluation_complete')
    bind(bound, complete_file)
    return {'status': 'PASS_MERGED_SIX_MODEL_AUDIT_WITH_LIMITATIONS' if not audit.issues else 'FAIL_REVIEW_REQUIRED',
        'reviewed_at': datetime.now().astimezone().isoformat(), 'issues': audit.issues, 'models': models,
        'bound_file_sha256': bound, 'CPU_benchmark_independent_review_pending': True,
        'formal_documents_programs_and_final_bundle_review_pending': True,
        'interpretation': '保留原A/B六模型并逐架构比较新旧标签任务；raw peace/two_up与共同负类已按同一测试身份派生。任务定义改变与单次实验分数不能当作已证明的泛化改善。'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=('A', 'B'))
    parser.add_argument('--arch', choices=ARCHES)
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.aggregate:
        report, path = aggregate(), RUN / 'aggregate_review.json'
    else:
        if not args.protocol or not args.arch:
            parser.error('逐模型审核需要--protocol与--arch；六模型汇总用--aggregate')
        report = review(args.protocol, args.arch)
        path = folder(args.protocol, args.arch) / 'model_review.json'
    if report['status'] != 'WAITING_FOR_COMPLETION' and not args.dry_run:
        commit(path, report)
    print(json.dumps({'status': report['status'], 'protocol': report.get('protocol'), 'architecture': report.get('architecture'),
        'issues': len(report.get('issues', [])), 'first_issues': report.get('issues', [])[:10],
        'missing': report.get('missing'), 'checks': report.get('total_checks'), 'output': str(path) if not args.dry_run else None}, ensure_ascii=False))
    return 0 if report['status'].startswith('PASS_') else 2


if __name__ == '__main__':
    raise SystemExit(main())
