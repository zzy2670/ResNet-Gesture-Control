"""非作者CAM源码预审：只用标准库比公式和固定选样，不导入模型库。"""
from __future__ import annotations

import ast
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def assignments(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('weight', 'native', 'up', 'scaled', 'overlay'):
                result[node.targets[0].id] = ast.dump(node.value, include_attributes=False)
    return result


def main():
    source = RUN / 'make_merged_trained_gradcam.py'
    protocol = RUN / 'MERGED_GRADCAM_PROTOCOL.md'
    assert sha(source) == '134fa605c4ed59b6fe61b0179e5630a5718ed1a1b0607e43f2d8185c33f72f07'
    assert sha(protocol) == 'b64149af6fee7f374ff26b4c26c6939e6da7bd0efd2ed1a40116ffae8f6fbe1c'
    bound = {str(path): sha(path) for path in (source, protocol, ROOT / 'make_A_trained_gradcam.py',
        ROOT / 'make_B_trained_gradcam.py', ROOT / 'benchmark_label_variants_cpu.py', Path(__file__).resolve())}
    formula = assignments(source)
    assert set(formula) == {'weight', 'native', 'up', 'scaled', 'overlay'}
    assert formula == assignments(ROOT / 'make_A_trained_gradcam.py') == assignments(ROOT / 'make_B_trained_gradcam.py')
    spec = importlib.util.spec_from_file_location('independently_reviewed_merged_CAM_source', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not {'torch', 'numpy', 'matplotlib'} & set(sys.modules)
    selections = {}
    for protocol_name in ('A', 'B'):
        current = RUN / protocol_name / 'resnet18/crops_test.json'
        old_root = ROOT / f'runs/{protocol_name}_20261102_TPE'
        old_meta = old_root / 'trained_gradcam/metadata.json'
        old_index = old_root / 'crops_test.json'
        for path in (current, old_meta, old_index):
            bound[str(path)] = sha(path)
        selected = module.select(protocol_name)
        rows, prior = read(current), read(old_meta)
        lookup = {(r['image_id'], r['hand_index']): r for r in rows}
        assert len(selected) == 6 and {r['target'] for r in selected} == set(range(6))
        seen, output = set(), []
        for label, chosen in enumerate(selected):
            previous = next(r['source'] for r in prior['rows'] if r['truth'] == label)
            previous_key = previous['image_id'], previous['hand_index']
            expected = lookup[previous_key]
            if expected['label'] != label:
                expected = min((r for r in rows if r['label'] == label), key=lambda r: (r['image_id'], r['hand_index']))
            assert chosen['source'] == expected and chosen['source']['label'] == label
            key = expected['image_id'], expected['hand_index']
            assert key not in seen
            seen.add(key)
            assert label != 0 or expected['raw_label'] not in {'peace', 'peace_inverted', 'two_up', 'two_up_inverted'}
            output.append({'class': label, 'image_id': key[0], 'hand_index': key[1], 'raw_label': expected['raw_label'],
                'original_label': chosen['original_label'], 'previous_key': list(previous_key),
                'selection': chosen['selection'], 'same_original_identity': key == previous_key})
        selections[protocol_name] = output
    expired_rejected = False
    try:
        module.validate_permit({'purpose': 'merged_gradcam_cpu', 'confirmed_by': 'root',
            'cpu_window_confirmed': True, 'expires_at': '2000-01-01T00:00:00+08:00'})
    except ValueError:
        expired_rejected = True
    assert expired_rejected and not {'torch', 'numpy', 'matplotlib'} & set(sys.modules)
    report = {'status': 'PASS_GRADCAM_SOURCE_PREFLIGHT_WITH_LIMITATIONS',
        'reviewed_at': datetime.now().astimezone().isoformat(), 'reviewer': 'merged_independent_preflight',
        'scope': '非作者静态源码、AST公式与标准库固定选样/许可拒绝核验；没有加载模型、权重、图像或运行CAM前向反向。',
        'bound_file_sha256': bound, 'fixed_selections_independently_recomputed': selections,
        'checks': {'old_A_B_formula_AST_exact_equal_for_weight_native_upsample_minmax_overlay': True,
            'four_last_residual_blocks_true_logit_gradients': True,
            'ground_truth_target_fixed_wrong_predictions_kept': True,
            'independent_per_sample_per_layer_minmax_constant_map_zero': True,
            'old_CAM_image_bytes_never_relabelled_or_rewritten': True,
            'new_class0_excludes_four_forward_raw_poses': True,
            'sample_identity_prior_reused_or_first_legal_immutable_key_no_prediction_selection': True,
            'source_GT_crop_and_detector_crop_scope_explicit': True,
            'source_science_aggregate_six_reviews_full_SHA_gate_before_torch_import': True,
            'real_speed_complete_and_review_data_SHA_bound': True,
            'root_CPU_permit_identity_expiry_and_exact_sha_required': True,
            'expired_permit_rejected_in_memory_without_creating_authorization_file': expired_rejected,
            'strict_local_new_ResNet18_checkpoint_load_and_actual_CPU_required_at_execution': True,
            'output_only_new_protocol_resnet18_trained_gradcam_reject_existing': True,
            'NPZ_RGB_input_logits_activation_gradient_weight_native_up_normalized_all_layers': True,
            'frozen_oracle_probabilities_and_whole_prediction_attached_without_false_bitwise_claim': True,
            'NPZ_PNG_materialized_in_buffers_and_exclusively_fsynced': True,
            'metadata_payload_materialized_before_live_sources_and_exact7_output_SHA_recheck': True,
            'permit_bytes_reread_SHA_and_current_expiry_checked_last_before_same_metadata_commit': True,
            'completion_status_pending_independent_CAM_review': True},
        'resolved_findings': ['协议明确固定真实类logit，删除与argmax目标相矛盾的措辞',
            'metadata成功提交前精确核六NPZ及PNG集合与SHA', '全部大hash之后紧邻提交重核同一许可字节及实时expiry'],
        'issues': [], 'model_loaded': False, 'inference_or_backward_executed': False,
        'new_CAM_generated': False, 'scientific_CAM_result_pass': False,
        'limitations': ['本预审不能替代真实CPU加载/前向反向或48组四层CAM数值独审。',
            'A沿用旧六身份；B原0类身份因two_up_inverted变新1类而替换为固定首个合法0手，其余五行保留。',
            '主图为人工GT手框分类解释，不是检测器CAM或检测裁图解释，不建立物理因果。',
            '每map独立归一化，颜色深浅不能跨层/样本当绝对强度或性能。',
            '未来必须逐层从原gradient/activation复算、核各格图像与原图手部位置和概率差异；仅生成metadata不代表独审PASS。'],
        'execution_gate': '只有科学总独审、12CPU测速与真实独审、root当前有效许可均满足后，root才能安排实际--run-cpu。'}
    payload = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    for path, digest in bound.items():
        assert sha(path) == digest
    path = RUN / 'merged_gradcam_preflight_review.json'
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'report': str(path), 'sha256': sha(path),
        'model_loaded': False, 'selection_counts': {p: len(rows) for p, rows in selections.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
