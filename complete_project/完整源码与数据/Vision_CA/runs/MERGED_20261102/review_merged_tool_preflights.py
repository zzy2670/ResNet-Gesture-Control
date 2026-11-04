"""只读预审构建/测速工具，缺模型不构建、不加载权重、不计时。"""
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
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def commit(name, report):
    # 编码报告之后，复核当前实际预审字节，再独占提交。
    payload = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    for path, digest in report['bound_file_sha256'].items():
        if sha(path) != digest:
            raise ValueError('预审提交前源码变化：' + path)
    with (RUN / name).open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, 'E:/python/Lib/site-packages')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    cpu_path = ROOT / 'benchmark_label_variants_cpu.py'
    protocol = RUN / 'LABEL_VARIANTS_CPU_PROTOCOL.md'
    reference = ROOT / 'runs/B_20261102_TPE/matched_cpu_benchmark.json'
    for path in (cpu_path, ROOT / 'build_merged_gesture_delivery.py', ROOT / 'build_label_comparison_bundle.py'):
        ast.parse(path.read_text(encoding='utf-8'))
    cpu = module('new_cpu_protocol_preflight', cpu_path)
    assert 'torch' not in sys.modules
    try:
        cpu.scientific_readiness()
    except FileNotFoundError as error:
        gate = {'status': 'BLOCKED_AS_REQUIRED', 'missing': str(error.filename),
                'no_torch_import_before_readiness': 'torch' not in sys.modules}
    else:
        raise RuntimeError('本预审预期新六模型尚未全部完成，完成状态发生变化，请按新状态审核')
    assert sha(reference) == cpu.REFERENCE_SHA256
    prior = json.loads(reference.read_text(encoding='utf-8'))
    assert prior['status'] == 'complete' and len(prior['photos']) == len(prior['classifier_inputs']) == 100
    expected = [f'{v}_{p}_{a}' for a in ('resnet18', 'resnet34', 'resnet50') for p in ('A', 'B') for v in ('unmerged', 'merged')]
    assert cpu.MODEL_KEYS == expected and cpu.ORDERS == [expected, list(reversed(expected)), expected[6:] + expected[:6]]
    # 实际导入依赖只证明环境，不构建任何模型，也不读权重。
    import psutil
    import torch
    import torchvision
    assert psutil.__version__ == '7.0.0'
    bindings = {str(p): sha(p) for p in (cpu_path, protocol, reference, Path(__file__).resolve())}
    deps = {'psutil': psutil.__version__, 'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'actual_import': True, 'models_constructed': False, 'weights_loaded': False, 'inference_executed': False}
    report = {'status': 'PASS_CPU_BENCHMARK_SOURCE_PREFLIGHT_WITH_LIMITATIONS',
        'reviewed_at': datetime.now().astimezone().isoformat(), 'reviewer': 'merged_independent_preflight',
        'scope': '仅源码/协议/固定输入身份与依赖实际import预审；未运行测速，不是7200次结果或成品验收',
        'bound_file_sha256': bindings, 'dependency_actual_import': deps, 'incomplete_scientific_gate_test': gate,
        'checks': {'AST_valid': True, 'default_help_only_no_models': True,
            'before_any_model_load_complete_aggregate_and_six_real_reviews_required': True,
            'real_aggregate_schema_and_empty_issues_six_unique_identities': True,
            'all_review_bound_files_live_SHA_before_model_load_and_precommit': True,
            'review_calibration_weights_and_selected_thresholds_consistent': True,
            'same100_photo_and100_crop_keys_order_pixels_tensor_SHA_as_prior_reference': True,
            'two_label_variants_recalculate_truth_metadata_no_accuracy_selection': True,
            '12_models_strict_architecture_and_fulltail_cpu_load': True,
            'same_protocol_detector_original_weight_identity': True,
            'batch1_float32_8threads20warm3x100_all12models': True,
            '72_blocks7200_calls_all_raw_times_and_workloads_retained': True,
            'fixed_interleaved_reverse_rotate6_orders': True,
            'original_whole_photo_function_and_preprocessed_classifier_scope': True,
            'background_raw_counters_RSS_available_memory_recorded': True,
            'minimum6GiB_available_before_all12_model_load': True,
            'exclusive_attempt_no_silent_retry_or_best_round_selection': True,
            'materialize_bytes_then_full_live_precommit_then_xb_fsync_same_bytes': True},
        'execution_gate': '仅在全部新模型训练与真实科学独审完成且root给独占CPU窗口后显式--run；当前门禁正确拒绝未完成状态。',
        'result_exists_at_review': cpu.OUTPUT.exists(), 'attempt_exists_at_review': cpu.OUTPUT.with_suffix('.attempt.jsonl').exists(),
        'issues': [], 'resolved_findings': ['补充任何模型加载之前的真实completion+aggregate+六model_review身份门禁及提交前全SHA核验'],
        'limitations': ['A-val固定输入用于速度比较，不能用作十二模型泛化准确率。',
            '模型各自冻结阈值带来检测框/接受手数量差异，不能把整图时延差直接归因于深度或标签语义。',
            '三轮固定顺序只能部分控制漂移，背景负载为原始计数的估计，不能据此剔除慢轮。',
            '资源约3GiB是协议预估，不是已测RSS；模型实际加载与推理必须留待正式窗口。',
            '执行后还须独审72块、7200原始ms、模型/输入身份及所有汇总；本预审不替代结果独审。']}
    assert report['result_exists_at_review'] is False and report['attempt_exists_at_review'] is False
    commit('label_variants_cpu_preflight_review.json', report)
    package_path = ROOT / 'build_merged_gesture_delivery.py'
    bundle_path = ROOT / 'build_label_comparison_bundle.py'
    transport_path = ROOT / 'build_ab_project_bundle.py'
    bundle = module('label_bundle_structure_preflight', bundle_path)
    spec = bundle.default_spec(bundle.DEFAULT_BUNDLE)
    expected_packages = {(v, p, a) for v in ('不合并', '合并') for p in ('A', 'B') for a in ('resnet18', 'resnet34', 'resnet50')}
    assert len(spec['inference_packages']) == 12 and {(r['variant'], r['protocol'], r['architecture']) for r in spec['inference_packages']} == expected_packages
    assert {'交接说明.docx', '基于ResNet18的手势识别项目方案_v12.docx', 'A与B协议ResNet训练与对比实验报告_v3.docx'} <= {Path(r['destination']).name for r in spec['documents']}
    bindings = {str(p): sha(p) for p in (package_path, bundle_path, transport_path, ROOT / 'deployment_src/gesture_predict.py', Path(__file__).resolve())}
    findings = [
        {'severity': 'blocking', 'file': package_path.name, 'location': 'ready and --review default',
         'finding': '仍消费MERGED/final_independent_review.json且status=PASS；真实权威为aggregate_review.json与PASS_MERGED_SIX_MODEL_AUDIT_WITH_LIMITATIONS。'},
        {'severity': 'blocking', 'file': bundle_path.name, 'location': 'default_spec reviews',
         'finding': '继承的旧B/final_independent_review.json与新增MERGED同名兼容PASS均不是真实科学权威schema。旧B应消费ab_final_independent_review，合并消费aggregate_review。'},
        {'severity': 'blocking', 'file': bundle_path.name, 'location': 'checked_reviews',
         'finding': '只看review status，没有live复核科学review绑定的实际输出/权重/校准/源SHA。'},
        {'severity': 'blocking', 'file': bundle_path.name, 'location': 'check_ready package gate',
         'finding': '十二ZIP虽核当前SHA，但并排十二目录也必须与各ZIP的SHA256SUMS清单逐文件一致，避免目录审核后变化而ZIP不变。'},
        {'severity': 'required', 'file': bundle_path.name, 'location': 'DEFAULT_BUNDLE',
         'finding': '默认根名仍ResNet_标签方案对照完整交接_20261103，需改为约定ResNet_合并与不合并完整交接_20261103。'}]
    delivery = {'status': 'NEEDS_FIX_BEFORE_BUILD', 'reviewed_at': datetime.now().astimezone().isoformat(),
        'reviewer': 'merged_independent_preflight', 'scope': '只读源码、help/list与不落盘spec结构核验；没有复制、构建、压缩、加载权重或推理',
        'bound_file_sha256': bindings, 'issues': findings,
        'checks_already_supported': {'new_class1_four_raw_labels': True, 'validation_truth_recomputed_from_raw': True,
            'validation_examples_fixed_by_ID_no_prediction_filter': True, 'six_new_packages_and_twelve_total_identities': True,
            'original_six_model_directories_not_written': True, 'merged_weight_must_differ_from_old_classifier': True,
            'new_calibration_weight_hash_and_thresholds_used': True,
            'new_doc_gate_requires_v12_v3_and_new_handover': True,
            'archive_temporarily_replaces_old_transport_doc_gate': True,
            'source_README_redirected_to_merged_packaged_README': True,
            'raw_A_fixed_B_fixed_each_once_and_all_merged_scientific_outputs': True,
            'crops_features_bytecode_personal_notes_media_excluded_scientific_figures_kept': True,
            'twelve_unique_ZIP_SHA_CRC_interface_review_required': True,
            'archive_streams_frozen_SHA_and_verifies_all_members_before_rename': True},
        'tools_executed': ['build_merged_gesture_delivery.py list', 'build_label_comparison_bundle.py --help', 'default_spec structure only'],
        'product_built': False, 'model_loaded': False, 'next_step': 'writer修复上述门禁后按新SHA独立复审；不能将本预审当十二程序或最终ZIP PASS。'}
    commit('merged_delivery_tools_preflight_review.json', delivery)
    print(json.dumps({'cpu': report['status'], 'delivery': delivery['status'], 'delivery_findings': len(findings)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
