"""修复后只读复审两交付工具；保留旧NEEDS_FIX审核，不构建任何成品。"""
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


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    sys.path.insert(0, str(ROOT))
    expected = {'build_merged_gesture_delivery.py': 'c159f13f359f699984c3591218512909a557090c6daf07d7ef2d2e9d29518fde',
                'build_label_comparison_bundle.py': 'ec8827f2a479f64172232d02058ad4f25e2ac108846c7d0d5e18d8177a8f3a90'}
    bound = {str(ROOT / name): digest for name, digest in expected.items()}
    for name, digest in expected.items():
        assert sha(ROOT / name) == digest
        ast.parse((ROOT / name).read_text(encoding='utf-8'))
    for path in (ROOT / 'build_ab_project_bundle.py', ROOT / 'deployment_src/gesture_predict.py', Path(__file__).resolve()):
        bound[str(path)] = sha(path)
    delivery = load('delivery_recheck', ROOT / 'build_merged_gesture_delivery.py')
    bundle = load('bundle_recheck', ROOT / 'build_label_comparison_bundle.py')
    spec = bundle.default_spec(bundle.DEFAULT_BUNDLE)
    assert bundle.DEFAULT_BUNDLE.name == 'ResNet_合并与不合并完整交接_20261103'
    identities = {(r['variant'], r['protocol'], r['architecture']) for r in spec['inference_packages']}
    assert len(spec['inference_packages']) == 12 and identities == {(v, p, a) for v in ('合并', '不合并') for p in 'AB' for a in ('resnet18', 'resnet34', 'resnet50')}
    assert len({r['archive'] for r in spec['inference_packages']}) == 12
    assert spec['project_readme_source'] == 'Vision_CA/runs/MERGED_20261102/packaged_project_README.md'
    assert spec['reviews'][0]['accepted'] == ['PASS_AB_RESULT_AUDIT_WITH_LIMITATIONS']
    assert spec['reviews'][1]['source'].endswith('/aggregate_review.json')
    assert spec['reviews'][1]['accepted'] == ['PASS_MERGED_SIX_MODEL_AUDIT_WITH_LIMITATIONS']
    assert bundle.FORMAL_NAMES == {'交接说明.docx', '基于ResNet18的手势识别项目方案_v12.docx', 'A与B协议ResNet训练与对比实验报告_v3.docx'}
    assert bundle.FORMAL_NAMES <= {Path(r['destination']).name for r in spec['documents']}
    # 仅进程内替换read以验证泛用PASS不会变成真实科学PASS；不写任何兼容文件。
    original_read = delivery.read
    delivery.read = lambda _: {'status': 'PASS', 'issues': [], 'bound_file_sha256': {'unused': 'unused'}}
    generic_pass_rejected = False
    try:
        delivery.verify_bound_review(Path('unused'), delivery.AGGREGATE_REVIEW_STATUS)
    except ValueError:
        generic_pass_rejected = True
    finally:
        delivery.read = original_read
    assert generic_pass_rejected
    incomplete_rejected = False
    try:
        delivery.ready(RUN / 'aggregate_review.json')
    except FileNotFoundError as error:
        incomplete_rejected = True
        missing = str(error.filename)
    assert incomplete_rejected
    # 只测试archive回调路由，无调用真实归档或任何来源树遍历。
    old_archive, old_gate = bundle.transport.archive, bundle.transport.check_ready
    route = []
    bundle.transport.archive = lambda *_: route.append(bundle.transport.check_ready is bundle.check_ready)
    try:
        bundle.archive(Path('unused'), Path('unused.zip'))
    finally:
        bundle.transport.archive = old_archive
    assert route == [True] and bundle.transport.check_ready is old_gate
    prior_path = RUN / 'merged_delivery_tools_preflight_review.json'
    prior_bytes = prior_path.read_bytes()
    prior_digest = hashlib.sha256(prior_bytes).hexdigest()
    prior = json.loads(prior_bytes)
    assert prior['status'] == 'NEEDS_FIX_BEFORE_BUILD' and len(prior['issues']) == 5
    report = {'status': 'PASS_DELIVERY_TOOLS_SOURCE_PREFLIGHT_WITH_LIMITATIONS',
        'reviewed_at': datetime.now().astimezone().isoformat(), 'reviewer': 'merged_independent_preflight',
        'scope': '两新交付工具源码与只读结构预审；未复制、build、freeze、archive、加载权重或模型推理，未运行大量科学产物散列',
        'bound_file_sha256': bound, 'superseded_review_sha256': prior_digest,
        'resolved_findings': [{**item, 'status': 'RESOLVED_IN_REVIEWED_SOURCE'} for item in prior['issues']],
        'issues': [], 'structure_checks': {'AST_valid': True,
            'real_aggregate_and_model_WITH_LIMITATIONS_statuses': True,
            'generic_PASS_rejected_in_memory_without_compatibility_file': generic_pass_rejected,
            'actual_incomplete_completion_rejected_before_build': incomplete_rejected,
            'incomplete_missing_file': missing,
            'all_aggregate_bound_files_live_SHA_and_subordinate_review_inclusion_required': True,
            'old_AB_real_schema_weight_and_available_artifact_bindings': True,
            'current_program_directory_set_manifest_SHA_and_ZIP_members_CRC_SHA_three_way_check': True,
            'twelve_unique_program_identities_and_twelve_ZIP_paths': True,
            'new_class1_four_raw_and_validation_truth_recalculation': True,
            'examples_fixed_by_validation_ID_without_prediction_filter': True,
            'new_classifier_weight_and_own_frozen_val_thresholds': True,
            'original_six_program_directories_not_written': True,
            'formal_v12_v3_new_handover_gate_no_old_doc_gate_fallback': True,
            'archive_routes_new_gate_and_restores_transport_in_finally': True,
            'default_bundle_name_and_packaged_source_README_path': True,
            'raw_A_fixed_B_fixed_each_once_and_all_merged_scientific_outputs': True,
            'crops_features_bytecode_personal_notes_media_excluded_scientific_figures_kept': True,
            'twelve_ZIP_unique_current_SHA_actual_interface_and_CRC_review_required': True,
            'streaming_archive_reads_frozen_SHA_and_verifies_all_bytes_before_final_rename': True},
        'execution_gate': '全部六模型与真实aggregate独审完成后才可构建新六程序；十二程序实际离线接口/ZIP独审、三Word终稿独审及实际全成员freeze/CRC/SHA核验后方可发布总包。',
        'product_built': False, 'model_loaded': False,
        'limitations': ['这是源码预审；未对尚未生成的六合并程序、十二ZIP或正式三Word给成品PASS。',
            '不运行ready全量产物SHA/verify_package_archive，以免与训练并行争用大量I/O；真实build/freeze/archive必须实际执行门禁。',
            '原A34/50下级旧review没有artifact_sha256字段，其对应空循环不构成完整旧val/test字节SHA证明；当前逻辑核原权重、状态、已有速度SHA，最终合并aggregate还锚定旧oracle/whole配对输出。',
            '历史裁图路径保留为证据、缓存排除；跨机训练需包内原图和固定清单重建新缓存/身份，历史controller不是迁移入口。']}
    history = RUN / 'review_history'
    history.mkdir(exist_ok=True)
    old = history / f'merged_delivery_tools_preflight_review.{prior_digest}.json'
    if not old.exists():
        with old.open('xb') as stream:
            stream.write(prior_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    assert sha(old) == prior_digest
    payload = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    for path, digest in bound.items():
        assert sha(path) == digest
    assert sha(prior_path) == prior_digest
    temporary = RUN / f'merged_delivery_tools_preflight_review.{os.getpid()}.tmp'
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, prior_path)
    print(json.dumps({'status': report['status'], 'resolved_issues': len(report['resolved_findings']), 'sha256': sha(prior_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
