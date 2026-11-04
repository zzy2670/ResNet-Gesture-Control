"""对新适配做无模型schema和路径拒绝自检，交root独立复审。"""
import ast
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    adapter = RUN / 'audit_merged_gesture_delivery.py'
    ast.parse(adapter.read_text(encoding='utf-8'))
    spec = importlib.util.spec_from_file_location('adapter_source_check', adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    schema = []
    for protocol, directory in (('A', ROOT.parent / 'ResNet手势识别交付'), ('B', ROOT.parent / 'ResNet_B手势识别交付')):
        prior, bindings = module.legacy_identity_bindings(protocol, directory, ROOT / 'runs/B_20261102_TPE/six_program_readiness.json')
        schema.append({'protocol': protocol, 'models': len(prior['models']), 'expected_bound_files': len(bindings),
                       'top_level_identity_present': 'identity_sha256' in prior})
    outside_rejected = False
    try:
        module.plain_contained(ROOT / 'train_A.py', RUN / 'merged_deployment_validation')
    except ValueError:
        outside_rejected = True
    assert outside_rejected
    files = [adapter, RUN / 'MERGED_DELIVERY_VALIDATION_PROTOCOL.md', ROOT / 'audit_B_gesture_delivery.py',
             ROOT / 'build_merged_gesture_delivery.py', RUN / 'delivery_storage_budget.json', Path(__file__).resolve()]
    bound = {str(path): sha(path) for path in files}
    report = {'status': 'AUTHOR_CHECKS_COMPLETE_AWAITING_ROOT_INDEPENDENT_SOURCE_REVIEW',
        'recorded_at': datetime.now().astimezone().isoformat(), 'scope': '源码准备与实际小文件schema/路径拒绝检查；不是推理包成品或清理执行PASS。',
        'bound_file_sha256': bound, 'AST_valid': True, 'actual_legacy_schema_checks': schema,
        'schema_fix': 'A使用six_program_readiness锚定原review/manifest/离线ZIP及每文件SHA；B保留非空顶层identity_sha256并与同readiness一致。',
        'outside_cleanup_boundary_rejected_without_IO_mutation': outside_rejected,
        'cleanup_design': {'only_new_mkdtemp_two_extracted_weight_files': True,
            'no_recursive_delete': True, 'plain_file_link_count1_and_no_symlink_or_junction_required': True,
            'absolute_resolved_containment_and_each_component_checked': True,
            'all_offline_assertions_success_then_source_program_ZIP_reverification': True,
            'target_manifest_size_SHA_checked_and_rechecked_immediately_before_unlink': True,
            'plan_and_receipt_and_CLI_raw_stdout_stderr_retained': True,
            'interface_failure_preserves_all_extracted_files': True,
            'partial_cleanup_failure_retains_remaining_and_exact_removed_receipt': True},
        'models_loaded': False, 'inference_executed': False, 'ZIP_built_or_extracted': False,
        'cleanup_executed': False, 'old_tools_or_packages_modified': False,
        'root_archive_plan': 'E:/ResNet_合并与不合并完整交接_20261103.zip',
        'remaining_gate': 'root按这些实际SHA独立复审新适配；正式CPU窗口实际运行后核72例/离线结果/每份清理回执与原完整包身份。'}
    payload = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    for file, digest in bound.items():
        assert sha(file) == digest
    output = RUN / 'merged_delivery_adapter_preflight_review.json'
    with output.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'adapter_sha256': bound[str(adapter)], 'review': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
