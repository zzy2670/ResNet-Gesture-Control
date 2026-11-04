"""最终验收器轻量准备自检，不读取实际最终ZIP或做大散列。"""
import ast
from datetime import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import zipfile

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
SOURCE = ROOT.parent / 'audit_final_label_bundle.py'


def main():
    ast.parse(SOURCE.read_text(encoding='utf-8'))
    spec = importlib.util.spec_from_file_location('final_label_auditor_source_check', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.local_target('README.md', '../escape.md')[0] == 'escaping_relative'
    assert module.local_target('old/report.md', 'C:/原图/图.png')[0] == 'absolute_local'
    assert module.local_target('old/report.md', 'file:///C:/原图/图.png')[0] == 'absolute_local'
    assert module.local_target('子目录/说明.md', '../README.md') == ('relative_local', 'README.md')
    assert module.local_target('README.md', 'https://example.com/a')[0] == 'external_not_fetched'
    assert module.markdown_links('[图](<图 1.png>) [图2](图(2).png)\n```\n[忽略](C:/private.png)\n```') == ['<图 1.png>', '图(2).png']
    # 小型内存OOXML关系仅验证解析规则；这不是生成或消费实际正式Word。
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as word:
        word.writestr('_rels/.rels', '<Relationships><Relationship Target="word/document.xml"/></Relationships>')
        word.writestr('word/document.xml', '<document/>')
        word.writestr('word/styles.xml', '<styles/>')
        word.writestr('word/_rels/document.xml.rels', '<Relationships><Relationship Target="/word/styles.xml"/><Relationship Target="C:/原机/参考.docx" TargetMode="External"/></Relationships>')
    external, issues = module.docx_links(buffer.getvalue())
    assert not issues and external == [{'relationship_part': 'word/_rels/document.xml.rels', 'target': 'C:/原机/参考.docx'}]
    assert module.local_target('交接说明.docx', external[0]['target'])[0] == 'absolute_local'
    # 分别核根内成员缺失和试图越出包根，保持外部绝对路径规则不变。
    negative_cases = []
    for target, expected_kind in (('/word/missing.xml', 'OPC_internal'), ('../../outside.xml', 'escaping_OPC_internal')):
        bad = io.BytesIO()
        with zipfile.ZipFile(bad, 'w') as word:
            word.writestr('word/document.xml', '<document/>')
            word.writestr('word/_rels/document.xml.rels', f'<Relationships><Relationship Target="{target}"/></Relationships>')
        _, issues = module.docx_links(bad.getvalue())
        assert len(issues) == 1 and issues[0]['kind'] == expected_kind
        negative_cases.append({'target': target, 'issue_kind': issues[0]['kind']})
    rejected = False
    try:
        module.safe('root/../escape')
    except ValueError:
        rejected = True
    assert rejected
    paths = [SOURCE, RUN / 'FINAL_LABEL_BUNDLE_AUDIT_PROTOCOL.md', ROOT.parent / 'audit_final_ab_bundle.py', Path(__file__).resolve()]
    bound = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report = {'status': 'PREPARATION_CHECKS_COMPLETE_NOT_FINAL_ARCHIVE_AUDIT',
        'recorded_at': datetime.now().astimezone().isoformat(), 'author': 'merged_independent_preflight',
        'source_sha256': bound, 'AST_pass': True, 'default_and_help_executed': True,
        'small_in_memory_MD_and_OOXML_link_structure_checks': True,
        'OPC_package_rooted_existing_target_passes': True,
        'OPC_missing_and_escape_targets_reported': negative_cases,
        'external_C_drive_remains_absolute_host_link': True,
        'relative_escape_rejected': True, 'historical_absolute_paths_classified_without_mutation': True,
        'actual_final_archive_read': False, 'large_hash_executed': False, 'model_loaded_or_inference': False,
        'final_result_not_created': not (RUN / 'final_bundle_independent_review.json').exists(),
        'required_future_schema': {'CPU_review': 'status PASS; data_sha256 actual12CPU result',
            'CAM_review': 'status PASS; nonempty bound_file_sha256 anchors both new protocols 6NPZ+PNG+metadata',
            'program_review': 'status PASS; 12 unique archives with actual offline interface and member CRC/SHA plus actual ZIP sha256',
            'formal_docs': 'status PASS; documents source/status/sha256 exact new3Word'},
        'root_independent_source_review_still_required': True}
    payload = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    for path, digest in bound.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    path = RUN / 'final_bundle_auditor_preparation.json'
    if path.exists():
        history = RUN / 'review_history'
        history.mkdir(exist_ok=True)
        old = path.read_bytes()
        old_hash = hashlib.sha256(old).hexdigest()
        archived = history / ('final_bundle_auditor_preparation.' + old_hash + '.json')
        if not archived.exists():
            with archived.open('xb') as stream:
                stream.write(old)
                stream.flush()
                os.fsync(stream.fileno())
        assert hashlib.sha256(archived.read_bytes()).hexdigest() == old_hash
        destination = path.with_name('final_bundle_auditor_preparation.' + str(os.getpid()) + '.tmp')
    else:
        destination = path
    with destination.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if destination != path:
        os.replace(destination, path)
    print(json.dumps({'status': report['status'], 'auditor_sha256': bound[str(SOURCE)], 'report': str(path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
