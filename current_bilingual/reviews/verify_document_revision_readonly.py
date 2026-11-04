"""Independent read-only checks for the two current-version pointer fixes.

Writes only this reviewer's own QA result; never edits authored Word/source files.
"""
from pathlib import Path
import hashlib
import json
from zipfile import ZipFile
from lxml import etree as E

S = Path(__file__).resolve().parents[1]
R = S / 'reviews'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    baseline = json.loads((R / 'complete_document_review_baseline_plan_parts.json').read_text(encoding='utf-8'))
    baseline_render = json.loads((R / 'complete_document_review_baseline_manifest.json').read_text(encoding='utf-8'))
    current_render = json.loads((S / 'docx_qa/current_render_manifest.json').read_text(encoding='utf-8-sig'))
    plan = next((S / 'documents').glob('*v13*.docx'))
    old = E.fromstring((R / 'complete_document_review_baseline_plan.xml').read_bytes())
    with ZipFile(plan) as z:
        current_parts = {n: sha(z.read(n)) for n in z.namelist()}
        new = E.fromstring(z.read('word/document.xml'))
    assert set(current_parts) == set(baseline)
    changed_parts = [n for n in baseline if baseline[n] != current_parts[n]]
    assert changed_parts == ['word/document.xml'], changed_parts
    old_t = old.findall('.//w:t', NS)
    new_t = new.findall('.//w:t', NS)
    assert len(old_t) == len(new_t)
    replacements = []
    for a, b in zip(old_t, new_t):
        if a.text != b.text:
            before, after = a.text, b.text
            assert after == before.replace('报告v3', '报告v4').replace('本方案v12', '本方案v13'), (before, after)
            replacements.append({'before': before, 'after': after})
            a.text = b.text
    assert len(replacements) == 2, replacements
    assert E.tostring(old) == E.tostring(new), 'OOXML changed beyond the two approved text replacements'
    current_hash = sha(plan.read_bytes())
    assert current_render[0]['source_sha256'] == current_hash
    assert current_render[0]['pages'] == baseline_render[0]['pages'] == 50
    changed_pages = []
    for before, after in zip(baseline_render[0]['page_images'], current_render[0]['page_images']):
        assert before['page'] == after['page']
        assert sha(Path(after['path']).read_bytes()) == after['sha256']
        if before['sha256'] != after['sha256']:
            changed_pages.append(after['page'])
    assert changed_pages == [41, 49], changed_pages
    for index in (1, 2):
        assert current_render[index]['source_sha256'] == baseline_render[index]['source_sha256']
        assert sha(Path(current_render[index]['source']).read_bytes()) == current_render[index]['source_sha256']
        assert current_render[index]['page_images'] == baseline_render[index]['page_images']
        for page in current_render[index]['page_images']:
            assert sha(Path(page['path']).read_bytes()) == page['sha256']
    result = {
        'status': 'PASS_STRUCTURAL_AND_RENDER_IDENTITY',
        'reviewer': '/root/complete_document_review',
        'plan': str(plan), 'current_plan_sha256': current_hash,
        'changed_zip_parts': changed_parts,
        'approved_text_replacements': replacements,
        'changed_pages': changed_pages,
        'unchanged_plan_pages': 48,
        'report_and_handover_unchanged': True,
        'actual_visual_review_of_changed_pages': 'PENDING',
    }
    (R / 'complete_document_revision_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('approved_text_replacements',)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
