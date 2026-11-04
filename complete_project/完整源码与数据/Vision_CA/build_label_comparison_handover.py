"""在十二程序与双标签结果独审后，另存新交接根三文件；不写旧交接。"""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import re
from io import BytesIO

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from build_project_plan_v12 import check_facts, check_output, RESPONSIBILITY_PARAGRAPHS
from build_AB_handover_docs import build_docx as build_baseline_docx

V = Path(__file__).resolve().parent
P = V.parent
R = V / 'runs/MERGED_20261102'
OUT = P / 'ResNet_合并与不合并完整交接_20261103'


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def build_docx(markdown):
    doc = Document(BytesIO(build_baseline_docx(markdown)))
    # Word/LibreOffice会合并相邻表；空段保持每张表自己的列宽与重复表头。
    for element in list(doc.element.body):
        following = element.getnext()
        if element.tag == qn('w:tbl') and following is not None and following.tag == qn('w:tbl'):
            spacer = OxmlElement('w:p')
            element.addnext(spacer)
    doc.core_properties.title = 'ResNet合并与不合并标签任务完整交接说明'
    doc.core_properties.subject = '十二识别程序、双标签真实实验、阅读顺序与小组后续协作'
    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def with_responsibility_section(markdown):
    # 交接第1章先建立检测/分类/汇总及整图API边界，不把多手能力归给ResNet。
    block = '\n\n'.join(RESPONSIBILITY_PARAGRAPHS)
    if block in markdown:
        return markdown
    heading = re.search(r'^## 1(?:[ .．、]|$)[^\n]*', markdown, flags=re.M)
    if heading is None:
        raise ValueError('新交接须保留第1章，以安放模块职责和两类接口说明')
    position = heading.end()
    return markdown[:position] + '\n\n### 模块职责与两类接口\n\n' + block + '\n' + markdown[position:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accepted-facts', type=Path)
    parser.add_argument('--expected-output-hashes', type=Path,
                        help='修订已生成新交接时给出根三文件当前SHA；不用于旧根目录')
    args = parser.parse_args()
    if not args.accepted_facts:
        print('仅准备生成器。最终事实、十二程序和独审齐全后使用--accepted-facts。')
        return
    facts_sha = sha(args.accepted_facts)
    facts = json.loads(args.accepted_facts.read_text(encoding='utf-8'))
    check_facts(facts)
    markdown = with_responsibility_section(facts['handover_markdown'])
    readme = facts['root_readme_markdown']
    if len(re.findall(r'^## ', markdown, flags=re.M)) < 8:
        raise ValueError('完整交接至少保留原八章承担的内容')
    if len(facts.get('handover_sections_covered', [])) < 8:
        raise ValueError('没有逐章保全交接内容的说明')
    for value in [markdown, readme]:
        if any(s in value for s in ['待补', '品牌可见', '内部工作稿', 'FINAL_ACCEPTED', 'file:///']):
            raise ValueError('正式交接含占位或不可携带个人路径')
        if '程序/不合并/' not in value or '程序/合并/' not in value:
            raise ValueError('两版本程序的实际逻辑目录未说明')
    outputs = {'README.md': readme.encode('utf-8'), '交接说明.md': markdown.encode('utf-8'),
               '交接说明.docx': build_docx(markdown)}
    expected = {} if not args.expected_output_hashes else json.loads(args.expected_output_hashes.read_text(encoding='utf-8'))
    for name in outputs:
        check_output(OUT / name, expected.get(name))
    # 所有字节先形成，核证据后才提交；新目录与旧ResNet_AB根没有写路径交集。
    check_facts(facts)
    if sha(args.accepted_facts) != facts_sha:
        raise ValueError('文档生成期间事实文件改变')
    for name in outputs:
        check_output(OUT / name, expected.get(name))
    OUT.mkdir(exist_ok=True)
    for name, data in outputs.items():
        (OUT / name).write_bytes(data)
    audit = {'status': 'AUTHOR_TEXT_READY_AWAITING_RENDER_AND_INDEPENDENT_REVIEW',
             'bundle_root': str(OUT), 'facts_sha256': facts_sha,
             'outputs': [{'source': str(OUT / name), 'sha256': sha(OUT / name)} for name in outputs],
             'old_formal_files_overwritten': False,
             'links_require_final_ZIP_member_validation': True}
    (R / 'label_comparison_handover_author_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
