"""从已审v11保留图表，按已独审双标签结果另存v12；无事实文件时不生成Word。"""
from pathlib import Path
from collections import Counter
from hashlib import sha256
from io import BytesIO
import argparse
import json
import re
from zipfile import ZipFile

from docx import Document
from docx.shared import Inches, RGBColor
from docx.oxml.ns import qn

# 复用既有排版工具，不调用旧生成入口，也不修改旧脚本或旧Word。
from build_project_plan_v11 import add_after, add_table_after, table_geometry, font_xml

V = Path(__file__).resolve().parent
P = V.parent
R = V / 'runs/MERGED_20261102'
SOURCE = P / '基于ResNet18的手势识别项目方案_v11.docx'
SOURCE_SHA256 = 'e05c3935d302d895ffc2b87535caaf2e1f88a5bf708b4292d546453c85022c1b'
TARGET = P / '基于ResNet18的手势识别项目方案_v12.docx'
REVIEW_ROLES = {'merged_science', 'twelve_programs', 'matched_speed', 'raw_label_comparison'}
REQUIRED_COVERAGE = {'label_semantics', 'same_fixed_splits', 'new_training_from_official',
                     'unmerged_results_retained', 'twelve_model_results', 'raw_subgroups',
                     'truth_denominators', 'old_CAM_label_identity', 'dual_program_layout',
                     'control_interface_and_team_roles',
                     'detector_classifier_rule_responsibilities',
                     'whole_image_and_classifier_interfaces',
                     'single_final_integer_and_per_hand_details'}
RESPONSIBILITY_PARAGRAPHS = (
    '整图识别由三个部分协作完成：SSDLite在完整照片中检测手框，可以给出多个框，也可能漏检或误检；'
    'ResNet分别对每个检测框裁出的手部图计算六类分数，也可能分错。多张裁图可以成批计算，但ResNet本身不负责找手或多手汇总。'
    '程序再用独立Python规则处理达到分类置信阈值的结果：存在已接受的3就输出3；否则非0指令只有一种时输出该指令，'
    '多种指令冲突或没有非0有效指令时输出0。0类手不会抵消另一只手的有效指令。这套流程不能保证照片中的所有手都被找到并正确分类。',
    '整图程序接口与分类器接口应分开理解：GesturePredictor.predict读取完整照片，经过检测、裁图、分类和规则汇总后返回一个0—5整数；'
    'predict_details还返回框、各类概率、是否接受和汇总原因。ResNet分类器只读取预处理后的单手图张量并返回六类分数。'
    '控制约定中，0表示无新指令，不发起或续期运动，也不作为取消当前运动的停止命令；3才是明确停止。命令有效期和运动响应仍需控制端实际联调。',
    '一张照片的公开输出始终只有一个数字。两只手都通过检测与分类阈值后，1＋1汇总为1，1＋2因动作冲突汇总为0，'
    '1＋3因停止优先汇总为3，1＋0仍汇总为1。程序内部保留每只手的框、类别和置信度，predict_details或可选JSON可查看这些明细；'
    'GesturePredictor.predict返回一个整数，CLI成功时stdout也只打印这个最终整数，不逐只手返回多个数字。',
)
REQUIRED_REPLACEMENT_PREFIXES = [
    '0类占手框总数63.79%',
    '本项目由本人和组员共同负责ResNet识别分支',
    '上述合并直接用于训练标签',
    '0行是并拢伸出两指的two_up_inverted',
]


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def images(doc):
    return Counter(sha256(doc.part.related_parts[s._inline.graphic.graphicData.pic.blipFill.blip.embed].blob).hexdigest()
                   for s in doc.inline_shapes)


def check_facts(facts):
    if facts.get('phase') != 'FINAL_ACCEPTED':
        raise ValueError('仅接受六合并模型、十二程序及比较均已独审的最终事实')
    if not REQUIRED_COVERAGE <= set(facts.get('coverage', {})) or not all(facts['coverage'].get(k) is True for k in REQUIRED_COVERAGE):
        raise ValueError('双标签有效要求尚未完整落实')
    if len(facts.get('comparison_rows', [])) != 12:
        raise ValueError('需要不合并/合并×A/B×三架构的十二行真实结果')
    identities = {(x['variant'], x['protocol'], x['architecture']) for x in facts['comparison_rows']}
    expected = {(v, p, f'resnet{n}') for v in ['不合并', '合并'] for p in 'AB' for n in [18, 34, 50]}
    if identities != expected:
        raise ValueError('十二行结果身份不完整或重复')
    files = facts.get('verified_files', [])
    bound = {str(Path(x['path']).resolve()).casefold(): x['sha256'] for x in files}
    if not files:
        raise ValueError('没有绑定证据SHA')
    for item in files:
        if sha(item['path']) != item['sha256']:
            raise ValueError(f'事实来源变化：{item["path"]}')
    reviews = facts.get('reviews', [])
    if len(reviews) != 4 or {x['role'] for x in reviews} != REVIEW_ROLES:
        raise ValueError('科学、十二程序、共同速度和raw子群比较的独审均须齐全')
    for item in reviews:
        path = Path(item['path'])
        if bound.get(str(path.resolve()).casefold()) != sha(path):
            raise ValueError('独审文件没有绑定当前SHA')
        value = json.loads(path.read_text(encoding='utf-8'))
        for key in item.get('status_field', 'status').split('.'):
            value = value[key]
        if value not in item.get('accepted_statuses', ['PASS']):
            raise ValueError(f'独审未通过：{item["role"]}')


def check_output(path, expected_sha):
    if path.exists() and (not expected_sha or sha(path) != expected_sha):
        raise ValueError('已有新稿须显式提供其当前SHA后才能修订；旧版永不覆盖')


def append_blocks(doc, anchor, blocks):
    current = anchor
    for block in blocks:
        kind = block['kind']
        if kind == 'paragraph':
            current = add_after(doc, current, block['text'], block.get('style', 'Normal'))
        elif kind == 'table':
            table = add_table_after(doc, current, block['headers'], block['rows'], block['widths'])
            current = doc.add_paragraph()
            table._tbl.addnext(current._p)
        elif kind == 'image':
            picture = doc.add_paragraph()
            current._p.addnext(picture._p)
            picture.alignment = 1
            shape = picture.add_run().add_picture(block['path'], width=Inches(block.get('width_inches', 6.5)))
            shape._inline.docPr.set('descr', block['alt'])
            picture.paragraph_format.keep_with_next = True
            current = add_after(doc, picture, block['caption'], 'Caption')
        else:
            raise ValueError(f'未知内容类型：{kind}')
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accepted-facts', type=Path)
    parser.add_argument('--expected-output-sha256')
    args = parser.parse_args()
    if not args.accepted_facts:
        print('仅准备生成器。正式执行需要--accepted-facts；要求见runs/MERGED_20261102/document_comparison_preparation.md。')
        return
    facts_sha = sha(args.accepted_facts)
    facts = json.loads(args.accepted_facts.read_text(encoding='utf-8'))
    check_facts(facts)
    if sha(SOURCE) != SOURCE_SHA256:
        raise ValueError('已审v11来源变化，须重新对照')
    check_output(TARGET, args.expected_output_sha256)
    doc = Document(SOURCE)
    source_tables = list(doc.tables)
    responsibility_matches = [p for p in doc.paragraphs if p.text.startswith('项目总体流程如图1')]
    if len(responsibility_matches) != 1:
        raise ValueError('未找到唯一整图流程说明，须核对源稿后安放模块职责')
    responsibility_anchor = responsibility_matches[0]
    original_images = images(doc)
    original_tables = len(doc.tables)
    original_headings = [p.text for p in doc.paragraphs if p.style.name.startswith('Heading')]
    # 旧版红色修改已成为既有正文，本次新增和实质改文再标红。
    for run in doc.element.xpath('.//w:rPr/w:color'):
        run.set(qn('w:val'), '000000')
    for item in facts['body_replacements']:
        matched = [p for p in doc.paragraphs if p.text == item['old']]
        if len(matched) != 1 or matched[0]._p.xpath('.//w:drawing'):
            raise ValueError('替换段落不唯一或包含须保留图形')
        p = matched[0]
        for node in list(p._p):
            if node.tag != qn('w:pPr'):
                p._p.remove(node)
        run = p.add_run(item['new'])
        run.font.color.rgb = RGBColor.from_string('C00000')
        font_xml(run._r.get_or_add_rPr())
    old_replacements = [x['old'] for x in facts['body_replacements']]
    if not all(any(x.startswith(prefix) for x in old_replacements) for prefix in REQUIRED_REPLACEMENT_PREFIXES):
        raise ValueError('标签主线、总体目标或旧CAM标签身份未完成全局修订')
    for item in facts.get('insertions', []):
        matches = [p for p in doc.paragraphs if p.text == item['after']]
        if len(matches) != 1:
            raise ValueError('插入锚点必须唯一且对应当前正文')
        for block in item['blocks']:
            if block['kind'] == 'image':
                bound = {str(Path(x['path']).resolve()).casefold(): x['sha256'] for x in facts['verified_files']}
                if bound.get(str(Path(block['path']).resolve()).casefold()) != sha(block['path']):
                    raise ValueError('新增图片必须绑定已审核证据SHA')
        append_blocks(doc, matches[0], item['blocks'])
    for item in facts.get('table_annotations', []):
        table = source_tables[item['source_table_index']]
        p = doc.add_paragraph(item['text'])
        table._tbl.addprevious(p._p)
    # 明确源码中的职责边界；不能只在facts勾选而未进入最终Word。
    for text in RESPONSIBILITY_PARAGRAPHS:
        responsibility_anchor = add_after(doc, responsibility_anchor, text)
    table_geometry(doc)
    doc.core_properties.subject = '不合并与合并标签下的A/B真实训练、任务语义与程序交接'
    buffer = BytesIO()
    doc.save(buffer)
    data = buffer.getvalue()
    revised = Document(BytesIO(data))
    if original_images - images(revised) or len(revised.tables) < original_tables:
        raise ValueError('v11已有图表丢失')
    headings = [p.text for p in revised.paragraphs if p.style.name.startswith('Heading')]
    if any(x not in headings for x in original_headings):
        raise ValueError('v11原章节丢失')
    with ZipFile(BytesIO(data)) as package:
        if package.testzip() is not None:
            raise ValueError('Word包校验失败')
        text = package.read('word/document.xml').decode('utf-8')
        if any(x in text for x in ['待补', '品牌可见', 'FINAL_ACCEPTED', '内部工作稿']):
            raise ValueError('正式正文含占位或内部文字')
    # 先形成确切字节，再复核活证据和输出身份，提交同一份字节。
    check_facts(facts)
    if sha(args.accepted_facts) != facts_sha or sha(SOURCE) != SOURCE_SHA256:
        raise ValueError('生成期间事实或源稿变化')
    check_output(TARGET, args.expected_output_sha256)
    TARGET.write_bytes(data)
    audit = {'status': 'AUTHOR_TEXT_READY_AWAITING_RENDER_AND_INDEPENDENT_REVIEW',
             'source': str(SOURCE), 'source_sha256': SOURCE_SHA256, 'output': str(TARGET),
             'output_sha256': sha(TARGET), 'facts_sha256': facts_sha,
             'original_images_preserved': sum(original_images.values()), 'images': len(revised.inline_shapes),
             'original_tables_preserved': original_tables, 'tables': len(revised.tables),
             'old_formal_files_overwritten': False}
    (R / 'project_plan_v12_author_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
