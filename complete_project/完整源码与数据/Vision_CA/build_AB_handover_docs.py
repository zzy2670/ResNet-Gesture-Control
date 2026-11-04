"""从已审核A/B事实生成新版根交接三文件，旧交接保持原样。

此脚本不读取模型、不重新推理、不打包数据。调用前由root提供审核事实文件。
"""
from pathlib import Path
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile
import argparse
import importlib.util
import json
import re

from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

V = Path(__file__).resolve().parent
P = V.parent
B = V / 'runs/B_20261102_TPE'
OUT = P / 'ResNet_AB项目完整交接_20261102'
SOURCE = B / 'handover_working.md'


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


spec = importlib.util.spec_from_file_location('old_handover_helpers', V / 'build_handover_docx.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def cjk(style, size, bold=False, east='宋体'):
    style.font.name = 'Calibri'
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.get_or_add_rFonts()
    for key in list(fonts.attrib):
        if 'theme' in key.lower():
            del fonts.attrib[key]
    fonts.set(qn('w:eastAsia'), east)
    fonts.set(qn('w:ascii'), 'Calibri')
    fonts.set(qn('w:hAnsi'), 'Calibri')
    sz = rpr.find(qn('w:sz'))
    cs = rpr.find(qn('w:szCs'))
    if cs is None:
        cs = OxmlElement('w:szCs')
        sz.addnext(cs)
    cs.set(qn('w:val'), str(round(size * 2)))


def table_widths(headers):
    if headers[:2] == ['数字', '手势标签']:
        return [.55, 4.65, 1.5]
    if headers[0] == '协议与子集':
        return [2.2, 1.35, 1.55, 1.6]
    if headers[0] == '工作' and len(headers) == 3:
        return [1.1, 3.65, 1.95]
    if headers[0] in ['A/B模型', 'A模型']:
        return [1.15, 1.25, 1.2, 1.25, 1.85]
    if headers[0] == '顺序':
        return [.5, 2.8, 3.4]
    if len(headers) == 4:
        return [.95, 1.6, 2.5, 1.65]
    return [6.7 / len(headers)] * len(headers)


def format_table(doc, values):
    table = doc.add_table(rows=len(values), cols=len(values[0]))
    table.autofit = False
    widths = table_widths(values[0])
    pr = table._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for name in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        x = OxmlElement('w:' + name)
        x.set(qn('w:val'), 'single')
        x.set(qn('w:sz'), '4')
        x.set(qn('w:color'), 'D9D9D9')
        borders.append(x)
    pr.append(borders)
    for col, width in zip(table.columns, widths):
        col.width = Cm(width * 2.54)
    for i, row in enumerate(table.rows):
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement('w:cantSplit'))
        if i == 0:
            trpr.append(OxmlElement('w:tblHeader'))
        for j, (cell, value) in enumerate(zip(row.cells, values[i])):
            cell.width = Cm(widths[j] * 2.54)
            tcpr = cell._tc.get_or_add_tcPr()
            mar = OxmlElement('w:tcMar')
            for edge, amount in [('top', '75'), ('left', '85'), ('bottom', '75'), ('right', '85')]:
                x = OxmlElement('w:' + edge)
                x.set(qn('w:w'), amount)
                x.set(qn('w:type'), 'dxa')
                mar.append(x)
            tcpr.append(mar)
            vertical = OxmlElement('w:vAlign')
            vertical.set(qn('w:val'), 'center')
            tcpr.append(vertical)
            if i == 0:
                shd = OxmlElement('w:shd')
                shd.set(qn('w:val'), 'clear')
                shd.set(qn('w:fill'), 'E7EDF4')
                tcpr.append(shd)
            p = cell.paragraphs[0]
            helper.inline(p, value)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.1
            p.paragraph_format.keep_with_next = False
            for r in p.runs:
                r.font.size = Pt(10.5)
                r.bold = i == 0
    return table


def build_docx(markdown):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21)
    sec.page_height = Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2)
    sec.top_margin = sec.bottom_margin = Cm(1.8)
    for name, size, bold, east in [('Normal', 11.5, False, '宋体'), ('Title', 22, True, '黑体'),
                                   ('Heading 1', 15, True, '黑体'), ('Heading 2', 12.5, True, '黑体'),
                                   ('Heading 3', 11.5, True, '黑体')]:
        cjk(doc.styles[name], size, bold, east)
    doc.styles['Normal'].paragraph_format.line_spacing = 1.2
    doc.styles['Normal'].paragraph_format.space_after = Pt(6)
    for name in ('Title', 'Heading 1', 'Heading 2', 'Heading 3'):
        pp = doc.styles[name].paragraph_format
        pp.keep_with_next = True
        pp.space_before = Pt(12 if name != 'Title' else 0)
        pp.space_after = Pt(6)
        pr = doc.styles[name].element.get_or_add_pPr()
        for x in list(pr.findall(qn('w:pBdr'))):
            pr.remove(x)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    lines = markdown.splitlines()
    i, code = 0, False
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith('```'):
            code = not code
            continue
        if not code and line.startswith('|'):
            rows = []
            while True:
                parts = [x.strip() for x in line.strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', x) for x in parts):
                    rows.append(parts)
                if i >= len(lines) or not lines[i].strip().startswith('|'):
                    break
                line = lines[i].strip()
                i += 1
            format_table(doc, rows)
            continue
        h = re.match(r'^(#{1,4})\s+(.+)$', line) if not code else None
        if h:
            depth, text = len(h.group(1)), h.group(2)
            p = doc.add_paragraph(style='Title' if depth == 1 else f'Heading {depth-1}')
            helper.inline(p, text)
            continue
        p = doc.add_paragraph()
        if code:
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.05
            r = p.add_run(line)
            r.font.name = 'Consolas'
            r.font.size = Pt(9.5)
            rf = r._r.get_or_add_rPr().get_or_add_rFonts()
            rf.set(qn('w:eastAsia'), '宋体')
        else:
            helper.inline(p, line)
    doc.core_properties.title = 'ResNet手势识别项目A与B交接说明'
    doc.core_properties.subject = '识别程序、A与B实验、资料阅读与后续小组合作'
    b = BytesIO()
    doc.save(b)
    data = b.getvalue()
    with ZipFile(BytesIO(data)) as z:
        assert z.testzip() is None
    return data


def final_markdown(facts):
    text = SOURCE.read_text(encoding='utf-8')
    replacements = facts['paragraph_replacements']
    for old, new in replacements.items():
        assert old in text, f'未找到交接段落：{old[:45]}'
        text = text.replace(old, new)
    report = facts['report_filename']
    plan = '基于ResNet18的手势识别项目方案_v11.docx'
    text = text.replace('历史与正式文档中的A/B综合报告v2', f'[A与B综合实验报告](<历史与正式文档/{report}>)')
    text = text.replace('历史与正式文档中的项目方案v11', f'[项目方案v11](历史与正式文档/{plan})')
    text = text.replace('历史与正式文档中的手势合并与数字编码', '[手势合并与数字编码](历史与正式文档/手势合并与数字编码.docx)')
    text = text.replace('完整源码与数据/Vision_CA/README.md |', '[完整项目README](完整源码与数据/Vision_CA/README.md) |')
    # 用A/B六模型真实结果替换旧的A三行。
    start = text.index('| A模型 |')
    end = text.index('\n\n人工框手图', start)
    a = facts.get('A_comparison')
    if a is None:
        a = json.loads((V / 'runs/A_20261102_TPE/comparison_results.json').read_text(encoding='utf-8'))
    rows = ['| A/B模型 | 人工框手图准确率 | 整图准确率 | 整图宏F1 | CPU整图均值／P95 |',
            '|---|---:|---:|---:|---:|']
    for protocol, models in [('A', a), ('B', facts['B_comparison'])]:
        assert len(models) == 3
        for x in models:
            rows.append(f'| {protocol} {x["model"]} | {x["test_crop_accuracy"]*100:.2f}% | {x["test_photo_accuracy"]*100:.2f}% | {x["test_photo_macro_f1"]*100:.2f}% | {x["whole_cpu_mean_ms"]:.2f}／{x["whole_cpu_p95_ms"]:.2f} ms |')
    text = text[:start] + '\n'.join(rows) + text[end:]
    assert '内部工作稿' not in text and '仍在运行' not in text and '正在按' not in text
    assert 'mapping.json' not in text and 'from gesture_predict' not in text
    assert '待补' not in text and '品牌可见' not in text
    return text


def readme(facts):
    report = facts['report_filename']
    return f'''# ResNet手势识别A与B完整项目

本包由我和组员共同负责的ResNet分支整理，包含A按照片划分和B按人员划分的三模型实验、六个独立识别程序，以及完整源码、真实数据、预训练起点和运行证据。

先读[交接说明](交接说明.md)，需要批注时使用[可编辑Word](交接说明.docx)；实际结果查[A与B综合报告](<历史与正式文档/{report}>)，方案原理查[项目方案v11](历史与正式文档/基于ResNet18的手势识别项目方案_v11.docx)。历史方案v1—v10与A报告v1保留当时原文，当前状态以新版和实际配置为准。

## 直接运行

按[所选模型README](程序/B/resnet18/README.md)安装依赖，从总包根目录执行：

```powershell
cd "程序/B/resnet18"
python predict.py "examples/1.jpg"
python predict.py "你的照片.jpg" --json "result.json"
```

程序/A和程序/B各有resnet18、resnet34、resnet50三个独立目录，自带各协议权重和阈值。成功stdout仅为0—5整数；异常写stderr并非0退出。0是不发起或续期运动，3是明确停止；控制端接收和运动行为仍需实际联调。

## 文件位置

| 位置 | 内容 |
|---|---|
| 历史与正式文档 | 方案v11、A/B报告、A报告v1、历史方案、编码与必要课程材料 |
| 程序/A及程序/B | 六个离线识别程序、配置、示例及各自校验清单 |
| 完整源码与数据/Vision_CA | 原图、A/B固定分组、源码、训练日志、TPE数据库、完整预测、CAM与四官方预训练权重 |
| SHA256SUMS.json | ZIP成员的大小与SHA256，用于接收核包 |

本包完整数据有31,833张原图、7,973名人员、38,884个手框，A与B各自使用全部原图；B三组人员、图片编号及内容SHA均无交叉。人工框分类与完整照片输出是不同评价，结果和真实分母见综合报告。

个人图文学习笔记及视频讲解、会议大录屏不分发。正式报告里的科学图表、错误照片和任务Grad-CAM保留。

先核包再改实验。原结果和历史路径保持留档，换机器按[完整项目说明](完整源码与数据/Vision_CA/README.md)恢复当前路径；新实验另用目录。另一识别路线的公平比较、控制集成、小组PPT和课堂演示由小组继续完成，建议分工见交接说明第8节。
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accepted-facts', required=True, type=Path)
    args = parser.parse_args()
    facts = json.loads(args.accepted_facts.read_text(encoding='utf-8'))
    assert facts['final_independent_review'] == 'PASS'
    assert facts['six_programs_verified'] is True
    for f in facts['verified_files']:
        assert sha(f['path']) == f['sha256'], f['path']
    markdown = final_markdown(facts)
    word = build_docx(markdown)
    # exact字节先形成，随后再次核对输入证据再写，防止文档生成期间依据变化。
    for f in facts['verified_files']:
        assert sha(f['path']) == f['sha256'], f['path']
    OUT.mkdir(exist_ok=True)
    (OUT / '交接说明.md').write_text(markdown, encoding='utf-8')
    (OUT / 'README.md').write_text(readme(facts), encoding='utf-8')
    (OUT / '交接说明.docx').write_bytes(word)
    audit = {'status': 'TEXT_AND_PACKAGE_CHECKS_READY_FOR_INDEPENDENT_RENDER_REVIEW',
             'facts': str(args.accepted_facts), 'facts_sha256': sha(args.accepted_facts),
             'outputs': [{'path': str(OUT / name), 'sha256': sha(OUT / name)} for name in ['交接说明.md', '交接说明.docx', 'README.md']],
             'old_handover_overwritten': False, 'rendered_QA_claimed': False,
             'link_validation_requires_final_freeze_member_set': True}
    (B / 'handover_AB_author_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
