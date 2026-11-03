"""将项目交接说明排成Word，保留本包内可点击的相对文件链接。"""
from pathlib import Path
import re
import sys
from urllib.parse import quote
from docx import Document
from docx.shared import Cm, Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image

ROOT = Path(__file__).resolve().parent


def font_style(style, size, east='SimSun', bold=False):
    style.font.name = 'Calibri'
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    props = style.element.get_or_add_rPr()
    fonts = props.get_or_add_rFonts()
    # 清除模板主题字体，避免主题覆盖显式中文字体并产生错误字宽。
    for attribute in ('asciiTheme', 'eastAsiaTheme', 'hAnsiTheme', 'cstheme'):
        fonts.attrib.pop(qn('w:' + attribute), None)
    fonts.set(qn('w:eastAsia'), east)
    fonts.set(qn('w:cs'), 'Calibri')
    complex_size = props.find(qn('w:szCs'))
    if complex_size is None:
        complex_size = OxmlElement('w:szCs')
        props.append(complex_size)
    complex_size.set(qn('w:val'), str(round(size * 2)))
    for tag in ('kern', 'spacing'):
        for element in list(props.findall(qn('w:' + tag))):
            props.remove(element)


def hyperlink(paragraph, text, url):
    # 链接直接挂在可读文字上，不让长网址挤破版心。
    relation = paragraph.part.relate_to(url, 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', is_external=True)
    link = OxmlElement('w:hyperlink')
    link.set(qn('r:id'), relation)
    run = OxmlElement('w:r')
    prop = OxmlElement('w:rPr')
    color = OxmlElement('w:color')
    color.set(qn('w:val'), '24557A')
    prop.append(color)
    run.append(prop)
    node = OxmlElement('w:t')
    node.text = text
    run.append(node)
    link.append(run)
    paragraph._p.append(link)


def inline(paragraph, text):
    # 仅解释笔记需要的少量Markdown；避免把标记符号带到Word。
    pattern = r'(\[([^\]]+)\]\((<?(?:[^()]|\([^)]*\))+>?)\)|\*\*([^*]+)\*\*|`([^`]+)`)'
    position = 0
    for match in re.finditer(pattern, text):
        paragraph.add_run(text[position:match.start()])
        if match.group(2):
            hyperlink(paragraph, match.group(2), quote(match.group(3).strip('<>'), safe='/:#?=&%'))
        elif match.group(4):
            paragraph.add_run(match.group(4)).bold = True
        else:
            run = paragraph.add_run(match.group(5))
            run.font.name = 'Consolas'
        position = match.end()
    paragraph.add_run(text[position:])


def build(source, output):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.top_margin = sec.bottom_margin = Cm(1.8)
    sec.left_margin = sec.right_margin = Cm(1.9)
    font_style(doc.styles['Normal'], 11.5)
    normal = doc.styles['Normal'].paragraph_format
    normal.line_spacing = 1.22
    normal.space_after = Pt(6)
    for name, size in [('Title', 22), ('Heading 1', 16), ('Heading 2', 13), ('Heading 3', 12)]:
        font_style(doc.styles[name], size, 'Microsoft YaHei', True)
        doc.styles[name].paragraph_format.keep_with_next = True
    # 明确覆盖模板标题装饰线，标题只保留黑色字体与留白。
    borders = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'nil')
    borders.append(bottom)
    title_props = doc.styles['Title'].element.get_or_add_pPr()
    for old_border in list(title_props.findall(qn('w:pBdr'))):
        title_props.remove(old_border)
    title_props.append(borders)
    font_style(doc.styles['Caption'], 10)
    doc.styles['Caption'].paragraph_format.space_after = Pt(9)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    count = 0
    code = False
    lines = source.read_text(encoding='utf-8').splitlines()
    skip = set()
    for index, line in enumerate(lines):
        if index in skip:
            continue
        line = line.strip()
        if not line:
            continue
        if line.startswith('|'):
            # 结构参数表保留为真正的Word表格，不能把Markdown竖线当正文。
            rows = []
            position = index
            while position < len(lines) and lines[position].strip().startswith('|'):
                values = [x.strip() for x in lines[position].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', x) for x in values):
                    rows.append(values)
                skip.add(position)
                position += 1
            table = doc.add_table(rows=len(rows), cols=len(rows[0]))
            table.style = 'Table Grid'
            table.autofit = True
            for row_number, values in enumerate(rows):
                # 同一数据行不得被拆成跨页空续行。
                table.rows[row_number]._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
                for cell, value in zip(table.rows[row_number].cells, values):
                    inline(cell.paragraphs[0], value)
                    cell.paragraphs[0].paragraph_format.space_before = Pt(0)
                    cell.paragraphs[0].paragraph_format.space_after = Pt(2)
                    cell.paragraphs[0].paragraph_format.line_spacing = 1.08
                    if len(rows) <= 15 and row_number < len(rows) - 1:
                        cell.paragraphs[0].paragraph_format.keep_with_next = True
                    for run in cell.paragraphs[0].runs:
                        run.font.size = Pt(10)
                        run.bold = row_number == 0
                    if row_number == 0:
                        shade = OxmlElement('w:shd')
                        shade.set(qn('w:val'), 'clear')
                        shade.set(qn('w:fill'), 'EAF0F5')
                        cell._tc.get_or_add_tcPr().append(shade)
            continue
        if line.startswith('```'):
            code = not code
            continue
        picture = re.fullmatch(r'!\[([^\]]*)\]\((.+)\)', line)
        if picture:
            count += 1
            label, path = picture.groups()
            path = Path(path)
            width, height = Image.open(path).size
            # 宽图给足版心；竖图限制高度，不能挤到页脚或割裂图注。
            max_width, max_height = 17.6, 21.7 if 'A_20261102_TPE' in str(path) else 9.5
            if path.name.endswith('_confusion.png'):
                max_width = 16.0
            chosen_width = min(max_width, max_height * width / height)
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            shape = p.add_run().add_picture(str(path), width=Cm(chosen_width))
            shape._inline.docPr.set('descr', label)
            caption = doc.add_paragraph(style='Caption')
            caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption.add_run(f'图{count} {label}')
            continue
        heading = re.match(r'^(#{1,4})\s+(.+)$', line)
        if heading:
            level = len(heading.group(1))
            p = doc.add_paragraph(style='Title' if level == 1 else f'Heading {min(level - 1, 3)}')
            inline(p, heading.group(2))
            continue
        p = doc.add_paragraph()
        if code:
            p.paragraph_format.line_spacing = 1.05
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(line)
            run.font.name, run.font.size = 'Consolas', Pt(9.5)
        else:
            if line.startswith('- '):
                line = '• ' + line[2:]
                p.paragraph_format.left_indent = Cm(0.3)
            inline(p, line)
    doc.core_properties.title = doc.paragraphs[0].text
    doc.core_properties.subject = doc.paragraphs[0].text
    doc.save(output)
    print(f'{output} 图片{count}', flush=True)


if __name__ == '__main__':
    build(Path(sys.argv[1]), Path(sys.argv[2]))
