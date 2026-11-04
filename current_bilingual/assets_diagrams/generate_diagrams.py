"""从包内原生Visio源制作中英两版，仅修改文字与必要字号，不运行任何模型。"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json, re, shutil, math
from lxml import etree as E
from PIL import Image, ImageFont
import comtypes.client

ROOT=Path(__file__).resolve().parent
NS='http://schemas.microsoft.com/office/visio/2012/main'
T='{'+NS+'}'
def sha(p):
    return sha256(Path(p).read_bytes()).hexdigest()
def get_text(node):
    return ''.join(node.itertext()).strip()
def numbers(texts):
    return Counter(re.findall(r'\d+(?:\.\d+)?',' '.join(texts)))
def wrap_text(text, size, width_pt):
    # 采用同一系统字体估算行宽，Visio仍使用原布局；只在单词边界换行。
    f=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',max(1,round(size*4)))
    result=[]
    for paragraph in text.split('\n'):
        words=paragraph.split(' ');line=''
        for word in words:
            candidate=(line+' '+word).strip()
            if line and f.getlength(candidate)>width_pt*4:
                result.append(line);line=word
            else:line=candidate
        result.append(line)
    return '\n'.join(result)
def fit_shape(shape, translated):
    # 保留所有图形、连线和坐标，仅适度缩小英文文字以免覆盖原有面板。
    size=shape.CellsU('Char.Size').ResultIU*72
    width=shape.CellsU('Width').ResultIU*72
    height=shape.CellsU('Height').ResultIU*72
    try:
        width-=72*(shape.CellsU('LeftMargin').ResultIU+shape.CellsU('RightMargin').ResultIU)
        height-=72*(shape.CellsU('TopMargin').ResultIU+shape.CellsU('BottomMargin').ResultIU)
    except Exception:pass
    chosen=size
    for trial in [size-i*.25 for i in range(0,max(1,math.ceil((size-7.5)/.25)+1))]:
        chosen=trial
        wrapped=wrap_text(translated,trial,max(8,width*.90-1))
        lines=wrapped.split('\n')
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',max(1,round(trial*4)))
        longest=max((font.getlength(x)/4 for x in lines),default=0)
        if longest<=width*.94 and len(lines)*trial*1.30<=height:
            chosen=trial;break
    else:
        wrapped=wrap_text(translated,chosen,max(8,width*.90-1))
    shape.Text=wrapped
    shape.CellsU('Char.Size').FormulaU=f'{chosen:.2f} pt'
    try:shape.CellsU('Char.Font').FormulaU='FONT("Arial")'
    except Exception:pass
    return {'shape_id':int(shape.ID),'font_pt_original':round(size,3),'font_pt_en':round(chosen,3),'text':wrapped,'width_pt':round(width,3),'height_pt':round(height,3)}
def native_texts(path):
    with ZipFile(path) as z:
        texts=[]
        for p in z.namelist():
            if re.match(r'visio/pages/page\d+\.xml$',p):
                r=E.fromstring(z.read(p));texts.extend(get_text(t) for t in r.iter(T+'Text'))
        return texts
def walk(shapes):
    for i in range(1,shapes.Count+1):
        s=shapes.Item(i)
        yield s
        try:
            if s.Shapes.Count:yield from walk(s.Shapes)
        except Exception:pass
def main():
    spec=json.loads((ROOT/'diagram_spec.json').read_text(encoding='utf-8'))
    translations=json.loads((ROOT/'translation_map.json').read_text(encoding='utf-8'))
    for folder in ['zh','en']:(ROOT/folder).mkdir(exist_ok=True)
    app=comtypes.client.CreateObject('Visio.InvisibleApp',dynamic=True)
    app.AlertResponse=7
    result=[]
    try:
        app.Settings.SetRasterExportResolution(2,220,220,0)
        for item in spec:
            ident=item['id'];original=ROOT/item['original_vsdx'];out=ROOT/'en'/f'{ident}_en.vsdx'
            source_texts=native_texts(original);target=translations[ident]
            if len(source_texts)!=len(target):raise ValueError((ident,len(source_texts),len(target)))
            if numbers(source_texts)!=numbers(target):raise ValueError((ident,'number mismatch',numbers(source_texts)-numbers(target),numbers(target)-numbers(source_texts)))
            # 中文版完全保留原生文件及已嵌入正式Word的PNG字节。
            shutil.copy2(original,ROOT/'zh'/f'{ident}_zh.vsdx')
            shutil.copy2(ROOT/item['original_png'],ROOT/'zh'/f'{ident}_zh.png')
            doc=app.Documents.OpenEx(str(original.resolve()),2+8+64+128)
            mapping={s:t for s,t in zip(source_texts,target)}
            shapes=[]
            try:
                page=doc.Pages.Item(1)
                for shape in walk(page.Shapes):
                    text=str(shape.Text).strip()
                    if not text:continue
                    if text not in mapping:raise ValueError((ident,'unknown native text',text))
                    shapes.append(fit_shape(shape,mapping[text]))
                if out.exists():out.unlink()
                doc.SaveAs(str(out.resolve()))
                page.Export(str((ROOT/'en'/f'{ident}_en.png').resolve()))
            finally:doc.Close()
            en_texts=native_texts(out)
            if any(re.search('[\u4e00-\u9fff]',s) for s in en_texts):raise ValueError((ident,'Chinese remained'))
            if numbers(source_texts)!=numbers(en_texts):raise ValueError((ident,'saved number mismatch'))
            png=ROOT/'en'/f'{ident}_en.png'
            result.append({**item,'original_sha256':sha(ROOT/item['original_png']),'zh_png':f'zh/{ident}_zh.png','en_png':f'en/{ident}_en.png',
                'editable_sources':[f'zh/{ident}_zh.vsdx',f'en/{ident}_en.vsdx',item['original_vsdx']],
                'generator':'generate_diagrams.py','required_data':[item['original_vsdx'],item['original_png'],'translation_map.json','diagram_spec.json'],
                'translation_checks':{'native_text_shape_count':len(source_texts),'translated_shape_count':len(shapes),'english_chinese_remnants':False,'numeric_tokens_equal':True,'original_geometry_and_embedded_media':'preserved; only text, font and line wrapping edited','visual_review':'pending'},
                'fitting':shapes,'en_png_size':list(Image.open(png).size),'sha256':{'zh_png':sha(ROOT/'zh'/f'{ident}_zh.png'),'en_png':sha(png),'zh_vsdx':sha(ROOT/'zh'/f'{ident}_zh.vsdx'),'en_vsdx':sha(out)}})
            print(ident,'exported',flush=True)
    finally:app.Quit()
    (ROOT/'manifest.json').write_text(json.dumps({'items':result,'scope':'11 plan mechanism diagrams; original sources retained; no model execution','runtime':'Python + comtypes + Pillow + lxml; Microsoft Visio 16.0 InvisibleApp; Arial for English labels'},ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
