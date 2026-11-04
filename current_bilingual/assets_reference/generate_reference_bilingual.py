"""只重排已保存图像的文字层；不加载模型，不读取原始数据目录。"""
from pathlib import Path
import base64
import hashlib
import html
import json
import shutil
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FONT = Path('C:/Windows/Fonts/msyh.ttc')
BOLD = Path('C:/Windows/Fonts/msyhbd.ttc')
NAVY = '#233950'
CORAL = '#be584a'
CAPTIONS_ZH = {
    'D1-I01':'手势编码表示例照片：peace，掌心朝向镜头的V形；编码1，前进。',
    'D1-I02':'手势编码表示例照片：peace_inverted，手背朝向镜头的V形；编码1，前进。',
    'D1-I03':'手势编码表示例照片：stop，三种张开手掌姿态之一；编码2，后退。',
    'D1-I04':'手势编码表示例照片：stop_inverted，三种张开手掌姿态之一；编码2，后退。',
    'D1-I05':'手势编码表示例照片：palm，三种张开手掌姿态之一；编码2，后退。',
    'D1-I06':'手势编码表示例照片：fist，握拳；编码3，停止。',
    'D1-I07':'手势编码表示例照片：like，拇指向上；编码4，左转。',
    'D1-I08':'手势编码表示例照片：dislike，拇指向下；编码5，右转。',
    'D1-I09':'手势编码表示例照片：no_gesture，闲置手示例；编码0，不发送指令。',
    'D1-I20':'图10 电缆项目的解释路径原图参考。原图来自本地电缆项目报告图3.9，展示前向推理与目标层解释计算；图中的逐位置特征与梯度相乘后跨通道平均对应原项目实现。中文版为原图的翻译适配。本图只作为计算流程参考，不是本项目的手势实验结果。',
    'D1-I22':'图12 电缆论文的分类与分层总览原图参考。原图来自本地电缆论文图4.14。七行对应七类测量状态，四列依次为原始STFT、block3、block1和block2。中文版为原图的翻译适配。本图用于展示类别与模块维度的呈现方式，不是手势实验结果。',
    'D1-I23':'图13 任务训练前整理的六行手图分层布局示例。各行按来源标签0至5组织，各列依次为输入、layer4主解释、layer1、layer2和layer3。热图由通用ImageNet ResNet18实际计算，每行标明其实际ImageNet目标。本图保留用于说明布局，不能作为本项目六类手势识别结果。两个语言版本均复用已保存的原始证据像素，不重新运行模型。',
    'D1-I24':'图14 原图、响应与手指局部细节的对应观察示例。标框和放大图对应同一输入的相同位置，可观察伸直手指的轮廓与指间空隙。该独立示例以通用ImageNet模型中的nail（钉子，索引677）为目标，不能说明模型已正确识别V形手势。正式局部分析将按实验需要补充。两个语言版本均复用已保存的原始证据像素，不重新运行模型。'
}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Layer:
    def __init__(self, image):
        self.original = Image.open(image).convert('RGB')
        self.canvas = self.original.copy()
        self.ops = []
        self.png = Path(image)
        self.texts = []
        self.evidence_regions = []

    def clear(self, box, color='white'):
        self.ops.append(('rect', list(box), color))
        ImageDraw.Draw(self.canvas).rectangle(box, fill=color)

    def text(self, xy, text, size=26, fill=NAVY, anchor='mm', bold=False):
        font=ImageFont.truetype(str(BOLD if bold else FONT), size)
        self.ops.append(('text', list(xy), text, size, fill, anchor, bold))
        self.texts.append(text)
        ImageDraw.Draw(self.canvas).text(xy, text, font=font, fill=fill, anchor=anchor,
                                        spacing=max(2,size//5), align='center' if anchor=='mm' else 'left')

    def vertical(self, box, text, size=23, fill=NAVY):
        # 旋转整条标签，不旋转或改动任何证据像素。
        x0,y0,x1,y1=box
        font=ImageFont.truetype(str(FONT),size)
        d=ImageDraw.Draw(self.canvas)
        bw=d.textbbox((0,0),text,font=font)[2]
        tile=Image.new('RGBA',(bw+10,size+14),(255,255,255,0))
        ImageDraw.Draw(tile).text((5,3),text,font=font,fill=fill)
        tile=tile.rotate(90,expand=True)
        self.canvas.paste(tile,(int((x0+x1-tile.width)/2),int((y0+y1-tile.height)/2)),tile)
        self.ops.append(('vertical',list(box),text,size,fill))
        self.texts.append(text)

    def save(self, stem):
        self.canvas.save(ROOT/(stem+'.png'))
        w,h=self.canvas.size
        raw=base64.b64encode(self.png.read_bytes()).decode()
        out=[f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
             '<title>Editable translation and presentation adaptation; original evidence raster preserved</title>',
             f'<image x="0" y="0" width="{self.original.width}" height="{self.original.height}" xlink:href="data:image/png;base64,{raw}"/>']
        if self.canvas.size != self.original.size:
            out.insert(2,f'<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>')
        for op in self.ops:
            if op[0]=='rect':
                _,b,c=op;x0,y0,x1,y1=b
                out.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0+1}" height="{y1-y0+1}" fill="{c}"/>')
            elif op[0]=='text':
                _,xy,t,s,c,a,bold=op
                x,y=xy;lines=t.split('\n'); step=s*1.2
                start=y-(len(lines)-1)*step/2 if a in ('mm','lm') else y
                anch='middle' if a=='mm' else 'start'
                for i,line in enumerate(lines):
                    out.append(f'<text x="{x}" y="{start+i*step}" text-anchor="{anch}" dominant-baseline="central" font-family="Microsoft YaHei, Arial, sans-serif" font-size="{s}" font-weight="{700 if bold else 400}" fill="{c}">{html.escape(line)}</text>')
            elif op[0]=='vertical':
                _,b,t,s,c=op;x0,y0,x1,y1=b;x=(x0+x1)/2;y=(y0+y1)/2
                out.append(f'<text x="{x}" y="{y}" transform="rotate(-90 {x} {y})" text-anchor="middle" dominant-baseline="central" font-family="Microsoft YaHei, Arial, sans-serif" font-size="{s}" fill="{c}">{html.escape(t)}</text>')
        out.append('</svg>')
        (ROOT/(stem+'.svg')).write_text('\n'.join(out),encoding='utf-8')
        (ROOT/(stem+'_text_layers.json')).write_text(json.dumps(self.ops,ensure_ascii=False,indent=2),encoding='utf-8')
        return [stem+'.svg',stem+'_text_layers.json']

def cable_mechanism(src):
    q=Layer(src)
    # 遮盖自然语言标签，保留原图的特征、梯度、颜色、曲线、数字与箭头位置。
    masks=[(442,5,700,29),(363,36,507,65),(535,36,680,65),(715,36,860,65),
           (142,173,332,188),(900,43,1023,74),(909,90,1022,140),(902,155,1022,185),
           (1060,45,1205,66),(1078,68,1206,188),(1255,79,1392,108),(1255,115,1392,143),
           (589,251,850,282),(605,384,846,402),(1198,254,1458,281),(1203,369,1455,388),
           (904,251,1136,265),(175,397,334,416),(392,490,487,539),(553,568,676,587),
           (952,486,1089,512),(1785,809,2224,830)]
    for m in masks:q.clear(m)
    q.text((570,17),'前向推理',15,bold=True)
    q.text((435,53),'模块1',14,bold=True)
    q.text((607,53),'模块2',14,bold=True)
    q.text((787,53),'目标层',14,bold=True)
    q.text((239,181),'对数功率STFT · 128 × 128',12,bold=True)
    q.text((962,58),'最大池化',14,bold=True)
    q.text((962,113),'全局平均\n池化',14,bold=True)
    q.text((962,169),'分类器',14,bold=True)
    q.text((1133,55),'七类得分',14,bold=True)
    labels=['背景','电晕','高压背景','局部放电','局放＋高压干扰','局放＋电晕','局放＋电晕＋高压干扰']
    for i,t in enumerate(labels):q.text((1080,79+18*i),t,12,CORAL if i==3 else NAVY,'lm',i==3)
    q.text((1327,94),'选定类别',14,CORAL,bold=True)
    q.text((1327,132),'局放得分 y_pd',14,CORAL,bold=True)
    q.text((738,267),'模块3目标激活 A_k',14,'#7f63a8',bold=True)
    q.text((724,395),'按 mean |A_k ⊙ G_k| 排序通道',12)
    q.text((1327,267),'目标层对局放得分的敏感性',14,CORAL,bold=True)
    q.text((1327,380),'每通道一张32 × 32梯度图',13)
    q.text((1008,259),'逐位置特征与梯度相乘',8)
    q.text((267,407),'模块3的Grad-CAM',14,'#f34b33',bold=True)
    q.text((439,513),'双线性\n上采样',14,bold=True)
    q.text((614,578),'32 × 32类别图',12,bold=True)
    q.text((1018,499),'跨通道平均',14,'#ed872a',bold=True)
    # 只替换坐标轴外自然语言；数字、边框和色图像素保持原样。
    q.clear((146,83,156,146));q.vertical((146,83,155,146),'频率（MHz）',6,'#111111')
    q.clear((224,163,272,172));q.text((248,168),'时间（ms）',6,'#111111')
    for x in [895,973,1051]:
        q.clear((x,292,x+7,334));q.vertical((x,292,x+7,334),'特征图行',5,'#111111')
    for x in [937,1015,1092]:
        q.clear((x-23,361,x+23,369));q.text((x,365),'特征图列',5,'#111111')
    q.clear((1131,291,1138,347));q.vertical((1131,291,1138,347),'特征与梯度乘积',5,'#111111')
    q.clear((567,444,653,454));q.text((610,448),'上采样前的32 × 32类别图',5,'#111111')
    q.clear((545,480,552,516));q.vertical((545,480,552,516),'特征图行',5,'#111111')
    q.clear((586,552,632,560));q.text((609,556),'特征图列',5,'#111111')
    q.text((1120,659),'依据电缆项目原图3.9的中文翻译适配；仅作计算流程参考',24,bold=True)
    q.text((1120,707),'原实现：逐位置 A_k ⊙ G_k 后跨通道平均；与本项目标准Grad-CAM区分。',22)
    q.text((1120,753),'亮区表示对局放得分的响应证据，不表示物理因果关系。',22)
    q.evidence_regions=[(166,54,322,158),(376,70,484,164),(548,70,660,164),(730,70,839,164),
                        (594,289,836,383),(908,277,966,353),(985,277,1044,353),(1063,277,1121,353),
                        (1115,270,1122,361),(125,425,325,586),(561,455,656,548),(661,455,667,548)]
    return q

def cable_matrix(src):
    q=Layer(src);w,h=q.canvas.size
    q.clear((0,0,w-1,59))
    for x,t in zip([543,1345,2146,2948],['原始对数功率STFT','block3 Grad-CAM','block1 Grad-CAM','block2 Grad-CAM']):q.text((x,29),t,32, '#111111')
    # 状态名称与近似频率坐标的中文标签使用原轴外留白，不触及色图与刻度数字。
    rows=[(63,651),(682,1270),(1301,1889),(1920,2508),(2539,3127),(3158,3746),(3777,4365)]
    states=['背景 · C1_backgroundv2_43','电晕 · C1_corona_101','高压背景 · C1_hv_bg_95',
            '局部放电 · C2_pd_100','局放＋高压干扰 · C1_pd_HI_105','局放＋电晕 · C1_pd_corona_103',
            '局放＋电晕＋高压干扰 · C2_pd_corona_HI_92']
    for (a,b),t in zip(rows,states):
        q.clear((0,a,113,b))
        q.vertical((0,a,56,b),t,23,'#111111')
        q.vertical((57,a,113,b),'近似频率（MHz）',23,'#111111')
    q.clear((0,4413,w-1,h-1))
    for x in [543,1345,2146,2948]:q.text((x,4434),'近似时间（ms）',28,'#111111')
    # 下方独立增加来源身份，避免将译图误认为本项目结果。
    tail=Image.new('RGB',(w,h+102),'white');tail.paste(q.canvas,(0,0));q.canvas=tail
    q.text((w/2,h+47),'依据电缆论文图4.14翻译适配；七类测量状态原图，非手势实验结果',31,CORAL,bold=True)
    q.evidence_regions=[(x,a,x+735,b) for a,b in rows for x in [173,979,1781,2588]]
    return q

def imagenet_matrix(src, lang):
    q=Layer(src);w,h=q.canvas.size
    for b in [(0,0,w-1,331),(0,332,562,2600),(0,2836,w-1,h-1)]:q.clear(b)
    zh=lang=='zh'
    q.text((1275,76),'六类来源手图的分层Grad-CAM呈现' if zh else 'Layer-wise Grad-CAM layout for six source labels',54,bold=True)
    q.text((1275,160),'通用ImageNet预训练模型布局示例，非六类手势实测结果' if zh else 'Generic ImageNet layout example; not six-gesture experiment results',40,CORAL)
    columns=['输入手图\n224 × 224','layer4 · 主解释\n7 × 7','layer1\n56 × 56','layer2\n28 × 28','layer3\n14 × 14'] if zh else ['Input hand crop\n224 × 224','layer4 · main view\n7 × 7','layer1\n56 × 56','layer2\n28 × 28','layer3\n14 × 14']
    for x,t in zip([760,1150,1540,1930,2320],columns):q.text((x,278),t,42,bold=True)
    meta=json.loads((ROOT/'source_evidence/v9_cam_matrix_metadata.json').read_text(encoding='utf-8'))
    descriptors=['其他／闲置手','V形','张开手掌','握拳','拇指向上','拇指向下']
    english=['other / idle hand','V sign','open palm','fist','thumb up','thumb down']
    for i,r in enumerate(meta['records']):
        row=[f'来源类别{i}：{descriptors[i]}',r['label'],'ImageNet目标：',r['target_name'],f'索引 {r["target_index"]}'] if zh else [f'Source label {i}:',english[i],r['label'],'ImageNet target:',r['target_name'],f'index {r["target_index"]}']
        q.text((35,420+378*i),'\n'.join(row),35,NAVY,'lm')
    q.text((1425,2883),'各图独立归一化的相对响应' if zh else 'Relative response, normalized separately within each panel',35)
    q.text((1275,2932),'按HaGRID来源标签组织；各行解释所标ImageNet目标。上采样不增加空间分辨率。' if zh else 'Rows follow HaGRID source labels. CAMs explain the stated ImageNet targets.',31)
    if not zh:q.text((1275,2975),'Upsampling does not add spatial resolution.',28)
    q.evidence_regions=[(573,334,2506,2598),(750,2748,2110,2836)]
    return q

def local_evidence(src,lang):
    q=Layer(src);zh=lang=='zh'
    for b in [(0,0,1633,224),(415,295,583,375),(985,300,1135,380),(0,625,420,748),(400,869,1515,937),(0,951,1633,1017)]:q.clear(b)
    q.text((817,53),'原始照片、分类输入与局部响应对应关系' if zh else 'Original image, network input and local response',38,bold=True)
    q.text((817,124),'ImageNet目标：nail（钉子，索引677）；通用模型观察示例，非手势识别结果' if zh else 'ImageNet target: nail (index 677); generic-model example, not a gesture result',30,CORAL,bold=True)
    titles=['原始照片与人工手框','送入网络的手图','同一手图＋layer4响应'] if zh else ['Original and manual hand box','Hand crop fed to network','Same crop + layer4 response']
    for x,t in zip([260,797,1335],titles):q.text((x,195),t,28,bold=True)
    q.text((499,336),'按手框裁图\n等比补边' if zh else 'Crop to box\npad to square',23 if zh else 21)
    q.text((1061,341),'叠加响应' if zh else 'Overlay\nresponse',25)
    q.text((222,693),'红框为人工选定的\n形态观察区，非热区分割' if zh else 'Red boxes: manually chosen\nshape inspection regions;\nnot heatmap segmentation',27,CORAL,bold=True)
    q.text((651,899),'同一区域：手指轮廓与指间空隙' if zh else 'Same region: finger contour and gap',27,bold=True)
    q.text((1183,899),'同坐标响应：位置与扩散范围' if zh else 'Same coordinates: response spread',27,bold=True)
    q.text((817,983),'宽范围响应覆盖指根、掌部及部分背景；不能据此声称模型已识别V形手势。' if zh else 'Broad response includes finger bases, palm and background; it does not establish V-sign recognition.',26)
    q.evidence_regions=[(122,225,399,595),(614,225,982,595),(1151,225,1518,595),(538,615,762,844),(1065,615,1289,844)]
    return q

def verify_evidence(q):
    # 每个证据块逐像素比较；新增文字层不能覆盖图像、热图、色尺或ROI。
    checks=[]
    for box in q.evidence_regions:
        before=q.original.crop(box).tobytes();after=q.canvas.crop(box).tobytes()
        checks.append({'box_xyxy':list(box),'same_pixels':before==after,'sha256':hashlib.sha256(before).hexdigest()})
    assert all(v['same_pixels'] for v in checks),checks
    return checks

def main():
    config=json.loads((ROOT/'reference_inputs.json').read_text(encoding='utf-8'))
    entries=[]
    for f in config:
        f['caption']=CAPTIONS_ZH[f['id']]
        f['caption_en']=f['caption_en'].replace('0�C5','0–5').replace('project��s',"project's")
        fid=f['id'];src=ROOT/f['local_original'];stem=fid.replace('-','_')
        assert digest(src)==f['sha256'],fid
        zh=stem+'_zh';en=stem+'_en';base={'id':fid,'original_sha256':f['sha256'],'zh_png':zh+'.png','en_png':en+'.png','zh_svg':zh+'.svg','en_svg':en+'.svg','generator':Path(__file__).name,'required_data':[src.relative_to(ROOT).as_posix()],'captions_zh':f['caption'],'captions_en':f['caption_en'],'caption_zh':f['caption'],'caption_en':f['caption_en']}
        if int(fid[-2:])<=9:
            for v in [zh,en]:shutil.copyfile(src,ROOT/(v+'.png'))
            ed=[]
            for v in [zh,en]:ed.extend(Layer(src).save(v))
            # Restore exact PNG file bytes after creating SVG wrappers.
            for v in [zh,en]:shutil.copyfile(src,ROOT/(v+'.png'))
            base.update({'editable_sources':ed+[src.relative_to(ROOT).as_posix()],'samepixel':True,'editability':'Photographs have no internal language labels; SVG embeds the unchanged photograph. Photograph pixels are not editable vector geometry.','translation_checks':{'no_internal_text':True,'both_sha256_equal_original':digest(ROOT/(zh+'.png'))==digest(ROOT/(en+'.png'))==f['sha256']}})
        else:
            checks={};ed=[]
            if fid in ['D1-I20','D1-I22']:
                ed.extend(Layer(src).save(en));shutil.copyfile(src,ROOT/(en+'.png'))
                q=cable_mechanism(src) if fid=='D1-I20' else cable_matrix(src)
                ed.extend(q.save(zh));checks['zh_evidence_pixel_checks']=verify_evidence(q)
                checks['english_original_byte_preserved']=digest(ROOT/(en+'.png'))==f['sha256']
                base['translation_adaptation']=True
            else:
                for lang,v in [('zh',zh),('en',en)]:
                    q=imagenet_matrix(src,lang) if fid=='D1-I23' else local_evidence(src,lang)
                    ed.extend(q.save(v));checks[lang+'_evidence_pixel_checks']=verify_evidence(q)
                    if lang=='en':assert not any('\u4e00'<=c<='\u9fff' for t in q.texts for c in t)
                base['required_data'] += ['source_evidence/'+v for v in f.get('required_evidence',[])]
            base['editable_sources']=ed
            base['editability']='Translation labels and presentation overlays are editable SVG text over unchanged scientific raster evidence. English external originals are raster-only SVG wrappers; native vector source was not found.' if fid in ['D1-I20','D1-I22'] else 'SVG text and annotation overlays are editable; scientific photos, response maps and numeric color scale remain saved original raster evidence.'
            if fid in ['D1-I23','D1-I24']:
                native='v9_cam_matrix.vsdx' if fid=='D1-I23' else 'v7_fig12_local_evidence.vsdx'
                base['native_source_provenance']=['source_evidence/'+native]
                base['required_data'].append('source_evidence/'+native)
                base['native_source_note']='Historical original source retained for provenance; it is not the final bilingual adaptation.'
            base['translation_checks']=checks
        base['output_sha256']={Path(base[k]).name:digest(ROOT/base[k]) for k in ['zh_png','en_png','zh_svg','en_svg']}
        entries.append(base)
    (ROOT/'manifest.json').write_text(json.dumps({'figures':entries,'limitations':'External originals retain original bytes. SVG files are editable translation/presentation adaptations over original evidence rasters, not original author Visio files. No model execution; historical sources unchanged.','qa':'Evidence raster regions passed exact pixel equality. Independent review is required before integrated delivery.','figure_count':len(entries),'raster_count':len(entries)*2,'svg_count':len(entries)*2,'scientific_model_execution':False},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Created',len(entries),'bilingual figure entries; evidence pixels verified.')

if __name__=='__main__':main()
