"""对三个正式Word作定向OOXML补丁，保留原正文、表格及历史源文件。"""
from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile, ZIP_DEFLATED
import argparse, hashlib, json, re
from urllib.parse import quote
from lxml import etree as E
from PIL import Image

STAGE=Path(__file__).resolve().parent
OUT=STAGE/'documents'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'pr':'http://schemas.openxmlformats.org/package/2006/relationships'}
OUTPUTS=['基于ResNet18的手势识别项目方案_v13_中英文配图.docx',
         'A与B协议ResNet训练与对比实验报告_v4_中英文配图.docx',
         '交接说明_v2_中英文配图.docx']
PHOTO_LABELS={
 'D1-I01':'peace + peace_inverted\nV sign with palm-facing and back-of-hand poses',
 'D1-I03':'stop + stop_inverted + palm\nThree open-palm poses',
 'D1-I06':'fist: closed fist',
 'D1-I07':'like: thumb up',
 'D1-I08':'dislike: thumb down',
 'D1-I09':'Other raw labels and no_gesture; no hand or unreliable results also return 0'}
EXPLANATION=[
 '这里的整图准确率，是让程序读入一张完整照片，经过找手、裁图、分类、置信度筛选和多手汇总后，检查最后那个0—5数字是否符合该任务的整图正确答案。B测试集有3,102张完整照片，每张只计一次。人工框手图准确率约99%，则是提前给定正确手框后评价单手分类，不能直接当作完整照片识别成功率；整图数字正确也不表示照片中的每只手都已找到。',
 '以B ResNet18为例，不合并版在3,102张照片中正确2,887张，整图准确率93.07%；合并版正确2,849张，为91.84%，少38张。可把这38张分开看：336张two_up与two_up_inverted照片的正确数字由0改为1；旧版在这336张均输出0，按旧任务全部算对，新版有307张正确输出1，其余29张仍输出0。其中19张没有检测框，另10张虽有框，但没有被接收的有效非0指令；这10张并非多手指令冲突。',
 '另外2,766张照片的新旧正确数字不变，正确数仍由2,551降为2,542，净少9张，其中21张由原来正确变成错误，12张由原来错误变成正确。相同的1,359张共同负类照片中，误发运动指令也由5张增加到16张，因此不能把全部变化都解释为计分口径改变。两版使用相同的检测器权重，检测阈值与分类阈值也都为0.3；这些配对结果描述了最终数字的变化，但不足以把变化唯一归因于定位、分类或汇总中的某一个环节，也不能据此笼统断言模型整体变差。']

def q(prefix,name):return '{'+NS[prefix]+'}'+name
def text(p):return ''.join(p.xpath('.//w:t/text()',namespaces=NS))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def xml(root):return E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)

def pprop(p,name,val):
    pp=p.find('w:pPr',NS)
    if pp is None:pp=E.Element(q('w','pPr'));p.insert(0,pp)
    el=pp.find('w:'+name,NS)
    if el is None:el=E.SubElement(pp,q('w',name))
    el.set(q('w','val'),val)

def clean_clone(p):
    for key in list(p.attrib):
        if 'paraId' in key or 'textId' in key:del p.attrib[key]
    return p

def paragraph_like(template,words,english=False):
    p=E.Element(q('w','p'),nsmap=template.nsmap)
    pp=template.find('w:pPr',NS)
    if pp is not None:p.append(deepcopy(pp))
    r=E.SubElement(p,q('w','r'))
    rp=template.find('.//w:rPr',NS)
    if rp is not None:r.append(deepcopy(rp))
    else:rp=E.SubElement(r,q('w','rPr'))
    if english:
        rp=r.find('w:rPr',NS)
        fonts=rp.find('w:rFonts',NS)
        if fonts is None:fonts=E.SubElement(rp,q('w','rFonts'))
        for k in ['ascii','hAnsi','cs']:fonts.set(q('w',k),'Arial')
        lang=rp.find('w:lang',NS)
        if lang is None:lang=E.SubElement(rp,q('w','lang'))
        lang.set(q('w','val'),'en-GB')
    for i,line in enumerate(words.split('\n')):
        if i:E.SubElement(r,q('w','br'))
        E.SubElement(r,q('w','t')).text=line
    pprop(p,'keepNext','0')
    return p

def replace_once_preserving_runs(p,old,new):
    before=text(p);assert before.count(old)==1
    start=before.index(old);end=start+len(old);offset=0;inserted=False
    for node in p.findall('.//w:t',NS):
        value=node.text or '';stop=offset+len(value)
        if offset<end and stop>start:
            left=value[:max(0,start-offset)]
            right=value[max(0,end-offset):] if stop>end else ''
            node.text=left+('' if inserted else new)+right;inserted=True
        offset=stop
    assert text(p)==before.replace(old,new,1)

def translated_assets():
    mapping={}
    for branch in ['assets_results','assets_diagrams','assets_reference']:
        path=STAGE/branch/'manifest.json'
        if not path.exists():raise RuntimeError(f'等待完整图源清单 {branch}')
        raw=read(path)
        items=raw if isinstance(raw,list) else raw.get('items',raw.get('figures',[]))
        if not items:raise RuntimeError(f'图源清单为空 {branch}')
        for item in items:
            item=deepcopy(item);item['_branch']=branch
            item['_zh']=STAGE/branch/item['zh_png'];item['_en']=STAGE/branch/item['en_png']
            assert item['_zh'].is_file() and item['_en'].is_file()
            ids=list(item.get('caption_translations',{})) or [item['id']]
            for fid in ids:
                m=deepcopy(item)
                cap=item.get('caption_translations',{}).get(fid)
                if isinstance(cap,dict):m['_caption_en']=cap['en']
                elif isinstance(cap,str):m['_caption_en']=cap
                else:m['_caption_en']=item.get('caption_en',item.get('captions_en',''))
                mapping[fid]=m
    return mapping

def update_handover(root):
    ps=root.findall('.//w:p',NS)
    changes={
      1:'ResNet识别分支由我和组员共同负责。现保留不合并实验，并在相同固定A照片划分、B人员划分上另训合并标签的六个分类器，共十二个自带权重的CPU识别程序。当前阅读方案v13与综合报告v4的中英文配图版，两种任务的科学结果与原方案v12、报告v3一致；原稿和更早历史版本保留。组员可先接收复核、主笔ResNet汇报，再与另一识别子组和控制组继续比较及集成。',
      132:'当前方案v13、报告v4中英文配图版；十二模型共同输入CPU测速与旧计时记录均保留',
      295:'第一次接手先读本交接v2，再读《A与B协议ResNet训练与对比实验报告_v4_中英文配图.docx》和《基于ResNet18的手势识别项目方案_v13_中英文配图.docx》，不用从v1逐版读。报告给两任务真实结果、参数和错误，方案保留完整系统原理及原图；每张图均在正文同时给出中文和英文版本。',
      300:'本交接v2与根README',303:'综合实验报告v4 中英文配图版',306:'项目方案v13 中英文配图版',
      310:'旧编码辅助理解张掌2和握拳3；two_up新版以方案v13和报告v4为准',
      309:'原完整项目包内：历史与正式文档/手势合并与数字编码.docx',
      312:'原完整项目包内：程序/不合并/或程序/合并/下A或B及相应架构README',
      315:'原完整项目包内：完整源码与数据/Vision_CA/README.md',
      317:'方案v12和报告v3是本轮双语配图修订之前的正式原稿，科学实验身份保留；方案v11、报告v2保留不合并已完成记录。方案v1—v10、A报告v1、总体理解v1和问题解答用于历史追溯。旧稿“B未分组”“六分类未训练”属于当时状态，当前阅读方案v13、报告v4和对应配置；旧“读图失败返回0”也已由CLI异常退出契约替代。历史修改意见不自动成为新待办。',
      332:'6 原完整项目包与本轮双语图源包',
      335:'├─ 历史与正式文档/         原方案v12、报告v3、v11/v2及必要历史/课程',
      343:'接手时先完整解压ZIP，再按上述布局核对文件。根SHA清单记录成员和校验值，包外核验记录给出最终检查结果；检查通过后，原图、分组、模型和日志才作为同一套交接材料使用。原大型项目ZIP仍保存于E:/ResNet_合并与不合并完整交接_20261103.zip；它是本轮双语图源修订之前的完整科学项目交接包，保留原身份。当前双语图源入口为E:/ResNet_当前正式文档与双语图源_20261103.zip，提供当前三份正式Word、每图中文与英文图片、可编辑图源、绘图脚本、重绘所需固定输入与图号对应清单，不替代原大型包中的数据、模型和训练证据。',
      363:'我和组员；v4两任务报告及目标设备'}
    for i,new in changes.items():
        old=ps[i];replacement=paragraph_like(old,new)
        if i in (303,306):
            rid={303:'rId10',306:'rId11'}[i]
            link=E.SubElement(replacement,q('w','hyperlink'));link.set(q('r','id'),rid)
            for run in list(replacement.findall('w:r',NS)):link.append(run)
        old.getparent().replace(old,replacement)
    # 一句原文提及当前报告的位置只作局部版本替换。
    p=ps[202]
    for t in p.findall('.//w:t',NS):
        if t.text:t.text=t.text.replace('报告v3另保留','报告v4另保留')
    note=paragraph_like(ps[295],'本轮双语图源ZIP中的正式文件为：基于ResNet18的手势识别项目方案_v13_中英文配图.docx、A与B协议ResNet训练与对比实验报告_v4_中英文配图.docx、交接说明_v2_中英文配图.docx。中文与英文图内文字逐图对应，原实验数据、样本、数值和结论不因翻译而改变；不以旧版页数描述本轮重新分页的文档。')
    anchor=root.findall('.//w:p',NS)[295];anchor.addnext(note)
    return changes

def build_document(docindex,inv,assets):
    source=Path(inv['documents'][docindex]['path'])
    assert sha(source)==inv['documents'][docindex]['sha256'],'正式源文件身份已变化'
    with ZipFile(source) as z:
        payload={n:z.read(n) for n in z.namelist()}
    root=E.fromstring(payload['word/document.xml'])
    rels=E.fromstring(payload['word/_rels/document.xml.rels'])
    ps=root.findall('.//w:p',NS)
    original_text=[text(p) for p in ps]
    plan_version_changes=[]
    if docindex==0:
        for pos,old,new in [(556,'报告v3','报告v4'),(798,'先读最新报告v3与本方案v12','先读最新报告v4与本方案v13')]:
            before=text(ps[pos]);replace_once_preserving_runs(ps[pos],old,new)
            original_text[pos]=text(ps[pos])
            plan_version_changes.append({'source_paragraph_index':pos,'before':before,'after':text(ps[pos])})
    rel_next=max([int(r.get('Id')[3:]) for r in rels if re.fullmatch('rId[0-9]+',r.get('Id',''))]+[0])+1
    draw_next=max([int(d.get('id')) for d in root.findall('.//wp:docPr',NS)]+[0])+1
    records=[]
    if docindex==2:
        handover_changes=update_handover(root)
        for rel in list(rels):
            if rel.get('Id') in ('rId10','rId11'):
                rel.set('Target',quote(OUTPUTS[1 if rel.get('Id')=='rId10' else 0]))
            elif rel.get('Id') in ('rId12','rId13'):
                rels.remove(rel)
    else:
        figures=[f for f in inv['figures'] if f['document']==str(source)]
        grouped={}
        for f in figures:grouped.setdefault(f['paragraph_index'],[]).append(f)
        def embed_image(drawing,f,lang):
            nonlocal rel_next,draw_next
            item=assets[f['id']];path=item['_'+lang]
            media=f'media/bilingual_{f["id"]}_{lang}.png';rid=f'rId{rel_next}';rel_next+=1
            E.SubElement(rels,q('pr','Relationship'),Id=rid,Type=NS['r']+'/image',Target=media)
            payload['word/'+media]=path.read_bytes()
            blip=drawing.find('.//a:blip',NS);blip.set(q('r','embed'),rid)
            ex=drawing.find('.//wp:extent',NS)
            width=int(ex.get('cx'));imwidth,imheight=Image.open(path).size
            height=round(width*imheight/imwidth)
            # 正文竖图预留完整图注空间，按比例缩小，避免高于可打印页面。
            max_height=round(7.75*914400)
            if height>max_height:
                width=round(width*max_height/height);height=max_height
            ex.set('cx',str(width))
            ex.set('cy',str(height))
            for ex2 in drawing.findall('.//a:xfrm/a:ext',NS):ex2.set('cx',str(width));ex2.set('cy',str(height))
            dp=drawing.find('.//wp:docPr',NS)
            if lang=='en':dp.set('id',str(draw_next));draw_next+=1
            dp.set('name','English figure' if lang=='en' else '中文图')
            dp.set('descr',item['_caption_en'] if lang=='en' else f['caption'])
            return {'id':f['id'],'language':lang,'relationship':rid,'source_png':str(path.relative_to(STAGE)).replace('\\','/'),'sha256':sha(path),'width_emu':width,'height_emu':height}
        for pos,fs in grouped.items():
            p=ps[pos];drawings=p.findall('.//w:drawing',NS)
            assert len(drawings)==len(fs)
            en_p=clean_clone(deepcopy(p))
            if E.QName(p.getparent()).localname=='tc':
                # 同一单元格内的两图或三图保持原顺序和同段布局。
                en_p=paragraph_like(p,PHOTO_LABELS[fs[0]['id']],english=True)
                for drawing,f in zip(drawings,fs):
                    d=deepcopy(drawing);r=E.SubElement(en_p,q('w','r'));r.append(d)
                    records.append(embed_image(d,f,'en'))
                    spacer=E.SubElement(E.SubElement(en_p,q('w','r')),q('w','t'));spacer.set('{http://www.w3.org/XML/1998/namespace}space','preserve');spacer.text='  '
                for drawing,f in zip(drawings,fs):records.append(embed_image(drawing,f,'zh'))
                pprop(p,'keepNext','0');pprop(en_p,'keepNext','0');p.addnext(en_p)
            else:
                assert len(fs)==1
                f=fs[0];caption=p.getnext()
                assert caption.tag==q('w','p') and text(caption)==f['caption'],f'原图注不对应 {f["id"]}'
                records.append(embed_image(drawings[0],f,'zh'))
                records.append(embed_image(en_p.find('.//w:drawing',NS),f,'en'))
                pprop(p,'keepNext','1');pprop(caption,'keepNext','0');pprop(en_p,'keepNext','1')
                en_caption=paragraph_like(caption,assets[f['id']]['_caption_en'],english=True)
                pprop(caption,'keepLines','1');pprop(en_caption,'keepLines','1')
                caption.addnext(en_p);en_p.addnext(en_caption)
        if docindex==1:
            anchor=ps[1706]
            for words in EXPLANATION:
                new=paragraph_like(anchor,words);anchor.addnext(new);anchor=new
    payload['word/document.xml']=xml(root)
    payload['word/_rels/document.xml.rels']=xml(rels)
    settings=E.fromstring(payload['word/settings.xml'])
    upd=settings.find('w:updateFields',NS)
    if upd is None:upd=E.SubElement(settings,q('w','updateFields'))
    upd.set(q('w','val'),'true');payload['word/settings.xml']=xml(settings)
    # 删除旧缓存页数，避免旧页数冒充新版分页结果。
    if 'docProps/app.xml' in payload:
        app=E.fromstring(payload['docProps/app.xml'])
        for node in list(app):
            if E.QName(node).localname in ['Pages','Words','Characters','CharactersWithSpaces','Paragraphs','Lines']:app.remove(node)
        payload['docProps/app.xml']=xml(app)
    target=OUT/OUTPUTS[docindex]
    with ZipFile(target,'w',ZIP_DEFLATED) as z:
        for n,data in payload.items():z.writestr(n,data)
    # 逐段验证未删改原正文与表格，交接仅允许明确版本说明变更。
    if docindex!=2:
        after=[text(p) for p in root.findall('.//w:p',NS)]
        cursor=iter(after)
        assert all(any(v==t for v in cursor) for t in original_text),'原正文段落丢失或改写'
    expected=[58,30,0][docindex]
    assert len(root.findall('.//a:blip',NS))==expected
    assert sha(source)==inv['documents'][docindex]['sha256']
    return {'output':target.name,'source':source.name,'source_sha256':sha(source),'output_sha256':sha(target),
            'image_occurrences':expected,'image_bindings':records,'original_body_and_tables_preserved':docindex!=2,
            'handover_targeted_version_changes':list(handover_changes) if docindex==2 else [],
            'plan_targeted_version_changes':plan_version_changes,
            'page_counts':'pending official render','report_new_explanation':EXPLANATION if docindex==1 else []}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--document-index',type=int,choices=range(3))
    args=parser.parse_args()
    assets=translated_assets();inv=read(STAGE/'figure_inventory.json')
    evidence=read(STAGE/'analysis'/'whole_command_accuracy_B18.json')
    assert evidence['status']=='RECOMPUTED_FROM_FROZEN_PREDICTIONS'
    assert [(x['n'],x['old_correct'],x['new_correct']) for x in evidence['groups']]==[(336,336,307),(2766,2551,2542),(3102,2887,2849)]
    assert (evidence['newly_required_failed'],evidence['no_detected_boxes'],evidence['boxes_without_accepted_nonzero_command'])==(29,19,10)
    assert (evidence['unchanged_old_correct_new_wrong'],evidence['unchanged_old_wrong_new_correct'])==(21,12)
    required={f['id'] for f in inv['figures']}
    assert required<=assets.keys(),f'图清单缺失 {sorted(required-assets.keys())}'
    assert len(required)==44
    for f in inv['figures']:
        assert assets[f['id']]['_caption_en'],f'缺英文图注 {f["id"]}'
    OUT.mkdir(exist_ok=True)
    if args.document_index is None:
        result=[build_document(i,inv,assets) for i in range(3)]
    else:
        result=read(OUT/'document_build_manifest.json')
        untouched={i:sha(OUT/OUTPUTS[i]) for i in range(3) if i!=args.document_index}
        result[args.document_index]=build_document(args.document_index,inv,assets)
        assert all(sha(OUT/OUTPUTS[i])==value==result[i]['output_sha256'] for i,value in untouched.items()),'非选中文档字节变化'
    (OUT/'document_build_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k!='image_bindings'} for r in result],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
