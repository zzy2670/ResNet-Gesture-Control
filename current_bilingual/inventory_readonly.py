from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
from xml.etree import ElementTree as E
from io import BytesIO
import json, os, re
from PIL import Image, ImageOps, ImageDraw

P=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
docs=[P/'基于ResNet18的手势识别项目方案_v12.docx',P/'A与B协议ResNet训练与对比实验报告_v3.docx',P/'ResNet_合并与不合并完整交接_20261103/交接说明.docx']
records=[]; documents=[]; unique={}
for di,p in enumerate(docs):
    with ZipFile(p) as z:
        rel={e.get('Id'):dict(e.attrib) for e in E.fromstring(z.read('word/_rels/document.xml.rels'))}
        root=E.fromstring(z.read('word/document.xml'))
        paras=list(root.iter('{'+NS['w']+'}p'))
        texts=[''.join(x.itertext()) if False else ''.join(t.text or '' for t in x.findall('.//w:t',NS)) for x in paras]
        figs=[]
        for pi,pa in enumerate(paras):
            for blip in pa.findall('.//a:blip',NS):
                rid=blip.get('{'+NS['r']+'}embed'); rr=rel.get(rid,{})
                target=rr.get('Target',''); name='word/'+target
                if not name or name not in z.namelist(): continue
                b=z.read(name); h=sha256(b).hexdigest()
                ext=Path(name).suffix
                dest=OUT/'embedded_originals'/(h+ext)
                dest.parent.mkdir(exist_ok=True)
                if not dest.exists(): dest.write_bytes(b)
                im=Image.open(BytesIO(b)); seq=len(figs)+1
                near=texts[pi+1:pi+5]
                caption=next((s for s in near if re.match(r'^图\s*\d',s)), '')
                if not caption: caption=next((s for s in near if s),'')
                dp=pa.find('.//wp:docPr',NS)
                rec={'id':f'D{di+1}-I{seq:02d}','document':str(p),'sequence':seq,'paragraph_index':pi,'media':name,'relationship':rid,'sha256':h,'size':list(im.size),'caption':caption,'context_before':texts[max(0,pi-2):pi],'context_after':near,'docpr':dict(dp.attrib) if dp is not None else {},'original_extract':str(dest),'exact_source_files':[],'generator_candidates':[]}
                figs.append(rec); records.append(rec); unique.setdefault(h,[]).append(rec['id'])
        documents.append({'path':str(p),'sha256':sha256(p.read_bytes()).hexdigest(),'visible_images':len(figs),'media_parts':len([n for n in z.namelist() if n.startswith('word/media/')]),'all_relationships':rel})
prune={'HaGRID','data','crops','train','val','test','images','dataset','.git','__pycache__','node_modules','embedded_originals','双语图与图源交付_20261103','ResNet双视频学习'}
files=[]; scripts=[]; editable=[]; evidence=[]
for base,dirs,names in os.walk(P,followlinks=False):
    dirs[:]=[d for d in dirs if d not in prune and not os.path.islink(Path(base)/d) and not any(x in d for x in ['排版','渲染','预览','终审']) and 'render' not in d.lower() and d not in {'audio','chunk_01','chunk_02','chunk_03','logs','language_check'}]
    for n in names:
        p=Path(base)/n; e=p.suffix.lower()
        if e in {'.png','.jpg','.jpeg','.svg','.drawio','.vsdx','.py','.ps1','.json','.csv','.npz','.npy'}:
            if re.match(r'page[-_]\d+',n) or p.stat().st_size>100_000_000:continue
            files.append(p)
            if e in {'.py','.ps1'}:scripts.append(p)
            if e in {'.vsdx','.drawio','.svg'}:editable.append(str(p))
            if e in {'.png','.jpg','.jpeg'}:
                h=sha256(p.read_bytes()).hexdigest()
                for rec in records:
                    if rec['sha256']==h:rec['exact_source_files'].append(str(p))
script_texts={p:p.read_text(encoding='utf-8',errors='replace') for p in scripts}
for rec in records:
    basenames={Path(x).name for x in rec['exact_source_files']}
    for p in scripts:
        s=script_texts[p]
        if basenames and any(n in s for n in basenames): rec['generator_candidates'].append(str(p))
summary={'scope':'Current three formal documents only. Read-only source discovery excludes HaGRID, all data directories and page-render QA directories. No training/inference or existing file changes.','documents':documents,'figures':records,'unique_image_sha256_groups':unique,'candidate_editable_sources':editable,'scanned_candidate_files':len(files),'script_files':[str(p) for p in scripts]}
(OUT/'figure_inventory.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
# 联系表仅用于检查真实图内文字，不能替代正式文档。
for di in range(1,3):
    group=[r for r in records if r['id'].startswith(f'D{di}-')]
    for k in range(0,len(group),6):
        batch=group[k:k+6]; sheet=Image.new('RGB',(1800,((len(batch)+1)//2)*580),'#eee'); dr=ImageDraw.Draw(sheet)
        for j,r in enumerate(batch):
            im=Image.open(r['original_extract']).convert('RGB'); im.thumbnail((880,535))
            x=(j%2)*900; y=(j//2)*580; sheet.paste(im,(x+(900-im.width)//2,y+28));dr.text((x+10,y+8),r['id']+' '+r['media'],fill='black')
        sheet.save(OUT/f'contact_D{di}_{k//6+1}.jpg',quality=90)
print(json.dumps({'documents':[{k:v for k,v in d.items() if k!='all_relationships'} for d in documents],'unique':len(unique),'scanned_files':len(files),'editable_count':len(editable),'figures':[{'id':r['id'],'caption':r['caption'],'sources':r['exact_source_files'],'generators':r['generator_candidates']} for r in records]},ensure_ascii=False,indent=2))
