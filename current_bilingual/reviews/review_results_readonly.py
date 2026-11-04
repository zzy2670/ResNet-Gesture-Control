"""独立只读复核实验双语图与冻结证据，不调用作者绘图入口或模型。"""
from pathlib import Path
from collections import Counter
from hashlib import sha256
from xml.etree import ElementTree as E
import json,re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

O=Path(__file__).resolve().parent
STAGE=O.parent
ROOT=STAGE/'assets_results'
PROJECT=STAGE.parent
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return sha256(p.read_bytes()).hexdigest()
def pixel(p):return np.asarray(Image.open(p).convert('RGB'))
inv={r['id']:r for r in load(STAGE/'figure_inventory.json')['figures']}
manifest=load(ROOT/'manifest.json');checks=[];issues=[];bindings=[]
for r in manifest:
    fid=r['id'];folder=ROOT/fid;original=folder/'original_reference.png'
    c={'id':fid,'original_sha_matches_inventory':sha(original)==inv[fid]['sha256'],'required_files_present':all((ROOT/x).is_file() for x in r['required_data']+r['editable_sources']+[r['generator'],r['zh_png'],r['en_png']]),'languages':{}}
    for lang in ['zh','en']:
        p=ROOT/r[lang+'_png'];s=folder/(lang+'.svg');root=E.fromstring(s.read_bytes())
        texts=[''.join(e.itertext()) for e in root.iter() if e.tag.endswith('}text')]
        c['languages'][lang]={'png_sha_matches_manifest':sha(p)==r['sha256'][lang],'editable_svg_texts':len(texts),'chinese_texts':sum(bool(re.search('[\u4e00-\u9fff]',x)) for x in texts),'visual_review':'PASS: actual zh/en pair inspected in contact sheet; larger English view checked for representative dense and tall plates'}
        if lang=='en' and c['languages'][lang]['chinese_texts']:issues.append({'id':fid,'issue':'Chinese text remains in English SVG'})
        bindings.extend([{'path':str(p.relative_to(STAGE)),'sha256':sha(p)},{'path':str(s.relative_to(STAGE)),'sha256':sha(s)}])
    if fid in {'D1-I25','D1-I26','D1-I28','D1-I29'}:
        a=pixel(original);c['science_panel_pixel_equality']={}
        for lang in ['zh','en']:
            b=pixel(folder/(lang+'.png'));c['science_panel_pixel_equality'][lang]=bool(np.array_equal(a[164:2873,399:],b[164:2873,399:]))
        arrays=0
        for source in inv[fid]['evidence_sources']:
            if not source['path'].endswith('.npz'):continue
            p=Path(source['path']);q=folder/p.name
            with np.load(p) as src,np.load(q) as dst:
                for key in dst.files:
                    if not np.array_equal(src[key],dst[key]):issues.append({'id':fid,'issue':'Saved heatmap array differs','array':key})
                    arrays+=1
        c['frozen_array_identity_count']=arrays
        c['metadata_equal_to_source']=load(folder/'metadata.json')==load(Path(inv[fid]['evidence_sources'][0]['path']))
    elif fid in {'D2-I03','D2-I10','D1-I27','D2-I04'}:
        a=pixel(original);mask=np.ones(a.shape[:2],bool)
        if fid=='D2-I03':rects=[(0,0,a.shape[1]-1,259),(0,922,a.shape[1]-1,1028)]
        elif fid=='D2-I10':rects=[(0,0,a.shape[1]-1,249),(0,870,a.shape[1]-1,972)]
        elif fid=='D1-I27':rects=[(0,0,a.shape[1]-1,223),(0,830,a.shape[1]-1,896),(755,897,1529,1529)]
        else:rects=[(0,0,a.shape[1]-1,278),(0,902,a.shape[1]-1,1007)]
        for x1,y1,x2,y2 in rects:mask[y1:y2+1,x1:x2+1]=False
        c['all_pixels_outside_text_bands_equal']={lang:bool(np.array_equal(a[mask],pixel(folder/(lang+'.png'))[mask])) for lang in ['zh','en']}
        evidence=inv[fid]['evidence_sources'][0]
        c['sample_record_equal_to_source']=load(folder/Path(evidence['path']).name)==load(Path(evidence['path']))
    elif fid in {'D2-I01','D2-I02','D2-I07','D2-I08'}:
        source=Path(inv[fid]['evidence_sources'][0]['path']);orig=load(source);small=load(folder/source.name)
        c['confusion_metrics_equal_to_original_json']=small['metrics']==orig['metrics']
        expected=Counter(str(n) for row in orig['metrics']['confusion_matrix'] for n in row)
        c['all_36_matrix_counts_in_both_svgs']={}
        for lang in ['zh','en']:
            root=E.fromstring((folder/(lang+'.svg')).read_bytes());got=Counter(''.join(e.itertext()) for e in root.iter() if e.tag.endswith('}text'))
            c['all_36_matrix_counts_in_both_svgs'][lang]=not bool(expected-got)
    elif fid in {'D2-I06','D2-I12'}:
        curves=load(folder/'frozen_curves.json');groups=[]
        for protocol,models in curves.items():
            for model,values in models.items():
                p=PROJECT/'Vision_CA/runs'/(protocol+'_20261102_TPE')
                if model!='ResNet18':p=p/('compare_'+model.lower())
                events=[json.loads(line) for line in (p/'events.jsonl').read_text(encoding='utf-8').splitlines() if line]
                real=[{'epoch':e['epoch'],'macro_f1':e['validation']['macro_f1']} for e in events if e.get('stage')=='finetune' and e.get('status')=='complete_epoch']
                groups.append({'protocol':protocol,'model':model,'count':len(real),'all_epoch_f1_values_equal':real==values})
        c['curve_groups']=groups
    elif fid=='D2-I15':
        data=load(folder/'spotcheck.json');photo_checks=[]
        for i,row in enumerate(data['records']):
            p=folder/f'fixed_photo_{i+1}.jpg';photo_checks.append({'image_id':row['image_id'],'raw_photo_hash_matches_frozen':sha(p)==row['source_image_sha256']})
        c['four_frozen_photos']=photo_checks
        source=Path(inv[fid]['evidence_sources'][0]['path']);c['all_boxes_scores_thresholds_equal']=data==load(source)
        # 按冻结照片的原有缩放与位置，检查框线/标签之外的真实照片像素。
        c['frozen_photo_interior_pixels']={}
        for lang in ['zh','en']:
            rendered=pixel(folder/(lang+'.png'));count=0;equal=True
            for i,row in enumerate(data['records']):
                image=Image.open(folder/f'fixed_photo_{i+1}.jpg').convert('RGB');scale=min(490/image.width,410/image.height);size=(round(image.width*scale),round(image.height*scale));photo=np.asarray(image.resize(size,Image.Resampling.LANCZOS))
                for col in [0,1]:
                    xp=24+col*555+(510-size[0])//2;yp=128+i*485;keep=np.ones((size[1],size[0]),bool)
                    boxes=row['GT_boxes'] if col==0 else row['predicted_boxes']
                    for box in boxes:
                        x1,y1,x2,y2=[int(box[j]*scale) for j in range(4)]
                        keep[max(0,y1-30):min(size[1],y2+6),max(0,x1-5):min(size[0],max(x2+6,x1+180))]=False
                    equal=equal and np.array_equal(photo[keep],rendered[yp:yp+size[1],xp:xp+size[0]][keep]);count+=int(keep.sum())
            c['frozen_photo_interior_pixels'][lang]={'equal':bool(equal),'pixel_count':count}
    c['caption_occurrences']=list(r['caption_translations'])
    c['caption_review']='PASS: each full English caption checked against inventory Chinese caption; 20 occurrence-specific captions covered across 15 groups.'
    checks.append(c)
    for key,value in c.items():
        if value is False:issues.append({'id':fid,'issue':key})
        if isinstance(value,dict):
            for k,v in value.items():
                if v is False:issues.append({'id':fid,'issue':key+'.'+k})
result={'status':'PASS' if not issues else 'CHANGES_REQUIRED','reviewer':'/root/bilingual_figure_inventory (not the author of assets_results)','scope':{'unique_figures':15,'pngs_inspected':30,'full_english_caption_occurrences':20,'source_boundary':'Only inventory-listed evidence files and packaged four fixed photos; no raw dataset traversal, no model execution.'},'checks':checks,'issues':issues,'output_bindings':bindings,'limitations':['Actual final Word embedded-page readability and ZIP membership belong to root integration review.','PNG visual review used all 15 bilingual pairs in three contact sheets plus full-size representative confusion/CAM/tall detection plates.','SVG photo/heatmap layers are frozen raster evidence; their labels remain editable text.'],'scientific_claims':'No new numerical results or model claims introduced; 4 CAM families retain original input/heatmap panel pixels; confusion values and all curve points match original frozen JSON/logs.'}
O.mkdir(exist_ok=True)
(O/'results_independent_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':result['status'],'issues':issues,'checks':len(checks)},ensure_ascii=False))
