"""只读核对参考图最终清单；必须显式绑定作者声明的最终manifest哈希。"""
from pathlib import Path
from hashlib import sha256
from xml.etree import ElementTree as E
import argparse,json,re
import numpy as np
from PIL import Image

O=Path(__file__).resolve().parent
STAGE=O.parent
ROOT=STAGE/'assets_reference'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return sha256(p.read_bytes()).hexdigest()
def path(p):
    x=Path(p)
    return x if x.is_absolute() else ROOT/x
def pixel(p):return np.asarray(Image.open(p).convert('RGB'))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--expected-manifest-sha256',required=True);args=parser.parse_args()
    mp=ROOT/'manifest.json'
    if sha(mp)!=args.expected_manifest_sha256:raise ValueError('作者最终清单身份不匹配，不能预判通过')
    d=load(mp);entries=d.get('figures',d.get('items',[]));inv={r['id']:r for r in load(STAGE/'figure_inventory.json')['figures']}
    expected={f'D1-I{i:02d}' for i in range(1,10)}|{'D1-I20','D1-I22','D1-I23','D1-I24'}
    if {r['id'] for r in entries}!=expected:raise ValueError('参考图身份集合不匹配')
    checks=[];issues=[];bindings=[]
    for r in entries:
        fid=r['id'];zh=path(r['zh_png']);en=path(r['en_png'])
        orig=ROOT/'originals'/(fid.replace('-','_')+'_original.png')
        c={'id':fid,'original_sha_matches_inventory':sha(orig)==inv[fid]['sha256'],'required_files_present':all(path(p).is_file() for p in r.get('required_data',[])+r.get('editable_sources',[])+[r['generator'],r['zh_png'],r['en_png']]),'caption_en':r.get('caption_en',r.get('captions_en'))}
        c['english_caption_present']=bool(c['caption_en'])
        for p in [zh,en]+[path(p) for p in r['editable_sources']]:bindings.append({'path':str(p.relative_to(STAGE)),'sha256':sha(p)})
        if int(fid[-2:])<=9:
            c['both_photo_bytes_equal_original']=sha(zh)==sha(en)==sha(orig)
        else:
            tc=r.get('translation_checks',{})
            if fid in {'D1-I20','D1-I22'}:
                c['english_original_byte_preserved']=sha(en)==sha(orig)
            # 对作者声明的每个证据矩形，重新从原始图和实际最终PNG逐像素独立比较。
            actual_checks={}
            for key,values in tc.items():
                if 'evidence_pixel_checks' not in key:continue
                lang='zh' if key.startswith('zh') else 'en';a=Image.open(orig).convert('RGB');b=Image.open(zh if lang=='zh' else en).convert('RGB');results=[]
                for cell in values:
                    box=cell.get('box_xyxy',cell.get('box'))
                    same=a.crop(box).tobytes()==b.crop(box).tobytes()
                    results.append({'box':box,'same_pixels':same})
                    if not same:issues.append({'id':fid,'issue':'Evidence pixels changed','box':box,'lang':lang})
                actual_checks[lang]=results
            c['independently_rechecked_evidence_regions']=actual_checks
            labels={};svg_checks={}
            for lang in ['zh','en']:
                sp=ROOT/(fid.replace('-','_')+'_'+lang+'.svg')
                if sp.is_file():
                    sr=E.fromstring(sp.read_bytes());embedded=[e for e in sr.iter() if e.tag.endswith('}image')]
                    iw,ih=Image.open(orig).size
                    svg_checks[lang]=bool(embedded) and all(float(e.get('width'))==iw and float(e.get('height'))==ih and float(e.get('x','0'))==0 and float(e.get('y','0'))==0 and not e.get('transform') for e in embedded)
                    if not svg_checks[lang]:issues.append({'id':fid,'issue':'SVG evidence raster size/position differs from original','lang':lang})
                jp=ROOT/(fid.replace('-','_')+'_'+lang+'_text_layers.json')
                if jp.is_file():
                    ops=load(jp);texts=[op[2] for op in ops if op[0] in {'text','vertical'}];labels[lang]=texts
                    c[lang+'_visible_overlay_texts']=texts
                    if lang=='en' and any(re.search('[\u4e00-\u9fff]',s) for s in texts):issues.append({'id':fid,'issue':'Chinese in English overlay'})
            c['svg_evidence_size_position_equal_original']=svg_checks
            if fid in {'D1-I20','D1-I22'}:
                c['chinese_translation_adaptation_marked']=any('翻译适配' in s for s in labels.get('zh',[]))
            if fid=='D1-I24':
                txt=' '.join(labels.get('en',[]));c['nail_677_identity_and_non_gesture_disclaimer']=all(s in txt for s in ['ImageNet','nail','677','not a gesture'])
                meta=load(ROOT/'source_evidence/v7_cam_metadata.json');c['saved_demo_metadata']=meta
            if fid=='D1-I23':
                meta=load(ROOT/'source_evidence/v9_cam_matrix_metadata.json');txt=' '.join(labels.get('en',[]))
                c['all_six_original_targets_labeled']=all(str(row['target_index']) in txt and row['target_name'] in txt for row in meta['records'])
                c['generic_imagenet_identity_marked']='ImageNet' in txt and 'not six-gesture' in txt
            if fid in {'D1-I23','D1-I24'}:
                source_meta=Path(inv[fid]['evidence_sources'][0]['path'])
                c['metadata_source_bytes_identical']=sha(ROOT/'source_evidence'/source_meta.name)==sha(source_meta)
                source_native=Path(inv[fid]['native_sources'][0]['path'])
                c['original_native_source_preserved']=sha(ROOT/'source_evidence'/source_native.name)==sha(source_native)
        c['visual_review']='FINAL_VISUAL_CHECK_PENDING'
        checks.append(c)
        for key,value in c.items():
            if value is False:issues.append({'id':fid,'issue':key})
    result={'status':'PENDING_FINAL_VISUAL_REVIEW','reviewer':'/root/bilingual_figure_inventory, not the author of assets_reference','manifest_sha256':args.expected_manifest_sha256,'checks':checks,'issues':issues,'output_bindings':bindings,'scope':'13 bilingual groups; 9 unaltered photographs, 2 cable originals plus translation adaptations, 2 earlier ImageNet demonstrations. No model computation or document edits.','limitations':['Final Word embedding/page readability and final ZIP membership require separate integration review.']}
    O.mkdir(exist_ok=True)
    (O/'reference_independent_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'state':result['status'],'issues':issues,'count':len(checks)},ensure_ascii=False))
if __name__=='__main__':main()
