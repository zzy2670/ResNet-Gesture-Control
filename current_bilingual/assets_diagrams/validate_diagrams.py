"""检查双语图的原生形状、数值、嵌入媒体与自包含文件对应。"""
from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
from collections import Counter
from xml.etree import ElementTree as E
import json,re

ROOT=Path(__file__).resolve().parent
NS='{http://schemas.microsoft.com/office/visio/2012/main}'
GEOMETRY={'PinX','PinY','Width','Height','LocPinX','LocPinY','Angle','BeginX','BeginY','EndX','EndY','FlipX','FlipY'}
def h(p):return sha256(Path(p).read_bytes()).hexdigest()
def normalize(x):return re.sub(r'\s+',' ',x).strip()
def numeric(x):return Counter(re.findall(r'\d+(?:\.\d+)?',' '.join(x)))
def native(p):
    with ZipFile(p) as z:
        media=Counter(sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('visio/media/'))
        texts=[];shapes={};external=[]
        for name in z.namelist():
            if re.match(r'visio/pages/page\d+\.xml$',name):
                root=E.fromstring(z.read(name))
                for s in root.iter(NS+'Shape'):
                    cells={c.get('N'):float(c.get('V')) for c in s.findall(NS+'Cell') if c.get('N') in GEOMETRY and c.get('V') is not None}
                    shapes[(name,s.get('ID'))]=cells
                texts.extend(normalize(''.join(x.itertext())) for x in root.iter(NS+'Text'))
            if name.endswith('.rels'):
                external.extend(dict(x.attrib) for x in E.fromstring(z.read(name)) if x.get('TargetMode')=='External')
        return media,texts,shapes,external
def main():
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    trans=json.loads((ROOT/'translation_map.json').read_text(encoding='utf-8'))
    checks=[]
    for r in manifest['items']:
        ident=r['id'];orig=ROOT/r['original_vsdx'];zh=ROOT/r['editable_sources'][0];en=ROOT/r['editable_sources'][1]
        om,ot,og,ox=native(orig);em,et,eg,ex=native(en)
        geometry_equal=og.keys()==eg.keys() and all(og[k].keys()==eg[k].keys() and all(abs(og[k][c]-eg[k][c])<1e-9 for c in og[k]) for k in og)
        c={'id':ident,'source_vsdx_unchanged':h(orig)==h(zh),'source_png_unchanged':h(ROOT/r['original_png'])==h(ROOT/r['zh_png']),
            'media_hash_multiset_equal':om==em,'geometry_equal':geometry_equal,'source_shape_count':len(og),'en_shape_count':len(eg),
            'all_english_labels_present':Counter(normalize(x) for x in trans[ident])==Counter(et),
            'numeric_tokens_equal':numeric(ot)==numeric(et),'no_chinese_in_english_native':not any(re.search('[\u4e00-\u9fff]',x) for x in et),
            'all_required_files_present':all((ROOT/x).is_file() for x in r['required_data']+r['editable_sources']+[r['zh_png'],r['en_png'],r['generator']]),
            'external_relationships':ex}
        c['status']='PASS' if all(v for k,v in c.items() if isinstance(v,bool)) and not ex else 'REVIEW'
        checks.append(c)
        r['translation_checks'].update({'source_geometry_equal':geometry_equal,'embedded_media_hashes_equal':om==em,'source_vsdx_identical':h(orig)==h(zh),'all_english_labels_present':c['all_english_labels_present'],'validation_status':c['status']})
    (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'validation.json').write_text(json.dumps({'checks':checks,'status':'PASS' if all(c['status']=='PASS' for c in checks) else 'REVIEW'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
