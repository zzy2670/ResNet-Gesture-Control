"""从指定图清单抽取紧凑输入；只读原成果，不访问数据集或模型。"""
from pathlib import Path
import json, hashlib, shutil
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
STAGE = ROOT.parent
PROJECT = STAGE.parent
RUNS = PROJECT / 'Vision_CA/runs'
IDS = ['D1-I25','D1-I26','D1-I27','D1-I28','D1-I29','D2-I01','D2-I02','D2-I03','D2-I04','D2-I06','D2-I07','D2-I08','D2-I10','D2-I12','D2-I15']

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    inv = read(STAGE / 'figure_inventory.json')
    figures = [f for f in inv['figures'] if f['id'] in IDS]
    records = []
    for f in figures:
        dest = ROOT / f['id']
        dest.mkdir(exist_ok=True)
        shutil.copy2(f['original_extract'], dest / 'original_reference.png')
        info = {'id': f['id'], 'original_sha256': f['sha256'], 'source_records': [],
                'occurrences': {x['id']:x['caption'] for x in inv['figures'] if x['sha256']==f['sha256']}}
        for source in f.get('evidence_sources', []):
            p = Path(source['path'])
            info['source_records'].append({'name':p.name, 'sha256':sha(p)})
            if p.suffix == '.npz':
                # 保存现有归一化数组，不重新计算热图。
                with np.load(p) as z:
                    arrays = {k:z[k] for k in z.files if 'normalized_224_cam' in k or k in ['input_rgb','logits']}
                    info.setdefault('array_equality', {})[p.name] = {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}
                    np.savez_compressed(dest / p.name, **arrays)
            elif p.suffix=='.json':
                data=read(p)
                if 'confusion' in f['preferred_exact_source']['path']:
                    # 冻结矩阵直接保留，不从图像重新诊断性能。
                    data={'metrics':data['metrics'], 'checkpoint_sha256':data.get('checkpoint_sha256'),
                          'source_sha256':sha(p)}
                (dest/p.name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
                if f['id']=='D2-I15' and p.name=='spotcheck.json':
                    # 只读取冻结清单中明确指定的四张照片，不遍历数据目录。
                    for i,row in enumerate(data['records']):
                        photo=Path(row['source_image'])
                        assert sha(photo)==row['source_image_sha256']
                        shutil.copy2(photo,dest/f'fixed_photo_{i+1}.jpg')
        if f['id'] in ['D2-I06','D2-I12']:
            curves={}
            for protocol in (['A'] if f['id']=='D2-I06' else ['A','B']):
                curves[protocol]={}
                for model in ['ResNet18','ResNet34','ResNet50']:
                    base=RUNS/(protocol+'_20261102_TPE')
                    if model!='ResNet18': base=base/('compare_'+model.lower())
                    p=base/'events.jsonl'
                    events=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]
                    rows=[r for r in events if r.get('stage')=='finetune' and r.get('status')=='complete_epoch']
                    curves[protocol][model]=[{'epoch':r['epoch'],'macro_f1':r['validation']['macro_f1']} for r in rows]
                    info['source_records'].append({'name':protocol+'/'+model+'/events.jsonl','sha256':sha(p)})
            (dest/'frozen_curves.json').write_text(json.dumps(curves,indent=2),encoding='utf-8')
        (dest/'provenance.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        records.append(info)
        a=np.asarray(Image.open(dest/'original_reference.png').convert('RGB'))
        if f['id'] in ['D1-I27','D2-I03','D2-I04','D2-I10']:
            c=(a.min(2)<230).sum(1)
            ok=c>a.shape[1]*.45
            breaks=np.where(np.diff(np.r_[False,ok,False]))[0].reshape(-1,2)
            print(f['id'], a.shape, breaks[(breaks[:,1]-breaks[:,0])>20].tolist())
    (ROOT/'frozen_input_index.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in records:
        for k,v in r['occurrences'].items():print(k,v)

if __name__=='__main__': main()
