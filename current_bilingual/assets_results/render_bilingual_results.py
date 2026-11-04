"""只排版冻结成果，离线生成两语言PNG和可编辑文字SVG；不调用模型。"""
from pathlib import Path
import argparse, base64, hashlib, io, json
from xml.sax.saxutils import escape
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                     'axes.unicode_minus':False,'svg.fonttype':'none','font.size':10,'savefig.dpi':180})
FONT=font_manager.findfont(font_manager.FontProperties(family=['Microsoft YaHei','DejaVu Sans']))
CAPTIONS={
 'D1-I25':'Figure 15. Six-class, layer-wise Grad-CAM for the trained ResNet18 under protocol A. Real test hand images and fixed task weights are used; each map is normalized independently. The figure does not show absolute response strength across layers or overall performance.',
 'D2-I05':'Figure 5. Six-class, layer-wise Grad-CAM on hand images for the trained ResNet18.',
 'D1-I26':'Figure 16. Six-class, layer-wise Grad-CAM for the trained ResNet18 under protocol B. The target is the true-class logit; each map is normalized independently. Colors are not probabilities and cannot be used to compare absolute strength across layers.',
 'D2-I09':'Figure 9. Six-class, layer-wise Grad-CAM for the trained ResNet18 under protocol B.',
 'D1-I27':'Figure 17. Comparison of manual-box classification inputs and automatic localization inputs for B test photos. The upper-right image uses only the box with the highest detection score; a match to the manual box on the left is not claimed. In the lower example, no automatic box passes the threshold.',
 'D2-I11':'Figure 11. Actual manual-box and automatic-localization inputs from the same B photos.',
 'D1-I28':'Figure 18. Six-class, layer-wise Grad-CAM for merged-label A ResNet18. Manual ground-truth boxes and true-class logits are used; each map is normalized independently. Color intensity cannot be used to compare absolute contributions across layers.',
 'D2-I13':'Figure 13. Six-class, layer-wise Grad-CAM on manual-box hand images for the trained merged-label A ResNet18.',
 'D1-I29':'Figure 19. Six-class, layer-wise Grad-CAM for merged-label B ResNet18. Manual ground-truth boxes and true-class logits are used; each map is normalized independently. Color intensity cannot be used to compare absolute contributions across layers.',
 'D2-I14':'Figure 14. Six-class, layer-wise Grad-CAM on manual-box hand images for the trained merged-label B ResNet18.',
 'D2-I01':'Figure 1. Test confusion matrix for ResNet18 on manual-box hand images.',
 'D2-I02':'Figure 2. Test confusion matrix for ResNet18 on whole photos.',
 'D2-I03':'Figure 3. Fixed whole-photo error examples from four classes for ResNet18.',
 'D2-I04':'Figure 4. Actual manual-box and automatic-box crops from the same photos.',
 'D2-I06':'Figure 6. Classifier learning curves of validation macro-F1 at each epoch.',
 'D2-I07':'Figure 7. Test confusion matrix for B ResNet18 on manual-box hand images.',
 'D2-I08':'Figure 8. Test confusion matrix for B ResNet18 on whole photos.',
 'D2-I10':'Figure 10. Fixed whole-photo error examples from four classes for B ResNet18.',
 'D2-I12':'Figure 12. Full fine-tuning validation curves for three architectures under protocols A and B.',
 'D2-I15':'Figure 15. Manual annotations and frozen detections on four two-hand photos for unmerged-label B18.'}

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def label(text,x,y,size=28,anchor='middle',color='#111111'):
    return dict(text=text,x=x,y=y,size=size,anchor=anchor,color=color)

def raster_plate(dest, lang, masks, labels, fixed_base=None):
    original=Image.open(dest/'original_reference.png').convert('RGB')
    base=fixed_base.copy() if fixed_base is not None else original.copy(); dr=ImageDraw.Draw(base)
    changed=np.zeros((original.height,original.width),bool)
    for rect,*fill in masks:
        # 只遮盖原图中的文字区域，实验照片、框和热图保持逐像素一致。
        dr.rectangle(rect,fill=fill[0] if fill else 'white')
        x1,y1,x2,y2=rect; changed[y1:y2+1,x1:x2+1]=True
    canvas=base.copy();dr=ImageDraw.Draw(canvas)
    svg=[]
    for item in labels:
        text=item['text']; x=item['x'];y=item['y'];s=item['size'];anchor=item['anchor'];color=item['color']
        font=ImageFont.truetype(FONT,s)
        pillow_anchor={'middle':'mm','end':'rm','start':'lm'}[anchor]
        lines=text.split('\n');lineheight=s*1.22
        for n,line in enumerate(lines):
            yy=y+(n-(len(lines)-1)/2)*lineheight
            # 两语言使用同一几何位置，只有可编辑文字不同。
            dr.text((x,yy),line,fill=color,font=font,anchor=pillow_anchor)
            svg.append(f'<text x="{x}" y="{yy}" font-family="Microsoft YaHei,DejaVu Sans,sans-serif" font-size="{s}" text-anchor="{anchor}" dominant-baseline="central" fill="{color}">{escape(line)}</text>')
    out=dest/(lang+'.png');canvas.save(out)
    if fixed_base is None:
        assert np.array_equal(np.asarray(original)[~changed],np.asarray(canvas)[~changed]),'文字蒙版以外发生像素变化'
    buf=io.BytesIO();base.save(buf,format='PNG')
    encoded=base64.b64encode(buf.getvalue()).decode('ascii')
    (dest/(lang+'.svg')).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{base.width}" height="{base.height}" viewBox="0 0 {base.width} {base.height}"><image width="{base.width}" height="{base.height}" xlink:href="data:image/png;base64,{encoded}"/>'+''.join(svg)+'</svg>',encoding='utf-8')
    # 文本新绘制可能超出蒙版，因此用科学面板专门验证，不把这类文字越界视为科学变化。
    return {'masked_pixels':int(changed.sum()),'original_reference_sha256':sha(dest/'original_reference.png')}

def cam(dest,lang):
    fid=dest.name
    meta=load(dest/'metadata.json')
    tag={'D1-I25':'A','D1-I26':'B','D1-I28':'合并标签 A' if lang=='zh' else 'Merged-label A','D1-I29':'合并标签 B' if lang=='zh' else 'Merged-label B'}[fid]
    masks=[((0,0,2499,162),),((0,163,397,2872),),((0,2873,2499,2999),)]
    title=f'{tag}：已训练ResNet18六类分层Grad-CAM' if lang=='zh' else f'{tag}: six-class layer-wise Grad-CAM, trained ResNet18'
    items=[label(title,1250,66,38)]
    headers=['分类输入','layer4 主解释 7×7','layer1 56×56','layer2 28×28','layer3 14×14'] if lang=='zh' else ['Classifier input','layer4 main 7×7','layer1 56×56','layer2 28×28','layer3 14×14']
    for x,t in zip([603,1021,1439,1857,2275],headers):items.append(label(t,x,143,28))
    for i,r in enumerate(meta['rows']):
        truth,pred,target=r['truth'],r['prediction'],r['target']
        raw=r['source']['raw_label']
        t=f'真实{truth} / 预测{pred}\n{raw}\n目标为真实类{target}' if lang=='zh' else f'True {truth} / Pred {pred}\n{raw}\nTarget: true class {target}'
        items.append(label(t,365,372+i*457,25,'end'))
    note='每图独立归一化；解释真实类logit；颜色不是概率或准确率\n跨层颜色深浅不能用于绝对贡献比较' if lang=='zh' else 'Each map is normalized independently; target: true-class logit; colors are not probabilities or accuracy.\nColor intensity does not compare absolute contributions across layers.'
    items.append(label(note,1250,2945,25))
    result=raster_plate(dest,lang,masks,items)
    a=np.asarray(Image.open(dest/'original_reference.png').convert('RGB'));b=np.asarray(Image.open(dest/(lang+'.png')).convert('RGB'))
    # 四个热图面板及输入区域从原PNG完整保全。
    science=np.zeros(a.shape[:2],bool);science[164:2873,399:]=True
    assert np.array_equal(a[science],b[science]),'CAM科学面板像素发生改变'
    result['scientific_panel_pixels_identical']=True
    return result

def confusion(dest,lang):
    source=next(dest.glob('test_*.json'));d=load(source)
    m=np.asarray(d['metrics']['confusion_matrix']);n=d['metrics']['n']
    whole='whole' in source.name;protocol='B' if dest.name in ['D2-I07','D2-I08'] else 'A'
    names=['0 其他','1 V形','2 张掌','3 握拳/停止','4 拇指上','5 拇指下'] if lang=='zh' else ['0 Other','1 V sign','2 Open palm','3 Fist / stop','4 Thumb up','5 Thumb down']
    fig,ax=plt.subplots(figsize=(7.6,6.4))
    norm=m/np.maximum(m.sum(1,keepdims=True),1)
    im=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
    for i in range(6):
        for j in range(6):ax.text(j,i,str(m[i,j]),ha='center',va='center',color='white' if norm[i,j]>.55 else 'black')
    title=(f'{protocol} ResNet18 '+('整张照片' if whole else '人工框手图')+f'测试混淆矩阵（{n}'+('张）' if whole else '只手）')) if lang=='zh' else f'{protocol} ResNet18: '+('whole-photo' if whole else 'manual-box hand')+f' test confusion matrix (n={n})'
    ax.set(xticks=range(6),yticks=range(6),xticklabels=names,yticklabels=names,xlabel='预测标签' if lang=='zh' else 'Predicted label',ylabel='真实标签' if lang=='zh' else 'True label',title=title)
    plt.setp(ax.get_xticklabels(),rotation=25,ha='right')
    fig.colorbar(im,ax=ax,label='真实类别内的比例' if lang=='zh' else 'Proportion within true class')
    fig.tight_layout();fig.savefig(dest/(lang+'.png'));fig.savefig(dest/(lang+'.svg'));plt.close(fig)
    return {'frozen_matrix':m.tolist(),'n':n,'same_data_source_for_both_languages':True,'color_scale':[0,1],'color_normalization':'within_true_class'}

def curves(dest,lang):
    data=load(dest/'frozen_curves.json');both=len(data)==2
    fig,axes=plt.subplots(1,len(data),figsize=(12,4.5) if both else (7.3,4.2),sharey=True,squeeze=False)
    for ax,(protocol,models) in zip(axes[0],data.items()):
        for model,rows in models.items():ax.plot([r['epoch'] for r in rows],[100*r['macro_f1'] for r in rows],marker='o',markersize=3,label=model)
        title=(protocol+'协议验证学习曲线' if both else '三种分类网络的验证学习曲线') if lang=='zh' else (f'Protocol {protocol}: validation learning curves' if both else 'Validation learning curves of three classifiers')
        ax.set(title=title,xlabel='完整微调轮次' if lang=='zh' else 'Full fine-tuning epoch');ax.grid(alpha=.2);ax.legend()
    axes[0,0].set_ylabel('验证宏平均F1（%）' if lang=='zh' else 'Validation macro-F1 (%)')
    fig.tight_layout();fig.savefig(dest/(lang+'.png'));fig.savefig(dest/(lang+'.svg'));plt.close(fig)
    return {'same_frozen_curve_source':True,'curve_point_counts':{p:{m:len(r) for m,r in ms.items()} for p,ms in data.items()},'data_sha256':sha(dest/'frozen_curves.json')}

def photo_errors(dest,lang):
    data=load(dest/'error_examples.json');is_b=dest.name=='D2-I10'
    w,h=Image.open(dest/'original_reference.png').size
    if is_b:bands=[(0,249),(870,972)];centers=[(536,207),(1445,207),(536,929),(1445,929)];size=30
    else:bands=[(0,259),(922,1028)];centers=[(597,214),(1619,214),(597,982),(1619,982)];size=30
    masks=[((0,y1,w-1,y2),) for y1,y2 in bands]
    title=('B整图错误' if is_b else '整图错误的固定代表例')+'：绿色实线为人工标注，虚线为自动手框' if lang=='zh' else ('B: fixed whole-photo errors' if is_b else 'Fixed whole-photo errors')+'; green solid = manual GT, dashed = detected hand'
    items=[label(title,w/2,48,31)]
    for (x,y),r in zip(centers,data):
        text=(f'真实命令{r["truth"]} → 输出{r["prediction"]}' if lang=='zh' else f'True command {r["truth"]} → output {r["prediction"]}')+'\n'+r['image_id'][:14 if is_b else 12]
        items.append(label(text,x,y,size))
    result=raster_plate(dest,lang,masks,items)
    result['fixed_sample_ids']=[r['image_id'] for r in data]
    result['frozen_truth_prediction']=[[r['truth'],r['prediction']] for r in data]
    result['photo_and_box_pixels_unchanged_outside_text_bands']=True
    return result

def crop_compare(dest,lang):
    data=load(dest/'actual_crop_comparison.json');is_b=dest.name=='D1-I27'
    w,h=Image.open(dest/'original_reference.png').size
    if is_b:
        masks=[((0,0,w-1,223),),((0,830,w-1,896),),((755,897,1529,1529),)]
        centers=[(390,198),(1140,198),(390,871),(1140,871)]
        title='B同一照片的人工框输入与自动定位输入' if lang=='zh' else 'B: manual-box and automatic-localization inputs from the same photos'
        texts=['人工框输入：真实0，预测0','最高检测分数框：预测4，分类置信度0.570','人工框输入：真实3，预测3','自动定位阶段没有产生分类输入'] if lang=='zh' else ['Manual-box input: true 0, predicted 0','Highest-score box: predicted 4; class confidence 0.570','Manual-box input: true 3, predicted 3','Automatic localization produced no classifier input']
        items=[label(title,w/2,48,32)]+[label(t,x,y,26) for t,(x,y) in zip(texts,centers)]
        items.append(label('没有通过门槛的手框\n没有进入分类器的自动裁图' if lang=='zh' else 'No hand box passed the threshold\nNo automatic crop entered the classifier',1140,1198,28))
    else:
        masks=[((0,0,w-1,278),),((0,902,w-1,1007),)]
        centers=[(434,227),(1266,227),(434,956),(1266,956)]
        title='同一照片的人工框与自动框裁图对照' if lang=='zh' else 'Manual-box and automatic-box crops from the same photos'
        texts=['人工框输入 → 预测0\n真实类0（call）','自动框输入 → 预测4\n分类置信度0.804','人工框输入 → 预测3\n真实类3（fist）','自动框输入 → 预测0\n分类置信度0.594'] if lang=='zh' else ['Manual-box input → predicted 0\nTrue class 0 (call)','Automatic-box input → predicted 4\nClass confidence 0.804','Manual-box input → predicted 3\nTrue class 3 (fist)','Automatic-box input → predicted 0\nClass confidence 0.594']
        items=[label(title,w/2,48,34)]+[label(t,x,y,28) for t,(x,y) in zip(texts,centers)]
    result=raster_plate(dest,lang,masks,items)
    result['fixed_sample_ids']=[r['image_id'] for r in data]
    result['frozen_crop_rectangles']=[{k:v for k,v in r.items() if 'rectangle' in k} for r in data]
    result['photo_pixels_unchanged_outside_text_bands']=True
    return result

def spotcheck(dest,lang):
    data=load(dest/'spotcheck.json')
    bg='#f5f7fa';masks=[]
    base=Image.new('RGB',(1120,2050),bg);draw=ImageDraw.Draw(base)
    small=ImageFont.truetype(FONT,17)
    # 原照片、缩放、位置、冻结框、分数逐一沿用；只把无框文字移到空白边距。
    for i,r in enumerate(data['records']):
        image=Image.open(dest/f'fixed_photo_{i+1}.jpg').convert('RGB')
        scale=min(490/image.width,410/image.height)
        size=(round(image.width*scale),round(image.height*scale))
        for column in (0,1):
            x=24+column*555;xp=x+(510-size[0])//2;yp=128+i*485
            base.paste(image.resize(size,Image.Resampling.LANCZOS),(xp,yp))
            boxes=r['GT_boxes'] if column==0 else r['predicted_boxes']
            for n,box in enumerate(boxes):
                rect=[xp+box[0]*scale,yp+box[1]*scale,xp+box[2]*scale,yp+box[3]*scale]
                color='#08c77b' if column==0 else '#ff8a00'
                draw.rectangle(rect,outline=color,width=3)
                t=r['raw_GT_labels'][n] if column==0 else f'D{n+1} p={r["detector_scores"][n]:.2f}'
                tx,ty=rect[0],max(yp,rect[1]-22)
                draw.rectangle(draw.textbbox((tx,ty),t,font=small),fill='#142238')
                draw.text((tx,ty),t,font=small,fill=color)
    title='B协议 ResNet18：四张指定双手照片（冻结测试结果）' if lang=='zh' else 'B ResNet18: four fixed two-hand photos (frozen test results)'
    note='左：人工标注（绿色）  右：实际检测（橙色）；未重新推理、未改阈值' if lang=='zh' else 'Left: manual GT (green). Right: detections (orange). Inference and thresholds unchanged.'
    items=[label(title,24,27,22,'start','#172b4d'),label(note,24,62,18,'start','#344563')]
    desc=['两只手均正确定位','两只手均未检出','漏掉举起的拳头，只检出垂下的手','检测框偏移，手势也分错'] if lang=='zh' else ['Both hands localized correctly','Neither hand detected','Raised fist missed; only lowered hand detected','Shifted detection box and wrong gesture class']
    for i,r in enumerate(data['records']):
        pred=r['whole_prediction_record'];tail=f'真值{pred["truth"]} → 输出{pred["prediction"]}' if lang=='zh' else f'true {pred["truth"]} → output {pred["prediction"]}'
        items.append(label(f'{i+1}. {desc[i]}; {tail}',24,105+i*485,21,'start','#172b4d'))
    items.append(label('无检测框' if lang=='zh' else 'No boxes',583,649,18,'start','#e64a19'))
    result=raster_plate(dest,lang,masks,items,fixed_base=base)
    result.update(fixed_sample_ids=[r['image_id'] for r in data['records']],detector_threshold=data['detector_threshold'],classifier_threshold=data['classifier_threshold'],detector_score_is_not_classifier_probability=True)
    result['all_photos_match_frozen_source_hashes']=all(sha(dest/f'fixed_photo_{i+1}.jpg')==r['source_image_sha256'] for i,r in enumerate(data['records']))
    result['boxes_and_scores_reused_without_recalculation']=True
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--only',nargs='*');args=parser.parse_args()
    index=load(ROOT/'frozen_input_index.json');manifest=[]
    for record in index:
        fid=record['id'];dest=ROOT/fid
        if args.only and fid not in args.only:continue
        if fid in ['D1-I25','D1-I26','D1-I28','D1-I29']:fn=cam
        elif fid in ['D2-I01','D2-I02','D2-I07','D2-I08']:fn=confusion
        elif fid in ['D2-I06','D2-I12']:fn=curves
        elif fid in ['D2-I03','D2-I10']:fn=photo_errors
        elif fid in ['D1-I27','D2-I04']:fn=crop_compare
        else:fn=spotcheck
        checks={lang:fn(dest,lang) for lang in ['zh','en']}
        manifest.append({'id':fid,'original_sha256':record['original_sha256'],
                         'zh_png':f'{fid}/zh.png','en_png':f'{fid}/en.png',
                         'editable_sources':[f'{fid}/zh.svg',f'{fid}/en.svg'],
                         'generator':'render_bilingual_results.py','required_data':[str(p.relative_to(ROOT)).replace('\\','/') for p in dest.iterdir() if p.suffix in ['.json','.npz','.jpg'] or p.name=='original_reference.png'],
                         'caption_en':CAPTIONS[fid],'caption_translations':{k:{'zh':v,'en':CAPTIONS[k]} for k,v in record['occurrences'].items()},
                         'translation_checks':checks,'preservation':'Display-only translation; no model execution, new samples, or new experimental results.',
                         'sha256':{lang:sha(dest/(lang+'.png')) for lang in ['zh','en']}})
        print(fid,'PASS',flush=True)
    if not args.only:(ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()
