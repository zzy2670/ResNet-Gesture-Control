"""六类真实来源手图的Grad-CAM布局演示；不训练六分类器，也不生成手势预测。"""
from pathlib import Path
import json, hashlib, textwrap
import numpy as np
from PIL import Image, ImageOps
import torch, torchvision, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import functional as TF
W=Path(__file__).resolve().parent;P=W.parent.parent
D=P/'HaGRID/dataset/hagrid-sample-30k-384p'
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False})
torch.set_num_threads(4)
# 按来源标签和文件名预先固定选例；只检查文件与标注能否读取，不查看模型响应。
available={p.stem:p for p in sorted((D/'hagrid_30k').rglob('*.jpg'))}
selected=[]
for number,label,chinese in [(0,'no_gesture','其他／闲置手'),(1,'peace','V形'),(2,'palm','张开手掌'),(3,'fist','握拳'),(4,'like','拇指向上'),(5,'dislike','拇指向下')]:
 candidates=sorted((D/'ann_train_val').glob('*.json')) if number==0 else [D/'ann_train_val'/(label+'.json')]
 found=None
 for annotation in candidates:
  ann=json.loads(annotation.read_text(encoding='utf-8'))
  for image_id in sorted(ann):
   record=ann[image_id]
   if image_id in available and label in record['labels']:
    boxindex=record['labels'].index(label);found=(available[image_id],record,boxindex,annotation,image_id);break
  if found:break
 assert found, label
 path,record,idx,annotation,image_id=found
 original=Image.open(path).convert('RGB');iw,ih=original.size
 x,y,bw,bh=record['bboxes'][idx]
 box=[max(0,int(x*iw)),max(0,int(y*ih)),min(iw,int((x+bw)*iw)),min(ih,int((y+bh)*ih))]
 crop=original.crop(box)
 input_image=ImageOps.pad(crop,(224,224),method=Image.Resampling.BILINEAR,color=(124,116,104))
 input_image.save(W/f'v9_cam_row{number}_input.png')
 selected.append({'number':number,'label':label,'chinese':chinese,'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'image_id':image_id,'annotation':str(annotation),'annotation_box_index':idx,'bbox_xyxy':box,'user_id':record['user_id'],'image':input_image})
 print('selected',number,label,image_id,flush=True)
# 保留官方ImageNet1000类输出头。所有行由单次前向的真实argmax决定目标，不寻找好看的目标。
weight=W/'v4_feature_resnet18-f37072fd.pth'
model=resnet18(weights=None).cpu().eval();model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True))
features={}
def capture(name):
 def hook(module,inputs,output):
  output.retain_grad();features[name]=output
 return hook
for name in ['layer1','layer2','layer3','layer4']:getattr(model,name).register_forward_hook(capture(name))
cmap=matplotlib.colormaps['turbo'];categories=ResNet18_Weights.IMAGENET1K_V1.meta['categories']
records=[]
for row in selected:
 model.zero_grad(set_to_none=True);features.clear();im=row.pop('image');base=np.asarray(im)/255.
 tensor=TF.normalize(TF.to_tensor(im),[.485,.456,.406],[.229,.224,.225]).unsqueeze(0).requires_grad_(True)
 logits=model(tensor);target=int(logits.argmax(1));logits[0,target].backward()
 data={'input_rgb_uint8':np.asarray(im),'input_normalized':tensor.detach().numpy(),'logits':logits.detach().numpy()};shapes={}
 for name,a in features.items():
  # 标准Grad-CAM：目标logit梯度的空间均值作权重，通道加权求和，再ReLU。
  g=a.grad;alpha=g.mean((2,3),keepdim=True);raw=torch.relu((alpha*a).sum(1,keepdim=True))
  up=torch.nn.functional.interpolate(raw,size=(224,224),mode='bilinear',align_corners=False)[0,0].detach().numpy()
  norm=(up-up.min())/(up.max()-up.min()) if up.max()>up.min() else np.zeros_like(up)
  overlay=np.uint8(np.clip(.58*base+.42*cmap(norm)[:,:,:3],0,1)*255)
  Image.fromarray(overlay).save(W/f'v9_cam_row{row["number"]}_{name}.png')
  data.update({name+'_activation':a.detach().numpy(),name+'_gradient':g.detach().numpy(),name+'_weights':alpha.detach().numpy(),name+'_raw_cam':raw.detach().numpy(),name+'_normalized224':norm});shapes[name]=list(a.shape)
 np.savez_compressed(W/f'v9_cam_row{row["number"]}_tensors.npz',**data)
 row.update({'target_index':target,'target_name':categories[target],'target_logit':float(logits[0,target].detach()),'feature_shapes':shapes})
 records.append(row);print('computed',row['number'],target,categories[target],flush=True)
meta={'identity':'ImageNet-pretrained ResNet18 layout demonstration; not a six-gesture model/result','weight_sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'selection_rule':'for each predefined source label, annotation filenames then image IDs lexicographic first with local image; no model scores used; row0 first no_gesture in call.json','preprocess':'annotation bbox integer xyxy crop; ImageOps.pad224 bilinear preserving aspect; RGB124,116,104 padding; ImageNet normalization','target_rule':'each row independent unmodified ImageNet1000 argmax; four layers same target','method':'gradient spatial mean → activation-channel weighted sum → ReLU → bilinear224 align_cornersFalse → independently minmax per CAM','normalization':'each panel independently 0–1; not across-layer or across-row response magnitude','overlay':'0.58*RGB+0.42*turbo','device':'cpu; four threads; no training/GPU','torch':torch.__version__,'torchvision':torchvision.__version__,'records':records}
(W/'v9_cam_matrix_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
# 单幅六行五列矩阵：同一行始终保留相同像素坐标、朝向和输入，不翻转或重裁。
fig=plt.figure(figsize=(8.5,10),facecolor='white')
fig.text(.5,.986,'六类来源手图的分层Grad-CAM呈现',ha='center',va='top',fontsize=15,fontweight='bold')
fig.text(.5,.957,'通用预训练模型布局演示，非六类手势实验结果',ha='center',va='top',fontsize=12,color='#b34e40')
cols=[('input','输入手图\n224 × 224'),('layer4','layer4 · 主解释\n7 × 7'),('layer1','layer1\n56 × 56'),('layer2','layer2\n28 × 28'),('layer3','layer3\n14 × 14')]
xs=[.225,.378,.531,.684,.837]
for x,(_,title) in zip(xs,cols):fig.text(x+.072,.922,title,ha='center',va='top',fontsize=11.3,fontweight='bold')
for r,row in enumerate(records):
 y=.765-r*.126
 label=f"来源类别{row['number']}：{row['chinese']}\n{row['label']}\nImageNet目标：\n"+'\n'.join(textwrap.wrap(row['target_name'],21))+f"\n索引{row['target_index']}"
 fig.text(.013,y+.063,label,ha='left',va='center',fontsize=10.4,linespacing=1.28)
 for x,(stage,_) in zip(xs,cols):
  ax=fig.add_axes([x,y,.145,.12325]);ax.imshow(Image.open(W/f'v9_cam_row{r}_{stage}.png'));ax.set_xticks([]);ax.set_yticks([])
  for spine in ax.spines.values():spine.set_color('#687786');spine.set_linewidth(.6)
cax=fig.add_axes([.30,.071,.52,.013]);bar=np.tile(np.linspace(0,1,500),(10,1));cax.imshow(bar,cmap=cmap,aspect='auto',extent=[0,1,0,1]);cax.set_yticks([]);cax.set_xticks([0,1]);cax.tick_params(labelsize=10,length=0);cax.set_xlabel('各图独立归一化的相对响应',fontsize=10,labelpad=1)
fig.text(.5,.022,'行按HaGRID标注组织；热图解释各行标出的ImageNet目标。上采样不增加原始空间分辨率。',ha='center',va='bottom',fontsize=9.6)
for ext in ['png','svg','pdf']:fig.savefig(W/('v9_cam_matrix.'+ext),dpi=300,facecolor='white')
plt.close(fig)
print('DONE matrix',flush=True)
