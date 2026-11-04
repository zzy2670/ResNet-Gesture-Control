"""仅从冻结产物生成正式A/B报告，不运行模型。"""
from pathlib import Path
import json
import re
import hashlib
from copy import deepcopy
from docx import Document
from docx.shared import Inches, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

B = Path(__file__).resolve().parent
V = B.parent.parent
A = V / 'runs/A_20261102_TPE'
P = V.parent
SOURCE = P / 'A协议ResNet训练与对比实验报告_v1.docx'
OUTPUT = P / 'A与B协议ResNet训练与对比实验报告_v2.docx'
LABELS = ['0 其他', '1 V形', '2 手掌', '3 握拳', '4 向上', '5 向下']
CLASS_SCOPE='classifier_only_preprocessed_224x224_tensor'
WHOLE_SCOPE='whole_photo_decode_detection_crop_classification_rules'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def pct(x):
    return f'{100*x:.2f}%'

def mdtable(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |',
                      '|' + '|'.join(['---']*len(headers)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]) + '\n\n'

def figure(path, title, explanation):
    if not path.exists():
        raise FileNotFoundError(path)
    return f'![{title}]({path.as_posix()})\n\n{explanation}\n\n'

def mistakes(metrics, unit='张'):
    cm = metrics['confusion_matrix']
    return '；'.join(f'{LABELS[i]}→{LABELS[j]} {cm[i][j]}{unit}'
                    for i in range(6) for j in range(6) if i != j and cm[i][j]) + '。'

def perclass(metrics):
    return mdtable(['类别','真实分母','精确率','召回率','F1'],
                   [[LABELS[i], metrics['support'][i], pct(metrics['precision'][i]),
                     pct(metrics['recall'][i]), pct(metrics['f1'][i])] for i in range(6)])

def events(out, stage):
    return [x for x in [json.loads(s) for s in (out/'events.jsonl').read_text(encoding='utf-8').splitlines() if s]
            if x.get('stage') == stage and x.get('status') == 'complete_epoch']

def matched_cpu():
    # 主速度比较必须采用预注册的六模型同输入补测，历史时延另外完整保留。
    result=read(B/'matched_cpu_benchmark.json')
    if result['status']!='complete' or len(result['models'])!=6:
        raise RuntimeError('六模型同输入CPU补测尚未完成')
    review=read(B/'matched_cpu_independent_review.json')
    if review['status']!='PASS' or review['data_sha256']!=hashlib.sha256((B/'matched_cpu_benchmark.json').read_bytes()).hexdigest():
        raise RuntimeError('六模型同输入CPU补测尚未经独立后审')
    models={m['key']:m for m in result['models']}
    for model in models.values():
        for scope in (CLASS_SCOPE,WHOLE_SCOPE):
            if model['summary'][scope]['n']!=300:
                raise ValueError('统一CPU补测不满三轮300次')
    return result,models

def recovery_probability_text(archive, current):
    # 按不可变的照片和手编号对齐；这里只复算已保存概率，不再次推理。
    before, after = read(archive), read(current)
    def keyed(payload):
        result = {(r['image_id'], r['hand_index']): r for r in payload['predictions']}
        if len(result) != len(payload['predictions']):
            raise ValueError('恢复预测存在重复照片手编号')
        return result
    old, new = keyed(before), keyed(after)
    if old.keys() != new.keys() or before['checkpoint_sha256'] != after['checkpoint_sha256']:
        raise ValueError('恢复预测的样本或冻结权重身份变化')
    review=read(B/'ab_final_independent_review.json')
    if not review['status'].startswith('PASS'):
        raise ValueError('冻结身份与恢复差异尚未经最终独审')
    changed = 0
    largest = 0.0
    label_changes=[]
    for key, row in old.items():
        other = new[key]
        if row['truth'] != other['truth']:
            raise ValueError('恢复预测真值身份变化')
        if row['prediction'] != other['prediction']:
            label_changes.append((key,row,other))
        delta = max(abs(a-b) for a,b in zip(row['probabilities'],other['probabilities']))
        changed += delta > 0
        largest = max(largest, delta)
    text=f'两次人工框评价按照片与手编号逐一对应，{changed}／{len(old)}只手的概率向量变化，最大绝对差为{largest:.16g}。'
    if not label_changes and before['metrics']==after['metrics']:
        return text+'全部真值、预测标签和汇总指标相同。'
    text+=f'{len(label_changes)}只手的argmax标签改变；首次中断子产物准确率为{pct(before["metrics"]["accuracy"])}，自然完整恢复结果为{pct(after["metrics"]["accuracy"])}。'
    for (image_id,hand_index),row,other in label_changes:
        a,b=row['prediction'],other['prediction']
        text+=f'照片{image_id}手编号{hand_index}（原标签{row["raw_label"]}、真值{row["truth"]}）由{a}变为{b}：首次p{a}={row["probabilities"][a]:.16g}、p{b}={row["probabilities"][b]:.16g}，恢复后分别为{other["probabilities"][a]:.16g}和{other["probabilities"][b]:.16g}。'
    return text+'最终正式指标采用自然完成全部测试的恢复产物；不在中断子结果与完整结果之间择高，不将数值变化归因为某项已确定的CUDA机制。'

def report_text():
    # 先确认全部运行阶段完成；最终文档另由非作者独审。
    if read(B/'all_stages_complete.json')['status'] != 'PASS':
        raise RuntimeError('B execution has not completed')
    allrows = read(B/'ab_comparison_results.json')
    if len(allrows) != 6:
        raise ValueError('Need all six A/B comparison rows')
    cpu_result,cpu_models=matched_cpu()
    summary = read(V/'data/B_split_verified_20261102/summary.json')
    source = (A/'实验报告正文.md').read_text(encoding='utf-8')
    overview = '本实验完成A按照片划分和B按人员互斥划分下的ResNet18、34、50独立训练、验证选型和冻结测试。核心结果如下；人工框评价六类手势分类，整图评价检测、裁图、分类和多手规则后的最终数字，二者分母分别列出。\n\n'
    overview += mdtable(['协议／架构','人工框正确／分母','人工框准确率','整图正确／分母','整图准确率','整图宏F1'],[[r['protocol']+' '+r['model'],f"{r['crop_correct']}／{r['crop_n']}",pct(r['crop_accuracy']),f"{r['photo_correct']}／{r['photo_n']}",pct(r['photo_accuracy']),pct(r['photo_macro_f1'])] for r in allrows])
    overview += '方法与A完整证据见第1—8章；B人员隔离、训练、阈值和完整测试见第9—10章；实际手形和错误环节见第11章；同架构A/B差值和同输入CPU主比较见第12章，模型取舍与集成结论见第13章。第1—8章原A时延是历史测量；原六组时延另完整列在附录，当前速度结论采用第12章补充统一测量。\n\n'
    source = source.replace('# A协议手势识别训练与评测报告', '# A与B协议ResNet训练与对比实验报告\n\n'+overview)
    source = source.replace('## 1 实验任务与处理流程', '## 1 实验任务与处理流程\n\n本报告先完整呈现A按照片划分的实验，再报告B按人员互斥划分的独立训练与测试，最后比较同一架构在两种评价条件下的结果。第1—8章中的“本轮”指A阶段，第9章起为B阶段；两阶段使用同一来源材料，训练、验证和测试归属各自独立确定。')
    source = source.replace('人员互斥的B协议本轮没有运行。', '人员互斥的B协议在A阶段未运行，其独立结果见第9—12章。')
    source = source.replace('这样保留手周围约10%的边缘，同时避免把长方形手框直接拉伸变形。', '沿长边两端各增加原长边10%，短边方向补足同一正方形，避免把长方形手框直接拉伸变形。')
    source = source.replace('人员互斥测试、真实摄像头连续视频和机器人仿真联动未在本轮执行。', 'A阶段未执行人员互斥测试；B的同来源人员互斥评价见后文，真实摄像头连续视频和机器人仿真联动仍未执行。')
    source = source.replace('尚不能给出跨人员、空场景或摄像头连续输入下的性能排序。', '该A结果本身不能给出跨人员、空场景或摄像头连续输入下的性能排序；B条件的实际比较见第12章。')
    source = source.replace('## 8 实验结论与后续方向','## 8 A阶段实验结论')
    source = source.replace('满足16:50前结束训练的要求；B协议未运行。','满足A阶段16:50前结束训练的要求；B阶段随后另行独立运行，不沿用该历史截止时间。')
    source = source.replace('无手照片、人员互斥B和连续摄像头输入应作为后续独立评价条件，当前静态照片结果不代替机器人停止功能的联动测试。', 'A阶段结束时将人员互斥B列为后续评价，现已由后文报告实际结果。无手照片和连续摄像头输入仍未完成评价，当前静态照片结果不代替机器人停止功能的联动测试。')
    text = source.rstrip() + '\n\n## 9 B人员互斥划分与分类训练\n\n'
    text += '### 9.1 固定人员与照片清单\n\nB仍使用同一HaGRID Sample30K的31,833张照片、7,973名人员和38,884个人工手框，按user_id而非单张照片分组。随机种子为42，先打乱人员，再将同一人的全部照片归入同一组。例如某人进入B训练组，其照片及所有派生手图都不会进入B验证或测试组。A与B是同一材料上的两套评价，A照片与B照片之间没有互斥要求。\n\n'
    text += mdtable(['分组','人数','人数比例','完整照片','照片比例','手框'],
                    [[k, summary['splits'][s]['users'], pct(summary['splits'][s]['user_fraction']),
                      summary['splits'][s]['images'], pct(summary['splits'][s]['image_fraction']),
                      summary['splits'][s]['hands']] for s,k in [('train','训练'),('val','验证'),('test','测试')]])
    text += '8∶1∶1应用于人员数，实际照片比例为79.40%／10.86%／9.74%，不把人员比例写成照片比例。逐图独立核验三组image_id、user_id与原照片内容SHA256的交叉均为0，并确认全部照片副本与原始字节、全部保留标注与原标注一致。原始类别目录中训练fist照片有1,367张，而训练fist手框为1,366个；按真实标注保留这一差异，没有补造手框或删照片。原标注档含509,323条记录，其中只有本地31,833张照片对应记录进入本实验，不能把全标注记录数当作本地样本量。\n\n'
    text += mdtable(['分组','手框总数']+LABELS, [[k,summary['splits'][s]['hands']]+[summary['splits'][s]['hands_per_merged_label'][str(i)] for i in range(6)] for s,k in [('train','训练'),('val','验证'),('test','测试')]])
    text += 'B训练、验证和测试照片中的0输出真值分别为14,125、1,932和1,695张；它们与0类手框数不同。测试的3,759只手来自全部3,102张完整照片，所有照片保留在整图测试分母中。B裁图、归一化和训练专用增强沿用第2.2节。B三个分类器均从各架构官方ImageNet V1初始化，B检测器从官方MobileNetV3初始化；没有用A微调权重初始化B。\n\n'
    outs = [B, B/'compare_resnet34', B/'compare_resnet50']
    names = ['ResNet18','ResNet34','ResNet50']
    trials = [read(o/'tpe_summary.json') for o in outs]
    histories = [events(o,'finetune') for o in outs]
    bests = [max(h,key=lambda x:x['validation']['macro_f1']) for h in histories]
    text += '### 9.2 三种架构的TPE搜索与末组微调\n\nB分别提取30,799张训练手图与4,326张验证手图的冻结特征，再独立搜索分类头。每架构24次试验、每次18轮，随机种子42，前4次随机启动，目标为完整验证轮的最高宏平均F1；六项搜索空间与第3.1节相同。参数搜索之后解冻layer4与分类头，其余层及其BatchNorm固定，采用第3.2节相同微调规则。TPE搜索学习率只服务于冻结头，末组微调仍使用10⁻⁴／5×10⁻⁴。本实验没有合并训练和验证集重训。\n\n'
    keys = [('最佳试验编号','best_trial'),('冻结头最佳验证宏F1','best_value')]
    rows = [[title]+[pct(t[key]) if key=='best_value' else t[key] for t in trials] for title,key in keys]
    for title,key in [('优化器','optimizer'),('搜索学习率','lr'),('权重衰减','weight_decay'),('Dropout','dropout'),('标签平滑','label_smoothing'),('训练类别权重','class_weight')]:
        rows.append([title]+[('使用' if t['best_params'][key] else '不使用') if key=='class_weight' else (f"{t['best_params'][key]:.9g}" if isinstance(t['best_params'][key],float) else t['best_params'][key]) for t in trials])
    rows += [['完整微调轮数']+[len(h) for h in histories],['验证选中轮数']+[b['epoch'] for b in bests],['最佳验证准确率']+[pct(b['validation']['accuracy']) for b in bests],['最佳验证宏F1']+[pct(b['validation']['macro_f1']) for b in bests],['完整训练加验证／s']+[f"{sum(x['seconds'] for x in h):.2f}" for h in histories]]
    text += mdtable(['项目']+names, rows)
    text += '选型保存宏平均F1严格提高的完整轮；未把后续同分轮次另按损失排序。三种模型超参数由各自验证选出，因此模型比较包含架构和独立选型的共同差异。完成固定24次搜索不表示已找到全局最优。\n\n'
    text += '末组微调的预设上限为18轮，并保留原patience=4早停。B18与B34完成18轮；B50完整完成11轮，第7轮得到最高验证宏F1，随后第8—11轮连续四轮未提高，因此自然早停并使用第7轮权重。11轮是既定选型规则的正常结束，与后续测试中断分别记录。\n\n'
    for name,best in zip(names,bests):
        text += f'#### {name}最佳轮验证分类\n\n第{best["epoch"]}轮在全部4,326只验证手上得到准确率{pct(best["validation"]["accuracy"])}、宏平均F1 {pct(best["validation"]["macro_f1"])}。\n\n' + perclass(best['validation']) + '全部非零错误方向：' + mistakes(best['validation'], '只') + '\n\n'
    text += '## 10 B手检测、阈值与完整测试\n\n### 10.1 检测器训练和验证阈值\n\n'
    dh = events(B,'detector')
    dbest = max(dh,key=lambda x:2*x['validation']['precision']*x['validation']['recall']/(x['validation']['precision']+x['validation']['recall']))
    dm = dbest['validation']
    text += f'B检测器使用25,274张训练照片及30,799个手框，沿用第4.1节的16轮、batch24、SGD与增强设置，完整训练加验证合计{sum(x["seconds"] for x in dh):.2f} s。按score≥0.5、IoU≥0.5的验证检测F1选中第{dbest["epoch"]}轮，TP={dm["tp"]}、FP={dm["fp"]}、FN={dm["fn"]}，精确率{pct(dm["precision"])}、召回率{pct(dm["recall"])}。三种B架构共享这份B检测权重。\n\n'
    text += '各分类器在全部3,457张验证照片上保存固定预测，再比较12个检测／分类阈值组合，排序规则与第4.2节一致。以下完整列出B18候选，安全分母为1,932张真实0和195张真实停止照片。\n\n'
    cals=[read(o/'calibration.json') for o in outs]
    text += mdtable(['检测阈值','分类阈值','准确率','宏F1','0误发运动','停止误发运动'],[[c['detector_threshold'],c['class_threshold'],pct(c['metrics']['accuracy']),pct(c['metrics']['macro_f1']),c['safety']['zero_to_motion'],c['safety']['stop_to_motion']] for c in cals[0]['candidates']])
    text += mdtable(['架构','检测／分类阈值','验证准确率','验证宏F1'],[[n,f"{c['selected']['detector_threshold']}／{c['selected']['class_threshold']}",pct(c['selected']['metrics']['accuracy']),pct(c['selected']['metrics']['macro_f1'])] for n,c in zip(names,cals)])
    text += '### 10.2 测试冻结与中断恢复\n\n分类权重、检测权重和阈值在各自验证后冻结，最终测试不用于重新搜索、选轮或改阈值。B18校准冻结时刻为2026年11月2日21:53:06。evaluate attempt1已经读取测试并完成3,759手人工框预测，随后进程消失，完整整图结果未写出，具体中断原因未知。旧产物归档后，attempt2按相同冻结权重、阈值和清单恢复，并于22:05:00自然退出。\n\n两次人工框概率并非逐位相同：1,039／3,759只手的概率向量变化，最大绝对差为0.0010422468185425；全部预测标签、真值和汇总指标相同。运行使用CUDA混合精度、cudnn.benchmark=True且未强制确定性算法，浮点实现差异与该观察相容，未进行实验归因；没有据两次测试结果择优。calibration中的test_accessed=false只描述验证校准阶段，不表示整个项目从未访问测试。\n\n'
    text += 'B50在2026年11月2日22:44:30完成验证冻结；首次测试已保存3,759手人工框结果，随后被系统重启中断。Windows事件1074记录22:47:02由shutdown.exe代表Administrator账户发起重启，重启前测试进程于22:47:03退出，系统于22:48:28重新启动；具体哪个上游程序或操作者调用shutdown.exe未知。完整整图与检测输出尚未生成时，旧人工框结果和日志已先行归档，经独立恢复审查后，使用原权重、阈值0.3／0.5和固定清单恢复未成功的测试阶段，没有重训、追加搜索或重新选阈值。\n\n'
    text += recovery_probability_text(B/'recovery_resnet50_20261102_2256/test_oracle_crop.json', outs[2]/'test_oracle_crop.json') + '两次结果均保留，恢复结果没有用于挑选较高测试成绩；此次恢复不构成新的独立试验。\n\n'
    for name,out in zip(names,outs):
        crop=read(out/'test_oracle_crop.json'); whole=read(out/'test_whole_photo.json'); det=read(out/'test_detector.json')['metrics']
        cm=whole['metrics']['confusion_matrix']; correct=sum(cm[i][i] for i in range(6)); cc=sum(crop['metrics']['confusion_matrix'][i][i] for i in range(6)); safety=whole['safety']
        text += f'### 10.{3+names.index(name)} {name}最终测试与错误方向\n\n人工框分类在3,759只手中正确{cc}只、错误{3759-cc}只，准确率{pct(crop["metrics"]["accuracy"])}、宏平均F1 {pct(crop["metrics"]["macro_f1"])}。\n\n'+perclass(crop['metrics'])
        text += '人工框全部非零错误方向：'+mistakes(crop['metrics'], '只')+'\n\n'
        text += f'完整照片正确{correct}／3,102张、错误{3102-correct}张，准确率{pct(whole["metrics"]["accuracy"])}、宏平均F1 {pct(whole["metrics"]["macro_f1"])}。\n\n'+perclass(whole['metrics'])
        text += '整图全部非零错误方向：'+mistakes(whole['metrics'])+'\n\n'
        text += f'真实0误发运动为{safety["zero_to_motion"]}／{safety["zero_samples"]}（{pct(safety["zero_to_motion_rate"])}）；真实停止误发运动为{safety["stop_to_motion"]}／{safety["stop_samples"]}（{pct(safety["stop_to_motion_rate"])}），真实停止输出0另有{cm[3][0]}／{safety["stop_samples"]}（{pct(cm[3][0]/safety["stop_samples"])}）。输出0表示没有有效动作指令，与3明确停止不同；停止输出0仍属于识别失败。\n\n'
        text += f'固定检测阈值{det["score_threshold"]}、IoU≥{det["iou_threshold"]}的框级结果为TP={det["tp"]}、FP={det["fp"]}、FN={det["fn"]}，精确率{pct(det["precision"])}、召回率{pct(det["recall"])}。这是固定阈值检测匹配结果，并非AP或mAP；漏框数不能直接当作失败照片数。\n\n'
        if name=='ResNet18':
            text += figure(B/'report_figures/ResNet18_crop_confusion.png','B ResNet18人工框测试混淆矩阵','人工框错误仍主要是有效手势与0类互混；V形有13只、张掌有7只被判为0。')
            text += figure(B/'report_figures/ResNet18_whole_confusion.png','B ResNet18整图测试混淆矩阵','完整照片中握拳41张、向上43张、向下43张输出0。215张总错误中207张是有效手势输出0，另8张是真实0输出有效类别；0→3有3张。')
    text += '## 11 B真实手形与分层Grad-CAM\n\n'
    # 实际审看图像后才写解释正文。
    supplemental=B/'report_interpretation_sections.md'
    if not supplemental.exists():
        raise FileNotFoundError('Await inspected B/CAM interpretation: '+str(supplemental))
    text += supplemental.read_text(encoding='utf-8').rstrip()+'\n\n'
    text += '## 12 A/B同架构比较与CPU开销\n\n### 12.1 六模型结果与相同架构的协议差异\n\nA允许同一人员跨训练、验证和测试照片，B按人员互斥。两者仍来自同一数据来源，且训练样本归属、验证选中的参数与权重也不同；A/B差值描述这两套完整评价条件的结果变化，不能单独解释为人员因素的因果效应。\n\n'
    fields=[('人工框正确／分母',lambda r:f"{r['crop_correct']}／{r['crop_n']}"),('人工框准确率',lambda r:pct(r['crop_accuracy'])),('人工框宏F1',lambda r:pct(r['crop_macro_f1'])),('整图正确／分母',lambda r:f"{r['photo_correct']}／{r['photo_n']}"),('整图准确率',lambda r:pct(r['photo_accuracy'])),('整图宏F1',lambda r:pct(r['photo_macro_f1'])),('整图错误',lambda r:r['photo_errors']),('真实0误发运动',lambda r:f"{r['zero_to_motion']}／{r['zero_samples']}"),('真实停止误发运动',lambda r:f"{r['stop_to_motion']}／{r['stop_samples']}"),('真实停止输出0',lambda r:f"{r['stop_to_zero']}／{r['stop_samples']}")]
    for protocol in ['A','B']:
        rr=[next(r for r in allrows if r['protocol']==protocol and r['model']==n) for n in names]
        text += mdtable([protocol+'协议指标']+names,[[title]+[fn(r) for r in rr] for title,fn in fields])
    diffs=[]
    for n in names:
        ar=next(r for r in allrows if r['protocol']=='A' and r['model']==n); br=next(r for r in allrows if r['protocol']=='B' and r['model']==n)
        diffs.append([n]+[f"{100*(br[k]-ar[k]):+.3f}" for k in ['crop_accuracy','crop_macro_f1','photo_accuracy','photo_macro_f1']])
    text += mdtable(['B−A／百分点','人工框准确率','人工框宏F1','整图准确率','整图宏F1'],diffs)
    text += figure(B/'report_figures/AB_classifier_learning_curves.png','A/B三架构完整微调验证曲线','左右分别是A和B的验证宏平均F1，不能把各自验证峰值当作最终测试成绩。曲线展示已完成轮次及后期变化，最优轮按各自验证规则保留。')
    text += '### 12.2 六模型同输入CPU分类与整图时延\n\n原有A/B测速使用各自测试输入，B18曾测得比更深模型更长的平均整图时间；该现象本身不能证明架构更慢。为在相同输入下比较速度，事先固定补充协议，再一次执行六模型统一CPU测量。原测量不覆盖、不删除，全部列在附录；本节采用补充测量的全部记录，不挑选最快轮次。\n\n六模型在同一Intel Core Ultra 9 285H上使用CPU、float32和8个算子线程。完整照片取A验证清单按照片ID排序的前100张；纯分类独立取A验证手图按照片ID和手编号排序的前100只，六模型分别共享同一批照片及同一批预处理224×224张量。每个模型每项计时先预热20次，正式遍历100个输入3轮，共300次；三轮模型顺序分别为A18、B18、A34、B34、A50、B50，逆序，以及该基础顺序左移三位。模型提前常驻内存，加载、哈希与分类预处理不计时。\n\n纯分类只计分类器前向；整图每次调用原整图函数，含打开解码、手检测、实际裁图、分类及多手规则。每次一张完整照片，其中多只手仍按原接口一批分类，不能把整图写成每只手分别batch1。各模型保留自己的冻结阈值和实际框数，故整图比较的是相同照片上的完整程序，计算工作量不强行相等。这项验证集补测不重算准确率、不用于改权重或阈值。\n\n'
    new_cpurows=[]
    for r in allrows:
        model=cpu_models[r['protocol']+'_'+r['model'].lower()]
        c,w=model['summary'][CLASS_SCOPE],model['summary'][WHOLE_SCOPE]
        new_cpurows.append([model['classifier_parameters']]+[f'{x:.3f}' for x in
                             (c['mean_ms'],c['p50_ms'],c['p95_ms'],c['sd_ms'],
                              w['mean_ms'],w['p50_ms'],w['p95_ms'],w['sd_ms'])])
    new_metrics=['参数量','分类均值／ms','分类中位数／ms','分类P95／ms','分类标准差／ms',
                 '整图均值／ms','整图中位数／ms','整图P95／ms','整图标准差／ms']
    text += mdtable(['统一补测指标']+[r['protocol']+' '+r['model'] for r in allrows],
                    [[title]+[row[i] for row in new_cpurows] for i,title in enumerate(new_metrics)])
    text += '表中均值、中位数、P95和样本标准差均由固定100个输入重复三轮的300次调用计算，并非300张独立照片。原始计时、三轮均值、实际框数、背景CPU计数和输入哈希均保存。计时块中其他进程的估计CPU占用为7.4%—43.0%；例如B18整图三轮均值为36.71、34.38、50.54 ms，较慢的第3轮也完整计入。自然文件缓存及系统波动保留，三轮顺序变化仍不能完全平衡所有位置影响。因此本次均值仅描述本机这轮测量，不据同架构A/B小差异或A34纯分类均值高于A50推断协议或深度造成加速，也未进行显著性检验。\n\n'
    history_cpu_text='## 附录 原A与B各自输入的历史CPU记录\n\n以下保留补充统一测量之前的六组原记录。各协议内部纯分类用相同100只测试手图、float32、8线程、20次预热、3轮300次；A与B输入不同。整图分别用各自测试清单前100张照片、10次预热，每次一张照片，含打开解码、检测、裁图、分类和规则，模型加载不计时。与第12章补测的照片、预热和测量批次不同，不能把两表差值当作模型更新带来的改善。\n\n'
    cpurows=[]
    for r in allrows:
        out=(A if r['protocol']=='A' else B); out=out if r['model']=='ResNet18' else out/('compare_'+r['model'].lower())
        w=read(out/'cpu_latency.json'); c=read(out/'classifier_cpu_latency_unified.json')
        cmid=c.get('median_ms',c.get('p50_ms')); wmid=w.get('median_ms',w.get('p50_ms'))
        cpurows.append([r['classifier_parameters'],f"{c['mean_ms']:.3f}",f'{cmid:.3f}' if cmid is not None else '未记录',f"{c['p95_ms']:.3f}",f"{w['mean_ms']:.3f}",f'{wmid:.3f}' if wmid is not None else '未记录',f"{w['p95_ms']:.3f}",f"{r['finetune_seconds']:.2f}"])
    metrics=['参数量','分类均值／ms','分类中位数／ms','分类P95／ms','整图均值／ms','整图中位数／ms','整图P95／ms','微调加验证／s']
    history_cpu_text += mdtable(['历史指标']+[r['protocol']+' '+r['model'] for r in allrows],[[title]+[row[i] for row in cpurows] for i,title in enumerate(metrics)])
    text += '## 13 综合结论与小组集成\n\n'
    final=B/'report_final_conclusion.md'
    if final.exists():
        text += final.read_text(encoding='utf-8').rstrip()+'\n\n'
    else:
        br=[r for r in allrows if r['protocol']=='B']
        best=max(br,key=lambda r:r['photo_macro_f1'])
        def current_whole(r):
            return cpu_models[r['protocol']+'_'+r['model'].lower()]['summary'][WHOLE_SCOPE]['mean_ms']
        fastest=min(br,key=current_whole)
        b18=next(r for r in br if r['model']=='ResNet18'); b50=next(r for r in br if r['model']=='ResNet50')
        text += f'A与B均完成三架构各24次分类头TPE、各自完整末组微调、验证阈值选择、全部固定测试照片评价和CPU测速；两协议分别训练检测器。B按人员互斥后，三架构人工框准确率为{min(r["crop_accuracy"] for r in br)*100:.2f}%—{max(r["crop_accuracy"] for r in br)*100:.2f}%，完整照片准确率为{min(r["photo_accuracy"] for r in br)*100:.2f}%—{max(r["photo_accuracy"] for r in br)*100:.2f}%。分类器在真实手框输入上表现较高，自动定位后的整图错误仍需单独处理。\n\n'
        text += f'B的最高整图宏平均F1来自{best["model"]}（{pct(best["photo_macro_f1"])}），同输入补测的最低CPU整图均值来自{fastest["model"]}（{current_whole(fastest):.2f} ms）。B50相对B18的整图准确率变化为{100*(b50["photo_accuracy"]-b18["photo_accuracy"]):+.3f}个百分点，整图宏F1变化为{100*(b50["photo_macro_f1"]-b18["photo_macro_f1"]):+.3f}个百分点；在相同A验证照片上的平均整图耗时由{current_whole(b18):.2f} ms变为{current_whole(b50):.2f} ms。模型选择应把这项取舍与各类召回、误发运动和停止漏识别一起判断。\n\n'
        text += f'B50的整图总体指标较高，但真实0误发运动为{b50["zero_to_motion"]}／{b50["zero_samples"]}，多于B18的{b18["zero_to_motion"]}／{b18["zero_samples"]}；两者真实停止误发运动均为0，停止输出0仍分别有{b50["stop_to_zero"]}与{b18["stop_to_zero"]}张。因而提高整体宏F1与减少误发运动在本次结果中没有同时实现。B34人工框准确率略低于B18，人工框宏F1则略高（99.04%对98.95%）；其整图准确率与宏F1均较低，并出现1张真实停止误发运动，未显示比B18更有利的综合结果。18与50应作为不同开销和错误取舍的候选共同复核，不能仅按总准确率决定接入控制的模型。\n\n'
    text += 'B人员互斥结果属于同一HaGRID来源的内部留出，没有独立设备、独立环境或真实机器人外部测试；本次也没有多种子、标签打乱、近重复与背景shortcut对照。因此较高人工框分数不等于所有捷径来源已经排除。0误发运动、停止误发运动和停止输出0必须一起判断，静态照片中的0次停止误发不能替代真实停止功能的联动验证。\n\nResNet分支由我和组员共同负责，后续应共同复核模型选择和汇报内容，与另一识别子组在共同输入和评价口径下比较，并与控制组核对整数含义、异常处理、停止优先及接收端。真实摄像头连续视频、无手空场景和机器人通信／控制联调仍属于后续工作。\n'
    return text+'\n\n'+history_cpu_text

def set_font(run, size=None, heading=False):
    run.font.name='Calibri'
    if size: run.font.size=Pt(size)
    rp=run._element.get_or_add_rPr(); f=rp.rFonts
    if f is None: f=OxmlElement('w:rFonts'); rp.insert(0,f)
    for key in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
        f.attrib.pop(qn('w:'+key),None)
    for key,val in [('ascii','Calibri'),('hAnsi','Calibri'),('cs','Calibri'),('eastAsia','Microsoft YaHei' if heading else 'SimSun')]: f.set(qn('w:'+key),val)
    if size:
        z=rp.find(qn('w:szCs'))
        if z is None: z=OxmlElement('w:szCs'); rp.append(z)
        z.set(qn('w:val'),str(round(size*2)))

def inline(paragraph, text):
    # 将来源写成Word原生链接。
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    pos=0
    for m in re.finditer(r'\[([^\]]+)\]\((https?://[^)]+)\)',text):
        if m.start()>pos: set_font(paragraph.add_run(text[pos:m.start()]))
        h=OxmlElement('w:hyperlink'); h.set(qn('r:id'),paragraph.part.relate_to(m[2],RT.HYPERLINK,is_external=True))
        r=OxmlElement('w:r'); t=OxmlElement('w:t'); t.text=m[1]; r.append(t); h.append(r); paragraph._p.append(h); pos=m.end()
    if pos<len(text): set_font(paragraph.add_run(text[pos:]))

def build_docx(text, output=OUTPUT):
    d=Document(SOURCE)
    body=d._element.body
    for child in list(body):
        if child.tag!=qn('w:sectPr'): body.remove(child)
    d.core_properties.title='A与B协议ResNet训练与对比实验报告'
    d.core_properties.subject='固定照片划分与人员互斥划分的手势识别实验'
    lines=text.splitlines(); i=0; fig_num=0; table_num=0; current_title=''
    width=d.sections[0].page_width-d.sections[0].left_margin-d.sections[0].right_margin
    while i<len(lines):
        line=lines[i].strip()
        if not line: i+=1; continue
        if line.startswith('|'):
            data=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',c) for c in row): data.append(row)
                i+=1
            table_num+=1
            compact=current_title.startswith(('10.4 ResNet34','10.5 ResNet50')) and data[0][0]=='类别'
            t=d.add_table(rows=1,cols=len(data[0])); t.style='Table Grid'; t.autofit=True
            n=len(data[0]); first=1.5 if n<=5 else 1.1
            widths=[first]+[(width/914400-first)/(n-1)]*(n-1)
            for grid,w in zip(t.columns,widths): grid.width=Inches(w)
            for j,row in enumerate(data):
                cells=t.rows[0].cells if j==0 else t.add_row().cells
                for cell,val,w in zip(cells,row,widths):
                    cell.width=Inches(w); cell.text=val
                    tcpr=cell._tc.get_or_add_tcPr(); margins=OxmlElement('w:tcMar')
                    for edge in ['top','left','bottom','right']:
                        m=OxmlElement('w:'+edge); m.set(qn('w:w'),'30' if compact and edge in ['top','bottom'] else '60'); m.set(qn('w:type'),'dxa'); margins.append(m)
                    tcpr.append(margins)
                    for p in cell.paragraphs:
                        p.paragraph_format.space_after=Pt(1 if compact else 3); p.paragraph_format.line_spacing=1.05
                        for r in p.runs: set_font(r,9 if n>=8 else 10); r.bold=(j==0)
                trpr=t.rows[j]._tr.get_or_add_trPr(); trpr.append(OxmlElement('w:cantSplit'))
                if j==0: trpr.append(OxmlElement('w:tblHeader'))
            d.add_paragraph().paragraph_format.space_after=Pt(3)
            continue
        m=re.match(r'!\[([^\]]*)\]\((.+)\)$',line)
        if m:
            path=Path(m[2]); fig_num+=1
            from PIL import Image
            with Image.open(path) as im: ratio=im.height/im.width
            preferred=6.2992126 if 'confusion' in path.name else 6.9291338
            w=min(width/914400,preferred,8.32/max(ratio,1e-9))
            p=d.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next=True
            shape=p.add_run().add_picture(str(path),width=Inches(w)); shape._inline.docPr.set('descr',m[1])
            p=d.add_paragraph(f'图{fig_num} {m[1]}','Caption'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        elif line.startswith('#'):
            h=re.match(r'^(#+)\s+(.*)',line); level=len(h[1]); title=h[2]
            current_title=title
            p=d.add_paragraph(title,'Title' if level==1 else 'Heading '+str(min(level-1,3)))
            # 两个独立模型结果与短历史附录从新页开始，避免单行表尾落在次页。
            if title.startswith(('10.4 ResNet34','10.5 ResNet50','附录 原A与B')):
                p.paragraph_format.page_break_before=True
            for r in p.runs: set_font(r,heading=True)
        else:
            p=d.add_paragraph(); inline(p,line.removeprefix('图：'))
        i+=1
    # 交付前核原A九张表的全部单元格和六张图的原始字节均未丢失。
    original=Document(SOURCE)
    def table_values(doc):
        return [[[c.text for c in row.cells] for row in table.rows] for table in doc.tables]
    def image_hashes(doc):
        return [hashlib.sha256(doc.part.related_parts[s._inline.graphic.graphicData.pic.blipFill.blip.embed].blob).hexdigest()
                for s in doc.inline_shapes]
    if table_values(d)[1:10] != table_values(original):
        raise ValueError('原A九张表未完整保留')
    if image_hashes(d)[:6] != image_hashes(original):
        raise ValueError('原A六张图未按原顺序保留')
    d.save(output)
    return {'figures':fig_num,'tables':table_num,'paragraphs':len(d.paragraphs),
            'A_original_9_tables_exact_cells_preserved':True,
            'A_original_6_figures_exact_bytes_preserved':True}

def update_project_readme():
    # 完整结果和正式报告落盘后，再一次更新项目入口。
    if not OUTPUT.exists() or read(B/'all_stages_complete.json')['status']!='PASS':
        raise RuntimeError('正式结果或报告尚未生成')
    path=V/'README.md'
    text=path.read_text(encoding='utf-8')
    if '## B人员互斥实验与新实验入口' in text:
        raise RuntimeError('README已更新，避免重复插入')
    old=text.split('\n\n',2)[1]
    first='本项目输入一张完整RGB照片，先定位手，再用ResNet分类，最后输出0—5整数。1前进、2后退、3明确停止、4左转、5右转；0表示没有有效动作指令，不发起或续期运动，不能与3明确停止混同。A按照片划分和B按人员互斥划分下的ResNet18／34／50训练、冻结测试及CPU测速现均完成，下面分别保留各自数据和证据入口。程序负责图片识别，真实摄像头、无手场景及机器人控制联调仍待评价。'
    text=text.replace(old,first,1)
    text=text.replace('## prepare_datasets.py 怎样工作','## prepare_datasets.py 的原始数据准备逻辑\n\n本节解释原脚本的实际行为。它的旧p2分支按人归档，后续正式B三组划分由独立的`prepare_B_datasets.py`完成；不能把旧p2文件夹与新B固定清单混为一谈。')
    text=text.replace('B当前会给每个人建文件夹','原脚本的B分支会给每个人建文件夹')
    text=text.replace('原始数据通过Windows目录联接接入，真实保存位置是同级 `HaGRID/dataset/hagrid-sample-30k-384p/`。','原工作目录的原始数据通过Windows目录联接接入，真实保存位置是同级 `HaGRID/dataset/hagrid-sample-30k-384p/`；完整复现包将入口保存为独立文件，迁移后不依赖该桌面联接。')
    text=text.replace('B没有在本次运行中重新整理或划分。','这里记录的是A初次准备；随后B已独立生成三组照片、标注和清单，详见下节。')
    text=text.replace('## A模型训练与测试','## A阶段模型训练与测试\n\n本节保留A阶段原始结果与当时的时延。A/B六模型当前速度比较采用后文同输入CPU补测，原时延在正式报告附录完整保留。')
    text=text.replace('B没有在本轮划分或训练。','这段记录A阶段；B的真实运行和结果见后文。')
    rows=read(B/'ab_comparison_results.json')
    cpu_result,cpu_models=matched_cpu()
    addition='## B人员互斥实验与新实验入口\n\n'
    addition+='B使用同一31,833张照片，按7,973名人员以随机种子42固定分组。三组分别为训练6,379人／25,274照片／30,799手、验证797人／3,457照片／4,326手、测试797人／3,102照片／3,759手。8∶1∶1按人数实施，照片比例为79.40%／10.86%／9.74%。三组人员、照片ID及照片内容哈希无交叉。A允许人员跨组，两套协议不要求相互排除照片。B从各自官方ImageNet初始化训练，没有继承A微调权重。\n\n'
    addition+=mdtable(['协议／模型','人工框准确率','完整照片准确率','整图宏F1','统一CPU分类均值／ms','统一CPU整图均值／ms'],[[r['protocol']+' '+r['model'],pct(r['crop_accuracy']),pct(r['photo_accuracy']),pct(r['photo_macro_f1']),f"{cpu_models[r['protocol']+'_'+r['model'].lower()]['summary'][CLASS_SCOPE]['mean_ms']:.2f}",f"{cpu_models[r['protocol']+'_'+r['model'].lower()]['summary'][WHOLE_SCOPE]['mean_ms']:.2f}"] for r in rows])
    addition+='A人工框／整图测试分母为3,917只手／3,180张照片，B为3,759只手／3,102张照片。速度列来自一次预先固定协议的补充测量：六模型使用相同100只A验证手图和相同100张A验证完整照片，CPU8线程、float32，每模型每项20次预热、3轮300次调用，并非300张独立照片；纯分类不含预处理，整图含解码、检测、裁图、分类和汇总，每次一张照片，内部多只手按原接口成批分类。模型加载不计时，其他进程估计CPU占用为7.4%—43.0%，全部慢轮保留，不能据同架构的小差异判断加速。原有各协议测试输入的时延完整保留在正式报告附录，不用两次测量差值宣称模型改进，也没有据速度补测重算准确率或改模型。\n\n'
    addition+='B18真实0误发运动5／1,695，真实停止误发运动0／173，但停止输出0仍有41／173。B18首次评估中断后，在同冻结条件下恢复；两次3,759手预测标签和指标相同，1,039手概率变化、最大绝对差0.0010422468185425，未按测试结果改模型或阈值。B50按patience=4在11轮自然早停，使用第7轮权重；它的首次测试被系统重启中断，随后以同权重、阈值及清单恢复，具体概率差异见正式报告。B是同来源未见人员留出，不是新设备、真实机器人或背景变化条件的外部验收。各模型完整错误方向、停止漏识别与冻结条件见正式报告。\n\n'
    addition+='```text\nVision_CA/\n├─ data/B_split_verified_20261102/\n│  ├─ summary.json                 人员、照片、手框及互斥统计\n│  ├─ train/                       25,274张照片及逐图manifest.csv\n│  ├─ val/                         3,457张照片及逐图manifest.csv\n│  └─ test/                        3,102张照片及逐图manifest.csv\n├─ runs/B_20261102_TPE/             B18、B检测器与共享证据\n│  ├─ compare_resnet34/            B34权重、搜索、日志与测试\n│  ├─ compare_resnet50/            B50权重、搜索、日志与测试\n│  ├─ trained_gradcam/             B18真实六类CAM及逐层张量\n│  ├─ report_figures/              冻结预测派生的正式科学图\n│  ├─ ab_comparison_results.json   六模型准确率、安全与耗时大表\n│  └─ 实验报告正文.md              正式A/B报告正文\n├─ portable_ab_project.py          完整包内只读核对数据与官方初始化\n└─ portable_B_training.py          新目录中的显式单阶段实验入口\n```\n\n'
    addition+='`prepare_B_datasets.py`负责实际人员互斥三组；`run_B_training.py`监督本次正式B各阶段，`run_B_pipeline.py`选择架构、B清单、初始化与输出；`postprocess_AB_results.py`只从既存预测和日志派生大表与图，不重推测试。`PROJECT_STATUS_B.md`保存当前项目状态，正式统计以冻结JSON和原始日志为准。原A与B的权重、清单、TPE数据库、逐样本预测和异常记录均保留。\n\n'
    addition+='### 使用已经训练的B模型识别照片\n\n三个B独立包位于同级`ResNet_B手势识别交付/ResNet18`、`ResNet34`、`ResNet50`；A包继续保留在`ResNet手势识别交付/`。每个包都有`predict.py`、`config.json`、两份权重、`requirements.txt`和自己的README。只需要识别图片时使用这些包，模型已包含，不需要重新训练。依赖安装可能联网，推理不下载权重。\n\n从本README所在的`Vision_CA`文件夹，在PowerShell运行：\n\n```powershell\nSet-Location ../ResNet_B手势识别交付/ResNet18\npython -m pip install -r requirements.txt\npython predict.py examples/1.jpg\npython predict.py "自己的完整照片.jpg" --json result.json\n```\n\n成功时标准输出只有一个整数。读图、权重或执行错误写标准错误并非零退出，不伪装成0；Python API的异常向调用方传播。多图调用可按包内README创建一次`GesturePredictor`后重复调用。握拳3优先，其余有效指令唯一则输出该指令，冲突或无有效指令则输出0；0类手不抵消另一只手的唯一有效动作。示例按验证照片ID固定选择，可能识别错误，合成空白图仅是接口示例。\n\n'
    addition+='上述相邻目录是原工作区布局。完整ZIP解压后的六个程序统一位于包根目录的`程序/A/resnet18`、`程序/A/resnet34`、`程序/A/resnet50`及`程序/B/resnet18`、`程序/B/resnet34`、`程序/B/resnet50`；完整源码位于`完整源码与数据/Vision_CA`，正式与历史Word位于`历史与正式文档`。解压后应阅读包内已经适配目录的README，不把原工作区相对路径直接用于新包。\n\n'
    addition+='### 在完整复现包中另建B实验\n\n完整复现包的`Vision_CA`应同时包含固定A/B照片清单、原标注和`pretrained/checkpoints/`中的四个官方初始化；原工作目录的历史绝对路径和旧裁图索引不能直接用于另一台机器。先在完整包自己的`Vision_CA`打开终端，读取该包依赖说明并安装对应Python环境，再运行只读检查和帮助：\n\n```powershell\npython portable_ab_project.py check\npython portable_B_training.py --list\npython portable_B_training.py --help\n```\n\n确实需要一轮新实验时，首阶段使用尚不存在的新目录；下列只是单阶段入口示例，不是识别照片的步骤：\n\n```powershell\npython portable_B_training.py prepare --out runs/new_B --arch resnet18\npython portable_B_training.py smoke --out runs/new_B --arch resnet18\npython portable_B_training.py features --out runs/new_B --arch resnet18\npython portable_B_training.py search --out runs/new_B --arch resnet18\npython portable_B_training.py finetune --out runs/new_B --arch resnet18\npython portable_B_training.py detector --out runs/new_B --arch resnet18\npython portable_B_training.py calibrate --out runs/new_B --arch resnet18\npython portable_B_training.py evaluate --out runs/new_B --arch resnet18\npython portable_B_training.py latency --out runs/new_B --arch resnet18\npython portable_B_training.py classifier-latency --out runs/new_B --arch resnet18\n```\n\n每条命令只执行一个阶段，默认分类搜索24次×18轮、末组微调上限18轮、检测16轮。`prepare`根据当前包照片重建新裁图与路径索引；34／50在同一个新`--out`下各自先执行`prepare`、`smoke`，再按自己的`features→search→finetune→calibrate→evaluate→latency→classifier-latency`接续，并复用新18裁图和新18检测器，不重复训练检测器。入口核对源码、manifest及官方初始化身份，拒绝写入原`A_20261102_TPE`或`B_20261102_TPE`，同一阶段不重复执行，并保留中断证据与串行锁。异常退出后需核已有阶段记录，不能删除锁后盲目重跑。原`run_A_budget.py`有历史截止时间，原B监督脚本与索引保留原运行身份；迁移复现使用上述新入口。\n\n'
    marker='## 在 VS Code 中使用 Codex'
    if marker not in text: raise ValueError('README结构发生变化，需要重新整合')
    text=text.replace(marker,addition+marker)
    old='最终实验报告为同级的[A协议ResNet训练与对比实验报告_v1.docx](../A协议ResNet训练与对比实验报告_v1.docx)。报告包含三模型大对比表、分类与整图混淆矩阵、真实错误照片、裁图差异和已训练ResNet18的分层Grad-CAM。'
    new='正式报告为同级的[A与B协议ResNet训练与对比实验报告_v2.docx](../A与B协议ResNet训练与对比实验报告_v2.docx)，包含完整A证据、B三模型训练与测试、六模型CPU大表、全部非零错误方向、实际裁图与冻结ResNet18分层Grad-CAM。原[A阶段报告v1](../A协议ResNet训练与对比实验报告_v1.docx)作为历史版本保留。'
    if old not in text: raise ValueError('README原报告链接发生变化，需要重新核对')
    text=text.replace(old,new)
    path.write_text(text,encoding='utf-8')
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def update_packaged_readme():
    # 完整包采用独立目录映射，避免把原工作区的相邻链接带进ZIP。
    text=(V/'README.md').read_text(encoding='utf-8')
    if '## B人员互斥实验与新实验入口' not in text:
        raise RuntimeError('先完成原工作区README再生成包内适配版')
    text=text.replace('三个B独立包位于同级`ResNet_B手势识别交付/ResNet18`、`ResNet34`、`ResNet50`；A包继续保留在`ResNet手势识别交付/`。',
                      '三个B独立包位于包根目录的`程序/B/resnet18`、`程序/B/resnet34`、`程序/B/resnet50`；A包位于`程序/A/resnet18`、`程序/A/resnet34`、`程序/A/resnet50`。')
    text=text.replace('Set-Location ../ResNet_B手势识别交付/ResNet18',
                      'Set-Location ../../程序/B/resnet18')
    text=text.replace('现在此路径已连接到实际数据，可在 VS Code 中逐层展开查看照片及标注。',
                      '本完整包在此路径保存原始数据文件，可在 VS Code 中逐层展开查看照片及标注。')
    note='上述相邻目录是原工作区布局。完整ZIP解压后的六个程序统一位于包根目录的`程序/A/resnet18`、`程序/A/resnet34`、`程序/A/resnet50`及`程序/B/resnet18`、`程序/B/resnet34`、`程序/B/resnet50`；完整源码位于`完整源码与数据/Vision_CA`，正式与历史Word位于`历史与正式文档`。解压后应阅读包内已经适配目录的README，不把原工作区相对路径直接用于新包。'
    text=text.replace(note,'本README位于解压包的`完整源码与数据/Vision_CA`。以下程序命令和文档链接均以这个文件夹为起点；包根目录README另外说明六个独立程序的选择和整体交接顺序。')
    marker='## 训练记录与独立预测包'
    if marker not in text:
        raise ValueError('缺少原训练记录入口，不能静默丢弃其功能说明')
    text=text.split(marker,1)[0]+'''## 训练记录与独立预测包

完整的A与B训练证据保留在本目录的`runs/A_20261102_TPE`与`runs/B_20261102_TPE`，包括权重、TPE数据库、训练日志、验证冻结记录、完整测试逐样本预测及独审。旧JSON或日志中的原机绝对路径仅用于追溯，另建实验使用`portable_B_training.py`重新生成当前路径。

| 用途 | 包内入口 |
|---|---|
| A三个独立图片预测程序 | [ResNet18](../../程序/A/resnet18/README.md)、[ResNet34](../../程序/A/resnet34/README.md)、[ResNet50](../../程序/A/resnet50/README.md) |
| B三个独立图片预测程序 | [ResNet18](../../程序/B/resnet18/README.md)、[ResNet34](../../程序/B/resnet34/README.md)、[ResNet50](../../程序/B/resnet50/README.md) |
| 当前完整项目方案 | [项目方案v11](../../历史与正式文档/基于ResNet18的手势识别项目方案_v11.docx) |
| A与B正式实测报告 | [实验报告v2](../../历史与正式文档/A与B协议ResNet训练与对比实验报告_v2.docx) |
| A阶段历史实测报告 | [A报告v1](../../历史与正式文档/A协议ResNet训练与对比实验报告_v1.docx) |

六个程序各含`predict.py`、配置、检测与分类权重及README。识别照片按各包说明安装依赖并运行即可；无需训练，也不依赖原电脑的TorchHub缓存。模型加载后可复用同一Python实例处理多张照片。

`prepare_training_delivery.py`用于原工作区A阶段的训练证据精选副本，不是训练或推理入口。本完整包直接保留原运行证据和复现所需数据，不依赖原工作区的`ResNet手势识别交付/训练记录`。历史项目方案及理解问答用于追溯讨论过程，早期未训练、B未执行或接口说明以项目方案v11、正式报告v2及当前程序README为准。
'''
    output=B/'packaged_project_README.md'
    output.write_text(text,encoding='utf-8')
    return {'path':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'archive_destination':'完整源码与数据/Vision_CA/README.md'}

if __name__=='__main__':
    text=report_text()
    (B/'实验报告正文.md').write_text(text,encoding='utf-8')
    counts=build_docx(text)
    counts.update({'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                   'output_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                   'output':str(OUTPUT)})
    (B/'ab_report_build_audit.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(counts,ensure_ascii=False))
