"""只读汇编已审科学事实与本作者正文；齐报告载荷后才独占创建最终事实。"""
from pathlib import Path
from hashlib import sha256
import json
import os
import sys
from collections import Counter
from docx import Document

R = Path(__file__).resolve().parent
V = R.parents[1]
P = V.parent
sys.path.insert(0, str(V))
from build_project_plan_v12 import REQUIRED_COVERAGE, check_facts

bound = {}
def read(path):
    path = Path(path)
    data = path.read_bytes()
    bound[str(path.resolve())] = sha256(data).hexdigest()
    return json.loads(data.decode('utf-8'))

def bind(path):
    path = Path(path)
    bound[str(path.resolve())] = sha256(path.read_bytes()).hexdigest()
    return str(path)

def paragraph(text, style='Normal'):
    return dict(kind='paragraph', text=text, style=style)

def table(headers, rows, widths):
    return dict(kind='table', headers=headers, rows=rows, widths=widths)

def mdtable(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---']*len(headers))+'|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])

def percent(x):
    return f'{x*100:.2f}%'

def fraction(x):
    return f"{x['count']}/{x['n']}（{percent(x['rate'])}）"

cpu = read(R/'label_variants_cpu_benchmark.json')
cpu_review = read(R/'label_variants_cpu_independent_review.json')
science = read(R/'aggregate_review.json')
programs = read(R/'twelve_packages_independent_review.json')
cam_review = read(R/'merged_gradcam_independent_review.json')
assert cpu['status'] == 'complete' and cpu_review['status'] == programs['status'] == cam_review['status'] == 'PASS'
assert science['status'] == 'PASS_MERGED_SIX_MODEL_AUDIT_WITH_LIMITATIONS'
for fn in ['MERGED_PROTOCOL.md', 'USER_AMENDMENT.json', 'all_training_evaluation_complete.json',
           'preflight_data_independent_review.json', 'packaged_project_README.md']:
    bind(R/fn)
bind(V/'deployment_src/gesture_predict.py')
bind(V/'train_A.py')
source = P/'基于ResNet18的手势识别项目方案_v11.docx'
bind(source)
bind(P/'A与B协议ResNet训练与对比实验报告_v2.docx')
bind(V/'runs/B_20261102_TPE/实验报告正文.md')
old_md_path = P/'ResNet_AB项目完整交接_20261102/交接说明.md'
bind(old_md_path)
oldmd = old_md_path.read_text(encoding='utf-8').split('\n\n')
doc = Document(source)
paras = [p.text for p in doc.paragraphs]
reviews = {}
rows = []
training_rows = []
raw_rows = []
negative_rows = []
negative_direction_rows = []
for proto in 'AB':
    for arch in ['resnet18', 'resnet34', 'resnet50']:
        review = read(R/proto/arch/'model_review.json')
        reviews[proto, arch] = review
        t = review['training']
        for name in ['fixed_head_summary.json','finetune_summary.json','merged_data_identity.json',
                     'label_statistics.json','calibration.json','test_detector.json']:
            path=R/proto/arch/name
            bind(path)
            if name != 'label_statistics.json':
                assert bound[str(path.resolve())] == review['bound_file_sha256'][str(path)]
        training_rows.append([proto+' '+arch.replace('resnet', 'ResNet'), str(t['head_epochs']), str(t['head_selected_epoch']),
                              str(t['finetune_epochs']), str(t['selected_epoch']), 'layer4＋fc'])
        pair = review['paired_original_vs_merged']
        for raw in ['peace', 'peace_inverted', 'two_up', 'two_up_inverted']:
            h = next(x for x in pair['raw_GT_hand_output_rates'] if x['raw_label'] == raw)
            ph = next(x for x in pair['raw_photo_folder_output1'] if x['raw_photo_folder'] == raw)
            raw_rows.append([proto+' '+arch.replace('resnet', 'ResNet'), raw, fraction(h['old_output1']), fraction(h['merged_output1']),
                             fraction(ph['old_command1']), fraction(ph['merged_command1'])])
        nh, np = pair['common_other_raw_GT_hands'], pair['common_other_raw_photos']
        negative_rows.append([proto+' '+arch.replace('resnet', 'ResNet'), fraction(nh['old_false_motion']), fraction(nh['merged_false_motion']),
                              fraction(np['old_false_motion']), fraction(np['merged_false_motion'])])
        truth = read(R/proto/arch/'photo_truth_test.json')
        common_ids = {x['image_id'] for x in truth if x['raw_labels'] and all(
            raw in nh['raw_labels'] for raw in x['raw_labels'])}
        for variant, prefix in [('不合并','unmerged'), ('合并','merged')]:
            cm = next(x for x in cpu['models'] if x['key'] == f'{prefix}_{proto}_{arch}')
            folder = Path(cm['run_folder'])
            oracle = read(folder/'test_oracle_crop.json')
            whole = read(folder/'test_whole_photo.json')
            om, wm = oracle['metrics'], whole['metrics']
            safety = whole['safety']
            speed = cm['summary']
            rows.append(dict(variant=variant, protocol=proto, architecture=arch,
                             oracle_n=om['n'], oracle_correct=sum(om['confusion_matrix'][i][i] for i in range(6)),
                             oracle_accuracy=om['accuracy'], whole_n=wm['n'],
                             whole_correct=sum(wm['confusion_matrix'][i][i] for i in range(6)),
                             whole_accuracy=wm['accuracy'], whole_macro_f1=wm['macro_f1'],
                             whole_support=wm['support'], whole_confusion_matrix=wm['confusion_matrix'],
                             zero_to_motion=safety['zero_to_motion'], stop_to_motion=safety['stop_to_motion'],
                             stop_to_zero=wm['confusion_matrix'][3][0],
                             cpu_classifier=speed['classifier_only_preprocessed_224x224_tensor'],
                             cpu_whole=speed['whole_photo_decode_detection_crop_classification_rules'],
                             source_folder=str(folder)))
            pred = {x['image_id']: x['prediction'] for x in whole['predictions']}
            direction = Counter(pred[k] for k in common_ids)
            assert sum(direction[k] for k in [1,2,4,5]) == (np['old_false_motion'] if prefix=='unmerged' else np['merged_false_motion'])['count']
            negative_direction_rows.append([variant, proto+' '+arch.replace('resnet','ResNet'), str(len(common_ids))] +
                                           [str(direction[k]) for k in [1,2,4,5]])

rows.sort(key=lambda x: (['不合并','合并'].index(x['variant']), x['protocol'], [18,34,50].index(int(x['architecture'][6:]))))
result_headers = ['任务／协议／模型','手图准确率','整图准确率','整图宏F1','CPU整图均值／P95(ms)']
result_rows = [[x['variant']+' '+x['protocol']+' '+x['architecture'].replace('resnet','ResNet'), percent(x['oracle_accuracy']),
                percent(x['whole_accuracy']), percent(x['whole_macro_f1']), f"{x['cpu_whole']['mean_ms']:.2f}／{x['cpu_whole']['p95_ms']:.2f}"] for x in rows]
safety_rows = [[x['variant']+' '+x['protocol']+' '+x['architecture'].replace('resnet','ResNet'),
                f"{x['zero_to_motion']}/{x['whole_support'][0]}",f"{x['stop_to_motion']}/{x['whole_support'][3]}",
                f"{x['stop_to_zero']}/{x['whole_support'][3]}"] for x in rows]
background = [b['cpu_background']['estimated_other_process_busy_percent'] for m in cpu['models'] for blocks in m['measurements'].values() for b in blocks]
speed_text = (f'本次十二模型主速度表来自一次共同输入CPU补测：A固定验证清单按图片编号排序的前100张完整照片，及按图片编号和手序号排序的前100张人工框手图。'
              f'CPU float32、8线程，模型提前加载，每模型每范围预热20次，三轮各100次，全部300次计入均值和P95，轮换模型顺序。'
              f'纯分类仅计预处理张量前向；整图计打开解码、检测、裁图、分类及规则，内部多手可成批分类。各模型使用自己的冻结阈值。'
              f'72个计时块的估计其他进程CPU占用为{min(background):.1f}%—{max(background):.1f}%，原始时间与自然慢轮全部保留。'
              f'该补测不重算准确率、不重选权重或阈值；小幅速度差不能归因于标签任务或A/B协议。原六模型补测及原测试时延保留为历史记录。')

replacement_text = {
6:'原六类表描述不合并版：0类占手框总数63.79%，原冻结头TPE比较训练频次加权，验证与测试保持原分布，未使用均衡采样。新版另训练合并任务，将two_up和two_up_inverted由0改为1，0类与1类分布随之变化；两版本的分母和逐类结果分开报告，不能将总体准确率变化直接解释为同一任务的泛化提升。',
7:paras[7]+'这项20,855计数沿用不合并标签；合并任务的手级与整图0分母另见第7章。',
20:'本项目由本人和组员共同负责ResNet识别分支：完整RGB照片先检测手框、逐手裁图分类，再由Python规则输出一个0—5整数。以ResNet18建立流程，并比较ResNet34与ResNet50；不合并和合并两种标签策略分别在固定A照片划分、B人员划分下完成真实训练与完整测试，共十二个独立识别程序。两种标签策略扩大了允许前进的姿态范围，须分别解释结果；控制端实际联调仍另行开展。',
23:'上述原编码表保留不合并版：peace和peace_inverted属于1，two_up和two_up_inverted属于0。新增合并版将这四种raw姿态全部定义为1，实际变更仅为两种two_up的0→1；其他映射、六个输出类别与原始标签均保留。数字2虽包含数据标签stop，控制含义仍是后退；握拳3才是明确停止，0是无新指令，不取消正在执行的运动。',
33:paras[33]+'这里的六行成绩均属不合并任务；新增合并训练与十二模型同输入速度见第7章。先确定控制端接受的姿态，再在相应任务内选择模型。',
114:paras[114]+'本节原A/B矩阵属于不合并任务，新增合并矩阵见第7.4节，原样本真值不翻写。',
130:paras[130]+'这里保留原不合并图的类别范围；合并任务1类还包含两种two_up，图中示例不代表已覆盖所有raw姿态。',
149:'0行是并拢伸出两指的two_up_inverted，在这张原不合并B模型图中真实类别为0；新合并任务把该raw定义为1，但不改写本图的旧模型、旧预测或旧真值。1行两指张开形成V形，主要响应覆盖手指下部与掌部；2行张掌背面、3行拳面与弯曲指节、4和5行拇指根及收拢手指均参与可见响应。浅层仍有衣服和背景边缘，不能说模型只看拇指方向；layer4的7×7响应上采样不会增加指尖定位证据。',
155:'（2）十二个独立程序保留不合并／合并、A／B和三架构的各自权重与配置。新六包完成72例CPU离线检查，同源保存结果逐项一致，其中67例预测与标签相同，5个自然错例原样保留；这不是新的72例准确率评估。接收组员先按任务选择目录，再复核接口，与控制组确认频率、有效期、0／3和异常处理，实际联调后另存整数与小车动作记录。',
156:'已形成两种标签任务共十二个A/B识别程序、整数与详细输出接口、真实测试结果和解释分析。接收核包、API复核及ResNet汇报主笔可由组员继续承担，我提供实验事实与技术复核；后续控制联调由识别与控制组共同完成，本次交付不等于课程整体演示已验收。',
160:paras[160]+'此为不合并映射；新增合并版只把two_up和two_up_inverted改为1，原raw标签、照片、手框及A/B组别不变。',
165:paras[165]+'该TPE属于不合并六模型。合并版每个协议／架构复用自己旧选参数，另训新标签分类头18轮，再从官方主干与新头微调，不新增TPE，也不载入旧已微调分类器。',
170:paras[170]+'这段记录原不合并六模型补测；本次两任务十二模型的主速度采用第7.2节新共同输入补测，两个计时批次不拼接。',
}
replacements = [dict(old=paras[i], new=text) for i,text in replacement_text.items()]
new_end = replacement_text[156]
chapter = [paragraph('7 合并与不合并标签任务的真实对照','Heading 1'),
 paragraph('7.1 任务定义、分母与新训练','Heading 2'),
 paragraph('不合并要求V形peace两姿态触发1，并拢两指two_up两姿态保持0；合并允许四种姿态都触发1。因此，相同输出1在两种two_up上会有不同对错。原图像素、人工手框、人员和A/B固定划分没有重做；这次比较首先回答控制指令覆盖范围和相伴错误如何变化。'),
 table(['协议测试','总手图／照片','旧手0→新手0','旧整图0→新整图0','旧1→新1'],
       [['A','3917／3180','2510→2149','1773→1412','351→712'],['B','3759／3102','2352→2016','1695→1359','354→690']],[1.05,1.35,1.4,1.5,1.2]),
 paragraph('上表1类手图数与整图1照片数在当前数据恰相等，仍属于不同计数单位。两协议停止照片均173。手图0分母不能代替整图0分母；共同其他负类也不能包括已改为1的two_up。'),
 paragraph('六个新分类器分别复用对应不合并协议与架构已选参数，固定新头各训18轮。其后加载官方ImageNet主干与这次新头，仅训练layer4和分类层，最多18轮、patience4；没有新TPE，没有旧微调分类器warm-start。检测器复用各自A或B原权重，不重新训练、不跨协议。最终六者都选择layer4微调阶段，实际轮次如下。'),
 table(['合并模型','新头轮数','头最佳轮','微调完整轮','微调最佳轮','选择阶段'],training_rows,[1.35,.85,.85,1.05,1.05,1.35]),
 paragraph('7.2 十二模型完整测试、安全错误与速度','Heading 2'),
 paragraph('人工框手图准确率只衡量已经裁好手的分类器；整图准确率和宏F1包含检测、实际裁图、阈值与多手汇总。以下两任务各用自己的固定真值，不能把分数变化当作同一任务的一次提升。'),
 table(result_headers,result_rows,[1.65,1.1,1.1,1.1,1.55]), paragraph(speed_text),
 paragraph('合并A18／34／50整图正确数为2924／2931／2954（各3180），合并B为2849／2848／2867（各3102）。原不合并六结果保留不变。两种任务中50的整图分数均较同协议18高，但更大架构耗时也更高，误运动和停止漏识别不随准确率单调改善。'),
 table(['任务／模型','真实0→运动','停止3→运动','停止3→0'], safety_rows,[2.1,1.5,1.45,1.45]),
 paragraph('运动指1、2、4、5。停止输出0表示没有识别到显式停止3，不是成功停止；0不取消当前运动。该表的真实0群体随任务变化，因此下一节另在同一共同负类上配对比较。'),
 paragraph('7.3 原始姿态与共同负类的配对结果','Heading 2'),
 paragraph('按同一(image_id,hand_index)连接人工框预测，四种raw各报告被预测为1的数目／支持数；整图按原照片文件夹归属报告最终1，保留完整照片上的检测和规则。这两种比率不能互换。下表箭头左为不合并、右为合并；完整逐模型细节同时保留在报告v3。'),
 table(['协议／模型','raw姿态','手图输出1：旧→新','整图输出1：旧→新'],
       [[a,b,c+' → '+d,e+' → '+f] for a,b,c,d,e,f in raw_rows],[1.3,1.3,1.9,2.0]),
 paragraph('共同其他手图仅取两任务都为0的九种raw：call、four、mute、no_gesture、ok、one、rock、three、three2。共同其他照片要求照片中每只标注手都属于这九种，不含新正类two_up，也不是把两版command0照片简单取交集。'),
 table(['协议／模型','手图旧误运动','手图新误运动','整图旧误运动','整图新误运动'],negative_rows,[1.3,1.3,1.3,1.3,1.3]),
 table(['任务','协议／模型','共同照片数','误发1','误发2','误发4','误发5'],negative_direction_rows,[.8,1.3,.9,.85,.85,.85,.85]),
 paragraph('扩大前进姿态覆盖并不保证共同负类更稳。A18共同负类整图误运动由15/1412变为30/1412；B50的共同负类手图误运动虽由18/2016降至16/2016，完整照片却由12/1359增至15/1359。应先选符合控制意图的标签策略，再权衡错误方向、停止漏识别和目标设备速度。'),
 paragraph('7.4 新合并模型的真实分层解释','Heading 2'),
 paragraph('下面是合并A和B已训练ResNet18，人工GT框裁出的单手输入，以真实类logit解释layer1—4，不是检测器或自动检测裁图CAM。A沿用旧六个样本，B的1—5行沿用旧样本；旧B第0行two_up_inverted在新任务为1，因而新B取排序首个合法0类no_gesture手图。原图15／16的旧身份完整保留。'),
]
for proto, fig in [('A',18),('B',19)]:
    path=R/proto/'resnet18/trained_gradcam/trained_six_class_matrix.png'
    bind(path);bind(path.with_name('metadata.json'))
    chapter.append(dict(kind='image',path=str(path),width_inches=6.5,alt=f'合并{proto}真实六类人工框单手分层Grad-CAM',
                        caption=f'图{fig} 合并{proto} ResNet18六类分层Grad-CAM。人工GT框、真实类logit，各图独立归一化；不能据颜色深浅跨层比较绝对贡献。'))
chapter += [paragraph('A的V形例主要覆盖伸指下部、指根与掌面；拳例覆盖弯曲指节与拳主体，拇指两例也包含收拢四指和掌面。B第0行是two_up照片中另一只自然下垂的no_gesture手，单手真值0而同照片整图真值1，说明0手不会抵消有效前进手。B握拳例的人工框分类为3，但冻结整图因没有保留检测框而输出0，仅说明该照片的定位漏检。'),
 paragraph('两张新矩阵的1行分别是peace、peace_inverted，没有展示two_up分类CAM；其新增姿态覆盖应读第7.3节全raw统计。浅层仍响应衣袖、窗口和背景边缘；layer4常覆盖掌心／指根或拳主体，不能称只看指尖或完全排除背景。原生7×7响应上采样不增加定位细节，位置关联也不是物理因果证据。'),
 paragraph('7.5 两种程序与后续协作','Heading 2'),
 paragraph('完整包按程序/不合并/{A,B}/resnet18、resnet34、resnet50与程序/合并/{A,B}/相应架构分开，共十二程序及十二独立ZIP。先核标签策略再选协议和模型，不能只换config中的标签文字冒充新训练结果。接收组员核包与API后主笔ResNet汇报，我提供实验事实与技术复核；与另一识别路线先约定同任务、分组、分母和速度范围，再共同比较。控制组提供入口与环境，双方保存真实联调记录。'),
 paragraph('先读最新报告v3与本方案v12；v11和报告v2保留不合并完整记录，更早Word仅追溯。完整源码与数据保留原图、固定清单、两任务运行证据、科学图和四份官方预训练；不包含个人媒体学习笔记及可重建裁图／特征缓存。移机训练须新目录及本机路径索引，不直接重跑旧固定controller或改冻结事实文件。')]

annotations=[dict(source_table_index=i,text=text) for i,text in {
1:'原六类数量表保留不合并标签统计；新合并版只改变two_up两raw的0→1，测试分母对照见第7.1节。',
4:'原数字与真实照片表对应不合并版；合并版在相同六输出中另将两种two_up定义为1，张掌2与握拳3不变。',
6:'以下A结果为不合并任务；表内CPU是原六模型计时批次，十二模型新主速度表见第7.2节。',
7:'以下B结果为不合并任务；表内CPU是原六模型计时批次，不能与第7.2节的新批次拼接。'}.items()]
insertions=[dict(after=paras[4],blocks=[paragraph('本方案同时保留不合并版与新增合并版。下方原统计及数字照片表属于不合并版；新增two_up并入1的任务定义、真实训练及十二模型对照集中见第7章。')]),
            dict(after=paras[79],blocks=[paragraph('本章保留不合并六模型的完整实施、结果及解释图；新增合并训练不重复TPE，其真实对照见第7章。以下原有TPE、六模型速度和CAM须按不合并身份阅读。')]),
            dict(after=new_end,blocks=chapter)]

# 在旧八章交接的基础上逐块更新，保留课程、接口、迁移与建议分工的实际内容。
h = list(oldmd)
h[0]='# ResNet合并与不合并标签任务完整交接说明'
h[1]='ResNet识别分支由我和组员共同负责。现保留不合并实验，并在相同固定A照片划分、B人员划分上另训合并标签的六个分类器，共十二个自带权重的CPU识别程序。方案v12和综合报告v3同时解释两种任务，旧模型、方案v11和报告v2原样保留。组员可先接收复核、主笔ResNet汇报，再与另一识别子组和控制组继续比较及集成。'
h[5]=mdtable(['数字','不合并版','合并版','控制约定'],[
 ['1','peace、peace_inverted','peace、peace_inverted、two_up、two_up_inverted','前进'],
 ['2','stop、stop_inverted、palm','相同','后退'],['3','fist','相同','明确停止'],['4','like','相同','左转'],['5','dislike','相同','右转'],
 ['0','其余raw手势；运行时无有效非0或冲突','相同，但不再包括两种two_up','无新指令']])
h[6]='0不发起或续期运动，也不是取消当前运动的命令；3才是明确停止。数据集名为stop的张掌标签属于后退2。两种two_up在不合并版为0、合并版为1，必须先选任务再看样例预期。识别器目前只产生数字，控制侧实际执行仍需联调。'
h[14]='原不合并六分类器各做24次冻结头TPE、每次18轮，再微调layer4和分类层；A实际18/18/18轮，B18/18/11轮，B50按原patience4自然早停、最佳第7轮。新增合并不再TPE：各协议／架构复用自己旧已选参数，按新标签另训新头18轮，再从官方ImageNet主干与这次新头微调，最多18轮／patience4；不从旧微调分类器继续训练。A和B检测器各自原权重复用、不重训。各模型的接收阈值在自己的验证结果中固定，测试不参与选型。\n\n'+mdtable(['合并模型','头轮数','头最佳','微调完整轮','微调最佳','阶段'],training_rows)
h[16]=mdtable(['工作','已完成内容','交接后用途'],[
 ['数据与语义','31,833原图标注、A/B固定名单、两种映射及真值分母核验','先确定允许触发1的姿态范围'],
 ['训练与测试','原六模型与新六模型均有真实完整冻结测试、配对raw与共同负类检查','比较各任务内部性能，保留旧身份'],
 ['解释与错误','旧A/B CAM保留；新A/B人工框CAM、实际检测对照与源张量','区分分类解释、定位失败和规则'],
 ['十二程序','六新包72例CPU离线检查，同源结果一致，67例标签相同、5自然错例保留；十二包总审核通过','接收复核与集成，不把接口样例当新准确率'],
 ['文档与速度','方案v12、报告v3、十二模型共同输入CPU测速，旧计时保留','按任务选型及准备展示']])
h[17]=mdtable(result_headers,result_rows)
h[18]='人工框手图先由标注裁好，只看分类器；整图包含自动找手、裁图、分类、阈值和多手规则。完整照片输入不能只用约99%的手图成绩描述。'+speed_text
h[19]='合并扩大了前进1的允许姿态，同一two_up的正确数字由0变1。手图0：A由2510变2149，B由2352变2016；整图command0：A由1773变1412，B由1695变1359。整体准确率、宏F1与类别不平衡因此变化，不能称为同一任务泛化改善。原A/B划分、像素及手框不变；B人员互斥仍只检验同一HaGRID来源，尚未证明真实摄像头外部泛化。\n\n共同负类只取两任务都为0的九种raw；照片要求全部标注手都属于这九种，排除two_up。下表运动为1、2、4、5；报告v3另保留四raw输出1与共同负类各误发方向。\n\n'+mdtable(['协议／模型','共同手图旧误运动','共同手图新误运动','共同照片旧误运动','共同照片新误运动'],negative_rows)
h[20]='停止3与无新指令0必须分别核验；停止输出0未识别到明确停止，不是成功停止。以下分母取各任务自己的真实0与真实3完整照片。\n\n'+mdtable(['任务／模型','真实0→运动','停止3→运动','停止3→0'],safety_rows)+'\n\n原不合并B18停止错例无自动框但人工框分类为3，只证明该例定位漏检，不能泛化为所有停止输出0。'
h[21]='原不合并阶段的恢复记录：'+h[21]
h[22]='旧不合并A/B六类CAM原样保留，旧B的two_up_inverted真0不改写成新真1。新增合并A/B ResNet18矩阵使用人工GT手框、真实类logit，解释分类器而非检测器。layer4主要响应常包含指根、掌心和拳主体；浅层仍有衣服、窗口或背景边缘。每图独立归一化，颜色不是概率或跨层绝对贡献，不能说只看指尖或排除全部背景。新两图1行是peace／peace_inverted，two_up表现读完整raw统计。新B0行是two_up照片中的no_gesture手，单手0而整图1，二者分母与任务不同。'
h[24]='第一次接手先读本交接，再读综合报告v3和方案v12，不用从v1逐版读。报告给两任务真实结果、参数和错误，方案保留完整系统原理及原图，并加入新任务对照。'
h[25]=mdtable(['顺序','文档或入口','阅读目的'],[
 ['1','根README、交接说明','完成范围、两种标签策略、目录与协作'],
 ['2','[综合实验报告v3](历史与正式文档/A与B协议ResNet训练与对比实验报告_v3.docx)','十二结果、raw子群、共同负类、安全、CPU与CAM'],
 ['3','[项目方案v12](历史与正式文档/基于ResNet18的手势识别项目方案_v12.docx)','来源、数字、完整流程、残差网络与两种真实训练'],
 ['4','[手势合并与数字编码](历史与正式文档/手势合并与数字编码.docx)','旧编码辅助理解张掌2和握拳3；two_up新版以v12/v3为准'],
 ['5','程序/不合并/或程序/合并/下A或B及相应架构README','先选任务，再运行CLI或详细接口'],
 ['6','[完整项目README](完整源码与数据/Vision_CA/README.md)','原图、固定清单、源码、训练证据与迁移边界']])
h[26]='方案v11、报告v2保留不合并已完成记录；方案v1—v10、A报告v1、总体理解v1和问题解答用于历史追溯。旧稿“B未分组”“六分类未训练”属于当时状态，当前以v12/v3和对应配置为准；旧“读图失败返回0”也已由CLI异常退出契约替代。历史修改意见不自动成为新待办。'
h[30]=h[30].replace('程序/A/resnet18','程序/不合并/A/resnet18')
h[31]='若选择合并，进入`程序/合并/A/resnet18`；也可选择相应B协议或resnet34／resnet50。十二目录各含predict.py、config.json、requirements.txt、weights、examples、samples.json、SHA256SUMS.json和README，使用各自真实权重、阈值与映射。推理小包是已核CPU便携入口；先按README准备兼容依赖，首次安装可联网，预测不下载模型。已核环境见各包README，新机器仍需检查本机依赖。'
h[37]='''```text
ResNet_合并与不合并完整交接_20261103/
├─ README.md、交接说明.md、交接说明.docx
├─ 历史与正式文档/         方案v12、报告v3、v11/v2及必要历史/课程
├─ 程序/不合并/{A,B}/{resnet18,resnet34,resnet50}/
├─ 程序/合并/{A,B}/{resnet18,resnet34,resnet50}/
├─ 独立程序ZIP/{不合并,合并}/{A,B}/   共12个独立ZIP
├─ 完整源码与数据/Vision_CA/   raw、固定A/B、源码、旧runs、MERGED、官方预训练
└─ SHA256SUMS.json           最终成员与校验清单
```'''
h[38]=h[38].replace('训练日志、TPE数据库、完整预测及解释证据','训练日志、原TPE数据库、两任务完整预测及解释证据').replace('程序/A中','程序/不合并/A中')+'合并科学记录在runs/MERGED_20261102，正式科学图和CAM保留。个人视频、学习笔记、关键帧、转写、会议大录屏及排版渲染不交付。'
h[39]='历史日志与Word保留原机路径，不整体替换。推理用十二小包即可；完整项目用于阅读和开发复核，移机训练详见其README。`python portable_ab_project.py check`只读核包；原不合并A/B各有portable_project.py和portable_B_training.py新输出入口。新增合并源码是原实验固定RUN的执行入口：移机新实验须独立目录、重建本机路径索引、裁图与特征缓存，核新标签和整图真值，再独立执行并审核。包内四份官方pretrained保留；裁图／特征缓存未打包。不能直接重跑旧controller覆盖原运行，也不能改科学JSON冒充原实验。'
h[40]+='最终大型ZIP保存于`E:/ResNet_合并与不合并完整交接_20261103.zip`；桌面项目同时保留新根交接、程序和源码。'
h[42]=h[42].replace('六程序','十二程序')
h[47]=h[47].replace('A/B报告及目标设备','v3两任务报告及目标设备').replace('比较整图错误、停止漏识别、时延与内存','先确认两种标签语义，再比较整图错误、停止漏识别、时延与内存').replace('写方法、数据、训练、A/B比较、错例与CAM','写方法、数据、两种真实训练、A/B与raw配对比较、错例与CAM')
h[48]=h[48].replace('A/B结果','两种任务与A/B结果')
handover='\n\n'.join(h).strip()+'\n'
rootreadme='''# ResNet合并与不合并完整交接

先读[交接说明](交接说明.md)或[Word版](交接说明.docx)，再读[综合报告v3](历史与正式文档/A与B协议ResNet训练与对比实验报告_v3.docx)和[方案v12](历史与正式文档/基于ResNet18的手势识别项目方案_v12.docx)。v11／v2保留不合并已完成记录，更早版本用于追溯。

两版都输出0—5一个整数。不合并只将peace两姿态定义为前进1；合并另将two_up两姿态定义为1。张掌2是后退，握拳3是明确停止，0无新指令且不取消正在执行的运动。先确定允许姿态，再选A照片划分／B人员划分和ResNet18／34／50。

程序入口为`程序/不合并/A/resnet18/`或`程序/合并/A/resnet18/`，其余协议和架构按同一目录规则选择，共十二程序。进入选定目录执行：

```powershell
python predict.py "examples/1.jpg"
python predict.py "你的照片.jpg" --json "result.json"
```

完整照片先经SSDLite检测、逐手ResNet分类，再按Python规则汇总。CLI成功仅打印一个整数；可选JSON保留每手框、概率和接收原因。错误写stderr并非0退出，不能当正常0。1＋1→1、1＋2→0、1＋3→3、1＋0→1均以手框和分类通过阈值为前提，不能保证所有手都被找到。

各程序README给依赖和CPU使用方法；[完整源码README](完整源码与数据/Vision_CA/README.md)给原图、固定清单、训练证据与移机开发边界。十二小包已离线核验，样例检查保留自然错例；完整测试准确率和十二模型共同输入速度请读v3。

`独立程序ZIP/不合并/`与`独立程序ZIP/合并/`保存十二独立ZIP。完整源码保留两任务冻结结果、真实科学CAM和四份官方预训练；可重建裁图／特征缓存及个人媒体学习笔记不收进总包。移机训练须独立新目录并重建本机路径索引，不覆盖原实验。

我与组员共同负责ResNet。组员可继续核包与API复核、主笔ResNet汇报；两路线比较依赖另一识别子组，控制实际联调依赖控制组入口，详见交接第8章建议分工。识别交付不等于ROS、闭环或课程整体验收。

根SHA清单按最终ZIP成员核验。大型ZIP位置：`E:/ResNet_合并与不合并完整交接_20261103.zip`。
'''

facts=dict(phase='PREPARATION', scope='双标签科学与程序技术事实审核完成；不表示用户已审阅新文稿',
           coverage={k:True for k in REQUIRED_COVERAGE}, comparison_rows=rows,
           body_replacements=replacements,insertions=insertions,table_annotations=annotations,
           handover_markdown=handover,root_readme_markdown=rootreadme,
           handover_sections_covered=['1任务与数字／模块职责','2数据与训练','3真实结果与错误','4阅读顺序与历史',
                                      '5CLI与API','6实际包内容与迁移','7课程未完工作','8共同ResNet角色、输入产物与依赖'],
           reviews=[dict(role='merged_science',path=str(R/'aggregate_review.json'),accepted_statuses=[science['status']]),
                    dict(role='twelve_programs',path=str(R/'twelve_packages_independent_review.json')),
                    dict(role='matched_speed',path=str(R/'label_variants_cpu_independent_review.json')),
                    dict(role='raw_label_comparison',path=str(R/'aggregate_review.json'),accepted_statuses=[science['status']],
                         scope='同一科学总审真实涵盖六模型已保存预测的raw配对及共同负类；不是另一份独立审核')])

# CAM全部新科学产物逐SHA继承独审绑定，不加载网络，不扫描数据或权重。
for path, expected in cam_review['bound_file_sha256'].items():
    if 'trained_gradcam' in path and 'MERGED_20261102' in path:
        assert sha256(Path(path).read_bytes()).hexdigest()==expected
        bind(path)
payload_path=R/'report_payload_v3.json'
if payload_path.exists():
    payload=read(payload_path)
    for key in ['report_analysis_markdown','report_execution_markdown','report_conclusion_markdown']:
        facts[key]=payload[key]
    bind(payload['report_analysis_markdown'])
    import re
    text=Path(payload['report_analysis_markdown']).read_text(encoding='utf-8')
    for image_path in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text): bind(image_path)
    # 报告作者明确列出的额外只读静态来源。
    for item in payload.get('source_files',[]):
        path=item['path'] if isinstance(item,dict) else item
        bind(path)
        if isinstance(item,dict): assert bound[str(Path(path).resolve())] == item['sha256']
facts['verified_files']=[dict(path=k,sha256=v) for k,v in sorted(bound.items())]
draft=R/'comparison_facts_author_draft.json'
draft.write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
if '--final' not in sys.argv:
    print(json.dumps({'draft':str(draft),'rows':len(rows),'bound_files':len(bound),'report_payload_present':payload_path.exists()},ensure_ascii=False))
else:
    assert payload_path.exists(), '报告载荷尚未齐'
    facts['phase']='FINAL_ACCEPTED'
    data=(json.dumps(facts,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    check_facts(facts)
    for path,expected in bound.items(): assert sha256(Path(path).read_bytes()).hexdigest()==expected, path
    output=R/'accepted_comparison_facts.json'
    with output.open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    print(json.dumps({'final':str(output),'sha256':sha256(data).hexdigest(),'bytes':len(data),'bound_files':len(bound)},ensure_ascii=False))
