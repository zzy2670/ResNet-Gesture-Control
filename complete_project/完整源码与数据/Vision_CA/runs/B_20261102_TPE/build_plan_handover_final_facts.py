"""汇总已冻结实验与共同输入CPU结果，生成方案、交接的最终事实文件。"""
from pathlib import Path
from hashlib import sha256
import argparse
import json

B = Path(__file__).resolve().parent
V = B.parent.parent
P = V.parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-review', required=True, type=Path)
    args = parser.parse_args()
    scientific = read(B / 'ab_final_independent_review.json')
    readiness = read(B / 'six_program_readiness.json')
    cpu_review = read(args.cpu_review)
    assert scientific['status'].startswith('PASS')
    assert readiness['status'] == 'PASS'
    assert cpu_review['status'].startswith('PASS')
    matched = read(B / 'matched_cpu_benchmark.json')
    assert matched['status'] == 'complete'
    # CPU复测只补速度；原有测试准确率和原始计时记录均不改写。
    comparison = read(B / 'ab_comparison_results.json')
    speed = {x['key']: x for x in matched['models']}
    normalized = {}
    for protocol in ['A', 'B']:
        normalized[protocol] = []
        for x in [r for r in comparison if r['protocol'] == protocol]:
            key = f'{protocol}_{x["model"].lower()}'
            summary = speed[key]['summary']
            whole_scope = next(k for k in summary if 'whole' in k)
            crop_scope = next(k for k in summary if k != whole_scope)
            assert summary[whole_scope]['n'] == summary[crop_scope]['n'] == 300
            normalized[protocol].append({
                **x, 'test_crop_accuracy': x['crop_accuracy'],
                'test_crop_macro_f1': x['crop_macro_f1'],
                'test_photo_accuracy': x['photo_accuracy'],
                'test_photo_macro_f1': x['photo_macro_f1'],
                'whole_cpu_mean_ms': summary[whole_scope]['mean_ms'],
                'whole_cpu_p95_ms': summary[whole_scope]['p95_ms'],
                'classifier_cpu_mean_ms': summary[crop_scope]['mean_ms'],
                'classifier_cpu_p95_ms': summary[crop_scope]['p95_ms'],
            })
    a, b = normalized['A'], normalized['B']
    reviews = [read(B / f'resnet{n}_final_independent_review.json') for n in [18, 34, 50]]
    assert [r['training']['finetune_complete_epochs'] for r in reviews] == [18, 18, 11]
    assert [r['training']['finetune_best_epoch'] for r in reviews] == [17, 18, 7]
    recovery = ('B18与B50的完整照片测试曾被中断，保留中断证据后，在模型、阈值和数据身份核验通过的条件下恢复同一测试阶段。'
                'B18两次已保存的3,759只人工框手图标签一致，1,039只手的概率有微小差异，最大绝对差0.00104225；'
                'B50有2,116只手的概率变化，最大绝对差0.00318676，其中一只真实2类stop_inverted手图由原平票输出0变为2。'
                '正式成绩统一采用恢复后自然完成的完整attempt2，不在两次结果中择高，也不把恢复视为独立重复实验。')
    cpu_text = ('六模型速度另作共同输入补测：使用A固定验证清单按图片编号排序的前100张完整照片，以及A验证人工框手图按图片编号和手序号排序的前100张手图。'
                '全部模型提前加载；CPU float32、8线程，每个范围预热20次，三轮均遍历全部100个输入，保留300次原始时间并轮换模型顺序。'
                '整图从打开解码到检测、裁图、分类和整数汇总，纯分类只计已预处理张量的前向。每次整图输入一张照片，内部多手按原接口成批分类；'
                '各模型保留已冻结阈值及实际手框工作量。计时块估计背景CPU占用约7.4%—43%，B18整图第三轮均值50.54 ms也完整保留；'
                '全部三轮共同汇总，不挑最快轮，也不能把小幅耗时差异直接归因于A/B划分。新补测不重算准确率，原测试集计时保留在各运行目录。')
    metrics_par = ('B三模型全部完成24次冻结头TPE搜索，每次18轮；随后18、34分别微调18个完整轮，最佳验证轮为17、18。'
                   '50按原连续4轮验证宏F1无提高的规则，在第11个完整轮后自然早停，最佳轮为7，没有为了补齐18轮而追加训练。'
                   'B手检测器独立训练16轮，各架构仅用B验证照片在12组阈值中选择，检测／分类阈值分别为0.3／0.3、0.3／0.3、0.3／0.5。')
    results_par = ('B18、34、50的人工框分类正确数分别为3,727、3,726、3,722／3,759；完整照片正确数分别为2,887、2,875、2,895／3,102。'
                   '人工框准确率为99.15%、99.12%、99.02%，整图准确率为93.07%、92.68%、93.33%，说明已经裁好手的分类成绩不能代替完整照片系统成绩。')
    safety_par = ('B三模型在1,695张真实0照片中分别有5、9、13张误发运动，在173张停止照片中分别有0、1、0张误发运动，另有41、42、40张停止照片输出0。'
                  '停止输出0并未识别出停止3。B18检测在score0.3、IoU0.5下有TP3,044、FP259、FN715，召回率80.98%；'
                  '漏框数与失败照片数不是同一个分母，单张照片也可能包含多只手。')
    compare_par = ('与各自A结果相比，B18、34、50整图准确率分别低0.55、0.46、1.01个百分点。B保持测试人员未参加训练，'
                   '但A与B测试照片构成、训练人员及独立训练检测器均不同，差值不能单独量出人员重叠的因果影响。'
                   '本次B50比B18少8张整图错误，却在真实0照片中多8张误发运动，人工框分类也未优于18；模型选择须一起考虑停止漏识别、误运动和目标设备速度。'
                   '人员互斥仍限于同一HaGRID来源，尚未证明真实摄像头或不同设备的外部泛化。')
    files = [B / 'ab_final_independent_review.json', B / 'six_program_readiness.json',
             B / 'ab_comparison_results.json', B / 'matched_cpu_benchmark.json', args.cpu_review,
             B / 'gradcam_independent_review.json', B / 'trained_gradcam/metadata.json',
             B / 'trained_gradcam/trained_six_class_matrix.png', B / 'report_figures/actual_crop_comparison.png',
             B / 'actual_crop_comparison.json', V / 'portable_B_training.py']
    files += [B / f'resnet{n}_final_independent_review.json' for n in [18, 34, 50]]
    verified = [{'path': str(p.resolve()), 'sha256': sha(p)} for p in files]
    b_rows = [
        ['人工框准确率', *[f'{x["crop_accuracy"]*100:.2f}%' for x in b]],
        ['人工框宏F1', *[f'{x["crop_macro_f1"]*100:.2f}%' for x in b]],
        ['整图准确率', *[f'{x["photo_accuracy"]*100:.2f}%' for x in b]],
        ['整图宏F1', *[f'{x["photo_macro_f1"]*100:.2f}%' for x in b]],
        ['整图错误数／3,102', *[str(x['photo_errors']) for x in b]],
        ['共同输入CPU整图均值 ms', *[f'{x["whole_cpu_mean_ms"]:.2f}' for x in b]],
        ['共同输入CPU整图P95 ms', *[f'{x["whole_cpu_p95_ms"]:.2f}' for x in b]],
        ['0误发运动／1,695', *[str(x['zero_to_motion']) for x in b]],
        ['停止误发运动／173', *[str(x['stop_to_motion']) for x in b]],
        ['停止输出0／173', *[str(x['stop_to_zero']) for x in b]],
    ]
    replacements = {
        '20': '本项目由本人和组员共同负责ResNet识别分支：输入一张完整RGB照片，自动找到手，按合并标签识别手势，输出0—5中的一个整数。以ResNet18建立完整流程，同时在A照片随机划分和B人员互斥协议下比较ResNet34与ResNet50。两协议的三模型训练、完整测试、解释分析和六个独立识别程序均已完成；控制端实际联调另行开展。',
        '33': ('本项目以ImageNet预训练ResNet18建立完整程序，并实测ResNet34与ResNet50。A三者整图准确率为93.62%、93.14%、94.34%，B为93.07%、92.68%、93.33%。'
               'A50比18少23张整图错误，B50比18少8张，但B50真实0误发运动更多。共同输入CPU补测见第6.5节；准确率、误运动、停止漏识别和速度需共同决定演示模型。'),
        '91': 'A三分类器各完成24次搜索与18轮微调，B三分类器各完成24次搜索，微调实际完成18、18、11轮；B50第7轮被选，第11轮按原规则早停。A、B检测器各训练16轮，各自保存完整验证与测试预测，不混用已训练参数。',
        '100': 'A、B均已形成完整测试预测、六类混淆、真实错误照片、实际裁图对照和CPU计时。人工框评价只检查已经找到手的分类能力，整图评价还包含检测、裁图、拒识与多手汇总；各项分母与原始结果分别保留。',
        '102': '解释分析使用已完成测试的固定模型、裁图方式和阈值，把分类响应对应到实际可见的手指、掌面与背景，并区分整图错误环节。A、B真实六类分层Grad-CAM均已生成，模型和预测身份保持不变。解释计算不更新参数，也未依据热图重新选模型或阈值；下面先保留参考原图和教学示例，再给任务模型的正式结果。',
        '127': 'A、B均保存真实六类分层总览及样本记录，包含原图、实际手图、原始标签、真值、预测、解释目标、目标层、裁图和权重身份。旧ImageNet示例用于解释计算，正式任务图使用对应协议训练后的ResNet18；整体性能以全部测试预测为准。',
        '130': '（2）六个独立程序均已检查六类、多手冲突、空白与错误路径，并在离线环境核验。接收组员可先复核同一接口，再与控制组确认数字、频率、有效期及异常处理；真正连接后另存整数与小车动作的实际记录。',
        '131': '已形成六个可运行的A/B识别程序、整数与详细输出接口、真实测试结果和解释分析。后续控制联调由识别与控制组共同完成；本次识别交付不等于课程整体演示已经验收。',
        '145': cpu_text + '控制端动作检查在实际接入后另记。',
    }
    cam_content = [
        {'kind': 'text', 'text': '图16按各真实类的图片编号和手序号排序取首个B测试手图，六行真实类0—5均预测正确；每行固定同一人工框输入，依次展示输入与layer4、layer1、layer2、layer3。它检验这些固定例子的类别响应，不能解释检测器有没有找到手。'},
        {'kind': 'image', 'path': str(B / 'trained_gradcam/trained_six_class_matrix.png'), 'width_inches': 6.5,
         'alt': 'B人员互斥协议已训练ResNet18六行五列Grad-CAM，同一行同一真实输入，六类真实与预测均相同。',
         'caption': '图16 B协议已训练ResNet18六类分层Grad-CAM。解释真实类logit，各图独立归一化；颜色不等于概率，不能比较跨层绝对强弱。'},
        {'kind': 'text', 'text': '0行是并拢伸出两指的two_up_inverted，属于本项目其他手势0；1行两指张开形成V形，主要响应同时覆盖手指下部与掌部。2行张掌背面含并拢手指与手背，3行握拳响应覆盖拳面和弯曲指节。4行拇指向上，5行拇指向下，响应还包含收拢手指与掌面，不能说模型只看拇指方向。浅层仍出现衣服和背景边缘响应，layer4的7×7响应经上采样也不会增加指尖定位证据。'},
        {'kind': 'image', 'path': str(B / 'report_figures/actual_crop_comparison.png'), 'width_inches': 6.5,
         'alt': 'B同一照片人工框与自动输入比较：0类人工裁图预测0但最高检测框输入预测4；握拳人工框预测3但自动检测未产生分类输入。',
         'caption': '图17 B测试照片的人工框分类与自动定位输入对照。上例右图仅为最高检测分数框，未声称一定与左侧真框匹配；下例没有通过阈值的自动框。'},
        {'kind': 'text', 'text': '上例真实0手图在人工框下分类为0，最高检测分数框形成的输入预测4，说明应先检查框对应和实际裁图内容；该对照没有证明哪个像素独自导致误判。下例握拳人工框分类为3，整图却因没有通过阈值的手框而输出0，属于该照片的定位漏检，不能据此归因全部停止输出0。没有自动分类输入时不补造分类CAM。'},
    ]
    plan_verified = verified + [{'path': str(V / 'build_project_plan_v11.py'), 'sha256': sha(V / 'build_project_plan_v11.py')}]
    plan = {'final_independent_review': 'PASS', 'verified_files': plan_verified, 'A_comparison': a,
            'replace_body_paragraphs': replacements, 'B_results_paragraphs': [metrics_par, results_par, safety_par, compare_par, recovery, cpu_text],
            'B_tables': [{'headers': ['指标', 'ResNet18', 'ResNet34', 'ResNet50'], 'rows': b_rows, 'widths': [2.55, 1.35, 1.35, 1.35]}],
            'B_cam_content': cam_content}
    old = (B / 'handover_working.md').read_text(encoding='utf-8')
    paragraphs = old.split('\n\n')
    replace = {}
    def set_prefix(prefix, text):
        original = next(x for x in paragraphs if x.startswith(prefix))
        replace[original] = text
    set_prefix('ResNet识别分支由我和组员', 'ResNet识别分支由我和组员共同负责。我们已经完成A按照片划分和B按人员划分的ResNet18、34、50实验，整理出六个自带权重的识别程序、完整源码与真实数据、正式方案和综合报告。组员可直接接收复核、使用现有结果准备汇报，并与另一识别子组和控制组继续比较及集成。')
    set_prefix('每种分类架构先固定ImageNet主干', '每种分类架构先固定ImageNet主干，对冻结特征上的分类头做24次TPE搜索，每次18轮，再解冻layer4和分类头微调。TPE比较优化器、学习率、权重衰减、Dropout、标签平滑与类别权重开关，不是整网搜索。A微调均完成18轮；' + metrics_par)
    set_prefix('| 工作 | 已完成内容', '| 工作 | 已完成内容 | 交接后用途 |\n|---|---|---|\n| 数据 | 原图标注、六类映射、A/B固定名单及隔离审核 | 确认每张照片归属，复现相同评价 |\n| A/B训练与测试 | 六分类器各24次TPE，A微调18/18/18轮，B为18/18/11轮；两检测器各16轮，全部固定测试完成 | 正式报告与六程序使用同一冻结结果 |\n| 解释与错误分析 | A/B真实六类分层CAM、错误照片、实际裁图对照及源张量 | 讲清真实形态与错误环节 |\n| 独立程序 | 六模型自带权重，完成离线、六类、多手、空白及异常检查 | 接收复核、集成与演示 |\n| 文档与速度 | 方案v11、A/B报告v2及共同输入CPU补测 | 理解方法、复核结果、选择演示模型 |')
    set_prefix('人工框手图先由标注裁好', '人工框手图先由标注裁好，只看分类器；整图包含自动找手、裁图、分类、阈值和多手汇总。实际输入是完整照片，不能只用约99%的手图成绩描述整套系统。上表CPU采用六模型共同输入补测。' + cpu_text)
    set_prefix('A中50较18少23张', compare_par)
    set_prefix('停止类必须保留漏识别', '停止类必须保留漏识别：A18/34/50在173张停止照片中分别有1/1/0张误发运动，还有35/32/32张输出0。' + safety_par + ' B18首个停止错例没有自动框，但人工框分类为3；只证明该例定位漏检。\n\n' + recovery)
    set_prefix('完整项目包含31,833张', '完整项目包含31,833张原图及标注、固定A和B各31,833张照片、对应手图索引、训练日志、TPE数据库、完整预测及解释证据。派生裁图和冻结特征缓存未重复收进总包，可按保留的原图、固定索引及预处理代码重建。四份官方预训练为ResNet18、34、50和MobileNetV3 ImageNet权重。程序目录用于迅速运行，完整项目用于开发复核；所有权威训练记录在完整源码与数据/Vision_CA/runs。程序/A中原README提到的“训练记录”是原独立交付布局，在本完整包请进入上述runs目录查阅，原程序文件和SHA保持不变。')
    set_prefix('原日志与历史Word保留原机器路径', '原日志与历史Word保留原机器路径；迁移时按完整项目README选择当前路径，不整体替换历史记录。在完整源码与数据/Vision_CA目录运行`python portable_ab_project.py check`只读核包；B新实验从`python portable_B_training.py prepare --out runs/new_B --arch resnet18`开始，使用包内四份pretrained/checkpoints权重并重建当前路径的手图索引。每条命令只执行一个阶段，按`--list`所列顺序继续；34/50复用这次新生成的裁图和18的检测器。新输出目录必须未被使用，历史A/B结果受到保护。迁移重训使用这一新入口，正式成绩仍对应本次已冻结的原运行。')
    set_prefix('下面是建议分工，供组内确认', '以下建议分工供组内确认。接收复核和汇报初稿可先推进，组员继续使用并检查已完成实验；两路线共同评价依赖另一子组提供程序和结果，控制联调依赖控制组提供接口和环境。')
    handover_verified = verified + [{'path': str(p), 'sha256': sha(p)} for p in [B / 'handover_working.md', V / 'build_AB_handover_docs.py']]
    handover = {'final_independent_review': 'PASS', 'six_programs_verified': True,
                'verified_files': handover_verified, 'report_filename': 'A与B协议ResNet训练与对比实验报告_v2.docx',
                'A_comparison': a, 'B_comparison': b, 'paragraph_replacements': replace}
    for name, value in [('project_plan_final_supplement.json', plan), ('handover_final_accepted_facts.json', handover)]:
        (B / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        print(name, sha(B / name))


if __name__ == '__main__':
    main()
