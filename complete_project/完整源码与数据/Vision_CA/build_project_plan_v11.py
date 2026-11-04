"""在v10原稿上修订研究方案，保留原图、历史文件和作者删改。

初稿仅写入B运行目录。只有显式给出已独审的B结果补充文件，才输出v11。
本脚本只编辑文档，不运行模型，不修改实验文件。
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile
import argparse
import json

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from lxml import etree

V = Path(__file__).resolve().parent
P = V.parent
B = V / 'runs/B_20261102_TPE'
SOURCE = P / '基于ResNet18的手势识别项目方案_v10.docx'
EXPECTED_SOURCE = '8d7ab102b43744335cb07d85055bec9d6fae04eaec48ec9c0f6df45c7d2e8b9d'
RED = RGBColor.from_string('C00000')
BLACK = RGBColor(0, 0, 0)


def sha(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def visible_image_hashes(doc):
    return Counter(sha256(doc.part.related_parts[x._inline.graphic.graphicData.pic.blipFill.blip.embed].blob).hexdigest()
                   for x in doc.inline_shapes)


def font_xml(rpr, east='宋体'):
    rf = rpr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts')
        rpr.insert(0, rf)
    for key in list(rf.attrib):
        if 'theme' in key.lower():
            del rf.attrib[key]
    for key in ('ascii', 'hAnsi', 'eastAsia', 'cs'):
        rf.set(qn('w:' + key), east)
    color = rpr.find(qn('w:color'))
    if color is not None:
        for key in list(color.attrib):
            if key != qn('w:val'):
                del color.attrib[key]
    sz = rpr.find(qn('w:sz'))
    szcs = rpr.find(qn('w:szCs'))
    if sz is not None:
        if szcs is None:
            szcs = OxmlElement('w:szCs')
            # szCs紧随sz，保持OpenXML规定的属性顺序。
            sz.addnext(szcs)
        szcs.set(qn('w:val'), sz.get(qn('w:val')))


def add_after(doc, anchor, text, style='Normal', red=True):
    p = doc.add_paragraph(style=style)
    anchor._p.addnext(p._p)
    run = p.add_run(text)
    run.font.color.rgb = RED if red and not style.startswith('Heading') else BLACK
    font_xml(run._r.get_or_add_rPr(), '黑体' if style.startswith('Heading') else '宋体')
    p.paragraph_format.keep_with_next = style.startswith('Heading')
    return p


def add_table_after(doc, anchor, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    anchor._p.addnext(table._tbl)
    table.autofit = False
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = value
    for values in rows:
        for cell, value in zip(table.add_row().cells, values):
            cell.text = str(value)
    for i, width in enumerate(widths):
        table.columns[i].width = Inches(width)
        for cell in table.columns[i].cells:
            cell.width = Inches(width)
    return table


def table_geometry(doc):
    for table in doc.tables:
        table.autofit = False
        pr = table._tbl.tblPr
        borders = pr.find(qn('w:tblBorders'))
        if borders is None:
            borders = OxmlElement('w:tblBorders')
            pr.append(borders)
        for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
            e = borders.find(qn('w:' + edge))
            if e is None:
                e = OxmlElement('w:' + edge)
                borders.append(e)
            for k, val in [('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')]:
                e.set(qn('w:' + k), val)
        for row_index, row in enumerate(table.rows):
            trpr = row._tr.get_or_add_trPr()
            if trpr.find(qn('w:cantSplit')) is None:
                cant = OxmlElement('w:cantSplit')
                header = trpr.find(qn('w:tblHeader'))
                if header is not None:
                    header.addprevious(cant)
                else:
                    trpr.append(cant)
            if row_index == 0:
                h = trpr.find(qn('w:tblHeader'))
                if h is None:
                    trpr.append(OxmlElement('w:tblHeader'))
            for cell in row.cells:
                tcpr = cell._tc.get_or_add_tcPr()
                margins = tcpr.find(qn('w:tcMar'))
                if margins is None:
                    margins = OxmlElement('w:tcMar')
                    tcpr.append(margins)
                for side, amount in [('top', '70'), ('left', '90'), ('bottom', '70'), ('right', '90')]:
                    x = OxmlElement('w:' + side)
                    x.set(qn('w:w'), amount)
                    x.set(qn('w:type'), 'dxa')
                    margins.append(x)
                if row_index == 0:
                    shd = tcpr.find(qn('w:shd'))
                    if shd is None:
                        shd = OxmlElement('w:shd')
                        tcpr.append(shd)
                    shd.set(qn('w:val'), 'clear')
                    shd.set(qn('w:fill'), 'E7EDF4')
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(3)
                    for run in p.runs:
                        font_xml(run._r.get_or_add_rPr())


def build(supplement=None):
    assert sha(SOURCE) == EXPECTED_SOURCE, '作者v10发生变化，须重新核对后再运行'
    s = None
    if supplement:
        supplement_sha256 = sha(supplement)
        s = read_json(supplement)
        assert s['final_independent_review'] == 'PASS'
        for f in s['verified_files']:
            assert sha(f['path']) == f['sha256'], f['path']
    doc = Document(SOURCE)
    source_paras = list(doc.paragraphs)
    source_tables = list(doc.tables)
    # 原稿此处人为换页在更新字号后产生仅两段的孤立页；允许自然排入本章。
    source_paras[8].paragraph_format.page_break_before = False
    original_images = visible_image_hashes(doc)
    source_headings = [(p.style.name, p.text) for p in source_paras if p.style.name.startswith('Heading')]
    # 原版红字转黑，最新改文保留红色，标题统一黑色。
    for rpr in doc.element.xpath('.//w:rPr'):
        font_xml(rpr)
        c = rpr.find(qn('w:color'))
        if c is not None:
            c.set(qn('w:val'), '000000')
    for style in doc.styles:
        if style.type == 1:
            font_xml(style.element.get_or_add_rPr(), '黑体' if style.name.startswith('Heading') or style.name == 'Title' else '宋体')
            style.font.color.rgb = BLACK
    doc.styles['Title'].font.size = Pt(22)
    doc.styles['Heading 1'].font.size = Pt(15)
    doc.styles['Heading 2'].font.size = Pt(12)
    doc.styles['Heading 3'].font.size = Pt(11)
    doc.styles['Normal'].font.size = Pt(11)
    for name in ('Title', 'Heading 1', 'Heading 2', 'Heading 3', 'Normal'):
        font_xml(doc.styles[name].element.get_or_add_rPr(), '黑体' if name != 'Normal' else '宋体')
    changes = []

    def replace(number, text):
        p = source_paras[number - 1]
        old = p.text
        assert not p._p.xpath('.//w:drawing'), f'不能替换图片段落{number}'
        for x in list(p._p):
            if x.tag != qn('w:pPr'):
                p._p.remove(x)
        r = p.add_run(text)
        r.font.color.rgb = BLACK if p.style.name.startswith('Heading') else RED
        font_xml(r._r.get_or_add_rPr())
        changes.append({'source_body_paragraph': number, 'old': old, 'new': text})
        return p

    replacements = {
        3: '本项目使用老师提供的HaGRID Sample 30K课程数据，训练与评价从完整照片输出数字指令的手势识别程序。HaGRID的全称为HAnd Gesture Recognition Image Dataset，是手势图像数据集。原始照片与JSON标注已在本地接入，A照片划分与B人员划分均已固定并核验。以下统计逐一读取18份JSON并与实际图片匹配得到，照片、手框和人员分别计数。',
        7: '0类占手框总数63.79%，明显多于各有效手势组。本次在冻结特征的参数搜索中比较是否按训练集频次加权，验证与测试均保留原分布；同时报告逐类结果，避免总体准确率被0类主导。实际训练没有使用均衡采样。',
        11: 'A按随机种子42在原始类别内随机分配照片，固定目录为Vision_CA/data/A_split_verified_20261102/P1_random_image。训练、验证、测试分别有25,468、3,185、3,180张照片，6,999、1,876、1,931名人员，31,095、3,872、3,917个手框。每组均有照片、标注与manifest.csv逐图清单；完整性、原图字节和标签已核验，三份清单已用于完成A实验。',
        12: 'A允许同一人的不同照片跨组，因此三组人数不能相加当作独立人员总数，A成绩也不能单独代表换人识别能力。旧P1_random_image中的summary.json只作为历史记录；当前实验始终使用上述新目录和已冻结清单。',
        13: 'B已经按随机种子42完成user_id人员分组，固定目录为Vision_CA/data/B_split_verified_20261102。训练、验证、测试分别有6,379、797、797人，25,274、3,457、3,102张照片，30,799、4,326、3,759个手框。例如某人被分入B训练组，其全部照片就只在训练组，验证和测试均不会再出现此人。',
        14: 'B的目标8∶1∶1按人数计算，实际人数比例为80.0075%、9.9962%、9.9962%；每个人照片数不同，照片比例相应为79.3956%、10.8598%、9.7446%。B三组人员编号、图片编号和精确内容SHA256两两交集均为0；31,833张原图的内容哈希全部不同，源照片和原标注均已保留。',
        15: '两套协议各自固定清单后分别训练，均覆盖全部18种原始主动手势及合并后的六类。定位器和分类器共用本协议训练照片；B从官方通用预训练参数重新开始，不加载A训练后的定位器或分类器。测试人员、照片和手框不参与B训练，也不根据测试成绩重新挑选划分。',
        17: '训练集用于学习定位和分类；验证集用于选择训练设置、模型轮次与接收阈值；测试集用于固定后的最终评价。已核对A、B的逐图清单、原图与标签完整性，并对B核实人员互斥。整图正确数字根据一张照片的全部人工手框按同一规则生成，不能用某个类别的手框数代替该类别整图照片数。',
        20: '本项目由本人和组员共同负责ResNet识别分支：输入一张完整RGB照片，自动找到手，按合并标签识别手势，输出0、1、2、3、4、5中的一个整数。以ResNet18建立完整流程，同时在相同协议下比较ResNet34与ResNet50。已完成A三模型实验、独立程序和离线接口检查；B状态及结果按本节和第6章的实际记录说明。',
        21: '识别程序提供数字输出接口，后续与控制组共同核对3取消运动、0不发起或续期运动的约定；当前识别结果不代表控制端已经联调或课程展示已完成。',
        33: '本项目以ImageNet预训练ResNet18建立完整程序，并实测ResNet34与ResNet50。A中三者整图准确率分别为93.62%、93.14%、94.34%，同机CPU整图均值分别为32.25、45.88、54.76 ms。ResNet50较18少23张错误，但每张照片平均增加约22.50 ms；34在本次A准确率与耗时上未取得更好的组合。模型选择因此同时考虑真实识别错误和运行开销，B人员互斥结果另行比较。',
        39: '网络结构规定卷积层、残差连接和分类层怎样连接；模型参数则是训练得到的具体数值。采用现成ResNet结构，不等于已经拥有本项目的手势分类器。本次从官方ImageNet 1000类预训练参数开始，保留前面的特征提取部分，把ResNet18和34的512→1000改为512→6，把ResNet50的2048→1000改为2048→6。新层先在冻结特征上训练，再解冻layer4与分类层进行微调，实际流程见第6.3节。',
        46: '图3为训练前制作的网络计算教学示例：手框取自HaGRID标注，彩色特征由通用ImageNet预训练ResNet18主干实际计算，并按各图自身范围着色；末端只展示当时512→6的结构，没有虚构预测概率。当前项目已有完成训练的六分类模型，但不把旧教学图改称其识别结果。图3解释中间特征计算；第6.6节Grad-CAM针对某类分数解释位置响应，二者作用不同。',
        66: '本次另用A_split_verified_20261102目录固定A照片清单，用prepare_B_datasets.py生成B_split_verified_20261102目录中的人员互斥清单，旧P1/P2统计保留为历史。两套新清单都已经核验。原始手框标签按第3章转换为六类数字，供定位训练、裁手分类和整图测试使用；实际分组规模见第1.2节。',
        68: '图7 早期Vision_CA数据准备示意。图中“仅有summary统计”的状态属于当时的旧输出；本次实际使用的新A、B三组清单均已生成并核验，数量与位置见第1.2节。',
        75: '上述资源承担不同工作：HaGRID及Sample 30K发布版本提供照片和标注；Vision_CA整理照片及其归属；TorchVision提供网络代码与通用参数；PyTorch执行训练。本项目已把数据核验、自动手定位、六类分类、裁图、多手规则和数字接口连接起来，A与B分别保存模型及实验记录。',
        80: '（1）读取A与B各自固定目录中的train、val、test三份manifest.csv逐图清单，核对照片路径、图片编号、人员编号和所属组。A按照片随机划分，B按人员划分；两套清单均已生成、核验并用于对应实验，不再从旧summary反推照片归属。',
        81: '（2）逐组统计照片、人员及各类手框，核实有效手势覆盖。同一原图及其裁图、增强版本保持在同一组，B三组人员完全互斥。A、B分别保存数据审核记录；定位与分类仅使用对应协议的训练照片，测试清单不参与选型。',
        82: '已形成：A、B逐图归属清单、完整照片与标注、照片和手框六类统计，以及B人员／图片编号／内容哈希隔离审核记录。',
        84: '（1）按第3章转换六类标签，并保留原始手势标签。完整照片及全部人工手框训练检测器；人工框裁出的手图及正确数字训练分类器。以框中心取边长为原长边1.2倍的正方形：长边两端各留原长边10%，短边方向补足正方形；超出照片处补黑，再缩放为224×224 RGB并作ImageNet标准化。',
        85: '（2）微调训练使用水平翻转、不超过10°旋转和轻度颜色扰动；检测训练同步翻转照片与手框。验证、测试没有随机增强。类别权重只从训练手框频次计算，并作为冻结头搜索开关；本次未使用均衡采样、随机手框扰动或预测框追加分类训练。',
        86: '已形成：检测器读取完整照片，分类器读取固定规则裁出的手图；训练增强与验证／测试处理分开保存，A与B的手图索引均保留原照片归属。',
        88: '（1）构建SSDLite320 MobileNetV3，采用官方ImageNet主干起点，检测输出设为背景和手两类，用本协议完整训练照片及人工手框训练。检测器内部将照片处理为320×320，输出框再对应回原照片。A、B各训练一套手检测器，同一协议三种分类架构共用该检测器。',
        89: '（2）先固定官方ImageNet主干，提取训练和验证手图特征，仅对Dropout与线性分类头进行Optuna TPE搜索：随机种子42、前4次随机启动、共24次试验，每次18轮。比较优化器、学习率、权重衰减、Dropout、标签平滑和类别权重开关，以验证宏平均F1选择。这是冻结特征上的头部搜索，不是整网或检测器的联合搜索。',
        90: '（3）再解冻layer4和分类层，其余层及其BatchNorm保持固定，用训练手图和正确数字微调：模型输出六类分数，与正确标签计算误差，再由AdamW更新可学习参数。微调上限18轮，连续4轮验证宏F1无提高可早停，保存完整验证轮中宏F1严格提高时的模型。检测器上限16轮，以验证检测F1选择轮次；本次不合并训练与验证组重训。',
        91: '已形成：各协议、各架构的冻结头搜索记录、微调日志与最佳分类参数，以及本协议手检测器记录。A三分类器各完成24次搜索和18轮微调，检测器完成16轮；B以最终运行记录为准。',
        93: '（1）仅用完整验证照片选择检测和分类接收阈值。检测分数比较0.3、0.5、0.7，最高类别概率比较0.3、0.5、0.7、0.9，共12组；按整图宏F1、停止误发运动、0误发运动及更高阈值的既定顺序选择。本次没有前两类分差阈值。模型身份和阈值冻结后再进入测试。',
        95: '（3）独立程序已检查六类、双手冲突和空白照片的数字输出。识别与控制端约定3取消当前运动，0不发起新运动，也不延长已有指令有效时间。该运动语义仍需在控制端实际接入后检查；CLI和Python调用方式及读图异常处理见附录A。',
        96: '已形成：协议内选定的分类器和检测器、验证冻结阈值，以及完整照片到单个数字的可运行流程。三种架构各有自己的分类阈值，不能直接套用另一模型的设置。',
        99: '（2）记录逐类混淆、所有非零误判方向、0误发运动、停止误发运动和停止输出0；回看人工框、自动框及实际裁图，区分找手、裁图、分类和拒识问题。A与B分别使用自己的完整固定测试清单；原计划的明暗、模糊、遮挡变换测试尚未实施，不计入原图测试成绩。',
        100: '已形成的A结果包括完整测试预测、六类混淆、真实错例、实际裁图对照和CPU时延。A三种架构均有人工框与整图结果；B在同样口径下单独记录。人工框评价只检查“已经找到手”的分类能力，整图评价还包含检测、裁图、拒识与多手汇总。',
        102: '解释分析使用已完成测试的固定模型、裁图方式和阈值，把分类响应对应回照片中实际可见的手指、掌面与背景，并区分整图错误环节。A已生成真实六类分层Grad-CAM，B只在对应最终测试锁定后生成。解释计算不更新参数，也不根据测试样本重新挑权重或改阈值；下面保留参考原图和教学示例，再说明任务模型的正式结果。',
        116: '图13 训练前制作的六类手图分层布局示例。每行按来源标签0—5组织，列为输入、layer4主解释、layer1、layer2、layer3。热图由通用ImageNet ResNet18实际计算，每行标注其真实ImageNet目标；保留本图用于说明布局，不将它改称本项目六类识别结果。',
        117: '正式总览沿用图13的六行五列：一行始终使用同一张实际分类输入，各列方向、裁剪范围和坐标一致。A与B分别加载本协议已经测试的固定ResNet18，以该行真实类别的softmax前分数为目标。按image_id与hand_index排序选择各真实类的首个测试手图，保留真实标签、预测及目标；预测与真值不同时另比较预测目标，不能为图好看而重选模型或划分。',
        127: 'A已保存按0—5组织的真实分层总览及对应样本记录，记录包含原图、实际手图、原始标签、真值、预测、目标层、解释类别、裁图和权重身份。旧ImageNet示例只承担教学作用，正式任务解释使用对应协议训练后的模型。局部放大与错误目标对照根据真实病例需要补充，整体性能仍以全部测试预测为准。',
        129: '（1）整理A、B训练与预测源码、固定清单、标签映射、模型参数、依赖版本、逐样本预测、阈值及真实报告。运行说明写清照片位置、模型加载、CLI与Python接口及异常处理；Grad-CAM程序、正式矩阵和样本身份随实验记录保留。组员包保留项目方案和正式实验图表，不分发个人图文学习笔记及讲解视频素材。',
        130: '（2）A三种独立识别程序已用六类、双手和空白图片检查输出与离线调用；B程序须按自己的权重和阈值完成同样检查。与控制组连接后，再用真实照片核对收到的数字和动作、3取消运动、0不续期运动，保存实际联调记录。',
        131: 'ResNet由本人和组员共同负责。建议接手组员先核包并复现接口，再主笔ResNet汇报和结果比较，本人复核技术内容；两人共同与控制组确认接口并联调。这是后续分工建议，实际安排需由组内确定；现有识别实验不替代控制联调、PPT提交与课堂验收。',
        134: 'A固定目录为Vision_CA/data/A_split_verified_20261102/P1_random_image，B为Vision_CA/data/B_split_verified_20261102；各自train、val、test均包含images照片文件夹、annotations标注文件夹与manifest.csv逐图清单，根目录有summary.json统计文件。原始数据、旧P1/P2统计及A全部结果保持保留，当前实验只读对应固定清单。',
        137: '保存原图、人工框、预测框和实际裁图对照，核对原图与检测器内部缩放的坐标变换。分类裁图以框中心和长边1.2倍确定正方形，超出照片补黑，缩放224×224；ImageNet均值为0.485、0.456、0.406，标准差为0.229、0.224、0.225。验证与测试不随机扰动框，也不改变类别分布。',
        139: '定位器用ssdlite320_mobilenet_v3_large构建，加载官方ImageNet MobileNetV3主干起点，设置背景／手两类。检测器训练batch24、SGD学习率0.003、动量0.9、权重衰减0.0005、余弦调度、混合精度、梯度裁剪10，计划16轮；在验证照片按score≥0.5和IoU≥0.5作一对一匹配，以F1选择完整轮。A、B分别训练，不使用A已学手检测器初始化B。',
        140: '分类器使用对应resnet18、resnet34或resnet50构建函数及官方ImageNet V1参数。冻结特征TPE搜索范围为AdamW或SGD、学习率0.0001至0.03、权重衰减0.000001至0.01、Dropout 0至0.4、标签平滑0至0.12、类别权重开关。每次试验18轮，24次试验后选验证宏F1最高者；该预算不代表已经找到全局最优。',
        141: 'layer4与分类头微调采用AdamW，起始学习率分别为0.0001和0.0005，batch128、余弦调度、混合精度、梯度裁剪5；前层保持冻结。记录实际轮数和被选轮次，并保存每轮完整验证结果。训练组与验证组没有合并重训，最终测试使用所选checkpoint；A与B参数、阈值及预测文件分别保存。',
        143: 'CLI调用格式为python predict.py 照片路径：成功时标准输出只有0—5中的一个整数；照片不存在、损坏或模型读取失败时，错误写入stderr并以非0退出，不返回正常整数0。Python接口为GesturePredictor.predict(image_path)，成功返回整数，异常由调用方处理；predict_details可读取框、分数和汇总原因。完整评价仍保留失败照片并单独记录，不能因异常剔除分母。',
        145: 'CPU计时先加载模型并预热，再从打开和解码完整照片开始，计入检测、裁图、预处理、分类与整数汇总；不包含冷启动导入和权重读取。同机Intel Core Ultra 9 285H、8线程、batch1，整图使用测试清单前100张照片、10次预热，报告均值和P95。模型纯分类另用相同100张预处理手图、20次预热、3轮共300前向，不与整图耗时混称。控制端动作检查在实际接入后另记。',
    }
    for number, value in replacements.items():
        replace(number, value)
    # 评价口径表先于新增实测结果；移动完整表块，不拆单元格内容。
    source_paras[99]._p.addprevious(source_tables[4]._tbl)

    # 新增表只作真实数量和指标的并列比较，原五表保持原位。
    a = add_after(doc, source_paras[14], 'A与B固定三组规模如下。A三组人员存在交叉；B三组人员可相加得到7,973人。')
    add_table_after(doc, a, ['协议与子集', '人员', '完整照片', '手框'], [
        ['A训练', '6,999', '25,468', '31,095'], ['A验证', '1,876', '3,185', '3,872'], ['A测试', '1,931', '3,180', '3,917'],
        ['B训练', '6,379', '25,274', '30,799'], ['B验证', '797', '3,457', '4,326'], ['B测试', '797', '3,102', '3,759']], [2.3, 1.3, 1.5, 1.5])

    a = add_after(doc, source_paras[41], 'ResNet18和34均使用两层3×3卷积的BasicBlock，四阶段分别为2／2／2／2与3／4／6／3个块；ResNet50使用1×1、3×3、1×1的Bottleneck，四阶段为3／4／6／3个块。50的最终特征维数为2048，18和34为512。三者都保留残差连接，差别包括块结构与计算量，不只是名称中的层数。')

    h = add_after(doc, source_paras[99], '6.5.1 A协议已完成结果', 'Heading 3', False)
    a = add_after(doc, h, 'A测试包含3,917个人工框手图和3,180张完整照片。三模型使用相同照片清单、预处理与训练规则，检测器共用；各架构独立搜索分类头和选择分类阈值。结果如下，完整逐类指标与错误方向保存在实验报告。')
    aa = s.get('A_comparison') if s else None
    if aa is None:
        aa = read_json(V / 'runs/A_20261102_TPE/comparison_results.json')
    table = add_table_after(doc, a, ['指标', 'ResNet18', 'ResNet34', 'ResNet50'], [
        ['人工框准确率', *[f'{x["test_crop_accuracy"]*100:.2f}%' for x in aa]],
        ['人工框宏F1', *[f'{x["test_crop_macro_f1"]*100:.2f}%' for x in aa]],
        ['整图准确率', *[f'{x["test_photo_accuracy"]*100:.2f}%' for x in aa]],
        ['整图宏F1', *[f'{x["test_photo_macro_f1"]*100:.2f}%' for x in aa]],
        ['整图错误数／3,180', *[str(round((1-x['test_photo_accuracy'])*3180)) for x in aa]],
        ['CPU整图均值 ms', *[f'{x["whole_cpu_mean_ms"]:.2f}' for x in aa]],
        ['CPU整图P95 ms', *[f'{x["whole_cpu_p95_ms"]:.2f}' for x in aa]],
        ['0误发运动／1,773', *[str(x['zero_to_motion']) for x in aa]],
        ['停止误发运动／173', *[str(x['stop_to_motion']) for x in aa]],
        ['停止输出0／173', '35', '32', '32']], [2.55, 1.35, 1.35, 1.35])
    a = doc.add_paragraph('A18人工框有26个错误，整图有203个错误，其中183张有效手势照片输出0、17张真实0照片输出有效手势、3张有效手势互混。检测器在score0.3、IoU0.5下有TP3,240、FP431、FN677，召回率82.72%；677个漏框不等于677张失败照片。优先检查小手定位与裁图，比只提高已经很高的人工框分类率更直接。')
    table._tbl.addnext(a._p)
    for r in a.runs: r.font.color.rgb = RED

    # 真正已训练的A CAM追加，不替换受作者要求保留的参考与教学图。
    h = add_after(doc, source_paras[126], '6.6.4 已训练模型的六类分层解释', 'Heading 3', False)
    a = add_after(doc, h, '图15使用A测试中每个真实类别按图片编号与手序号排序的首个手图，六例均预测正确；这是固定选例规则得到的结果。各行固定同一人工框输入，列顺序为输入、layer4、layer1、layer2、layer3，解释目标为真实类别。人工框解释不用于说明漏检。')
    pic = doc.add_paragraph()
    a._p.addnext(pic._p)
    pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    shape = pic.add_run().add_picture(str(V / 'runs/A_20261102_TPE/trained_gradcam/trained_six_class_matrix.png'), width=Inches(6.50))
    shape._inline.docPr.set('descr', 'A协议已训练ResNet18六类分层Grad-CAM；六行真实类0至5，五列同一输入及layer4、layer1、layer2、layer3。')
    pic.paragraph_format.keep_with_next = True
    c = add_after(doc, pic, '图15 A协议已训练ResNet18六类分层Grad-CAM。真实测试手图与固定任务权重，各图独立归一化；该图不表示跨层响应强弱或总体性能。', 'Caption')
    a = add_after(doc, c, '0类示例可见多指展开，主要响应包含掌心与指根；V形例覆盖两指下部和指根，张掌例覆盖掌面与指根，握拳例覆盖拳面和指节。拇指向上与向下两例含拇指根、收拢手指与掌部响应，不能据此说模型只根据拇指方向判断。浅层还对衣服和背景边缘有响应。layer4只有7×7原生网格，应结合输入中的指形观察，不能把平滑热图当作精细指尖定位。')

    if supplement:
        # 补充文件是root与独审者提供的冻结文稿片段，不自动读取正在写入的结果。
        for key, value in s.get('replace_body_paragraphs', {}).items():
            replace(int(key), value)
        bh = add_after(doc, a, '6.6.5 B协议已训练模型解释', 'Heading 3', False)
        bcur = bh
        for item in s.get('B_cam_content', []):
            if item['kind'] == 'text':
                bcur = add_after(doc, bcur, item['text'])
            elif item['kind'] == 'image':
                q = doc.add_paragraph()
                bcur._p.addnext(q._p)
                q.alignment = WD_ALIGN_PARAGRAPH.CENTER
                shape = q.add_run().add_picture(item['path'], width=Inches(item.get('width_inches', 6.5)))
                shape._inline.docPr.set('descr', item['alt'])
                q.paragraph_format.keep_with_next = True
                bcur = add_after(doc, q, item['caption'], 'Caption')
        # B结果置于A结果后，保持6.5节内实验次序。
        bh = add_after(doc, source_paras[99], '6.5.2 B协议已完成结果', 'Heading 3', False)
        # 调整到A结果块后：A表后的结果段落是唯一明确锚点。
        a_result = next(p for p in doc.paragraphs if p.text.startswith('A18人工框有26个错误'))
        a_result._p.addnext(bh._p)
        bcur = bh
        for text in s['B_results_paragraphs']:
            bcur = add_after(doc, bcur, text)
        for spec in s.get('B_tables', []):
            t = add_table_after(doc, bcur, spec['headers'], spec['rows'], spec['widths'])
            q = doc.add_paragraph()
            t._tbl.addnext(q._p)
            bcur = q
        target = P / '基于ResNet18的手势识别项目方案_v11.docx'
    else:
        a = add_after(doc, source_paras[90], 'B18目前已完成24次冻结头搜索和18轮微调，验证最佳轮为17；人工框验证准确率99.24%、宏F1为98.93%。B最终三模型测试、统一CPU测速和解释结论尚未完成审核，本内部稿不据训练时间推断结果。')
        target = B / 'project_plan_v11_working.docx'

    # 原表内容按当前已执行协议纠正，保留全部原表与真实手图。
    for table in doc.tables[:5]:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    if p.text == '观察负样本与成像差异':
                        continue
    # 所有无alt的旧图按相邻图注或表格标签补充可读描述。
    retained_alt = [
        '真实HaGRID手图：peace，V形掌面姿态，对应数字1。',
        '真实HaGRID手图：peace_inverted，V形背面姿态，对应数字1。',
        '真实HaGRID手图：stop，张掌姿态，对应后退数字2。',
        '真实HaGRID手图：stop_inverted，手背张掌姿态，对应数字2。',
        '真实HaGRID手图：palm，另一种张掌姿态，对应数字2。',
        '真实HaGRID手图：fist握拳，对应停止数字3。',
        '真实HaGRID手图：like拇指向上，对应左转数字4。',
        '真实HaGRID手图：dislike拇指向下，对应右转数字5。',
        '真实HaGRID闲置手示例，对应无指令数字0。',
        '普通CNN与残差CNN连接比较，主分支F(x)与旁路x相加。',
        '完整RGB照片经手检测、裁图、ResNet18分类和规则得到整数的总体流程。',
        'ResNet18真实结构、四组BasicBlock及512到6分类头。',
        '同一手图的网络结构、特征图、局部卷积、残差相加和平均池化教学展开。',
        'PyTorch训练与预测，误差反向更新参数，保存后加载推理。',
        'TorchVision网络代码、ImageNet通用权重与本项目六分类训练的关系。',
        'HaGRID原始数据、Sample30K第三方版本与课程副本及标注用途。',
        '历史Vision_CA归档示意，图内仅有summary状态属于早期输出。',
        'SSDLite320定位与ResNet18手势分类的输入输出衔接。',
        'HaGRID、Vision_CA、TorchVision与PyTorch共同形成项目训练和预测程序。',
        '电缆项目Grad-CAM计算机制原图参考，逐位置特征乘梯度后跨通道平均。',
        '本项目标准Grad-CAM计算机制仿制图，通道权重来自梯度空间平均，使用ImageNet教学目标。',
        '电缆论文七类分层矩阵原图参考，行是类别，列是输入和模块。',
        '六类手图五列分层布局教学示例，各行使用实际标出的ImageNet目标。',
        '同一原图、同一坐标局部放大的对应教学示例，使用ImageNet nail目标。',
    ]
    for index, shape in enumerate(doc.inline_shapes, 1):
        dp = shape._inline.docPr
        if index <= len(retained_alt):
            dp.set('descr', retained_alt[index-1])
    table_geometry(doc)
    for p in doc.paragraphs:
        if p.style.name.startswith('Heading') or p.style.name == 'Title':
            for run in p.runs:
                run.font.color.rgb = BLACK
                font_xml(run._r.get_or_add_rPr(), '黑体')
    # 兼容性预检：移除无部件支持的脚注终注引用，源稿若有正文注释则不得静默处理。
    assert not doc.element.xpath('.//w:footnoteReference|.//w:endnoteReference')
    doc.core_properties.title = '基于ResNet18的手势识别项目方案'
    doc.core_properties.subject = 'A照片划分与B人员互斥协议下的手势识别方法与实际进展'
    # 旧图上挂有原机器个人学习媒体的点击链接；交接包保留嵌入图，不分发个人媒体。
    removed_local_links = []
    for key, relation in list(doc.part.rels.items()):
        if relation.reltype.endswith('/hyperlink') and relation.target_ref.lower().startswith('file:'):
            removed_local_links.append(relation.target_ref)
            for node in list(doc.element.xpath('//*[@r:id]')):
                if node.get(qn('r:id')) != key:
                    continue
                if etree.QName(node).localname == 'hyperlink':
                    parent = node.getparent()
                    index = parent.index(node)
                    for child in list(node):
                        parent.insert(index, child)
                        index += 1
                    parent.remove(node)
                else:
                    assert etree.QName(node).localname in ('hlinkClick', 'hlinkHover')
                    node.getparent().remove(node)
            del doc.part.rels[key]
    buffer = BytesIO()
    doc.save(buffer)
    data = buffer.getvalue()
    with ZipFile(BytesIO(data)) as z:
        assert z.testzip() is None
        root = etree.fromstring(z.read('word/document.xml'))
        all_text = ''.join(root.xpath('//*[local-name()="t"]/text()'))
        assert '照片无法读取时记录读取错误并返回0' not in all_text
        assert '3个正确' not in all_text and '各选1' not in all_text
        assert '品牌可见' not in all_text and '待补' not in all_text
    reread = Document(BytesIO(data))
    images = visible_image_hashes(reread)
    assert not (original_images - images), '原始图片字节或出现次数丢失'
    final_headings = [(p.style.name, p.text) for p in reread.paragraphs if p.style.name.startswith('Heading')]
    assert all(x in final_headings for x in source_headings), '原有章节丢失'
    assert len(reread.tables) >= 7 and len(reread.inline_shapes) >= 25
    # 正式字节已经形成，再核对作者源稿与被接受的证据；不重新编码后写入。
    assert sha(SOURCE) == EXPECTED_SOURCE, '生成期间作者v10发生变化'
    if supplement:
        assert sha(supplement) == supplement_sha256, '生成期间最终补充文件发生变化'
        for f in s['verified_files']:
            assert sha(f['path']) == f['sha256'], f['path']
    target.write_bytes(data)
    record = {'status': 'FINAL_TEXT_READY_FOR_INDEPENDENT_RENDER_REVIEW' if supplement else 'INTERNAL_DRAFT_AWAITING_FINAL_B',
              'source': str(SOURCE), 'source_sha256': sha(SOURCE), 'output': str(target), 'output_sha256': sha(target),
              'original_tables_preserved': 5, 'tables': len(reread.tables), 'original_images_preserved': 24,
              'visible_images': len(reread.inline_shapes), 'all_original_image_bytes_and_multiplicities_preserved': True,
              'source_heading_inventory_preserved': True, 'changes': changes,
              'excluded_absolute_personal_media_links_removed': removed_local_links,
              'not_claimed': ['rendered_page_QA', 'independent_final_review', 'control_integration', 'course_acceptance'],
              'pending_final_B': [] if supplement else ['three_final_architecture_JSON', 'uniform_CPU_latency', 'B_CAM_and_case_facts', 'final_independent_review']}
    (B / 'project_plan_revision_audit.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k != 'changes'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--final-supplement', type=Path)
    args = parser.parse_args()
    build(args.final_supplement)
