"""只从完整且已独审的双标签产物生成v3；默认不写Word、不加载模型。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tempfile

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Inches, Pt
from docx.text.run import Run

R = Path(__file__).resolve().parent
V = R.parent.parent
P = V.parent
SOURCE = P / 'A与B协议ResNet训练与对比实验报告_v2.docx'
SOURCE_MD = V / 'runs/B_20261102_TPE/实验报告正文.md'
SOURCE_SHA256 = 'f0423c89427e68c19ce877185db4a547b608f213f58575a6d9c8fd9dc7504770'
SOURCE_MD_SHA256 = '667ee5ff1bd9ca74abfad77e91457cef66519745dffb3e268cfc98b7927e86a8'
TARGET = P / 'A与B协议ResNet训练与对比实验报告_v3.docx'
TARGET_MD = R / '实验报告正文_v3.md'
CPU_FILE = R / 'label_variants_cpu_benchmark.json'
CLASS_SCOPE = 'classifier_only_preprocessed_224x224_tensor'
WHOLE_SCOPE = 'whole_photo_decode_detection_crop_classification_rules'
ARCHES = ('resnet18', 'resnet34', 'resnet50')
RAWS = ('peace', 'peace_inverted', 'two_up', 'two_up_inverted')
MOVING = (1, 2, 4, 5)
OLD_MAP = {'peace': 1, 'peace_inverted': 1, 'stop': 2, 'stop_inverted': 2,
           'palm': 2, 'fist': 3, 'like': 4, 'dislike': 5}
NEW_MAP = dict(OLD_MAP, two_up=1, two_up_inverted=1)
REQUIRED_ROLES = {'merged_science', 'matched_speed', 'raw_label_comparison'}
REQUIRED_COVERAGE = {'label_semantics', 'same_fixed_splits', 'new_training_from_official',
                     'unmerged_results_retained', 'twelve_model_results', 'raw_subgroups',
                     'truth_denominators', 'old_CAM_label_identity', 'control_interface_and_team_roles'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    # 大权重只在最终事实核查时流式读取，不导入任何模型库。
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pct(value):
    return f'{100 * value:.2f}%'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |',
                      '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]) + '\n\n'


def folder(variant, protocol, arch):
    if variant == '合并':
        return R / protocol / arch
    base = V / f'runs/{protocol}_20261102_TPE'
    return base if arch == 'resnet18' else base / ('compare_' + arch)


def labels(variant):
    return ['0 其他', '1 双指前进姿态' if variant == '合并' else '1 V形',
            '2 手掌', '3 握拳', '4 向上', '5 向下']


def keyed(rows, hand=False):
    result = {(x['image_id'], x['hand_index']) if hand else x['image_id']: x for x in rows}
    if len(result) != len(rows):
        raise ValueError('照片或照片加手编号存在重复')
    return result


def check_facts(facts):
    # 共用方案与交接的事实接口；报告不额外依赖程序打包完成。
    if facts.get('phase') != 'FINAL_ACCEPTED':
        raise ValueError('当前不是已接受的最终事实，禁止生成正式稿')
    if not all(facts.get('coverage', {}).get(k) is True for k in REQUIRED_COVERAGE):
        raise ValueError('双标签比较仍有有效要求未覆盖')
    rows = facts.get('comparison_rows', [])
    expected = {(v, p, a) for v in ('不合并', '合并') for p in 'AB' for a in ARCHES}
    if len(rows) != 12 or {(x['variant'], x['protocol'], x['architecture']) for x in rows} != expected:
        raise ValueError('需要十二项唯一的真实实验身份')
    if sha(SOURCE) != SOURCE_SHA256 or sha(SOURCE_MD) != SOURCE_MD_SHA256:
        raise ValueError('不合并正式v2来源发生变化；禁止静默覆盖历史身份')
    files = facts.get('verified_files', [])
    bound = {str(Path(x['path']).resolve()).casefold(): x['sha256'] for x in files}
    if len(bound) != len(files) or not files:
        raise ValueError('最终事实文件清单为空或重复')
    for item in files:
        if sha(item['path']) != item['sha256']:
            raise ValueError(f'事实文件已改变：{item["path"]}')
    reviews = facts.get('reviews', [])
    if not REQUIRED_ROLES <= {x['role'] for x in reviews}:
        raise ValueError('科学、十二模型速度、原始子群比较独审尚未齐备')
    for item in reviews:
        path = Path(item['path'])
        if bound.get(str(path.resolve()).casefold()) != sha(path):
            raise ValueError('独审未绑定当前SHA')
        value = read(path)
        for key in item.get('status_field', 'status').split('.'):
            value = value[key]
        if (not isinstance(value, str) or not value.startswith('PASS')
                or value not in item.get('accepted_statuses', ['PASS'])):
            raise ValueError(f'独审未通过：{item["role"]}')
    if read(R / 'all_training_evaluation_complete.json')['status'] != 'PASS':
        raise ValueError('六组合并训练和完整测试尚未全部完成')
    aggregate = read(R / 'aggregate_review.json')
    if aggregate.get('status') != 'PASS_MERGED_SIX_MODEL_AUDIT_WITH_LIMITATIONS' or aggregate.get('issues') != []:
        raise ValueError('合并六模型总科学独审未通过')
    bound[str((R / 'all_training_evaluation_complete.json').resolve()).casefold()] = sha(R / 'all_training_evaluation_complete.json')
    bound[str((R / 'aggregate_review.json').resolve()).casefold()] = sha(R / 'aggregate_review.json')
    for protocol in 'AB':
        for arch in ARCHES:
            review_path = R / protocol / arch / 'model_review.json'
            report = read(review_path)
            if (report.get('status') != 'PASS_MERGED_MODEL_AUDIT_WITH_LIMITATIONS'
                    or report.get('issues') != [] or report['protocol'] != protocol or report['architecture'] != arch):
                raise ValueError(f'单模型独审未通过：{protocol}/{arch}')
            if aggregate['bound_file_sha256'].get(str(review_path)) != sha(review_path):
                raise ValueError('总独审没有绑定当前逐模型独审')
            bound[str(review_path.resolve()).casefold()] = sha(review_path)
            # 显式核本报告实际读取的全部JSON，不靠一份未绑定原产物的PASS文件。
            merged = R / protocol / arch
            original = folder('不合并', protocol, arch)
            used = [merged / name for name in ('fixed_head_summary.json', 'finetune_summary.json',
                    'merged_data_identity.json', 'photo_truth_test.json', 'calibration.json',
                    'test_oracle_crop.json', 'test_whole_photo.json', 'test_detector.json')]
            used += [original / name for name in ('test_oracle_crop.json', 'test_whole_photo.json')]
            for path in used:
                digest = report['bound_file_sha256'].get(str(path))
                if not digest or sha(path) != digest:
                    raise ValueError(f'报告原产物未由逐模型独审绑定：{path}')
                bound[str(path.resolve()).casefold()] = digest
            statistics = merged / 'label_statistics.json'
            stats, identity_splits = read(statistics), read(merged / 'merged_data_identity.json')['splits']
            if set(stats) != set(identity_splits) or any(
                    key not in identity_splits[split] or value != identity_splits[split][key]
                    for split, fields in stats.items() for key, value in fields.items()):
                raise ValueError('分母统计未与已审数据身份对应')
            bound[str(statistics.resolve()).casefold()] = sha(statistics)
    cpu = read(CPU_FILE)
    keys = {f'{v}_{p}_{a}' for v in ('unmerged', 'merged') for p in 'AB' for a in ARCHES}
    if cpu.get('status') != 'complete' or len(cpu['models']) != 12 or {m['key'] for m in cpu['models']} != keys:
        raise ValueError('十二模型同输入CPU测量尚未完整完成')
    for model in cpu['models']:
        for scope in (CLASS_SCOPE, WHOLE_SCOPE):
            if (model['summary'][scope]['n'] != 300 or len(model['measurements'][scope]) != 3
                    or any(len(b['all_ms']) != 100 for b in model['measurements'][scope])):
                raise ValueError('十二模型速度必须保留每项三轮全部300次调用')
    if bound.get(str(CPU_FILE.resolve()).casefold()) != sha(CPU_FILE):
        raise ValueError('十二模型速度结果不在最终事实SHA清单中')
    speed_review = next(x for x in reviews if x['role'] == 'matched_speed')
    speed_value = read(speed_review['path'])
    if 'data_sha256' in speed_value and speed_value['data_sha256'] != sha(CPU_FILE):
        raise ValueError('速度独审对应另一份数据')
    cam_path = R / 'merged_gradcam_independent_review.json'
    cam_review = read(cam_path)
    if not cam_review.get('status', '').startswith('PASS') or cam_review.get('issues') != []:
        raise ValueError('两套真实合并CAM尚未经独立验收')
    if bound.get(str(cam_path.resolve()).casefold()) != sha(cam_path):
        raise ValueError('CAM独审不在最终事实清单中')
    for protocol in 'AB':
        for name in ('metadata.json', 'trained_six_class_matrix.png'):
            path = R / protocol / 'resnet18/trained_gradcam' / name
            if cam_review['bound_file_sha256'].get(str(path)) != sha(path):
                raise ValueError('实际解释图或元数据与CAM独审身份不一致')
    analysis_path = Path(facts['report_analysis_markdown'])
    if bound.get(str(analysis_path.resolve()).casefold()) != sha(analysis_path):
        raise ValueError('真实合并案例解释稿没有绑定当前证据')
    for image_path in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', analysis_path.read_text(encoding='utf-8')):
        path = Path(image_path)
        if bound.get(str(path.resolve()).casefold()) != sha(path):
            raise ValueError('解释图不在最终核验清单中')
    for name in ('report_execution_markdown', 'report_conclusion_markdown'):
        if not isinstance(facts.get(name), str) or not facts[name].strip():
            raise ValueError('运行异常说明或实测结论尚未完成')
    return bound, cpu


def result_rows(cpu):
    # 只提取已经保存的正式指标；不再次访问模型或运行推理。
    speeds = {x['key']: x for x in cpu['models']}
    result = []
    for variant in ('不合并', '合并'):
        for protocol in 'AB':
            for arch in ARCHES:
                out = folder(variant, protocol, arch)
                crop, photo = read(out / 'test_oracle_crop.json'), read(out / 'test_whole_photo.json')
                c, w, s = crop['metrics'], photo['metrics'], photo['safety']
                speed = speeds[f'{"merged" if variant == "合并" else "unmerged"}_{protocol}_{arch}']
                result.append({'variant': variant, 'protocol': protocol, 'architecture': arch,
                    'crop': c, 'photo': w, 'safety': s,
                    'crop_correct': sum(c['confusion_matrix'][i][i] for i in range(6)),
                    'photo_correct': sum(w['confusion_matrix'][i][i] for i in range(6)),
                    'stop_to_zero': w['confusion_matrix'][3][0], 'speed': speed})
    return result


def summary_table(rows):
    return table(['标签任务／协议／架构', '人工框正确／分母', '人工框准确率', '人工框宏F1',
                  '整图正确／分母', '整图准确率', '整图宏F1'],
        [[f'{r["variant"]} {r["protocol"]} {r["architecture"]}', f'{r["crop_correct"]}／{r["crop"]["n"]}',
          pct(r['crop']['accuracy']), pct(r['crop']['macro_f1']), f'{r["photo_correct"]}／{r["photo"]["n"]}',
          pct(r['photo']['accuracy']), pct(r['photo']['macro_f1'])] for r in rows])


def perclass(metrics, variant):
    return table(['类别', '真实分母', '精确率', '召回率', 'F1'],
        [[labels(variant)[i], metrics['support'][i], pct(metrics['precision'][i]),
          pct(metrics['recall'][i]), pct(metrics['f1'][i])] for i in range(6)])


def errors(metrics, variant, unit):
    cm = metrics['confusion_matrix']
    return '；'.join(f'{labels(variant)[i]}→{labels(variant)[j]} {cm[i][j]}{unit}'
                     for i in range(6) for j in range(6) if i != j and cm[i][j]) + '。'


def common_command_counts(protocol, arch, report):
    # 在同一不可变样本交集上列出四个运动指令，避免换分母造成虚假的安全改善。
    old, new = folder('不合并', protocol, arch), folder('合并', protocol, arch)
    oh = keyed(read(old / 'test_oracle_crop.json')['predictions'], True)
    nh = keyed(read(new / 'test_oracle_crop.json')['predictions'], True)
    op = keyed(read(old / 'test_whole_photo.json')['predictions'])
    np = keyed(read(new / 'test_whole_photo.json')['predictions'])
    if oh.keys() != nh.keys() or op.keys() != np.keys():
        raise ValueError('新旧测试不可变样本集合不同')
    hands = [k for k, x in nh.items() if OLD_MAP.get(x['raw_label'], 0) == NEW_MAP.get(x['raw_label'], 0) == 0]
    truth = keyed(read(new / 'photo_truth_test.json'))
    photos = [k for k, x in truth.items() if all(OLD_MAP.get(t, 0) == NEW_MAP.get(t, 0) == 0 for t in x['raw_labels'])]
    output = []
    for scope, keys, before, after, audited in (
        ('人工框共同其他手', hands, oh, nh, report['paired_original_vs_merged']['common_other_raw_GT_hands']),
        ('完整照片共同纯其他负类', photos, op, np, report['paired_original_vs_merged']['common_other_raw_photos'])):
        a = [sum(before[k]['prediction'] == command for k in keys) for command in MOVING]
        b = [sum(after[k]['prediction'] == command for k in keys) for command in MOVING]
        if (len(keys) != audited['old_false_motion']['n'] or len(keys) != audited['merged_false_motion']['n']
                or sum(a) != audited['old_false_motion']['count'] or sum(b) != audited['merged_false_motion']['count']):
            raise ValueError('共同负类逐指令统计未与独审总数对应')
        output.append([protocol, arch, scope, len(keys), *a, sum(a), *b, sum(b)])
    return output


def comparison_sections(rows, cpu, facts):
    text = '## 14 两种前进姿态定义与同一数据上的比较\n\n'
    text += '不合并任务把peace、peace_inverted定义为前进1，two_up和two_up_inverted仍属于其他0；合并任务把这四种原始姿态都定义为前进1。实际改变的是后两种姿态的0→1，输出仍为六类。原始stop、stop_inverted、palm仍为后退2，fist仍为明确停止3。扩大前进姿态范围改变了正确答案和类别难度，不是单纯更换名称。\n\n'
    text += table(['原始姿态', '不合并真值', '合并真值', '控制含义变化'],
                  [[raw, OLD_MAP.get(raw, 0), NEW_MAP.get(raw, 0), '不发有效指令→前进' if raw.startswith('two_up') else '仍为前进'] for raw in RAWS])
    text += 'A继续使用原照片划分，B继续使用原人员互斥划分。照片、人员、手框、像素及由同一照片裁出的手图归属均未重分；原不合并权重与预测全部保留。相同照片在不同标签任务下有不同真值，因此两任务总体准确率或宏F1的差值只作描述，不能直接称为同一任务上的泛化提升。\n\n'
    counts = []
    for protocol in 'AB':
        statistics = read(R / protocol / 'resnet18/label_statistics.json')
        for split, name in [('train', '训练'), ('val', '验证'), ('test', '测试')]:
            x = statistics[split]
            for variant, hand_key, photo_key in [('不合并', 'old_six_class', 'old_photo_commands'), ('合并', 'new_six_class', 'new_photo_commands')]:
                counts.append([protocol, name, variant, x['photos'], x['hands'],
                    '／'.join(str(x[hand_key][str(i)]['hands']) for i in range(6)),
                    '／'.join(str(x[photo_key].get(str(i), 0)) for i in range(6))])
    text += table(['协议', '分组', '任务', '完整照片', '手图', '手图六类支持0—5', '整图真值六类支持0—5'], counts)
    text += '“某类手框出现在多少张照片中”与“整图聚合真值等于该类的照片数”分别统计。A测试0类手支持由2,510变为2,149，整图0由1,773变为1,412；B分别由2,352变为2,016、由1,695变为1,359。两协议明确停止的整图分母均为173，未因合并而改变。\n\n'
    text += '## 15 合并任务的固定参数训练与验证选型\n\n'
    text += '原不合并六组各24次TPE已经完成，本次没有新增TPE。每组合并实验沿用自己协议与架构的旧best_params，只用新训练标签重新训练一个Dropout加线性分类头18轮，再以该新头和官方ImageNet V1主干开始layer4与fc微调。冻结特征数值仅在官方初始化、输入与张量身份相同后复用，并重建标签；旧已微调分类器没有作为新初始化。各自A或B检测器只按原字节复用，没有跨协议或重新训练。\n\n'
    text += '新分类头以验证宏F1选轮；后续微调上限18轮、patience4，且必须优于新冻结头才替换最终权重。若未提高，正式模型保持新冻结头，不能把微调尝试误写成最终都采用微调权重。训练与验证保持分开，没有合并后重训。\n\n'
    training = []
    parameters = []
    for protocol in 'AB':
        for arch in ARCHES:
            out = R / protocol / arch
            head, fine, review = read(out / 'fixed_head_summary.json'), read(out / 'finetune_summary.json'), read(out / 'model_review.json')
            train = review['training']
            stage = {'merged_fixed_head': '新冻结头', 'layer4_finetune': '末组微调'}[fine['selected_stage']]
            training.append([protocol, arch, head['epochs_completed'], head['selected_epoch'], pct(head['validation']['macro_f1']),
                fine['epochs_completed'], stage, fine['selected_epoch'], pct(train['selected_validation']['macro_f1'])])
            pp = head['best_params']
            parameters.append([protocol, arch, pp['optimizer'], f'{pp["lr"]:.9g}', f'{pp["weight_decay"]:.9g}',
                               f'{pp["dropout"]:.9g}', f'{pp["label_smoothing"]:.9g}', '使用' if pp['class_weight'] else '不使用'])
    text += table(['协议', '架构', '新头完整轮', '新头选中轮', '新头验证宏F1', '微调实际轮', '正式选中阶段', '正式轮', '正式验证宏F1'], training)
    text += table(['协议', '架构', '优化器', '冻结头学习率', '权重衰减', 'Dropout', '标签平滑', '类别权重'], parameters)
    text += '表中学习率属于冻结头训练。微调固定AdamW：layer4为10⁻⁴、fc为5×10⁻⁴，batch128、梯度裁剪5、余弦调度与原增强，保留CUDA混合精度。六组各自在全部验证完整照片上比较12组检测／分类阈值并冻结，完整测试未参与参数、选轮或阈值选择。新任务沿用旧任务已选参数，故这也是固定既有训练流程下的任务变化比较，而非两标签分别重新充分优化后的上限比较。\n\n'
    text += '### 15.1 冻结身份与实际运行异常\n\n' + facts['report_execution_markdown'].rstrip() + '\n\n'
    text += '## 16 十二项结果与合并任务的完整错误方向\n\n'
    text += '人工框列评价单手分类，整图列评价自动定位、裁图、分类、接受阈值和多手规则后的最终数字。数字相同不能推出检测框完整，人工框分类正确也不代表在整图里成功找到该手。各标签任务按自己的正确答案计分，下表安全项保留各自真实分母。\n\n'
    text += table(['任务／协议／架构', '真实0误发运动', '真实停止误发运动', '真实停止输出0'],
        [[f'{r["variant"]} {r["protocol"]} {r["architecture"]}', f'{r["safety"]["zero_to_motion"]}／{r["safety"]["zero_samples"]}',
          f'{r["safety"]["stop_to_motion"]}／{r["safety"]["stop_samples"]}', f'{r["stop_to_zero"]}／{r["safety"]["stop_samples"]}'] for r in rows])
    for r in rows:
        if r['variant'] != '合并':
            continue
        protocol, arch = r['protocol'], r['architecture']
        out = R / protocol / arch
        calibration = read(out / 'calibration.json')['selected']
        det = read(out / 'test_detector.json')['metrics']
        text += f'### 16.{1 + list("AB").index(protocol) * 3 + list(ARCHES).index(arch)} 合并{protocol} {arch}完整测试\n\n'
        text += f'冻结检测／分类阈值为{calibration["detector_threshold"]}／{calibration["class_threshold"]}。人工框正确{r["crop_correct"]}／{r["crop"]["n"]}只手，完整照片正确{r["photo_correct"]}／{r["photo"]["n"]}张。\n\n'
        text += perclass(r['crop'], '合并') + '人工框全部非零错误方向：' + errors(r['crop'], '合并', '只') + '\n\n'
        text += perclass(r['photo'], '合并') + '整图全部非零错误方向：' + errors(r['photo'], '合并', '张') + '\n\n'
        text += f'固定阈值检测匹配TP={det["tp"]}、FP={det["fp"]}、FN={det["fn"]}，精确率{pct(det["precision"])}、召回率{pct(det["recall"])}。这是score≥{det["score_threshold"]}、IoU≥{det["iou_threshold"]}的框级结果，未计算AP或mAP；漏框个数与失败照片数不能互换。\n\n'
    text += '## 17 四种原始姿态与共同负类上的配对比较\n\n'
    text += '本章按不可变照片ID与手编号连接新旧保存预测，不重新推理。人工框表按每只手的raw标签分组；整图表另按完整照片的原始目录类别分组。两种two_up在旧任务应输出0、新任务应输出1，因此“输出1率”在这里描述接口语义改变的实现情况，不能把旧任务低输出1率解释为旧模型识别失败。\n\n'
    hand_rows, photo_rows, common_rows = [], [], []
    for protocol in 'AB':
        for arch in ARCHES:
            review = read(R / protocol / arch / 'model_review.json')
            paired = review['paired_original_vs_merged']
            for row in paired['raw_GT_hand_output_rates']:
                if row['raw_label'] in RAWS:
                    a, b = row['old_output1'], row['merged_output1']
                    hand_rows.append([protocol, arch, row['raw_label'], row['old_target'], row['merged_target'],
                                      f'{a["count"]}／{a["n"]}', pct(a['rate']), f'{b["count"]}／{b["n"]}', pct(b['rate'])])
            for row in paired['raw_photo_folder_output1']:
                a, b = row['old_command1'], row['merged_command1']
                photo_rows.append([protocol, arch, row['raw_photo_folder'], f'{a["count"]}／{a["n"]}', pct(a['rate']),
                                   f'{b["count"]}／{b["n"]}', pct(b['rate'])])
            common_rows += common_command_counts(protocol, arch, review)
    text += table(['协议', '架构', 'raw手姿态', '旧真值', '新真值', '不合并输出1／手数', '比例', '合并输出1／手数', '比例'], hand_rows)
    text += table(['协议', '架构', '原始照片目录', '不合并输出1／照片', '比例', '合并输出1／照片', '比例'], photo_rows)
    text += '共同其他负类排除两种新增正类，只保留两任务都映射0的相同样本。手级使用相同人工框手键；照片级要求每只标注手的raw标签在两任务都属于0。下面分别列误发1、2、4、5，合计是误运动，不把输出3计入运动；照片级“真值同时为0的交集”可包含动作冲突，概念上与纯其他照片不同，本表明确使用后者。\n\n'
    for scope in ('人工框共同其他手', '完整照片共同纯其他负类'):
        text += scope + '：\n\n'
        compact_rows = []
        for row in common_rows:
            if row[2] == scope:
                compact_rows.append([row[0], row[1], '不合并', row[3], *row[4:9]])
                compact_rows.append([row[0], row[1], '合并', row[3], *row[9:14]])
        text += table(['协议', '架构', '标签任务', '共同分母', '误发1', '误发2', '误发4', '误发5', '误运动合计'], compact_rows)
    text += '以上同样本比较与各任务自己的完整0类安全表同时保留。新任务完整0类已经排除two_up，分母缩小；只比较两张完整0类误运动率不足以判断是否变安全。停止3的误运动与输出0另按共同的173张照片判断。\n\n'
    text += '## 18 十二模型在共同输入上的CPU速度\n\n'
    text += '本次十二模型速度属于新的一次预先固定协议测量，原六模型历史时延及其补充基准均保留在前文与附录。十二模型继续使用同一批100张A验证完整照片与100只已预处理A验证人工框手图，CPU、float32、8线程；每模型每项预热20次，固定100个输入重复三轮，共300次调用，不是300张独立照片。模型加载与纯分类预处理不计时，整图包含打开解码、检测、裁图、分类及规则，一次调用一张照片，照片中的多只手仍成批分类。\n\n'
    speed_rows = []
    for r in rows:
        c, w = r['speed']['summary'][CLASS_SCOPE], r['speed']['summary'][WHOLE_SCOPE]
        speed_rows.append([f'{r["variant"]} {r["protocol"]} {r["architecture"]}'] +
                          [f'{x:.3f}' for x in (c['mean_ms'], c['p50_ms'], c['p95_ms'], c['sd_ms'],
                                              w['mean_ms'], w['p50_ms'], w['p95_ms'], w['sd_ms'])])
    text += table(['任务／协议／架构', '分类均值', '分类P50', '分类P95', '分类标准差', '整图均值', '整图P50', '整图P95', '整图标准差'], speed_rows)
    backgrounds = [block['cpu_background']['estimated_other_process_busy_percent']
                   for model in cpu['models'] for scope in (CLASS_SCOPE, WHOLE_SCOPE)
                   for block in model['measurements'][scope]]
    text += f'全部数值单位为ms。计时块中其他进程估计CPU占用为{min(backgrounds):.1f}%—{max(backgrounds):.1f}%。三轮原始慢值、顺序和真实检测框工作量均保留；同一照片在不同冻结阈值下保留框数可不同。未按最快轮次重选结果，模型和系统波动也会影响均值，不能把同架构小差异归因为标签映射或人员协议本身。背景CPU占用和三轮漂移须结合原始测量记录解释。\n\n'
    text += '## 19 实际手检测错误与分类解释\n\n'
    text += '前文第6章和第11章的CAM仍属于原不合并模型，原two_up_inverted真0的身份保持；它在新任务的真值变为1，并不使旧热图自动成为新模型解释。分类CAM解释实际送入分类器的手图分数，不能代替完整照片的手检测与多手检查。\n\n'
    text += Path(facts['report_analysis_markdown']).read_text(encoding='utf-8').rstrip() + '\n\n'
    text += '## 20 两种任务的取舍与小组集成\n\n'
    text += '是否接收两指并拢姿态作为前进信号，首先是接口允许姿态范围的选择。合并后，这两种姿态触发1属于预期行为；不合并时它们应保持0。两种模型均保留，接收程序必须先选择标签任务，再选择A/B与架构，不能让同一控制场景混用两套预期。\n\n'
    text += '比较模型时，应分别看同一任务下的人工框与整图表现、四个raw姿态的前进输出、共同其他负类的误运动、真实停止的错误，以及本次共同输入速度。整体宏F1提高并不保证共同负类误运动减少；人工框高分也不保证整图多手定位完整。本次结果仍是同一HaGRID来源的静态照片内部评价，未完成真实摄像头连续输入和机器人控制联动验收。\n\n'
    text += facts['report_conclusion_markdown'].rstrip() + '\n\n'
    text += 'ResNet分支由我和组员共同负责。后续共同复核两种任务的接口与模型取舍，与另一识别子组在相同输入和评价口径下比较，并与控制组确认0不发起或续期运动、3明确取消运动、异常非零退出以及接收端行为，之后再做真实联调。\n\n'
    return text


def report_text(facts, cpu):
    source = SOURCE_MD.read_text(encoding='utf-8')
    rows = result_rows(cpu)
    # 将原六行摘要移到历史附录，所有原表原值仍保留；开头直接呈现十二项结果。
    first_table = re.search(r'(?m)^\|.*(?:\n\|.*)+\n?', source)
    if not first_table:
        raise ValueError('旧正式报告缺少已知摘要表')
    old_overview = first_table.group(0)
    source = source[:first_table.start()] + summary_table(rows) + source[first_table.end():]
    source = source.replace('本实验完成A按照片划分和B按人员互斥划分下的ResNet18、34、50独立训练、验证选型和冻结测试。核心结果如下；人工框评价六类手势分类，整图评价检测、裁图、分类和多手规则后的最终数字，二者分母分别列出。',
        '本实验比较不合并与合并两种前进姿态定义，在A照片划分和B人员互斥划分下分别训练ResNet18、34、50，共十二项结果。合并任务允许peace、peace_inverted、two_up和two_up_inverted均触发前进1；两种two_up由原0改为1，其余输出含义保持。下表按各自真值报告人工框分类与完整照片最终数字。')
    title_end = source.find('\n\n')
    flow = ('程序先让SSDLite读取完整照片，检测一个或多个手框；它可能漏手，也可能给出位置不准的框。'
            '每个保留框各裁成一张224×224手图，ResNet只负责判断这只手属于哪类。'
            '最终0—5不是残差网络自带的多手功能，而由独立规则汇总已达到分类阈值的结果：'
            '有已接受的3就输出3；否则只有一种非0有效动作时输出该动作；不同有效动作冲突或没有有效动作时输出0。'
            '0类手不抵消另一只手的唯一有效动作；0表示不发起或续期运动，不能当作取消现有运动的3。'
            '例如已通过阈值的两手结果为1和1时输出1，1和2冲突时输出0，1和3时输出3，1和0时仍输出1。'
            'predict与命令行对一张照片最终只返回一个整数，predict_details另保留每只候选手的框、分类概率和接受状态供排查。'
            '因此程序能处理多个候选手框，但没有保证每只手都被找到或每张多手照片都识别正确。\n\n')
    source = source[:title_end + 2] + flow + source[title_end + 2:]
    source = source.replace('## 13 综合结论与小组集成', '## 13 不合并任务的综合结论与小组集成')
    marker = '## 附录 原A与B各自输入的历史CPU记录'
    if marker not in source:
        raise ValueError('旧历史CPU附录边界缺失')
    body, history = source.split(marker, 1)
    body = re.sub(r'(?m)^方法与A完整证据见第1—8章.*\n?', '', body)
    roadmap = ('第1—13章呈现不合并任务的方法、数值、错误、失败恢复与科学图，旧时延保持其历史身份。'
               '第14—17章说明标签改变、新训练与共同样本比较，第18章给出十二模型共同输入速度，'
               '第19—20章解释实际错误与接口取舍。新旧任务准确率的差值不直接代表泛化提高。\n\n')
    body = body.replace('## 1 实验任务与处理流程', roadmap + '## 1 实验任务与处理流程', 1)
    return (body.rstrip() + '\n\n' + comparison_sections(rows, cpu, facts) + marker + history +
            '\n\n### 不合并v2原六行摘要\n\n' + old_overview + '\n'), rows


def font(run, heading=False, size=None):
    # 明确中文字体并去掉主题选择器，避免Windows中文标题压缩。
    run.font.name = 'Calibri'
    if size:
        run.font.size = Pt(size)
    rp = run._element.get_or_add_rPr()
    rf = rp.rFonts
    if rf is None:
        rf = OxmlElement('w:rFonts')
        rp.insert(0, rf)
    for key in ('asciiTheme', 'hAnsiTheme', 'eastAsiaTheme', 'cstheme'):
        rf.attrib.pop(qn('w:' + key), None)
    for key, value in {'ascii': 'Calibri', 'hAnsi': 'Calibri', 'cs': 'Calibri',
                       'eastAsia': 'Microsoft YaHei' if heading else 'SimSun'}.items():
        rf.set(qn('w:' + key), value)
    if size:
        sz = rp.find(qn('w:szCs'))
        if sz is None:
            sz = OxmlElement('w:szCs')
            rp.append(sz)
        sz.set(qn('w:val'), str(round(size * 2)))


def table_signatures(doc):
    return Counter(json.dumps([[c.text for c in row.cells] for row in t.rows], ensure_ascii=False)
                   for t in doc.tables)


def image_hashes(doc):
    return [hashlib.sha256(doc.part.related_parts[s._inline.graphic.graphicData.pic.blipFill.blip.embed].blob).hexdigest()
            for s in doc.inline_shapes]


def inline(paragraph, text):
    # 原文正式来源仍保留为可点击的Word原生链接，不把Markdown标记打印进正文。
    start = 0
    for match in re.finditer(r'\[([^\]]+)\]\((https?://[^)]+)\)', text):
        if match.start() > start:
            font(paragraph.add_run(text[start:match.start()]))
        link = OxmlElement('w:hyperlink')
        link.set(qn('r:id'), paragraph.part.relate_to(match[2], RT.HYPERLINK, is_external=True))
        run = OxmlElement('w:r')
        content = OxmlElement('w:t')
        content.text = match[1]
        run.append(content)
        font(Run(run, paragraph))
        link.append(run)
        paragraph._p.append(link)
        start = match.end()
    if start < len(text):
        font(paragraph.add_run(text[start:]))


def build_docx(text):
    # 从原v2复用真实样式；不修改原Word，也不添加科研范围之外的教学图。
    doc = Document(SOURCE)
    for child in list(doc._element.body):
        if child.tag != qn('w:sectPr'):
            doc._element.body.remove(child)
    doc.core_properties.title = 'A与B协议ResNet训练与对比实验报告'
    doc.core_properties.subject = '合并与不合并两种前进姿态定义的十二项实验'
    lines = text.splitlines()
    i, figures = 0, 0
    width = (doc.sections[0].page_width - doc.sections[0].left_margin - doc.sections[0].right_margin) / 914400
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('|'):
            data = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+', c) for c in row):
                    data.append(row)
                i += 1
            n = len(data[0])
            t = doc.add_table(rows=1, cols=n)
            t.style = 'Table Grid'
            t.autofit = True
            first = 1.5 if n <= 5 or data[0][0] == '标签任务／协议／架构' else 1.0
            widths = [first] + [(width - first) / (n - 1)] * (n - 1)
            if data[0] == ['协议', '分组', '任务', '完整照片', '手图', '手图六类支持0—5', '整图真值六类支持0—5']:
                proportions = [.42, .48, .58, .74, .66, 2.1, 2.1]
                widths = [width * x / sum(proportions) for x in proportions]
            for column, w in zip(t.columns, widths):
                column.width = Inches(w)
            for j, row in enumerate(data):
                cells = t.rows[0].cells if j == 0 else t.add_row().cells
                for cell, value, w in zip(cells, row, widths):
                    cell.width = Inches(w)
                    cell.text = value
                    for para in cell.paragraphs:
                        para.paragraph_format.space_after = Pt(2)
                        para.paragraph_format.line_spacing = 1.05
                        # 短指标表保持完整；大型长表仍逐行正常跨页。
                        if len(data) <= 8:
                            para.paragraph_format.keep_with_next = j < len(data) - 1
                        for run in para.runs:
                            font(run, size=9 if n >= 8 else 10)
                            run.bold = j == 0
                trpr = t.rows[j]._tr.get_or_add_trPr()
                trpr.append(OxmlElement('w:cantSplit'))
                if j == 0:
                    trpr.append(OxmlElement('w:tblHeader'))
            doc.add_paragraph().paragraph_format.space_after = Pt(3)
            continue
        match = re.fullmatch(r'!\[([^\]]*)\]\((.+)\)', line)
        if match:
            from PIL import Image
            path = Path(match[2])
            with Image.open(path) as im:
                ratio = im.height / im.width
            para = doc.add_paragraph()
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para.paragraph_format.keep_with_next = True
            shape = para.add_run().add_picture(str(path), width=Inches(min(width, 6.93, 8.32 / ratio)))
            shape._inline.docPr.set('descr', match[1])
            figures += 1
            para = doc.add_paragraph(f'图{figures} {match[1]}', 'Caption')
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif line.startswith('#'):
            h = re.match(r'^(#+)\s+(.*)', line)
            level, title = len(h[1]), h[2]
            para = doc.add_paragraph(title, 'Title' if level == 1 else 'Heading ' + str(min(level - 1, 3)))
            if title.startswith('附录 原A与B'):
                para.paragraph_format.page_break_before = True
            for run in para.runs:
                font(run, heading=True)
        else:
            para = doc.add_paragraph()
            inline(para, line)
        i += 1
    original = Document(SOURCE)
    if table_signatures(original) - table_signatures(doc):
        raise ValueError('旧v2的29张表及全部单元格未完整保留')
    if image_hashes(doc)[:len(original.inline_shapes)] != image_hashes(original):
        raise ValueError('旧v2十二张科学图未按原字节和原顺序保留')
    payload = BytesIO()
    doc.save(payload)
    return payload.getvalue(), {'figures': figures, 'tables': len(doc.tables),
                                'original_29_tables_exact_cells_preserved': True,
                                'original_12_figures_exact_bytes_preserved': True}


def commit(path, payload, expected_sha, precommit):
    # 精确字节形成后再核活的事实边界；旧文件永不作为目标。
    if path in (SOURCE, SOURCE_MD) or path.parent != (P if path.suffix == '.docx' else R):
        raise ValueError('输出路径越过本作者的新稿边界')
    if path.exists() and (not expected_sha or sha(path) != expected_sha):
        raise ValueError(f'新稿已存在，修订须提供其当前SHA：{path}')
    precommit()
    with tempfile.NamedTemporaryFile(prefix=path.name + '.', suffix='.tmp', dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    try:
        precommit()
        if path.exists() and sha(path) != expected_sha:
            raise ValueError('提交前新稿已被其他写者改变')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accepted-facts', type=Path)
    parser.add_argument('--expected-output-sha256')
    parser.add_argument('--expected-markdown-sha256')
    args = parser.parse_args()
    if not args.accepted_facts:
        print('准备就绪：仅在六合并完整训练测试、科学总独审、十二模型CPU与raw比较独审齐备后，传入--accepted-facts生成新v3。')
        return
    facts_sha = sha(args.accepted_facts)
    facts = read(args.accepted_facts)
    bound, cpu = check_facts(facts)
    text, rows = report_text(facts, cpu)
    word, counts = build_docx(text)
    markdown = text.encode('utf-8')
    def precommit():
        if sha(args.accepted_facts) != facts_sha or sha(SOURCE) != SOURCE_SHA256 or sha(SOURCE_MD) != SOURCE_MD_SHA256:
            raise ValueError('提交前事实或旧正式来源发生变化')
        for path, digest in bound.items():
            if sha(path) != digest:
                raise ValueError('提交前证据文件已改变')
    commit(TARGET, word, args.expected_output_sha256, precommit)
    commit(TARGET_MD, markdown, args.expected_markdown_sha256, precommit)
    print(json.dumps(dict(counts, output=str(TARGET), output_sha256=hashlib.sha256(word).hexdigest(),
                          markdown=str(TARGET_MD), markdown_sha256=hashlib.sha256(markdown).hexdigest(),
                          fact_sha256=facts_sha, result_count=len(rows)), ensure_ascii=False))


if __name__ == '__main__':
    main()
