"""逐原始张量独立核新CAM；CPU单线程数值运算，不构建或前向任何网络。"""
from __future__ import annotations

from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.path.insert(0, 'E:/python/Lib/site-packages')
import numpy as np
import torch
from PIL import Image

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
LAYERS = ('layer4', 'layer1', 'layer2', 'layer3')
EXPECTED_SHAPES = {'layer1': (64, 56, 56), 'layer2': (128, 28, 28),
                   'layer3': (256, 14, 14), 'layer4': (512, 7, 7)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def max_error(left, right):
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def render_from_saved_arrays(protocol, rows, arrays):
    # 只重建呈现到内存，用已保存RGB和CAM核PNG每个像素，不重新计算网络。
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    figure, axes = plt.subplots(6, 5, figsize=(12.5, 15))
    titles = ['分类输入', 'layer4 主解释 7×7', 'layer1 56×56', 'layer2 28×28', 'layer3 14×14']
    for number, (row, saved) in enumerate(zip(rows, arrays)):
        rgb = saved['input_rgb']
        axes[number, 0].imshow(rgb)
        for column, layer in enumerate(LAYERS, 1):
            colored = plt.colormaps['turbo'](saved[layer + '_normalized_224_cam'])[..., :3]
            overlay = (0.58 * rgb + 0.42 * 255 * colored).clip(0, 255).astype(np.uint8)
            axes[number, column].imshow(overlay)
        axes[number, 0].set_ylabel(f'真实{row["truth"]} / 预测{row["prediction"]}\n{row["source"]["raw_label"]}\n目标为真实类{row["truth"]}',
                                  fontsize=10, rotation=0, ha='right', va='center', labelpad=15)
        for column, axis in enumerate(axes[number]):
            axis.set_xticks([])
            axis.set_yticks([])
            if number == 0:
                axis.set_title(titles[column], fontsize=11)
    figure.suptitle(f'合并标签 {protocol}：真实新训练ResNet18六类分层Grad-CAM', fontsize=15)
    figure.text(.5, .015, '每图独立归一化；解释真实类别分数；颜色不代表准确率，跨层颜色深浅不作绝对强度比较', ha='center', fontsize=10)
    figure.subplots_adjust(top=.945, bottom=.045, left=.16, right=.99, hspace=.1, wspace=.03)
    output = io.BytesIO()
    figure.savefig(output, format='png', dpi=200)
    plt.close(figure)
    output.seek(0)
    with Image.open(output) as image:
        return np.asarray(image.convert('RGB'))


def main():
    destination = RUN / 'merged_gradcam_independent_review.json'
    require(not destination.exists(), '已有CAM科学独审，拒绝覆盖')
    preflight_path = RUN / 'merged_gradcam_preflight_review.json'
    preflight = read(preflight_path)
    require(preflight['status'] == 'PASS_GRADCAM_SOURCE_PREFLIGHT_WITH_LIMITATIONS', 'CAM源码预审未通过')
    bound = {str(Path(__file__).resolve()): sha(__file__), str(preflight_path): sha(preflight_path)}
    def bind(path, expected=None):
        path = Path(path)
        digest = sha(path)
        require(expected is None or digest == expected, '身份SHA变化：' + str(path))
        key = str(path.resolve())
        require(key not in bound or bound[key] == digest, '重复依赖的SHA冲突')
        bound[key] = digest
    for path, digest in preflight['bound_file_sha256'].items():
        bind(path, digest)
    reports, formula_checks = [], []
    for protocol in ('A', 'B'):
        folder = RUN / protocol / 'resnet18'
        cam = folder / 'trained_gradcam'
        expected = {'metadata.json', 'trained_six_class_matrix.png'} | {f'row_{i}_tensors.npz' for i in range(6)}
        require({p.name for p in cam.iterdir()} == expected, 'CAM最终精确8文件集合错误')
        doc = read(cam / 'metadata.json')
        require(doc['status'] == 'COMPLETE_PENDING_INDEPENDENT_CAM_REVIEW' and doc['protocol'] == protocol
            and doc['device'] == 'cpu' and doc['architecture'] == 'resnet18' and doc['label_variant'] == 'merged'
            and doc['used_for_selection'] is False and doc['normalization'] == 'independent_per_sample_per_layer', 'CAM来源/范围不符')
        require(set(doc['output_sha256']) == expected - {'metadata.json'}, 'metadata必须锚定精确7产物')
        for name, digest in doc['output_sha256'].items():
            bind(cam / name, digest)
        bind(cam / 'metadata.json')
        for file, digest in doc['bound_file_sha256'].items():
            key = str(Path(file).resolve())
            require(key not in bound or bound[key] == digest, 'CAM bound冲突')
            bound[key] = digest
        bind(folder / 'classifier_best.pt', doc['checkpoint_sha256'])
        state = torch.load(folder / 'classifier_best.pt', map_location='cpu', weights_only=True)['model']
        fc_weight, fc_bias = state['fc.1.weight'], state['fc.1.bias']
        require(tuple(fc_weight.shape) == (6, 512), '新实际6类FC头形状错误')
        oracle_doc = read(folder / 'test_oracle_crop.json')
        require(oracle_doc['checkpoint_sha256'] == doc['checkpoint_sha256'], 'CAM与冻结测试不是同一新模型')
        oracle = {(r['image_id'], r['hand_index']): r for r in oracle_doc['predictions']}
        whole = {r['image_id']: r for r in read(folder / 'test_whole_photo.json')['predictions']}
        crops = read(folder / 'crops_test.json')
        rows = doc['rows']
        require(len(rows) == 6 and [r['truth'] for r in rows] == list(range(6)), '六真实类别不唯一完整')
        cached_arrays, row_reports = [], []
        for label, row in enumerate(rows):
            record = row['source']
            planned = preflight['fixed_selections_independently_recomputed'][protocol][label]
            require((record['image_id'], record['hand_index'], record['raw_label']) ==
                    (planned['image_id'], planned['hand_index'], planned['raw_label']), '固定选样偏离非作者预审')
            require(record == crops[row['index']] and row['target'] == row['truth'] == record['label'] == label,
                    'GT手索引或目标不是固定真实类')
            require(label != 0 or record['raw_label'] not in {'peace', 'peace_inverted', 'two_up', 'two_up_inverted'}, '新0冒用前进类')
            require(row['input_scope'] == 'ground_truth_crop_not_detector_crop', '解释输入范围不符')
            key = record['image_id'], record['hand_index']
            require(row['frozen_oracle_prediction'] == oracle[key] and row['frozen_whole_photo_prediction'] == whole[key[0]], '附带冻结输出被改写')
            bind(record['path'])
            bind(record['source_path'])
            with Image.open(record['path']) as image:
                rgb = np.asarray(image.convert('RGB'))
            with Image.open(record['source_path']) as source:
                width, height = source.size
            require(rgb.shape == (224, 224, 3), '原GT裁图不是224RGB')
            # 从已冻结bbox重算1.2方形边界，核原图坐标和GT裁图关系。
            x1, y1, x2, y2 = record['bbox_xyxy']
            side = max(x2 - x1, y2 - y1) * 1.2
            import math
            rectangle = [math.floor((x1 + x2 - side) / 2), math.floor((y1 + y2 - side) / 2),
                         math.ceil((x1 + x2 + side) / 2), math.ceil((y1 + y2 + side) / 2)]
            require(rectangle == record['crop_rectangle'] and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, 'GT框/裁图几何不符')
            with np.load(cam / f'row_{label}_tensors.npz', allow_pickle=False) as stored:
                saved = {name: stored[name] for name in stored.files}
            expected_keys = {'input_normalized_tensor', 'input_rgb', 'logits'} | {layer + '_' + suffix for layer in LAYERS
                for suffix in ('activation', 'gradient', 'channel_weights', 'native_cam', 'upsampled_224_cam', 'normalized_224_cam')}
            require(set(saved) == expected_keys and all(np.isfinite(v).all() for v in saved.values()), 'NPZ字段/finite错误')
            require(np.array_equal(rgb, saved['input_rgb']), 'NPZ输入RGB与实际原GT裁图不一致')
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).contiguous().float().div(255)
            normalized = tensor.sub(torch.tensor([.485, .456, .406])[:, None, None]).div(torch.tensor([.229, .224, .225])[:, None, None])
            require(np.array_equal(normalized.numpy(), saved['input_normalized_tensor']), '归一化输入张量不等')
            require(hashlib.sha256(saved['input_normalized_tensor'].tobytes()).hexdigest() == row['tensor_sha256'], '输入tensor SHA错误')
            logits = torch.from_numpy(saved['logits'])
            probability = logits.softmax(0).numpy()
            require(max_error(probability, row['probabilities']) < 1e-7 and int(probability.argmax()) == row['prediction'], '保存logits/CPU概率/预测不一致')
            delta = max_error(row['probabilities'], oracle[key]['probabilities'])
            require(abs(delta - row['max_probability_abs_difference']) < 1e-12, 'CPU与冻结GPU概率差异记录不符')
            # 只用保存layer4激活和实际FC张量做闭式代数检查，不调用任何网络forward。
            layer4 = torch.from_numpy(saved['layer4_activation'])
            algebra_logits = fc_weight @ layer4.mean(dim=(1, 2)) + fc_bias
            logit_error = max_error(algebra_logits.numpy(), saved['logits'])
            require(logit_error < 2e-5, '保存layer4与真实新FC头不能重建logits')
            target_gradient = (fc_weight[label] / 49)[:, None, None].expand_as(layer4)
            gradient_error = max_error(target_gradient.numpy(), saved['layer4_gradient'])
            require(gradient_error < 1e-8, 'layer4梯度不是该真实类别logit的实际FC权重/49')
            for layer in LAYERS:
                activation, gradient = saved[layer + '_activation'], saved[layer + '_gradient']
                require(activation.shape == gradient.shape == EXPECTED_SHAPES[layer], '分层activation/gradient形状错误')
                a, g = torch.from_numpy(activation), torch.from_numpy(gradient)
                weights = g.mean(dim=(1, 2))
                native = (a * weights[:, None, None]).sum(0).clamp_min(0)
                up = torch.nn.functional.interpolate(native[None, None], (224, 224), mode='bilinear', align_corners=False)[0, 0]
                scaled = ((up - up.min()) / (up.max() - up.min())).numpy() if up.max() > up.min() else np.zeros((224, 224))
                differences = {suffix: max_error(value, saved[layer + '_' + suffix]) for suffix, value in
                    (('channel_weights', weights.numpy()), ('native_cam', native.numpy()), ('upsampled_224_cam', up.numpy()), ('normalized_224_cam', scaled))}
                require(differences['channel_weights'] < 1e-8 and differences['native_cam'] < 2e-6
                    and differences['upsampled_224_cam'] < 2e-6 and differences['normalized_224_cam'] < 3e-6, 'CAM数值公式重算失败')
                require(scaled.min() >= 0 and scaled.max() <= 1, '逐map归一化范围错误')
                formula_checks.append({'protocol': protocol, 'class': label, 'layer': layer,
                    'activation_shape': list(activation.shape), 'gradient_absmax': float(np.abs(gradient).max()),
                    'native_min': float(native.min()), 'native_max': float(native.max()),
                    'normalized_min': float(scaled.min()), 'normalized_max': float(scaled.max()),
                    'errors': differences})
            row_reports.append({'class': label, 'raw_label': record['raw_label'], 'image_id': key[0], 'hand_index': key[1],
                'prediction': row['prediction'], 'frozen_oracle_prediction': oracle[key]['prediction'],
                'max_CPU_vs_frozen_probability_difference': delta, 'saved_layer4_FC_algebra_logit_error': logit_error,
                'true_logit_layer4_FC_weight_div49_gradient_error': gradient_error, 'whole_photo_truth': whole[key[0]]['truth'],
                'selection': row['selection'], 'crop_geometry_checked': True})
            cached_arrays.append(saved)
        reconstruction = render_from_saved_arrays(protocol, rows, cached_arrays)
        with Image.open(cam / 'trained_six_class_matrix.png') as image:
            actual = np.asarray(image.convert('RGB'))
        require(actual.shape == reconstruction.shape == (3000, 2500, 3), '矩阵图尺寸不符')
        pixel_error = int(np.max(np.abs(actual.astype(np.int16) - reconstruction.astype(np.int16))))
        changed_pixels = int(np.count_nonzero(np.any(actual != reconstruction, axis=2)))
        require(changed_pixels == 0, '大图不能由其保存RGB/CAM逐像素重建')
        reports.append({'protocol': protocol, 'checkpoint_sha256': doc['checkpoint_sha256'], 'rows': row_reports,
            'PNG_shape': list(actual.shape), 'PNG_entire_pixel_reconstruction_exact': True,
            'PNG_max_channel_difference': pixel_error, 'PNG_changed_pixels': changed_pixels,
            'original_resolution_visually_viewed': True})
        del state, cached_arrays, reconstruction, actual
    # 旧A的既有整图SHA提供直接不变证据；旧B沿用其冻结总包映射中的SHA。
    old_a = read(ROOT / 'runs/A_20261102_TPE/final_independent_review.json')['trained_gradcam_review']
    bind(ROOT / 'runs/A_20261102_TPE/trained_gradcam/trained_six_class_matrix.png', old_a['matrix_sha256'])
    bind(ROOT / 'runs/A_20261102_TPE/trained_gradcam/metadata.json', old_a['metadata_sha256'])
    old_map_path = ROOT.parent / 'ResNet_AB项目完整交接_20261102/冻结成员映射.json'
    old_map = read(old_map_path)['files']
    old_b = ROOT / 'runs/B_20261102_TPE/trained_gradcam'
    for name in ('trained_six_class_matrix.png', 'metadata.json'):
        path = old_b / name
        match = next(r for r in old_map if str(Path(r['source']).resolve()) == str(path.resolve()))
        bind(path, match['sha256'])
    require(len(formula_checks) == 48 and len(reports) == 2, '两协议48层数值范围不完整')
    require(sum(1 for path in bound if '/trained_gradcam/' in path.replace('\\', '/')
        and '/MERGED_20261102/' in path.replace('\\', '/')) >= 16, '未绑定全部16新CAM产物')
    result = {'status': 'PASS', 'issues': [], 'reviewed_at': datetime.now().astimezone().isoformat(),
        'scope': '非作者单CPU线程核12NPZ/48层原值、16新产物与两张完整PNG；只数学/图像呈现复核，没有构建或调用网络forward/backward。',
        'bound_file_sha256': bound, 'protocols': reports, 'formula_checks': formula_checks,
        'checks': {'protocols': 2, 'NPZ_files': 12, 'layer_maps_recomputed': 48, 'new_products_bound': 16,
            'entire_PNG_pixel_reconstructions_exact': 2, 'old_A_B_CAM_SHA_unchanged': True,
            'GT_RGB_and_ImageNet_input_tensor_exact': True, 'true_logit_target_verified_by_actual_FC_gradient': True},
        'visual_observations': ['A的layer4在各例主要覆盖掌心/指根或握拳主体；A三指其他类也在手掌附近有强相对响应，不能把热点等同于可执行手势。',
            'B0是two_up整图里另一只no_gesture手的GT裁图；本手目标0，但该完整照片真值仍1。新0例没有把旧two_up_inverted图改标。',
            'A/B的peace与张掌layer4多为粗粒度手部区域；部分layer2/layer3及layer1在衣袖、窗口、背景纹理或图边也有响应，不能说模型只看指尖或完全排除背景。',
            'B在缩放预览可能看似缺后3行文字；已看实际2500×3000原图，全部列名/6行标/底注存在且与内存逐像素重建完全一致，无需重绘。'],
        'limitations': ['人工GT裁图分类解释，不是检测器或整图检测裁图CAM；附带whole_photo记录只用于追踪边界。',
            '每map独立minmax，不能按跨层/样本颜色深浅比较绝对贡献，也不能据六例估计性能或解释因果。',
            '48层数学复算证明保存值与公式一致；全模型梯度实际执行沿用已冻结源码/产物及root执行记录，不在本审核重新运行。',
            'CPU和冻结测试概率差异逐样本量化保留，不宣称逐位相同；不改参数、阈值或测试输出。'],
        'models_constructed': False, 'model_forward_or_backward_executed': False,
        'GPU_executed': False, 'accuracy_or_threshold_selection_executed': False}
    payload = (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    for file, digest in bound.items():
        require(sha(file) == digest, '最终提交前CAM或来源身份改变：' + file)
    with destination.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': 'PASS', 'maps': 48, 'PNG_pixel_exact': 2,
        'sha256': sha(destination), 'report': str(destination)}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        path = RUN / 'merged_gradcam_independent_review_failure.json'
        if not path.exists():
            with path.open('x', encoding='utf-8') as stream:
                json.dump({'status': 'FAIL_REVIEW_REQUIRED', 'error': repr(error), 'traceback': traceback.format_exc(),
                    'source_outputs_untouched': True}, stream, ensure_ascii=False, indent=2)
        raise
