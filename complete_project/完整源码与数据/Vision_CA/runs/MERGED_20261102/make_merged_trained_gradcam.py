"""合并标签真实分层Grad-CAM；默认仅显示计划，正式运行必须通过全部外部门禁。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
VISION = RUN.parents[1]
LAYERS = ('layer4', 'layer1', 'layer2', 'layer3')
MERGED_ONE = {'peace', 'peace_inverted', 'two_up', 'two_up_inverted'}
SCHEMA = {
    'permit': {'purpose': 'merged_gradcam_cpu', 'confirmed_by': 'root',
               'cpu_window_confirmed': True, 'expires_at': 'ISO8601 with timezone',
               'speed_review_path': 'absolute path', 'speed_review_status': 'exact PASS status',
               'bound_file_sha256': {'absolute path': 'sha256'}},
    'outputs': ['trained_six_class_matrix.png', 'row_{0..5}_tensors.npz', 'metadata.json'],
    'npz_per_layer': ['activation', 'gradient', 'native_cam', 'upsampled_224_cam',
                      'normalized_224_cam', 'channel_weights'],
    'metadata': ['checkpoint_sha256', 'device', 'threads', 'rows', 'bound_file_sha256',
                 'formula', 'overlay', 'normalization', 'gate_identity', 'status'],
}


def read(path):
    # 默认计划和所有门禁只使用标准库。
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(bound):
    # 在加载前及最终提交前核验同一批实际文件。
    for name, digest in bound.items():
        if sha(name) != digest:
            raise ValueError(f'绑定文件已变化：{name}')


def commit(path, payload):
    # 所有新结果独占创建，绝不覆盖旧图或旧证据。
    with Path(path).open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def select(protocol):
    # 优先沿用旧图的固定身份，不按成功、置信度或热图挑图。
    folder = RUN / protocol / 'resnet18'
    rows = read(folder / 'crops_test.json')
    old = VISION / f'runs/{protocol}_20261102_TPE'
    prior = read(old / 'trained_gradcam/metadata.json')
    old_rows = read(old / 'crops_test.json')
    key = lambda row: (row['image_id'], row['hand_index'])
    lookup = {key(row): (i, row) for i, row in enumerate(rows)}
    old_lookup = {key(row): row for row in old_rows}
    if len(lookup) != len(rows) or set(lookup) != set(old_lookup):
        raise ValueError('新旧测试手身份不唯一或split已改变')
    chosen = []
    for label in range(6):
        previous = next(item['source'] for item in prior['rows'] if item['truth'] == label)
        index, record = lookup[key(previous)]
        reason = 'reuse_original_fixed_sample_identity'
        if record['label'] != label:
            index, record = min(((i, r) for i, r in enumerate(rows) if r['label'] == label),
                                key=lambda pair: key(pair[1]))
            reason = 'old_identity_changed_semantics_first_immutable_key_legal_class'
        if label == 0 and record['raw_label'] in MERGED_ONE:
            raise ValueError('新0类不能包含四种前进姿态')
        if label == 1 and record['raw_label'] not in MERGED_ONE:
            raise ValueError('新1类原始标签不符')
        original = old_lookup[key(record)]
        for field in ('path', 'source_path', 'bbox_xyxy', 'crop_rectangle', 'raw_label', 'user_id'):
            if original[field] != record[field]:
                raise ValueError(f'原图/GT裁图身份改变：{field}')
        chosen.append({'index': index, 'source': record, 'previous_sample_key': list(key(previous)),
                       'selection': reason, 'original_label': original['label'], 'target': label})
    return chosen


def validate_permit(permit):
    # 轻量实时检查单独保留，最后提交前不再重复大文件哈希。
    if (permit.get('purpose') != 'merged_gradcam_cpu' or permit.get('confirmed_by') != 'root'
            or permit.get('cpu_window_confirmed') is not True):
        raise ValueError('缺少root明确CPU窗口确认')
    expiry = datetime.fromisoformat(permit['expires_at'])
    if expiry.tzinfo is None or datetime.now().astimezone() >= expiry:
        raise ValueError('CPU窗口已过期或缺时区')


def gates(permit_path):
    # 复用已独审测速工具的六模型科学门禁，不复制一套较弱规则。
    path = VISION / 'benchmark_label_variants_cpu.py'
    spec = importlib.util.spec_from_file_location('merged_cam_readiness', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    science = module.scientific_readiness()
    permit = read(permit_path)
    validate_permit(permit)
    bound = dict(science['bound_file_sha256'])
    for name in (path, Path(__file__), RUN / 'MERGED_GRADCAM_PROTOCOL.md', Path(permit_path)):
        bound[str(name.resolve())] = sha(name)
    speed_path = RUN / 'label_variants_cpu_benchmark.json'
    review_path = Path(permit['speed_review_path']).resolve()
    speed, review = read(speed_path), read(review_path)
    status = permit['speed_review_status']
    if not status.startswith('PASS') or review.get('status') != status or review.get('issues', []) != []:
        raise ValueError('十二模型CPU速度真实独审未通过')
    if speed.get('status') != 'complete':
        raise ValueError('十二模型CPU测速尚未complete')
    permit_bound = permit['bound_file_sha256']
    for required in (speed_path, review_path):
        if str(required.resolve()) not in permit_bound:
            raise ValueError('root确认未绑定真实速度数据及独审')
    # root必须绑定已审核的速度数据，不能用不关联数据的PASS代替。
    review_bound = review.get('bound_file_sha256', {})
    if review.get('data_sha256') != sha(speed_path) and review_bound.get(str(speed_path)) != sha(speed_path):
        raise ValueError('速度独审没有绑定当前十二模型测速数据')
    for name, digest in permit_bound.items():
        if name in bound and bound[name] != digest:
            raise ValueError('许可与科学独审身份冲突')
        bound[name] = digest
    verify(bound)
    return bound


def run(args):
    # 此分支只允许root在最终测速后明确安排；本轮准备不调用。
    if not args.permit:
        raise ValueError('--run-cpu必须提供--permit')
    bound = gates(args.permit)
    folder = RUN / args.protocol / 'resnet18'
    selected = select(args.protocol)
    dest = folder / 'trained_gradcam'
    if dest.exists():
        raise FileExistsError('新CAM输出目录已存在，拒绝重跑或覆盖')
    for name in (folder / 'crops_test.json', VISION / 'train_A.py',
                 VISION / 'make_A_trained_gradcam.py', VISION / 'make_B_trained_gradcam.py',
                 VISION / f'runs/{args.protocol}_20261102_TPE/trained_gradcam/metadata.json',
                 VISION / f'runs/{args.protocol}_20261102_TPE/crops_test.json'):
        bound[str(name)] = sha(name)
    for item in selected:
        for field in ('path', 'source_path'):
            name = item['source'][field]
            bound[name] = sha(name)
    # 禁用CUDA可见性，所有张量和checkpoint均显式留在CPU。
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(VISION))
    import train_A as base
    torch.set_num_threads(args.threads)
    checkpoint = folder / 'classifier_best.pt'
    test = read(folder / 'test_oracle_crop.json')
    if sha(checkpoint) != test['checkpoint_sha256']:
        raise ValueError('新checkpoint与新完整人工框测试不一致')
    verify(bound)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    net = base.resnet(None, state['params']['dropout']).cpu().eval()
    net.load_state_dict(state['model'], strict=True)
    dataset = base.CropDataset(folder, 'test')
    saved_predictions = {(r['image_id'], r['hand_index']): r for r in test['predictions']}
    whole = read(folder / 'test_whole_photo.json')
    whole_rows = {r['image_id']: r for r in whole['predictions']}
    dest.mkdir()
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    figure, axes = plt.subplots(6, 5, figsize=(12.5, 15))
    titles = ['分类输入', 'layer4 主解释 7×7', 'layer1 56×56', 'layer2 28×28', 'layer3 14×14']
    metadata = []
    for number, chosen in enumerate(selected):
        record, captured, handles = chosen['source'], {}, []
        def hook(name):
            def save(module, inputs, output):
                output.retain_grad()
                captured[name] = output
            return save
        try:
            for name in LAYERS:
                handles.append(getattr(net, name)[-1].register_forward_hook(hook(name)))
            tensor, truth, _ = dataset[chosen['index']]
            tensor = tensor.unsqueeze(0).cpu().requires_grad_(True)
            net.zero_grad(set_to_none=True)
            scores = net(tensor)
            probabilities = scores.detach().softmax(1)[0].numpy()
            scores[0, truth].backward()
            with Image.open(record['path']) as source:
                image = np.asarray(source.convert('RGB'))
            if image.shape != (224, 224, 3):
                raise ValueError('原分类输入不是224×224，不能静默重采样')
            arrays = {'input_normalized_tensor': tensor.detach()[0].numpy(),
                      'input_rgb': image, 'logits': scores.detach()[0].numpy()}
            axes[number, 0].imshow(image)
            for column, name in enumerate(LAYERS, 1):
                activation = captured[name].detach()[0]
                gradient = captured[name].grad.detach()[0]
                weight = gradient.mean(dim=(1, 2))
                native = torch.relu((weight[:, None, None] * activation).sum(0))
                up = torch.nn.functional.interpolate(native[None, None], size=(224, 224),
                    mode='bilinear', align_corners=False)[0, 0]
                lo, hi = up.min(), up.max()
                scaled = ((up - lo) / (hi - lo)).numpy() if hi > lo else np.zeros((224, 224))
                colored = plt.colormaps['turbo'](scaled)[..., :3]
                overlay = (0.58 * image + 0.42 * 255 * colored).clip(0, 255).astype(np.uint8)
                axes[number, column].imshow(overlay)
                for suffix, value in (('activation', activation.numpy()), ('gradient', gradient.numpy()),
                    ('channel_weights', weight.numpy()), ('native_cam', native.numpy()),
                    ('upsampled_224_cam', up.numpy()), ('normalized_224_cam', scaled)):
                    if not np.isfinite(value).all():
                        raise ValueError('CAM存在非有限数值')
                    arrays[f'{name}_{suffix}'] = value
            # 保存真实原数值，供独审者从梯度均值独立重建各层CAM。
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            commit(dest / f'row_{truth}_tensors.npz', buffer.getvalue())
            prediction = int(probabilities.argmax())
            axes[number, 0].set_ylabel(f'真实{truth} / 预测{prediction}\n{record["raw_label"]}\n目标为真实类{truth}',
                                      fontsize=10, rotation=0, ha='right', va='center', labelpad=15)
            for column, axis in enumerate(axes[number]):
                axis.set_xticks([])
                axis.set_yticks([])
                if number == 0:
                    axis.set_title(titles[column], fontsize=11)
            saved = saved_predictions[(record['image_id'], record['hand_index'])]
            metadata.append({**chosen, 'truth': truth, 'prediction': prediction,
                'probabilities': probabilities.tolist(), 'frozen_oracle_prediction': saved,
                'max_probability_abs_difference': float(np.max(np.abs(probabilities - saved['probabilities']))),
                'frozen_whole_photo_prediction': whole_rows[record['image_id']],
                'input_scope': 'ground_truth_crop_not_detector_crop',
                'tensor_sha256': hashlib.sha256(arrays['input_normalized_tensor'].tobytes()).hexdigest()})
        finally:
            for handle in handles:
                handle.remove()
    figure.suptitle(f'合并标签 {args.protocol}：真实新训练ResNet18六类分层Grad-CAM', fontsize=15)
    figure.text(.5, .015, '每图独立归一化；解释真实类别分数；颜色不代表准确率，跨层颜色深浅不作绝对强度比较', ha='center', fontsize=10)
    figure.subplots_adjust(top=.945, bottom=.045, left=.16, right=.99, hspace=.1, wspace=.03)
    png = io.BytesIO()
    figure.savefig(png, format='png', dpi=200)
    plt.close(figure)
    commit(dest / 'trained_six_class_matrix.png', png.getvalue())
    result = {'status': 'COMPLETE_PENDING_INDEPENDENT_CAM_REVIEW', 'protocol': args.protocol,
        'architecture': 'resnet18', 'label_variant': 'merged', 'checkpoint_sha256': sha(checkpoint),
        'device': 'cpu', 'threads': args.threads, 'trained_project_model': True,
        'used_for_selection': False, 'rows': metadata, 'bound_file_sha256': bound,
        'gate_identity': str(Path(args.permit).resolve()),
        'formula': 'ReLU(sum_k(mean_spatial(gradient_k)*activation_k)); bilinear224; per_map_minmax',
        'overlay': '0.58*RGB + 0.42*turbo', 'normalization': 'independent_per_sample_per_layer',
        'limitations': '人工GT裁图解释；非检测器CAM、非物理因果、不新增性能分数；与原图手部关系须逐图独审。',
        'output_sha256': {p.name: sha(p) for p in dest.iterdir() if p.is_file()}}
    payload = (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    # 图和数值先形成确切字节，完成标记提交前再次核验实时许可和全部身份。
    gates(args.permit)
    verify(bound)
    # 成功标记必须绑定仍然存在且未改变的六份张量与一张矩阵图。
    expected_outputs = {'trained_six_class_matrix.png'} | {f'row_{i}_tensors.npz' for i in range(6)}
    if {p.name for p in dest.iterdir()} != expected_outputs:
        raise ValueError('CAM输出集合与六份张量加一张矩阵不一致')
    for name, digest in result['output_sha256'].items():
        if sha(dest / name) != digest:
            raise ValueError(f'CAM输出在成功提交前已变化：{name}')
    # 大文件身份全核后，紧邻提交重读同一许可字节并核当前有效期。
    permit_bytes = Path(args.permit).read_bytes()
    if hashlib.sha256(permit_bytes).hexdigest() != bound[str(Path(args.permit).resolve())]:
        raise ValueError('CPU许可在成功提交前已变化')
    validate_permit(json.loads(permit_bytes.decode('utf-8-sig')))
    commit(dest / 'metadata.json', payload)
    print(str(dest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-cpu', action='store_true')
    parser.add_argument('--permit', type=Path)
    parser.add_argument('--protocol', choices=('A', 'B'), default='A')
    parser.add_argument('--threads', type=int, default=8, choices=range(1, 9))
    parser.add_argument('--schema', action='store_true')
    parser.add_argument('--plan-samples', action='store_true')
    args = parser.parse_args()
    if args.run_cpu:
        run(args)
    elif args.schema:
        print(json.dumps(SCHEMA, ensure_ascii=False, indent=2))
    elif args.plan_samples:
        print(json.dumps(select(args.protocol), ensure_ascii=False, indent=2))
    else:
        print('PREPARATION_ONLY：A/B新ResNet18各6×5矩阵，input/layer4/layer1/layer2/layer3。')
        print('未加载模型、未推理。--run-cpu须六模型完成与科学总独审、十二CPU complete及独审、root绑定CPU窗口许可。')
        print('内部协议：' + str(RUN / 'MERGED_GRADCAM_PROTOCOL.md'))


if __name__ == '__main__':
    main()
