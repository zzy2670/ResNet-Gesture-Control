"""仅用既有冻结清单和少量文件大小估算余量，不重读原图或权重内容。"""
import json
import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import shutil

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PROJECT = ROOT.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    # 旧冻结名单已经保存逐文件字节数，本轮不枚举整个数据树。
    old = json.loads((PROJECT / 'ResNet_AB项目完整交接_20261102/SHA256SUMS.json').read_text(encoding='utf-8'))
    totals = {}
    for row in old['files']:
        parts = row['path'].split('/')
        if '/data/' in row['path']:
            group = 'old_fixed_data_' + parts[parts.index('data') + 1]
        elif '/runs/' in row['path']:
            group = 'old_scientific_outputs'
        elif row['path'].startswith('程序/'):
            group = 'old_six_programs'
        else:
            group = 'other_old_members'
        totals[group] = totals.get(group, 0) + row['bytes']
    # 相同架构的新checkpoint参数尺寸相同，使用旧实际checkpoint大小作字节级近似。
    classifiers, detectors, old_zips = [], [], []
    for protocol in 'AB':
        base = ROOT / 'runs' / f'{protocol}_20261102_TPE'
        for arch in ('resnet18', 'resnet34', 'resnet50'):
            folder = base if arch == 'resnet18' else base / ('compare_' + arch)
            classifiers.append((folder / 'classifier_best.pt').stat().st_size)
            detectors.append((base / 'detector_best.pt').stat().st_size)
            title = arch.replace('resnet', 'ResNet')
            delivery = PROJECT / ('ResNet手势识别交付' if protocol == 'A' else 'ResNet_B手势识别交付')
            archive = delivery / (f'{title}_图片识别程序.zip' if protocol == 'A' else f'{title}_B协议图片识别程序.zip')
            old_zips.append(archive.stat().st_size)
    package_weights = sum(classifiers) + sum(detectors)
    # 给新十二例照片和小文档每程序2MiB保守余量；ZIP上界用几乎不压缩估计。
    new_packages_upper = package_weights + 6 * 2 * 1024**2
    new_six_zips_upper = int(new_packages_upper * 1.002) + 1024**2
    extraction_peak = max(c + d for c, d in zip(classifiers, detectors)) + 2 * 1024**2
    # 新scientific目录包括每模型classifier与head/日志/预测/科学图，旧manifest不含合并版。
    new_science_upper = sum(classifiers) + 6 * 32 * 1024**2
    # 旧冻结清单已经收录旧六ZIP，新包只是换位置，不能再次加算。
    old_zips_already_included = sum(r['bytes'] for r in old['files'] if r['path'].startswith('程序/') and r['path'].endswith('.zip'))
    old_zip_increment = max(0, sum(old_zips) - old_zips_already_included)
    final_uncompressed_upper = sum(r['bytes'] for r in old['files']) + old_zip_increment + new_science_upper + new_packages_upper + new_six_zips_upper
    report = {'status': 'READONLY_STORAGE_ESTIMATE', 'recorded_at': datetime.now().astimezone().isoformat(),
        'C_free_bytes': shutil.disk_usage(PROJECT).free,
        'old_frozen_member_bytes': sum(r['bytes'] for r in old['files']), 'old_groups': totals,
        'new_six_program_weight_bytes_estimate': package_weights, 'new_six_programs_upper_bytes': new_packages_upper,
        'new_six_ZIP_upper_bytes': new_six_zips_upper, 'all_six_extracted_weights_bytes_if_kept': package_weights,
        'one_extraction_peak_bytes': extraction_peak, 'sequential_cleanup_saves_after_all6_bytes': package_weights,
        'old_six_ZIP_bytes_existing': sum(old_zips), 'old_six_ZIP_already_in_old_manifest_bytes': old_zips_already_included,
        'old_six_ZIP_increment_bytes': old_zip_increment, 'new_scientific_upper_bytes': new_science_upper,
        'final_ZIP_conservative_uncompressed_upper_bytes': final_uncompressed_upper,
        'C_extra_need_conservative_including_new_packages_ZIPs_finalZIP_and_one_temp': new_packages_upper + new_six_zips_upper + final_uncompressed_upper + extraction_peak,
        'root_selected_final_archive': 'E:/ResNet_合并与不合并完整交接_20261103.zip',
        'selected_plan': '项目、正式文档、12程序仍在桌面P；最后大ZIP及partial直接写已有E盘根目录。实际验完后才在桌面项目写位置/SHA回执，现不创建假最终入口。',
        'C_new_packages_ZIPs_and_one_extract_peak_with_E_final_plan': new_packages_upper + new_six_zips_upper + extraction_peak,
        'note': '最终ZIP实际压缩大小尚未测；已压缩JPEG/独立ZIP通常不能再显著压缩。新科学图表未最终生成，此为保守计划而非保证。现有训练已写部分新checkpoint，未来训练增量另需统计。',
        'drives_free_bytes': {drive: shutil.disk_usage(drive).free for drive in ('C:/', 'D:/', 'E:/', 'F:/')}}
    if args.write:
        report['budget_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        payload = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        path = RUN / 'delivery_storage_budget.json'
        with path.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
