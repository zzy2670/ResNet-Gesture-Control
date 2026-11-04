"""串行监督B协议三架构训练和测试；保存进程日志，不调用旧A预算或截止。"""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/B_20261102_TPE'
ENTRY = ROOT / 'run_B_pipeline.py'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def alive(pid):
    # Windows进程句柄只读查询；不对其他进程发信号或终止。
    if os.name == 'nt':
        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle: return False
        code = ctypes.c_ulong()
        try:
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def event(**data):
    data = {'time': datetime.now().astimezone().isoformat(), **data}
    with (RUN / 'controller_events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(json.dumps(data, ensure_ascii=False), flush=True)


def plan():
    operations = []
    for arch in ('resnet18', 'resnet34', 'resnet50'):
        operations += [(arch, 'prepare', []), (arch, 'smoke', ['--device', 'cpu', '--workers', '0']),
                       (arch, 'features', ['--batch-size', '128']),
                       (arch, 'search', ['--batch-size', '512', '--epochs', '18', '--trials', '24']),
                       (arch, 'finetune', ['--batch-size', '128', '--epochs', '18', '--lr', '0.0001', '--patience', '4'])]
        if arch == 'resnet18':
            operations.append((arch, 'detector', ['--batch-size', '24', '--epochs', '16', '--lr', '0.003']))
        operations += [(arch, 'calibrate', ['--batch-size', '32']), (arch, 'evaluate', ['--batch-size', '32'])]
    for arch in ('resnet18', 'resnet34', 'resnet50'):
        operations += [(arch, 'latency', ['--device', 'cpu', '--workers', '0', '--batch-size', '1']),
                       (arch, 'classifier-latency', ['--device', 'cpu', '--workers', '0', '--batch-size', '1'])]
    return operations


def outputs(stage):
    # 搜索选中的分类文件会被后续微调正常替换，故搜索凭据锚定不可变摘要和试验账本。
    return {'prepare': ['data_audit.json', 'crops_train.json', 'crops_val.json', 'crops_test.json', 'model_info.json'],
            'smoke': ['cpu_smoke.json'], 'features': ['features_train.pt', 'features_val.pt'],
            'search': ['tpe_summary.json', 'tpe.sqlite3'], 'finetune': ['classifier_best.pt'],
            'detector': ['detector_best.pt'], 'calibrate': ['calibration.json', 'val_whole_raw.json'],
            'evaluate': ['test_oracle_crop.json', 'test_whole_photo.json', 'test_detector.json'],
            'latency': ['cpu_latency.json'], 'classifier-latency': ['classifier_cpu_latency_unified.json']}[stage]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retry-failed', action='store_true', help='仅在故障已复盘后明确允许重试未成功阶段')
    parser.add_argument('--list', action='store_true', help='列出阶段，不启动任何工作')
    args = parser.parse_args()
    operations = plan()
    if args.list:
        print('\n'.join(f'{arch}/{stage}' for arch, stage, _ in operations)); return
    RUN.mkdir(parents=True, exist_ok=True)
    lock = RUN / 'controller_lock.json'
    if lock.exists():
        previous = json.loads(lock.read_text(encoding='utf-8'))
        if alive(previous['pid']):
            raise RuntimeError(f"已有B监督进程仍运行PID={previous['pid']}，不重复启动")
    write_json(lock, {'pid': os.getpid(), 'started_at': datetime.now().astimezone().isoformat()})
    code = {name: sha(ROOT / name) for name in ('run_B_pipeline.py', 'run_B_training.py', 'train_A.py', 'compare_resnets_A.py')}
    frozen = RUN / 'controller_source_identity.json'
    if frozen.exists() and json.loads(frozen.read_text(encoding='utf-8')) != code:
        raise RuntimeError('监督或训练源码身份已变，须明确复盘，不直接恢复')
    write_json(frozen, code)
    snapshots = RUN / 'code_snapshot'
    snapshots.mkdir(exist_ok=True)
    for name, digest in code.items():
        target = snapshots / name
        if target.exists() and sha(target) != digest: raise RuntimeError('已有代码快照身份不匹配')
        if not target.exists(): target.write_bytes((ROOT / name).read_bytes())
    event(status='controller_started', pid=os.getpid(), stages=len(operations), source_sha256=code)
    for arch, stage, extras in operations:
        out = RUN if arch == 'resnet18' else RUN / ('compare_' + arch)
        out.mkdir(parents=True, exist_ok=True)
        markers = out / 'stage_markers'; markers.mkdir(exist_ok=True)
        marker = markers / (stage + '.json')
        if marker.exists():
            saved = json.loads(marker.read_text(encoding='utf-8'))
            if saved['source_sha256'] != code or any(sha(out / name) != digest for name, digest in saved['outputs'].items()):
                raise RuntimeError(f'{arch}/{stage}成功凭据或文件变化，拒绝跳过')
            event(architecture=arch, stage=stage, status='verified_existing_success_skipped'); continue
        old_logs = list((out / 'stage_logs').glob(stage + '_attempt*.log')) if (out / 'stage_logs').exists() else []
        if old_logs and not args.retry_failed:
            raise RuntimeError(f'{arch}/{stage}存在未成功尝试，先查原日志再显式--retry-failed')
        free_bytes = shutil.disk_usage(RUN).free
        # 大缓存与checkpoint写入前保留少量真实余量；空间不足时停在阶段边界。
        if free_bytes < 512 * 1024 * 1024:
            raise RuntimeError(f'磁盘剩余不足512MiB，当前{free_bytes}字节；不启动新阶段')
        command = [sys.executable, str(ENTRY), stage, '--arch', arch, *extras]
        log = out / 'stage_logs' / f'{stage}_attempt{len(old_logs)+1}.log'; log.parent.mkdir(exist_ok=True)
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
        with log.open('w', encoding='utf-8') as handle:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            event(architecture=arch, stage=stage, status='started', pid=child.pid, command=command, log=str(log))
            write_json(RUN / 'active_stage.json', {'architecture': arch, 'stage': stage, 'pid': child.pid,
                       'supervisor_pid': os.getpid(), 'log': str(log), 'status': 'running'})
            exit_code = child.wait()
        event(architecture=arch, stage=stage, status='exited', pid=child.pid, exit_code=exit_code)
        write_json(RUN / 'active_stage.json', {'architecture': arch, 'stage': stage, 'pid': child.pid,
                   'supervisor_pid': os.getpid(), 'log': str(log), 'status': 'exit0' if exit_code == 0 else 'failed', 'exit_code': exit_code})
        if exit_code:
            raise RuntimeError(f'{arch}/{stage}退出{exit_code}，原日志保留，不自动盲重试')
        evidence = {name: sha(out / name) for name in outputs(stage)}
        write_json(marker, {'architecture': arch, 'stage': stage, 'status': 'exit0', 'source_sha256': code,
                            'outputs': evidence, 'completed_at': datetime.now().astimezone().isoformat()})
    event(status='all_B_training_test_latency_stages_exit0')
    write_json(RUN / 'all_stages_complete.json', {'status': 'PASS', 'scope': 'training_test_and_latency_execution',
               'completed_at': datetime.now().astimezone().isoformat(), 'independent_review_pending': True})


if __name__ == '__main__':
    main()
