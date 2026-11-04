"""串行监督六个合并标签模型；不启动搜索、检测器训练或CPU测速。"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_merged_pipeline import ROOT, RUN, SOURCES, STAGES, DEFAULT_PRETRAINED, make_identity, out_for, pin_identity, sha

BUNDLED_PYTHON = Path('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')


def now():
    return datetime.now().astimezone().isoformat()


def append_event(**fields):
    # 事件追加保存，健康查看只读此文件，绝不触发自动重启。
    row = {'time': now(), **fields}
    with (RUN / 'controller_events.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


def save_exclusive(path, data, precommit=None):
    # 成功凭据采用独占新文件，避免覆盖一次已完成运行。
    payload = json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')
    if precommit:
        precommit()
    with path.open('xb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def outputs(stage):
    # head的classifier_best会被正常微调替换，凭据锚定独立冻结头及摘要。
    return {'smoke': ['cpu_smoke.json'], 'head': ['fixed_head_best.pt', 'fixed_head_summary.json'],
        'finetune': ['classifier_best.pt', 'finetune_summary.json'],
        'reuse-detector': ['detector_best.pt', 'shared_detector_identity.json'],
        'calibrate': ['calibration.json', 'val_whole_raw.json'],
        'evaluate': ['test_oracle_crop.json', 'test_whole_photo.json', 'test_detector.json']}[stage]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--preflight-only', action='store_true', help='只进行六个CPU烟测')
    parser.add_argument('--pretrained-root', type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument('--python', type=Path, default=BUNDLED_PYTHON)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    stages = ('smoke',) if args.preflight_only else STAGES
    # 所有CPU烟测先完成，才进入任何长期GPU阶段。
    operations = [(p, a, 'smoke') for p in ('A', 'B') for a in ('resnet18', 'resnet34', 'resnet50')]
    if not args.preflight_only:
        operations += [(p, a, s) for p in ('A', 'B') for a in ('resnet18', 'resnet34', 'resnet50') for s in stages if s != 'smoke']
    if args.list:
        print('\n'.join('/'.join(op) for op in operations))
        return
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    import psutil
    RUN.mkdir(parents=True, exist_ok=True)
    lock = RUN / 'controller_lock.json'
    if lock.exists():
        saved = json.loads(lock.read_text(encoding='utf-8'))
        try:
            prior = psutil.Process(saved['pid'])
            if abs(prior.create_time() - saved['process_create_time']) < 0.01:
                raise RuntimeError(f"监督进程仍运行：PID={saved['pid']}")
        except psutil.NoSuchProcess:
            pass
        # 旧锁保留为独立历史；活着的子进程也阻止重新调度。
        for event in (RUN / 'controller_events.jsonl').read_text(encoding='utf-8').splitlines() if (RUN / 'controller_events.jsonl').exists() else []:
            row = json.loads(event)
            if row.get('status') == 'started' and psutil.pid_exists(row['pid']):
                child = psutil.Process(row['pid'])
                if abs(child.create_time() - row['process_create_time']) < 0.01:
                    raise RuntimeError(f"旧子进程仍运行：PID={row['pid']}")
        lock.rename(lock.with_name(f'controller_lock_history_{os.getpid()}_{int(psutil.Process().create_time())}.json'))
    save_exclusive(lock, {'pid': os.getpid(), 'process_create_time': psutil.Process().create_time(), 'started_at': now()})
    code = {n: sha(ROOT / n) for n in SOURCES}
    pin_identity(RUN / 'controller_source_identity.json', code)
    snapshots = RUN / 'code_snapshot'
    snapshots.mkdir(exist_ok=True)
    for name, digest in code.items():
        target = snapshots / name
        if target.exists() and sha(target) != digest:
            raise RuntimeError('源码快照变化')
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write((ROOT / name).read_bytes())
    append_event(status='controller_started', pid=os.getpid(), source_sha256=code, preflight_only=args.preflight_only)
    for protocol, arch, stage in operations:
        out = out_for(protocol, arch)
        identity = make_identity(protocol, arch, args.pretrained_root)
        pin_identity(out / 'merged_input_identity.json', identity)
        markers = out / 'stage_markers'
        markers.mkdir(exist_ok=True)
        marker = markers / (stage + '.json')
        if marker.exists():
            prior = json.loads(marker.read_text(encoding='utf-8'))
            if prior['identity'] != identity or sha(prior['log']) != prior['log_sha256'] or any(sha(out / n) != h for n, h in prior['outputs'].items()):
                raise RuntimeError(f'{protocol}/{arch}/{stage}成功凭据不匹配')
            append_event(protocol=protocol, architecture=arch, stage=stage, status='verified_success_skipped')
            continue
        logs = out / 'stage_logs'
        logs.mkdir(exist_ok=True)
        if list(logs.glob(stage + '_attempt*.log')):
            raise RuntimeError(f'{protocol}/{arch}/{stage}存在未成功尝试；保留原日志，须先独立复盘而非盲重试')
        if shutil.disk_usage(RUN).free < 1024 ** 3:
            raise RuntimeError('磁盘剩余不足1GiB，停止在阶段边界')
        command = [str(args.python), str(ROOT / 'run_merged_pipeline.py'), stage, '--protocol', protocol,
            '--arch', arch, '--pretrained-root', str(args.pretrained_root), '--workers', str(0 if stage == 'smoke' else args.workers),
            '--device', 'cpu' if stage == 'smoke' else 'cuda']
        log = logs / (stage + '_attempt1.log')
        env = dict(os.environ, PYTHONPATH='E:/python/Lib/site-packages', PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
        with log.open('x', encoding='utf-8') as handle:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            created = psutil.Process(child.pid).create_time()
            append_event(protocol=protocol, architecture=arch, stage=stage, status='started', pid=child.pid,
                process_create_time=created, command=command, log=str(log))
            code_return = child.wait()
        append_event(protocol=protocol, architecture=arch, stage=stage, status='exited', pid=child.pid, exit_code=code_return)
        if code_return:
            raise RuntimeError(f'{protocol}/{arch}/{stage}退出{code_return}；查看{log}')
        evidence = {n: sha(out / n) for n in outputs(stage)}
        # 先算完输出字节身份，再核验实时源码和输入，最后提交同一凭据。
        if make_identity(protocol, arch, args.pretrained_root) != identity:
            raise RuntimeError('提交前输入或源码身份变化')
        log_digest = sha(log)
        def verify_stage():
            # 凭据序列化完成后再次核验活输入及输出，提交同一批字节。
            if make_identity(protocol, arch, args.pretrained_root) != identity or sha(log) != log_digest:
                raise RuntimeError('提交前身份或日志变化')
            if any(sha(out / name) != digest for name, digest in evidence.items()):
                raise RuntimeError('提交前输出变化')
        save_exclusive(marker, {'protocol': protocol, 'architecture': arch, 'stage': stage, 'identity': identity,
            'pid': child.pid, 'process_create_time': created, 'exit_code': code_return, 'log': str(log),
            'log_sha256': log_digest, 'outputs': evidence, 'completed_at': now()}, verify_stage)
    target = RUN / ('all_cpu_smokes_complete.json' if args.preflight_only else 'all_training_evaluation_complete.json')
    def verify_all():
        # 总完成标记核验全部阶段，不能只依据最后一个退出码。
        for protocol, arch, stage in operations:
            out = out_for(protocol, arch)
            saved = json.loads((out / 'stage_markers' / (stage + '.json')).read_text(encoding='utf-8'))
            if saved['identity'] != make_identity(protocol, arch, args.pretrained_root):
                raise RuntimeError('总提交前输入身份变化')
            if sha(saved['log']) != saved['log_sha256'] or any(sha(out / n) != h for n, h in saved['outputs'].items()):
                raise RuntimeError('总提交前阶段产物变化')
    save_exclusive(target, {'status': 'PASS', 'completed_at': now(), 'source_sha256': code,
        'CPU_benchmark_pending': True, 'independent_review_pending': True}, verify_all)
    append_event(status='all_requested_stages_exit0', preflight_only=args.preflight_only)


if __name__ == '__main__':
    main()
