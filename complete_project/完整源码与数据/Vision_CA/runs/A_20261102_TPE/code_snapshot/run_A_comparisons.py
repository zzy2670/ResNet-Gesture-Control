"""主流程结束后按剩余时间运行34/50对比；不并行争用GPU。"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SHARED = ROOT / 'runs' / 'A_20261102_TPE'
ENTRY = ROOT / 'compare_resnets_A.py'
BASE_HASH = '93e03db74bf5218db2fc3bdf2553c605c2dded2d2b459cca46e8c42c904c6303'
COMPARE_HASH = '3bd9e96999917edf07a1738b0ab6eb8710405848cc2baa1691303e97ec944a3f'


def event(**fields):
    fields = dict(time=datetime.now().astimezone().isoformat(), **fields)
    with (SHARED / 'comparison_budget_events.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(fields, ensure_ascii=False) + '\n')
    print(json.dumps(fields, ensure_ascii=False), flush=True)


def run(arch, stage, extra, cutoff):
    # 仅监督本任务子进程树；截止后不把部分轮列作完整对比。
    assert hashlib.sha256(ENTRY.read_bytes()).hexdigest() == COMPARE_HASH
    assert hashlib.sha256((ROOT / 'train_A.py').read_bytes()).hexdigest() == BASE_HASH
    out = SHARED / ('compare_' + arch)
    end = datetime.fromisoformat(cutoff).timestamp()
    if time.time() >= end:
        raise TimeoutError('对比阶段启动前已到截止时间')
    command = [sys.executable, str(ENTRY), stage, '--arch', arch,
               '--split-root', str(ROOT / 'data/A_split_verified_20261102/P1_random_image'),
               '--shared-run', str(SHARED), '--out', str(out), '--deadline', cutoff, *extra]
    with (out / f'{stage}_console.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        event(architecture=arch, stage=stage, status='started', pid=child.pid, command=command)
        while child.poll() is None:
            if time.time() >= end:
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
                child.wait(timeout=30)
                event(architecture=arch, stage=stage, status='hard_deadline_terminated', exit_code=child.returncode)
                raise TimeoutError('对比子进程已截止')
            time.sleep(2)
    event(architecture=arch, stage=stage, status='exited', exit_code=child.returncode)
    if child.returncode:
        raise RuntimeError(f'{arch}/{stage}退出码{child.returncode}')


def main():
    # 主流程全部自然完成后才运行额外GPU模型；剩余预算不足则如实跳过。
    event(status='waiting_main_pipeline')
    while True:
        events = SHARED / 'budget_events.jsonl'
        rows = [json.loads(line) for line in events.read_text(encoding='utf-8').splitlines()] if events.exists() else []
        if any(row.get('status') == 'all_stages_exit0' for row in rows):
            break
        if datetime.now().astimezone() >= datetime.fromisoformat('2026-11-02T15:40:00+08:00'):
            event(status='optional_comparisons_skipped_insufficient_time')
            return
        time.sleep(5)
    for arch, training_end in [('resnet34', '2026-11-02T16:00:00+08:00'),
                               ('resnet50', '2026-11-02T16:33:00+08:00')]:
        for stage in ['features', 'search', 'finetune']:
            # 相同批量、搜索次数与轮数上限，具体完成轮数保留在日志中。
            extra = ['--batch-size', '512' if stage == 'search' else '128',
                     '--workers', '4', '--epochs', '18', '--trials', '24',
                     '--lr', '0.0001', '--patience', '4']
            run(arch, stage, extra, training_end)
        run(arch, 'calibrate', ['--batch-size', '32', '--workers', '4'], '2026-11-02T16:39:00+08:00')
        run(arch, 'evaluate', ['--batch-size', '32', '--workers', '4'], '2026-11-02T16:44:00+08:00')
    # 所有GPU训练退出后再顺序测速，减少模型间计时条件的差别。
    for arch in ['resnet34', 'resnet50']:
        for stage in ['latency', 'classifier-latency']:
            run(arch, stage, ['--device', 'cpu', '--batch-size', '1', '--workers', '0'],
                '2026-11-02T16:48:00+08:00')
    event(status='optional_comparisons_all_exit0')


if __name__ == '__main__':
    main()
