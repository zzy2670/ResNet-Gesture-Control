"""在本次授权的时间预算内串行占用GPU，保留完整阶段日志与退出码。"""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs' / 'A_20261102_TPE'
ENTRY = ROOT / 'train_A.py'
SPLIT = ROOT / 'data' / 'A_split_verified_20261102' / 'P1_random_image'
EXPECTED = '93e03db74bf5218db2fc3bdf2553c605c2dded2d2b459cca46e8c42c904c6303'


def event(**fields):
    # 预算监督不加载任何模型，只记录本次子进程。
    fields = dict(time=datetime.now().astimezone().isoformat(), **fields)
    with (OUT / 'budget_events.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(fields, ensure_ascii=False) + '\n')
    print(json.dumps(fields, ensure_ascii=False), flush=True)


def stage(name, extras, cutoff):
    # 每个阶段只启动一次；硬截止只结束该阶段所属的进程树。
    actual = hashlib.sha256(ENTRY.read_bytes()).hexdigest()
    if actual != EXPECTED:
        raise RuntimeError('训练入口已变化，停止自动调度以重新审查')
    end = datetime.fromisoformat(cutoff).timestamp()
    if time.time() >= end:
        raise TimeoutError(f'{name}启动前已到截止时间')
    command = [sys.executable, str(ENTRY), name, '--split-root', str(SPLIT),
               '--out', str(OUT), '--deadline', cutoff, *extras]
    with (OUT / f'{name}_console.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        event(stage=name, status='started', pid=child.pid, deadline=cutoff,
              command=command, code_sha256=actual)
        while child.poll() is None:
            if time.time() >= end:
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                               capture_output=True, check=False)
                child.wait(timeout=30)
                event(stage=name, status='hard_deadline_terminated', pid=child.pid,
                      exit_code=child.returncode)
                raise TimeoutError(f'{name}已达到硬截止，未宣称完整完成')
            time.sleep(2)
    event(stage=name, status='exited', pid=child.pid, exit_code=child.returncode)
    if child.returncode:
        raise RuntimeError(f'{name}退出码{child.returncode}，保留日志等待修复')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', choices=['finetune', 'detector', 'calibrate'], default='finetune')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # 分类微调先完成，检测训练最晚16:25退出，为校准和完整测试留余量。
    plan = [
        ('finetune', ['--epochs', '18', '--batch-size', '128', '--workers', '4',
                      '--lr', '0.0001', '--patience', '4'], '2026-11-02T15:05:00+08:00'),
        ('detector', ['--epochs', '16', '--batch-size', '24', '--workers', '4',
                      '--lr', '0.003'], '2026-11-02T16:25:00+08:00'),
        ('calibrate', ['--batch-size', '32', '--workers', '4'], '2026-11-02T16:35:00+08:00'),
        ('evaluate', ['--batch-size', '32', '--workers', '4'], '2026-11-02T16:43:00+08:00'),
        ('latency', ['--device', 'cpu', '--workers', '0', '--batch-size', '1'], '2026-11-02T16:48:00+08:00'),
    ]
    start = [item[0] for item in plan].index(args.start)
    for name, extra, cutoff in plan[start:]:
        stage(name, extra, cutoff)
    event(status='all_stages_exit0', training_hard_deadline='2026-11-02T16:25:00+08:00',
          user_deadline='2026-11-02T16:50:00+08:00')


if __name__ == '__main__':
    main()
