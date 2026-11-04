"""记录本轮实际软件与代码身份，不改变训练进程或用户程序。"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import importlib.metadata


def main():
    root = Path(__file__).resolve().parent
    dest = root / 'runs/A_20261102_TPE'
    snapshot = dest / 'code_snapshot'
    snapshot.mkdir(exist_ok=True)
    files = {}
    for name in ['train_A.py', 'compare_resnets_A.py', 'run_A_budget.py', 'run_A_comparisons.py',
                 'benchmark_A_classifiers.py', 'postprocess_A_results.py', 'make_A_trained_gradcam.py']:
        data = (root / name).read_bytes()
        target = snapshot / name
        # 同名快照只允许保持相同字节，不覆盖未知历史代码。
        if target.exists():
            assert target.read_bytes() == data
        else:
            target.write_bytes(data)
        files[name] = hashlib.sha256(data).hexdigest()
    versions = {}
    for package in ['torch', 'torchvision', 'optuna', 'numpy', 'pillow', 'scikit-learn']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not_installed'
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True).stdout.strip()
    record = {'recorded_at': datetime.now().astimezone().isoformat(),
              'python': sys.executable, 'python_version': platform.python_version(),
              'platform': platform.platform(), 'processor': platform.processor(),
              'verified_cpu_name': 'Intel Core Ultra 9 285H', 'logical_processors': 16,
              'gpu': gpu, 'package_versions': versions, 'code_sha256': files,
              'scope': 'observed_current_run_environment_and_snapshot_not_a_claim_about_earlier_installations'}
    (dest / 'environment.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
