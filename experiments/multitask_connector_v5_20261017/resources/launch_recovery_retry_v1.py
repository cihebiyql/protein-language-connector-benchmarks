"""Launch one detached loading-OOM retry manager, without a scheduler."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCOPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCOPE))
from src.mt_supervise import process_identity, write_json

def main():
    root = SCOPE / 'private/recovery_retry_v1_launch'
    root.mkdir(exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(SCOPE), CUDA_VISIBLE_DEVICES='')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): env[key] = '2'
    command = [sys.executable, str(SCOPE / 'resources/recovery_retry_manager_v1.py')]
    with (root / 'stdout.log').open('xb') as stdout, (root / 'stderr.log').open('xb') as stderr:
        child = subprocess.Popen(command, cwd=SCOPE, env=env, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, start_new_session=True)
    receipt = {'identity': process_identity(child.pid), 'launched_unix': time.time(),
        'scheduled_monitor': False, 'selected_fourth_gpu': 6, 'existing_gpu_devices': [3, 4, 5]}
    write_json(root / 'launch.json', receipt)
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
