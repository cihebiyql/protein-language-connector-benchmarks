"""Create pinned private recovery/publication configs and launch one detached finalizer."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCOPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCOPE))
from src.mt_case import sha
from src.mt_supervise import read_json, write_json, process_identity

def main():
    source_names = ('resources/recovery_worker_v1.py', 'resources/recovery_retry_manager_v1.py',
        'resources/launch_recovery_retry_v1.py', 'resources/reconcile_recovery_v1.py',
        'resources/recovery_summary_v1.py', 'resources/recovery_finalize_v1.py',
        'resources/prepare_launch_recovery_finalize_v1.py', 'tests/test_recovery_worker_v1.py',
        'tests/test_reconcile_recovery_v1.py', 'tests/test_recovery_summary_v1.py')
    pins = {str(SCOPE / name): sha(SCOPE / name) for name in source_names}
    base = read_json(SCOPE / 'private/node1_publish_config.json')
    base.update(queue_root=str(SCOPE / 'runs/reconciled_recovery_v1'), run_id='formal_v1_recovered_node1_v1')
    publisher_config = SCOPE / 'private/recovery_v1_publisher_config.json'
    if publisher_config.exists(): raise FileExistsError('Recovery publication config already retained')
    write_json(publisher_config, base); publisher_config.chmod(0o600)
    config = {'schema': 'v5-recovery-finalization.v1', 'scope_root': str(SCOPE),
        'maximum_distinct_owned_gpus': 4, 'source_pins': pins,
        'wait_timeout_seconds': 86400, 'publisher_config': str(publisher_config)}
    config_path = SCOPE / 'private/recovery_finalize_v1_config.json'
    if config_path.exists(): raise FileExistsError('Recovery finalizer config already retained')
    write_json(config_path, config); config_path.chmod(0o600)
    root = SCOPE / 'private/recovery_finalize_v1'
    root.mkdir(exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(SCOPE), CUDA_VISIBLE_DEVICES='')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): env[name] = '2'
    command = [sys.executable, str(SCOPE / 'resources/recovery_finalize_v1.py'), '--config', str(config_path)]
    with (root / 'stdout.log').open('xb') as stdout, (root / 'stderr.log').open('xb') as stderr:
        child = subprocess.Popen(command, cwd=SCOPE, env=env, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, start_new_session=True)
    receipt = {'identity': process_identity(child.pid), 'config_sha256': sha(config_path),
        'launched_unix': time.time(), 'node1_only': True, 'scheduled_monitor_created': False}
    write_json(root / 'launch.json', receipt)
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
