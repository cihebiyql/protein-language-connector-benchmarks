"""Version the waiting finalizer to add tested raw-ledger publication handling."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SCOPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCOPE))
sys.path.insert(0, str(SCOPE / 'resources'))
from src.mt_case import sha
from src.mt_supervise import read_json, write_json, process_identity, same_process

def main():
    config = read_json(SCOPE / 'private/recovery_finalize_v1_config.json')
    extra = ('resources/recovery_finalize_v2.py', 'resources/recovery_publication_policy_v1.py',
        'resources/prepare_launch_recovery_finalize_v2.py', 'tests/test_recovery_finalize_v1.py',
        'tests/test_recovery_finalize_v2.py', 'tests/test_recovery_publication_policy_v1.py')
    config['source_pins'].update({str(SCOPE / name): sha(SCOPE / name) for name in extra})
    config_path = SCOPE / 'private/recovery_finalize_v2_config.json'
    if config_path.exists(): raise FileExistsError('Prior v2 config must be preserved')
    write_json(config_path, config); config_path.chmod(0o600)
    root = SCOPE / 'private/recovery_finalize_v2'
    root.mkdir(exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(SCOPE), CUDA_VISIBLE_DEVICES='')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): env[name] = '2'
    command = [sys.executable, str(SCOPE / 'resources/recovery_finalize_v2.py'), '--config', str(config_path)]
    with (root / 'stdout.log').open('xb') as stdout, (root / 'stderr.log').open('xb') as stderr:
        child = subprocess.Popen(command, cwd=SCOPE, env=env, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, start_new_session=True)
    identity = process_identity(child.pid)
    receipt = {'identity': identity, 'config_sha256': sha(config_path), 'node1_only': True,
        'scheduled_monitor_created': False, 'reason': 'Preserve raw provenance ledger privately and keep normalized public history'}
    write_json(root / 'launch.json', receipt)
    until = time.monotonic() + 90
    while not (root / 'binding.json').exists():
        if not same_process(identity): raise RuntimeError('New finalizer exited; old waiting process retained')
        if time.monotonic() > until: raise TimeoutError('New finalizer not ready; old waiting process retained')
        time.sleep(.2)
    current = Path('/proc') / str(identity['pid'])
    fields = (current / 'stat').read_text().rsplit(')', 1)[1].split()
    assert int(fields[1]) == os.getpid() and identity['pid'] == identity['pgid'] == identity['sid']
    assert os.readlink(current / 'fd/0') == '/dev/null'
    assert (root / 'stderr.log').stat().st_size == 0
    old_root = SCOPE / 'private/recovery_finalize_v1'
    old = read_json(old_root / 'launch.json')['identity']
    assert read_json(old_root / 'status.json')['stage'] == 'waiting_for_existing_experiments'
    assert not (SCOPE / 'runs/reconciled_recovery_v1').exists()
    if same_process(old):
        raw = (Path('/proc') / str(old['pid']) / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        assert [x.decode() for x in raw][1:] == [str(SCOPE / 'resources/recovery_finalize_v1.py'),
            '--config', str(SCOPE / 'private/recovery_finalize_v1_config.json')]
        os.kill(old['pid'], signal.SIGTERM)
    receipt['superseded_waiting_finalizer_identity'] = old
    receipt['old_sources_config_logs_preserved'] = True
    write_json(SCOPE / 'private/recovery_finalize_v1_retired.json', receipt)
    print(json.dumps(receipt))

if __name__ == '__main__': main()
