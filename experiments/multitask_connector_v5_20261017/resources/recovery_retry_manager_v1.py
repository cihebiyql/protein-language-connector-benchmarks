"""Run the one authorized loading-OOM recovery on a fourth GPU, preserving all evidence."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

SCOPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCOPE))
sys.path.insert(0, str(SCOPE / 'resources'))
from src import mt_case, mt_queue, mt_supervise as supervision
import recovery_worker_v1 as adapter

TARGET = adapter.TARGET
ROOT = SCOPE / 'runs/recovery_retry_v1'
ORIGINAL = SCOPE / 'runs/formal_v1'
GPU = 6

def status(value):
    supervision.write_json(ROOT / 'recovery_status.json', {'at_unix': time.time(), 'case_id': TARGET, **value})

def loading_oom_only(attempt):
    failure = supervision.read_json(attempt / 'qualification/failed.json')
    progress = attempt / 'qualification/progress.jsonl'
    updates = [json.loads(line) for line in progress.read_text().splitlines()] if progress.exists() else []
    return (failure.get('exception_type') == 'OutOfMemoryError'
        and not any(row.get('event') == 'optimizer_step' for row in updates)
        and not (attempt / 'formal_process.json').exists())

def run():
    if socket.gethostname() != 'node1': raise RuntimeError('Recovery is restricted to node1')
    plan_path = SCOPE / 'configs/queue_formal_v1.json'
    plan = supervision.read_json(plan_path)
    mt_case.verify_pins(plan)
    cases = [case for case in plan['cases'] if case['case_id'] == TARGET]
    if len(cases) != 1: raise ValueError('Unique frozen recovery case required')
    case = cases[0]
    old = ORIGINAL / 'cases' / TARGET / 'attempt_001'
    if not loading_oom_only(old): raise ValueError('Recovery requires the recorded pre-update loading OOM')
    if supervision.group_members(supervision.read_json(old / 'supervision/started.json')['identity']):
        raise RuntimeError('Original failed process group still has live members')
    ROOT.mkdir(exist_ok=False)
    with mt_queue.lease(SCOPE / '.runtime_locks/recovery_retry_v1.lock'):
        supervision.write_json(ROOT / 'controller.json', {'identity': supervision.process_identity(os.getpid()),
            'original_plan_sha256': mt_case.sha(plan_path), 'adapter_sha256': mt_case.sha(adapter.ADAPTER)})
        policy = {'schema': 'v5-recovery-resource-policy.v1', 'scope_root': str(SCOPE),
            'original_plan_path': str(plan_path), 'original_plan_sha256': mt_case.sha(plan_path),
            'case_id': TARGET, 'original_gpu_devices': [3, 4, 5], 'authorized_gpu_devices': [3, 4, 5, 6],
            'maximum_distinct_owned_gpus': 4, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'reason': 'User authorized a fourth GPU and recovery of a pre-training loading OOM; no score-based retry',
            'original_failure_sha256': mt_case.sha(old / 'qualification/failed.json'),
            'scientific_source_config_data_and_evaluation_changed': False}
        policy_path = SCOPE / 'configs/recovery_resource_policy_v1.json'
        if policy_path.exists(): raise FileExistsError('Existing recovery policy must be preserved')
        supervision.write_json(policy_path, policy)
        policy_sha = mt_case.sha(policy_path)
        budget = plan['resource_policy']
        until = time.monotonic() + plan.get('resource_wait_timeout_seconds', 86400)
        supervision.PHYSICAL_GPUS = adapter.AUTHORIZED
        lane = None
        while time.monotonic() < until:
            mt_case.verify_pins(plan)
            try:
                lane = (SCOPE / '.runtime_locks/lane_6.lock').open('a+')
                import fcntl
                fcntl.flock(lane.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lane.close(); lane = None
                status({'status': 'resource_wait', 'reasons': ['authorized_fourth_lane_owned']})
                time.sleep(10); continue
            observation = supervision.probe(root=ROOT, gpu=GPU, rss_limit_bytes=budget['rss_limit_bytes'],
                gpu_memory_budget_bytes=budget['gpu_memory_budget_bytes'][case['backbone']],
                expected_host=budget['expected_host'], storage_reserve_bytes=budget['storage_reserve_bytes'])
            if observation['ready']: break
            lane.close(); lane = None
            status({'status': 'resource_wait', 'reasons': observation['reasons'], 'observation': observation})
            time.sleep(10)
        if lane is None:
            status({'status': 'not_started_resource_timeout'})
            return 1
        try:
            attempt = ROOT / 'cases' / TARGET / 'attempt_002'
            attempt.mkdir(parents=True, exist_ok=False)
            spec_path = attempt / 'spec.json'
            spec = mt_queue.build_spec(plan_path, plan, case, attempt, GPU)
            spec.update(recovery_policy_path=str(policy_path), recovery_policy_sha256=policy_sha,
                recovery_adapter_sha256=mt_case.sha(adapter.ADAPTER),
                command=[plan['paths']['python'], str(adapter.ADAPTER), '--mode', 'worker', '--spec', str(spec_path)])
            supervision.write_json(spec_path, spec)
            adapter.validate(spec_path)
            env = dict(os.environ, PYTHONPATH=str(SCOPE), CUDA_VISIBLE_DEVICES='',
                CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false')
            for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): env[key] = '2'
            command = [plan['paths']['python'], str(adapter.ADAPTER), '--mode', 'supervise',
                '--spec', str(spec_path), '--output', str(attempt / 'supervision')]
            with (attempt / 'manager_stdout.log').open('xb') as stdout, (attempt / 'manager_stderr.log').open('xb') as stderr:
                child = subprocess.Popen(command, cwd=SCOPE, env=env, stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr, start_new_session=True)
            child_identity = supervision.process_identity(child.pid)
            supervision.write_json(attempt / 'supervisor_owner.json', {'identity': child_identity})
            while child.poll() is None:
                status({'status': 'running', 'attempt': str(attempt), 'gpu': GPU,
                    'supervisor_identity': child_identity, 'maximum_distinct_owned_gpus': 4})
                time.sleep(5)
            exit_code = child.wait()
            try:
                binding = mt_case.accepted_attempt(plan, attempt, case)
                adapter.validate(spec_path)
                status({'status': 'accepted', 'attempt': str(attempt), 'binding': binding,
                    'supervisor_exit_code': exit_code, 'gpu': GPU, 'old_failure_preserved': True})
                return 0
            except Exception as error:
                status({'status': 'failed', 'attempt': str(attempt), 'error': repr(error),
                    'supervisor_exit_code': exit_code, 'gpu': GPU, 'old_failure_preserved': True})
                return 1
        finally:
            lane.close()

if __name__ == '__main__':
    raise SystemExit(run())
