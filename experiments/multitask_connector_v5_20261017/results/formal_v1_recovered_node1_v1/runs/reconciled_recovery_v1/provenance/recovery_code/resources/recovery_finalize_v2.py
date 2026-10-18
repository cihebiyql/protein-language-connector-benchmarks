"""Wait for the preserved queue and retry, reconcile all45, explain M3, and publish on node1."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

SCOPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCOPE))
sys.path.insert(0, str(SCOPE / 'resources'))
from src import mt_case, mt_queue, mt_supervise as supervision
import node1_publish
import finish_and_export
import reconcile_recovery_v1
import recovery_summary_v1
import recovery_publication_policy_v1

STATE = SCOPE / 'private/recovery_finalize_v2'

def progress(stage, **extra):
    supervision.write_json(STATE / 'status.json', {'utc': datetime.now(timezone.utc).isoformat(), 'stage': stage, **extra})

def verify_config(path, *, verify_assets=True):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(SCOPE / 'private'): raise ValueError('Private finalizer configuration required')
    config = supervision.read_json(path)
    if config['schema'] != 'v5-recovery-finalization.v1': raise ValueError('Wrong finalizer contract')
    if config['scope_root'] != str(SCOPE) or config['maximum_distinct_owned_gpus'] != 4:
        raise ValueError('Scope and four-card authorization must remain explicit')
    for source, expected in config['source_pins'].items():
        if not Path(source).resolve(strict=True).is_relative_to(SCOPE) or mt_case.sha(source) != expected:
            raise ValueError('Pinned recovery implementation changed')
    if verify_assets:
        mt_case.verify_pins(supervision.read_json(SCOPE / 'configs/queue_formal_v1.json'))
    return config

def run(config_path):
    if socket.gethostname() != 'node1': raise RuntimeError('This finalization is node1-local')
    config = verify_config(config_path)
    STATE.mkdir(exist_ok=True)
    with mt_queue.lease(SCOPE / '.runtime_locks/recovery_finalize_v2.lock'):
        binding = {'config_sha256': mt_case.sha(config_path), 'identity': supervision.process_identity(os.getpid())}
        supervision.write_json(STATE / 'binding.json', binding)
        original = SCOPE / 'runs/formal_v1'
        retry_root = SCOPE / 'runs/recovery_retry_v1'
        retry = retry_root / 'cases/elm__M3__qwen3_4b_2507__seed17/attempt_002'
        deadline = time.monotonic() + config['wait_timeout_seconds']
        retired = SCOPE / 'private/recovery_v1_old_publisher_retired.json'
        while time.monotonic() < deadline:
            verify_config(config_path, verify_assets=False)
            if not retired.exists():
                progress('waiting_for_verified_publisher_handoff')
                time.sleep(10); continue
            old = supervision.read_json(SCOPE / 'private/node1_publication/formal_v1_node1_terminal_v1/launch.json')
            if finish_and_export.identity_alive(old['identity']):
                raise RuntimeError('Old publication owner must be retired before recovered publication')
            outcome = supervision.read_json(retry_root / 'recovery_status.json') if (retry_root / 'recovery_status.json').exists() else {}
            original_terminal = (original / 'complete.json').exists() and (original / 'analysis_exit.json').exists()
            retry_terminal = outcome.get('status') in ('accepted', 'failed', 'not_started_resource_timeout')
            if original_terminal and retry_terminal: break
            counts = supervision.read_json(original / 'queue_state.json')['counts']
            progress('waiting_for_existing_experiments', original_counts=counts, retry_status=outcome.get('status'))
            time.sleep(15)
        else:
            raise TimeoutError('Bounded finalization wait expired; original evidence remains untouched')
        verify_config(config_path)
        progress('building_independently_reaccepted_45_case_view')
        view = SCOPE / 'runs/reconciled_recovery_v1'
        reconciliation = reconcile_recovery_v1.build_view(SCOPE, original, retry, view)
        code = view / 'provenance/recovery_code'
        code.mkdir()
        copied = []
        for source, digest in config['source_pins'].items():
            source = Path(source)
            target = code / source.relative_to(SCOPE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if mt_case.sha(target) != digest: raise ValueError('Recovery execution source copy differs')
            copied.append({'relative': target.relative_to(view).as_posix(), 'sha256': digest})
        supervision.write_json(view / 'provenance/recovery_code_manifest.json', {'files': copied,
            'resource_entrypoint_additional_to_original_source_closure': True, 'maximum_distinct_owned_gpus': 4})
        plan_path = SCOPE / 'configs/queue_formal_v1.json'
        plan = supervision.read_json(plan_path)
        analysis = SCOPE / 'analysis/recovery_reconciled_v1'
        env = dict(os.environ, PYTHONPATH=str(SCOPE), CUDA_VISIBLE_DEVICES='')
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): env[name] = '2'
        progress('recomputing_all45_with_frozen_metrics')
        command = [plan['paths']['python'], '-m', 'src.mt_analyze', '--plan', str(plan_path),
            '--queue-root', str(view), '--data-root', plan['paths']['data_root'], '--output', str(analysis)]
        with (STATE / 'analysis_stdout.log').open('xb') as stdout, (STATE / 'analysis_stderr.log').open('xb') as stderr:
            result = subprocess.run(command, cwd=SCOPE, env=env, stdout=stdout, stderr=stderr, timeout=7200)
        supervision.write_json(view / 'analysis_exit.json', {'exit_code': result.returncode, 'command': command})
        if result.returncode: raise RuntimeError('Frozen independent analysis failed; no result relabelling')
        progress('summarizing_paired_multiseed_M3_comparisons')
        summary_directory = view / 'provenance/M3_STABILITY'
        recovery_summary_v1.write_summary(analysis / 'analysis.json', summary_directory)
        document = SCOPE / 'docs/RECOVERY_M3_ANALYSIS_v1.md'
        if document.exists(): raise FileExistsError('Prior interpretation must not be overwritten')
        shutil.copy2(summary_directory / 'REPORT.md', document)
        progress('publishing_recovered_results_from_node1', retry_accepted=reconciliation['retry_accepted'])
        with recovery_publication_policy_v1.policy():
            receipt = node1_publish.run(config['publisher_config'])
        supervision.write_json(STATE / 'PUBLICATION_RECEIPT.json', receipt)
        progress('published', commit=receipt['commit'], retry_accepted=reconciliation['retry_accepted'],
            expected_cases=45, local_computer_required=False)
        return receipt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.config)))
    except Exception as error:
        if STATE.is_dir(): progress('failed', error=repr(error), evidence_preserved=True)
        raise

if __name__ == '__main__':
    main()
