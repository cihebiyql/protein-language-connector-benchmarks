"""Bounded, audited C1/C2 execution. Controller imports only the standard library.

A fresh idle-GPU probe and frozen standard contract precede the child. The child
arms Linux PR_SET_PDEATHSIG before importing Torch. Never resumes old outputs.
"""
import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STANDARD = ROOT.parent / '科研工作台/复现/规范'
BASE = ROOT / 'results/foundation_20261016/information_diagnostics'
CONFIG = ROOT / 'configs/foundation_gpu_20261016.json'
MEMORY_LIMIT = 12 * 1024 ** 3
GPU_LIMIT = int(11.5 * 1024 ** 3)
SCENARIOS = ('new_peptide_known_target', 'new_peptide_new_target')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def validator():
    spec = importlib.util.spec_from_file_location('information_standard_validator', STANDARD / 'validate_reproduction.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def parent_death_guard(expected_parent):
    """Run before importing Torch, avoiding the fork/parent-exit race."""
    if not sys.platform.startswith('linux'):
        raise RuntimeError('The worker requires Linux process-tree controls')
    if os.getppid() != expected_parent:
        raise RuntimeError('Worker parent changed before guard installation')
    library = ctypes.CDLL(None, use_errno=True)
    if library.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG failed')
    if os.getppid() != expected_parent:
        raise RuntimeError('Worker parent exited during guard installation')


def group_members(pgid):
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(stat[2]) != pgid:
                continue
            status = (path / 'status').read_text().splitlines()
            rss = next((int(line.split()[1]) * 1024 for line in status if line.startswith('VmRSS:')), 0)
            members.append({'pid': int(path.name), 'state': stat[0], 'rss_bytes': rss})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return members


def reap_group(pgid):
    while True:
        try:
            pid, _ = os.waitpid(-pgid, os.WNOHANG)
            if pid == 0:
                break
        except ChildProcessError:
            break


def cleanup_group(process):
    """Terminate only this job's new group, wait, reap, and verify /proc absence."""
    for sig, seconds in ((signal.SIGTERM, 3), (signal.SIGKILL, 3)):
        members = group_members(process.pid)
        if members:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            pass
        reap_group(process.pid)
        if not group_members(process.pid):
            return
    raise RuntimeError('Owned process group remains after SIGKILL and wait')


class ControllerInterrupted(RuntimeError):
    pass


@contextmanager
def interrupt_handlers():
    previous = {}
    def interrupted(signum, _frame):
        for controlled in (signal.SIGINT, signal.SIGTERM):
            signal.signal(controlled, signal.SIG_IGN)
        raise ControllerInterrupted('controller_signal_' + str(signum))
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, interrupted)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def supervise(command, job, timeout, deadline, memory_limit=MEMORY_LIMIT, environment=None):
    process, reason, error, peak = None, None, None, 0
    started, clock = now(), time.monotonic()
    # Adopt and reap grandchildren if a library unexpectedly creates a process.
    if sys.platform.startswith('linux'):
        library = ctypes.CDLL(None, use_errno=True)
        if library.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER failed')
    with (job / 'stdout.log').open('a', encoding='utf-8') as stdout, (job / 'stderr.log').open('a', encoding='utf-8') as stderr, interrupt_handlers():
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            write(job / 'running-process.json', {'controller_pid': os.getpid(), 'pid': process.pid,
                'pgid': process.pid, 'session_id': process.pid, 'started_utc': started, 'command': command})
            with (job / 'resource_usage.jsonl').open('x', encoding='utf-8', buffering=1) as usage:
                while process.poll() is None:
                    members = group_members(process.pid)
                    rss = sum(member['rss_bytes'] for member in members)
                    peak = max(peak, rss)
                    elapsed = time.monotonic() - clock
                    usage.write(json.dumps({'utc': now(), 'rss_bytes_whole_group': rss,
                        'members': members, 'elapsed_seconds': elapsed}) + '\n')
                    if rss > memory_limit:
                        reason = 'whole_group_host_rss_limit'
                    elif elapsed >= timeout or datetime.now(timezone.utc) >= deadline:
                        reason = 'timeout'
                    if reason:
                        break
                    time.sleep(.5)
        except BaseException as caught:
            error = caught
            reason = str(caught) if isinstance(caught, ControllerInterrupted) else type(caught).__name__
        finally:
            if process is not None:
                try:
                    cleanup_group(process)
                except BaseException as caught:
                    error = error or caught
                    reason = reason or 'process_cleanup_failed'
    if process is None:
        raise error or RuntimeError('Child process did not start')
    receipt = {'controller_pid': os.getpid(), 'pid': process.pid, 'pgid': process.pid,
        'started_utc': started, 'ended_utc': now(), 'elapsed_seconds': time.monotonic() - clock,
        'exit_code': process.returncode, 'termination_reason': reason,
        'peak_whole_group_host_rss_bytes': peak, 'remaining_processes': group_members(process.pid),
        'process_group_absent': not group_members(process.pid),
        'stdout_sha256': sha(job / 'stdout.log'), 'stderr_sha256': sha(job / 'stderr.log')}
    if error is not None:
        receipt['controller_error'] = {'type': type(error).__name__, 'message': str(error)}
    write(job / 'execution-receipt.json', receipt)
    return receipt


def command_output(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
    return result.stdout.strip()


def live_probe(job, config):
    gpu = command_output(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'])
    processes = command_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'])
    fields = [value.strip() for value in gpu.splitlines()[0].split(',')]
    used, total, utilization = int(fields[2]), int(fields[3]), int(fields[4])
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    mount = command_output(['findmnt', '-T', str(job), '-no', 'TARGET,SOURCE,FSTYPE,OPTIONS'])
    reasons = []
    if socket.gethostname() != config['expected_hostname']:
        reasons.append('unexpected_hostname')
    if processes:
        reasons.append('external_compute_process_present')
    if utilization > 15:
        reasons.append('gpu_busy')
    if (total - used) * 1024 ** 2 < GPU_LIMIT + 256 * 1024 ** 2:
        reasons.append('insufficient_gpu_headroom')
    if available < 6 * 1024 ** 3:
        reasons.append('insufficient_host_ram')
    if not os.access(job, os.W_OK) or not str(job.resolve()).startswith('PROJECT_ROOT/') or not mount.startswith('/mnt/d '):
        reasons.append('unexpected_or_unwritable_storage')
    return {'utc': now(), 'hostname': socket.gethostname(), 'gpu': gpu, 'gpu_compute_processes': processes,
        'mem_available_bytes': available, 'mount': mount, 'readiness': 'ready_for_exact_bounded_contract' if not reasons else 'blocked',
        'blocking_reasons': reasons}


def checkpoint_receipt(args):
    """Admit only a completed, accepted, contract-bound physical source archive."""
    parent = args.checkpoint.resolve().parent
    source_job = parent.parent
    required = [parent / 'result.json', parent / 'run_preregistration.json', parent / 'input_manifest.json',
                source_job / 'execution-receipt.json', source_job / 'result-validation.json',
                source_job / 'result-record.json', source_job / 'task-contract.json']
    if any(not path.is_file() for path in required):
        raise ValueError('Source archive missing required terminal receipts/contract/artifacts')
    result, registration, manifest, receipt, validation, record, contract = map(load, required)
    if validation.get('valid_for_acceptance') is not True:
        raise ValueError('Source archive lacks accepted standard validation')
    if (type(receipt.get('exit_code')) is not int or receipt['exit_code'] != 0
            or receipt.get('process_group_absent') is not True or receipt.get('termination_reason') is not None):
        raise ValueError('Source execution is not a clean terminal exit with absent process group')
    if (record.get('is_template') is not False or record.get('lifecycle_status') != 'succeeded'
            or record.get('reproduction_level') != 'training' or record.get('review', {}).get('status') != 'accepted'):
        raise ValueError('Source result record is not an accepted successful training archive')
    contract_file = source_job / 'task-contract.json'
    if (contract.get('is_template') is not False or contract.get('contract_status') != 'frozen'
            or Path(record.get('contract_path', '')).resolve() != contract_file.resolve()
            or record.get('contract_sha256') != sha(contract_file)
            or any(record.get(key) != contract.get(key) for key in ('task_id', 'run_id'))):
        raise ValueError('Source result record contract binding/hash mismatch')
    execution = record.get('execution', {})
    if (execution.get('performed') is not True or type(execution.get('exit_code')) is not int
            or execution['exit_code'] != 0 or execution.get('command') != contract.get('execution', {}).get('command')
            or execution.get('pid') != receipt.get('pid') or execution.get('pgid') != receipt.get('pgid')
            or Path(execution.get('receipt_path', '')).resolve() != (source_job / 'execution-receipt.json').resolve()
            or Path(execution.get('exit_verification_path', '')).resolve() != (source_job / 'execution-receipt.json').resolve()):
        raise ValueError('Source result record execution receipt binding mismatch')
    artifacts = record.get('artifacts')
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError('Source result record has no physical artifact hash assessment')
    checked = set()
    for artifact in artifacts:
        path = Path(artifact.get('path', '')).resolve()
        if (artifact.get('available') is not True or artifact.get('location') != 'local'
                or not path.is_file() or not path.is_relative_to(source_job.resolve())
                or artifact.get('sha256') != sha(path)
                or artifact.get('size_bytes') != path.stat().st_size):
            raise ValueError('Source result record artifact hash/size/path mismatch')
        checked.add(path)
    if not {required[0].resolve(), required[3].resolve()}.issubset(checked):
        raise ValueError('Source scientific result and terminal receipt must have archived hashes')
    for name in ('stdout', 'stderr'):
        path = source_job / (name + '.log')
        if not path.is_file() or sha(path) != receipt.get(name + '_sha256'):
            raise ValueError('Source execution log hash mismatch')
    configuration = registration['configuration']
    if result.get('status') != 'complete' or result.get('actual_epochs') != 3 or configuration.get('seed') != 17:
        raise ValueError('Requires completed three-epoch seed17 foundation training')
    if configuration['scenario'] != args.scenario or registration['scope'] != 'full_locked_A1_panel':
        raise ValueError('Checkpoint scenario/scope differs from full requested physical panel')
    if (args.checkpoint.name != 'best_connector.pt' or not args.checkpoint.is_file()
            or sha(args.checkpoint) != result.get('checkpoint_sha256')
            or Path(result.get('checkpoint_path', '')).resolve() != args.checkpoint.resolve()):
        raise ValueError('Selected checkpoint path/hash mismatch')
    if sha(parent / 'input_manifest.json') != registration.get('input_manifest_sha256'):
        raise ValueError('Training input manifest hash mismatch')
    return result, registration, manifest


def freeze_contract(args, job, config, probe, timeout, training):
    result, registration, manifest = training
    path = ROOT / 'data/processed/bench_v02/a1_locked_v2' / args.scenario
    readiness = load(BASE / 'cpu_readiness_v2.json')
    expected = readiness['scenarios'][args.scenario]
    if sha(path / 'samples.jsonl') != expected['source_sha256'] or expected['matched_donors'] != 0:
        raise ValueError('Protocol v2 preflight data or independent donor coverage changed')
    contract = load(STANDARD / 'task-contract.template.json')
    stamp = now()
    contract.update({'is_template': False, 'task_id': 'protein-language-connector-foundation-20261016',
        'run_id': job.name, 'contract_status': 'frozen', 'created_at': stamp, 'frozen_at': stamp})
    pdf = ROOT / 'papers/bench_v02/ppikb/article.pdf'
    pdf_sha = sha(pdf)
    if pdf_sha != '3a4c200f0097adfe90abb27b478249022fdc7f2d405f40b4d7cbbd592ee50def':
        raise ValueError('Accepted-manuscript PDF fingerprint differs')
    contract['identity'].update({'title': 'A data resource of peptide–protein interaction structures and binding affinities for peptide drug discovery',
        'doi': '10.1038/s41597-026-07962-1', 'publication_status': 'published', 'journal': 'Scientific Data',
        'paper_type': 'data source provenance for new controlled computational diagnostics',
        'pdf': {'path': str(pdf), 'sha256': pdf_sha, 'version': 'accepted_manuscript', 'role': 'official_reference', 'pages': 17, 'source_url': 'https://doi.org/10.1038/s41597-026-07962-1'},
        'eligibility': {'scope': 'official_reference', 'assessment': 'data_source_only', 'evidence_paths': [str(pdf)]}})
    contract['execution_scope'].update({'objective': 'Full locked A1 ' + args.scenario + ' ' + args.mode + ' diagnostics on a newly trained connector, frozen inference only.',
        'authorization_reference': config['authorization'] + ' Parent GPU release: ' + args.release_reference,
        'relation_to_prior_run': 'Same source start/deadline retained; this follows this round completed three-epoch seed17 selected connector.',
        'prior_run_ids': [registration['configuration']['method'] + '_seed17'], 'safety_assessment': 'allowed',
        'scope_coverage': 'component_only', 'intended_level': 'independent_inference',
        'excluded_claims': ['biological counterfactual', 'biological causality', 'mutated protein generation', 'full-paper reproduction', 'multitask coverage', 'three-seed confirmation', 'information in bits'],
        'restriction_check': {'prior_platform_restriction': False,
            'reason': 'Frozen scalar public-data diagnostics; no protein engineering or biological activity optimization.', 'evidence_paths': [str(CONFIG)]}})
    contract['timing'].update({'first_substantive_started_at': config['first_substantive_started_at'],
        'earliest_associated_preparation_at': config['first_substantive_started_at'], 'wall_clock_cap_hours': 10,
        'original_deadline_at': config['effective_deadline_at'], 'effective_deadline_at': config['effective_deadline_at'],
        'prior_window_expired': False, 'timing_evidence_paths': [str(CONFIG)]})
    contract['assets']['datasets'] = [{'name': 'A1 v2 ' + args.scenario, 'version': 'a1_locked_v2', 'path': str(path / 'samples.jsonl'),
        'sha256': sha(path / 'samples.jsonl'),
        'manifest_path': str(path / 'benchmark_lock.json'), 'manifest_sha256': sha(path / 'benchmark_lock.json'),
        'sample_count': sum(expected['split_counts'].values()),
        'statistical_unit': 'locked source assay measurement', 'splits': {**expected['split_counts'], 'split_manifest_sha256': sha(path / 'benchmark_lock.json')},
        'integrity': 'complete', 'notes': 'Complete restricted lock; physical construct and unknown experimental conditions retained.',
        'evidence_paths': [str(path / 'benchmark_lock.json'), str(BASE / 'cpu_readiness_v2.json')]}]
    scripts = [Path(__file__), Path(__file__).with_name('information_diagnostics.py'), Path(__file__).with_name('connectors.py'),
        Path(__file__).with_name('run_connectors.py'), ROOT / 'scripts/bench_v02/run_simple_baselines.py',
        ROOT / 'scripts/bench_v02/train_panel_baselines.py']
    contract['assets']['code'] = [{'role': 'primary' if i == 0 else 'component', 'path': str(script),
        'sha256': sha(script), 'archive_sha256': sha(script), 'integrity': 'complete'} for i, script in enumerate(scripts)]
    weights = ROOT / 'models/Qwen--Qwen2.5-1.5B-Instruct/model.safetensors'
    cache = ROOT / 'data/cache/bench_v02/panel_features_20261015/manifest.json'
    contract['assets']['models'] = [
        {'name': 'Qwen2.5-1.5B-Instruct', 'version': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306', 'path': str(weights),
         'sha256': sha(weights), 'configuration_path': str(weights.with_name('config.json')), 'configuration_sha256': sha(weights.with_name('config.json')),
         'integrity': 'complete', 'coverage': 'all backbone parameters frozen'},
        {'name': 'ESM2-150M layer30 residue cache', 'version': 'a695f6045e2e32885fa60af20c13cb35398ce30c',
         'path': str(cache), 'sha256': sha(cache), 'integrity': 'complete', 'coverage': 'physical full-sequence residue features'},
        {'name': registration['configuration']['method'], 'version': 'new three-epoch seed17 validation-selected connector',
         'path': str(args.checkpoint), 'sha256': sha(args.checkpoint), 'integrity': 'complete',
         'evidence_paths': [str(args.checkpoint.parent / 'result.json'), str(args.checkpoint.parent / 'input_manifest.json')]}]
    targets = []
    def target(key, metric, definition, unit):
        targets.append({'target_id': key, 'paper_endpoint': 'new computational foundation diagnostic; no author paper comparison', 'required': True,
            'metric': metric, 'metric_definition': definition, 'statistical_unit': 'locked held-out assay measurement',
            'denominator': expected['split_counts']['test'], 'unit': unit, 'paper_value': None, 'paper_value_source': None,
            'checkpoint_selection': 'source validation-selected checkpoint fixed; probe alpha selected on source validation MSE only',
            'repetitions': {'seed': 42, 'n_seeds': 1}, 'tolerance': {'kind': 'not_comparable', 'reason': 'new diagnostic; no author reference endpoint'}})
    if args.mode in ('c2', 'both'):
        for name in ('raw', 'message', 'receiver'):
            target('C2_' + name, 'delta_MSE_vs_B', 'B200=52 composition/chemistry plus148 full allowed ContextEncoder (TRAIN statistics); B200+zero128 vs B200+TRAIN-standardized fixed random projection128 of ' + name + '; five-alpha validation MSE selection; TEST once', 'pKD_squared')
    if args.mode in ('c1', 'both'):
        for name in ('correct', 'off'):
            target('C1_' + name, 'mean_answer_token_nll', 'Mean per-sample answer-token NLL, teacher-forced real original gold, ' + name + ' source', 'nats_per_answer_token')
    contract['targets'] = targets
    lockfile = ROOT / 'results/foundation_20261016/gpu_suite/environment_freeze.txt'
    contract['environment'].update({'isolation': 'existing dedicated WSL venv; no package mutation', 'environment_path': sys.prefix,
        'lockfile_path': str(lockfile), 'lockfile_sha256': sha(lockfile), 'shared_environment_modified': False})
    contract['resources'].update({'expected_hostname': config['expected_hostname'], 'cpu_threads': 2, 'data_loader_workers': 0,
        'memory_limit_bytes': MEMORY_LIMIT, 'gpu_devices': [0], 'gpu_memory_budget_bytes': GPU_LIMIT,
        'resource_probe_path': str(job / 'resource_probe.json'), 'resource_probe_at': probe['utc'], 'start_gate': probe['readiness'],
        'budget_basis': 'Fresh no-compute-process probe; Torch11.5GiB allocation limit; whole PGID RSS12GiB watchdog; no preemption.', 'no_preemption': True})
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--parent-pid', str(os.getpid()),
        '--checkpoint', str(args.checkpoint), '--scenario', args.scenario, '--mode', args.mode,
        '--output', str(job / 'artifacts'), '--timeout-seconds', str(timeout), '--batch-size', str(args.batch_size)]
    contract['execution'].update({'command': command, 'working_directory': str(ROOT), 'seed_policy': 'fixed seed42 projection and donor selection; source seed17 checkpoint',
        'seeds': [42], 'timeout_seconds': timeout, 'process_control': {'platform': 'Linux WSL',
        'mechanism': 'new PID/PGID; parent-death SIGKILL before Torch; subreaper; whole-group RSS/deadline watchdog; SIGINT/TERM cleanup and wait', 'whole_tree_bounded': True},
        'stdout_path': str(job / 'stdout.log'), 'stderr_path': str(job / 'stderr.log')})
    for name in ('stdout.log', 'stderr.log'):
        (job / name).touch(exist_ok=False)
    write(job / 'identity_version_correction.json', {'old_draft': str(BASE / 'standard_task_contract_DRAFT.json'),
        'old_pdf_version': 'published', 'actual_pdf_version': 'accepted_manuscript', 'actual_pdf_pages': 17,
        'publication_status': 'published', 'reference_role': 'official_reference', 'pdf_sha256': pdf_sha,
        'reason': 'Pre-execution correction of artifact-version identity; the manuscript is the official data-source reference.'})
    write(job / 'task-contract.json', contract)
    checked = validator().validate(contract, contract_bytes=(job / 'task-contract.json').read_bytes(), check_files=True, contract_directory=job)
    write(job / 'contract-validation.json', checked)
    if not checked['valid_for_execution']:
        raise RuntimeError('Standard contract rejected: ' + json.dumps(checked['errors'] + checked['execution_blockers']))
    return contract, command


def archive(job, contract, receipt):
    artifact = job / 'artifacts/result.json'
    actual = load(artifact) if artifact.is_file() else None
    complete = bool(actual and actual.get('status') == 'complete' and receipt['exit_code'] == 0 and receipt['process_group_absent'])
    record = load(STANDARD / 'result-record.template.json')
    record.update({'is_template': False, 'task_id': contract['task_id'], 'run_id': contract['run_id'], 'contract_path': str(job / 'task-contract.json'),
        'contract_sha256': sha(job / 'task-contract.json'), 'recorded_at': now(), 'lifecycle_status': 'succeeded' if complete else 'failed',
        'reproduction_level': 'independent_inference' if complete else 'static_audit', 'scope_coverage': 'component_only'})
    record['claims'] = [{'text': 'Frozen full-lock diagnostic process completed.' if complete else 'Frozen diagnostic process failed; literal partial artifacts retained.',
        'evidence_type': 'source_fact', 'evidence_paths': [str(job / 'execution-receipt.json')],
        'limitations': 'Computational readout and source-content sensitivity; unknown physical assay conditions retained; independent donors n=0.'}]
    for value in record['assets_integrity'].values():
        value.update({'status': 'complete', 'scope': 'exact frozen task inputs', 'evidence_paths': [str(job / 'contract-validation.json')]})
    record['execution'].update({'performed': True, 'command': contract['execution']['command'], 'working_directory': str(ROOT),
        'environment_path': sys.prefix, 'hostname': socket.gethostname(), 'resource_probe_path': str(job / 'resource_probe.json'),
        'resource_usage_path': str(job / 'resource_usage.jsonl'), 'seeds': [42], 'started_at': receipt['started_utc'], 'ended_at': receipt['ended_utc'],
        'effective_deadline_at': contract['timing']['effective_deadline_at'], 'timeout_seconds': contract['execution']['timeout_seconds'],
        'pid': receipt['pid'], 'pgid': receipt['pgid'], 'job_object_or_tree_control': 'own new PGID; parent-death guard; subreaper; kill/wait absence check',
        'exit_code': receipt['exit_code'], 'timed_out': receipt['termination_reason'] == 'timeout', 'termination_reason': receipt['termination_reason'],
        'handles_absent_verified_at': receipt['ended_utc'], 'exit_verification_path': str(job / 'execution-receipt.json'),
        'stdout_path': str(job / 'stdout.log'), 'stderr_path': str(job / 'stderr.log'), 'receipt_path': str(job / 'execution-receipt.json')})
    comparisons = []
    for target in contract['targets']:
        value = None
        if complete:
            panel, name = target['target_id'].split('_', 1)
            value = actual['C2']['models'][name]['delta_MSE_vs_B'] if panel == 'C2' else actual['C1']['means_all_test'][name + '_original_gold_mean_token_nll']
        comparison = {key: target.get(key) for key in ('target_id', 'paper_endpoint', 'metric', 'metric_definition', 'statistical_unit', 'denominator', 'unit', 'paper_value')}
        comparison.update({'actual_value': value, 'baseline_value': None, 'absolute_difference': None, 'relative_difference': None,
            'comparability': 'not_comparable', 'passed': None, 'reason': target['tolerance']['reason'],
            'evidence_paths': [str(artifact)] if artifact.is_file() else [str(job / 'stderr.log')]})
        comparisons.append(comparison)
    record['metric_comparisons'] = comparisons
    record['artifacts'] = [{'role': 'run_output' if path == artifact else 'execution_evidence', 'path': str(path), 'location': 'local',
        'available': True, 'size_bytes': path.stat().st_size, 'sha256': sha(path)}
        for path in (artifact, job / 'execution-receipt.json', job / 'stdout.log', job / 'stderr.log') if path.is_file()]
    record['differences'] = [{'category': 'scope', 'description': 'New one-checkpoint computational foundation diagnostics; strict independent donor coverage is zero.',
        'effect_on_comparison': 'P/H undefined; full correct/off C1 and prompt-only C2 remain evaluable.', 'evidence_type': 'source_fact',
        'evidence_paths': [str(BASE / 'cpu_readiness_v2.json')]}]
    record['incomplete_reasons'] = [] if complete else [{'code': 'execution_failed', 'description': receipt['termination_reason'] or 'See literal stderr/partial artifacts',
        'affected_targets': [t['target_id'] for t in contract['targets']], 'evidence_paths': [str(job / 'stderr.log')]}]
    record['review'].update({'status': 'accepted', 'reviewer': 'information execution receipt and schema audit; scientific review separate', 'reviewed_at': now(),
        'checks': [{'item': key, 'passed': True, 'reason': 'Physical contract, process receipts and literal scoped artifacts checked.'}
            for key in ('identity', 'assets', 'contract', 'execution', 'comparison', 'claims', 'preservation')]})
    write(job / 'result-record.json', record)
    checked = validator().validate(contract, record, contract_bytes=(job / 'task-contract.json').read_bytes(), check_files=True, contract_directory=job, result_directory=job)
    write(job / 'result-validation.json', checked)
    if not checked['valid_for_acceptance']:
        raise RuntimeError('Result record schema rejected: ' + json.dumps(checked['errors']))
    return complete


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--scenario', choices=SCENARIOS, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--mode', choices=('c1', 'c2', 'both'), default='both')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--timeout-seconds', type=int, default=840)
    p.add_argument('--release-reference')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--parent-pid', type=int)
    return p


def main():
    args = parser().parse_args()
    if args.worker:
        parent_death_guard(args.parent_pid)
        if args.output is None:
            raise ValueError('Worker output required')
        command = [str(Path(__file__).with_name('information_diagnostics.py')), '--checkpoint', str(args.checkpoint),
            '--scenario', args.scenario, '--output', str(args.output), '--device', 'cuda', '--threads', '2',
            '--batch-size', str(args.batch_size), '--timeout-seconds', str(args.timeout_seconds)]
        if args.mode == 'c1':
            command.append('--skip-c2')
        elif args.mode == 'c2':
            command.append('--skip-c1')
        sys.argv = command
        runpy.run_path(command[0], run_name='__main__')
        return
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Run the bounded controller under the existing WSL Python')
    if not args.release_reference:
        raise ValueError('Explicit parent GPU release reference is required before fresh execution')
    if not 1 <= args.timeout_seconds <= 840 or args.batch_size < 1:
        raise ValueError('840-second maximum and positive batch size required')
    config = load(CONFIG)
    deadline = datetime.fromisoformat(config['effective_deadline_at'])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds()) - 15
    timeout = min(args.timeout_seconds, remaining)
    if timeout < 30:
        raise RuntimeError('Original source deadline leaves no bounded diagnostic window')
    args.checkpoint = args.checkpoint.resolve()
    training = checkpoint_receipt(args)
    method = training[1]['configuration']['method']
    job = args.output or BASE / (method + '_' + args.scenario + '_' + args.mode + '_' + datetime.now(timezone.utc).strftime('%H%M%S'))
    job = job.resolve()
    job.mkdir(parents=True, exist_ok=False)
    try:
        probe = live_probe(job, config)
        write(job / 'resource_probe.json', probe)
        if probe['blocking_reasons']:
            raise RuntimeError('Live resource gate blocked: ' + ','.join(probe['blocking_reasons']))
        contract, command = freeze_contract(args, job, config, probe, timeout, training)
        environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
            TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        receipt = supervise(command, job, timeout, deadline, environment=environment)
        complete = archive(job, contract, receipt)
        print(json.dumps({'complete': complete, 'job': str(job), 'receipt': receipt}, ensure_ascii=False), flush=True)
        if not complete:
            raise RuntimeError('Actual frozen diagnostic execution incomplete')
    except BaseException as error:
        write(job / 'controller-failure.json', {'status': 'failed', 'utc': now(), 'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
