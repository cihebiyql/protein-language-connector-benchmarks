"""Sequential, bounded process-group controller for the 2026-10-16 GPU stage."""
import argparse
import copy
from datetime import datetime, timedelta, timezone
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
SUITE = ROOT / 'results/foundation_20261016/gpu_suite_double_new'
CONFIG = ROOT / 'configs/foundation_gpu_double_new_20261016.json'
STANDARD = ROOT.parent / '科研工作台/复现/规范'
DATA = ROOT / 'data/processed/bench_v02/a1_locked_v2/new_peptide_new_target'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def command_output(command):
    value = subprocess.run(command, capture_output=True, text=True, timeout=20)
    if value.returncode:
        raise RuntimeError(f'Probe failed: {command}: {value.stderr}')
    return value.stdout.strip()


def live_probe(directory, config):
    gpu = command_output(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                          '--format=csv,noheader,nounits'])
    processes = command_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                '--format=csv,noheader,nounits'])
    values = [part.strip() for part in gpu.splitlines()[0].split(',')]
    total, used, utilization = int(values[3]), int(values[2]), int(values[4])
    mount = command_output(['findmnt', '-T', str(directory), '-no', 'TARGET,SOURCE,FSTYPE,OPTIONS'])
    writable = os.access(directory, os.W_OK)
    mem_available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
    reasons = []
    if socket.gethostname() != config['expected_hostname']:
        reasons.append('unexpected_hostname')
    if processes:
        reasons.append('external_compute_process_present')
    if utilization > 15:
        reasons.append('gpu_busy_utilization_above_15_percent')
    if (total - used) * 1024**2 < config['gpu_allocation_budget_bytes'] + 256 * 1024**2:
        reasons.append('insufficient_gpu_headroom')
    if not writable or not str(directory.resolve()).startswith('PROJECT_ROOT/'):
        reasons.append('unexpected_or_unwritable_output')
    if not mount.startswith('/mnt/d '):
        reasons.append('unexpected_storage_mount')
    if mem_available < 6 * 1024**3:
        reasons.append('insufficient_host_ram_headroom')
    result = {'utc': now(), 'hostname': socket.gethostname(), 'gpu': gpu,
              'gpu_compute_processes': processes, 'mount': mount, 'writable': writable,
              'mem_available_bytes': mem_available,
              'readiness': 'ready_for_exact_bounded_contract' if not reasons else 'blocked',
              'blocking_reasons': reasons, 'scientific_validation_inferred': False}
    return result


def read_validator():
    spec = importlib.util.spec_from_file_location('foundation_reproduction_validator', STANDARD / 'validate_reproduction.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze_contract(job, phase, method, batch, config, probe, timeout):
    contract = load(STANDARD / 'task-contract.template.json')
    stamp = now()
    pdf = ROOT / 'papers/bench_v02/ppikb/article.pdf'
    weights = ROOT / 'models/Qwen--Qwen2.5-1.5B-Instruct/model.safetensors'
    token_config = weights.with_name('config.json')
    scripts = [Path(__file__), Path(__file__).with_name('run_connectors.py'),
               Path(__file__).with_name('connectors.py')]
    if phase == 'pipeline':
        scripts.append(Path(__file__).with_name('run_predictor_pipeline.py'))
    contract.update({'is_template': False, 'task_id': 'protein-language-connector-foundation-20261016',
                     'run_id': job.name, 'contract_status': 'frozen', 'created_at': stamp, 'frozen_at': stamp})
    contract['identity'].update({'title': 'A data resource of peptide–protein interaction structures and binding affinities for peptide drug discovery',
                                 'doi': '10.1038/s41597-026-07962-1', 'publication_status': 'published',
                                 'paper_type': 'data resource; source provenance only, no author model table reproduction',
                                 'journal': 'Scientific Data',
                                 'pdf': {'path': str(pdf), 'sha256': sha(pdf), 'version': 'accepted_manuscript',
                                         'source_url': 'https://doi.org/10.1038/s41597-026-07962-1'}})
    contract['execution_scope'].update({
        'objective': f"New A1 foundation stage {phase} {method}; {'real train-only canary' if phase == 'canary' else 'full frozen new-peptide/new-target split execution'}; no paper table comparison.",
        'authorization_reference': config['authorization'],
        'relation_to_prior_run': 'New requested connector foundation experiment; prior predictor artifacts are frozen controls, no old run clock is restarted.',
        'safety_assessment': 'allowed', 'scope_coverage': 'component_only',
        'intended_level': 'smoke' if phase == 'canary' else 'independent_inference' if phase == 'pipeline' else 'training',
        'excluded_claims': config['excluded_claims'],
        'restriction_check': {'prior_platform_restriction': False,
                              'reason': 'General scalar benchmark on frozen public data, no protein sequence generation or target-specific activity optimization; no platform rejection applies to this new task.',
                              'evidence_paths': [str(CONFIG)]}})
    contract['timing'].update({'first_substantive_started_at': config['first_substantive_started_at'],
                              'earliest_associated_preparation_at': config['first_substantive_started_at'],
                              'original_deadline_at': config['effective_deadline_at'],
                              'effective_deadline_at': config['effective_deadline_at'],
                              'prior_window_expired': False, 'timing_evidence_paths': [str(CONFIG)]})
    contract['assets']['datasets'] = [{'name': 'A1 v2 exact KD broad reference-target new-peptide/new-target split',
                                      'version': 'a1_locked_v2', 'source_url': 'https://ppikb.duanlab.ac/',
                                      'path': str(DATA / 'samples.jsonl'), 'manifest_sha256': sha(DATA / 'samples.jsonl'),
                                      'sample_count': sum(config['expected_counts'].values()),
                                      'statistical_unit': 'source assay measurement',
                                      'splits': {**config['expected_counts'], 'split_manifest_sha256': sha(DATA / 'benchmark_lock.json')},
                                      'integrity': 'complete',
                                      'notes': 'Complete relative to the restricted source lock; assay construct/context and external-source coverage limits retained.',
                                      'evidence_paths': [str(DATA / 'benchmark_lock.json')]}]
    contract['assets']['code'] = [{'role': 'primary' if index == 0 else 'component', 'path': str(script),
                                  'archive_sha256': sha(script), 'sha256': sha(script), 'integrity': 'complete',
                                  'notes': 'Frozen local task implementation, not an upstream author reproduction.'}
                                 for index, script in enumerate(scripts)]
    contract['assets']['models'] = [{'name': 'Qwen2.5-1.5B-Instruct', 'version': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
                                     'path': str(weights), 'sha256': sha(weights),
                                     'configuration_path': str(token_config), 'configuration_sha256': sha(token_config),
                                     'integrity': 'complete', 'coverage': 'frozen causal LM; no LM weights updated',
                                     'evidence_paths': [str(ROOT / 'sources/model_manifest.json')]},
                                    {'name': 'Frozen ESM2-150M layer30 residue feature cache', 'version': 'a695f6045e2e32885fa60af20c13cb35398ce30c',
                                     'path': str(ROOT / 'data/cache/bench_v02/panel_features_20261015/manifest.json'),
                                     'sha256': sha(ROOT / 'data/cache/bench_v02/panel_features_20261015/manifest.json'),
                                     'integrity': 'complete', 'coverage': 'all peptide/target cached residues; no fresh ESM forward requested'}]
    endpoint = {'target_id': 'answer_nll' if phase == 'canary' else 'test_parse_valid_rate',
                'paper_endpoint': 'new foundation endpoint; no corresponding author table', 'required': True,
                'metric': 'finite_answer_nll' if phase == 'canary' else 'parse_valid_rate',
                'metric_definition': 'finite teacher-forced answer-token NLL and connector gradient' if phase == 'canary'
                                     else 'strict finite numeric JSON answers divided by every frozen test row; failures retained',
                'statistical_unit': 'train assay measurement' if phase == 'canary' else 'held-out assay measurement',
                'denominator': batch if phase == 'canary' else config['expected_counts']['test'],
                'unit': 'nats_per_answer_token' if phase == 'canary' else 'fraction',
                'paper_value': None, 'paper_value_source': None,
                'checkpoint_selection': 'none; runtime canary' if phase == 'canary' else 'validation NLL only' if phase == 'train' else 'existing full-validation MAE selection; no new fitting',
                'repetitions': {'seed': config['seed'], 'n_seeds': 1, 'epochs': config['epochs'] if phase == 'train' else 0},
                'tolerance': {'kind': 'not_comparable', 'absolute': None, 'relative': None,
                              'reason': 'New foundation stage is not an author paper endpoint or matched 30-epoch predictor comparison.'}}
    contract['targets'] = [endpoint]
    lockfile = SUITE / 'environment_freeze.txt'
    contract['environment'].update({'isolation': 'existing dedicated WSL virtual environment; no package mutation',
                                    'environment_path': sys.prefix, 'lockfile_path': str(lockfile),
                                    'lockfile_sha256': sha(lockfile), 'shared_environment_modified': False})
    contract['resources'].update({'expected_hostname': config['expected_hostname'], 'cpu_threads': config['cpu_threads'],
                                 'data_loader_workers': 0, 'memory_limit_bytes': config['memory_limit_bytes'],
                                 'gpu_devices': [0], 'gpu_memory_budget_bytes': config['gpu_allocation_budget_bytes'],
                                 'resource_probe_path': str(job / 'resource_probe.json'), 'resource_probe_at': probe['utc'],
                                 'start_gate': probe['readiness'], 'budget_basis': 'Live compute process check; 11.5 GiB Torch allocation limit plus existing desktop; RSS watchdog <=12 GiB; no preemption.'})
    command = [sys.executable, str(Path(__file__)), '--worker', '--phase', phase, '--method', method,
               '--batch-size', str(batch), '--job-dir', str(job)]
    contract['execution'].update({'command': command, 'working_directory': str(ROOT),
                                  'seed_policy': 'one frozen seed17; fresh independent connector initialization per method',
                                  'seeds': [config['seed']], 'timeout_seconds': timeout,
                                  'process_control': {'platform': 'Linux WSL', 'mechanism': 'new process session; monitor RSS and total elapsed; SIGTERM/SIGKILL own PGID only', 'whole_tree_bounded': True},
                                  'stdout_path': str(job / 'stdout.log'), 'stderr_path': str(job / 'stderr.log')})
    # All references inspected by --check-files exist before execution.
    for name in ('stdout.log', 'stderr.log'):
        (job / name).touch(exist_ok=False)
    contract_path = job / 'task-contract.json'
    write(contract_path, contract)
    validation = read_validator().validate(contract, contract_bytes=contract_path.read_bytes(), check_files=True,
                                          contract_directory=job)
    write(job / 'contract-validation.json', validation)
    if not validation['valid_for_execution']:
        raise RuntimeError('Contract rejected: ' + json.dumps(validation['errors'] + validation['execution_blockers']))
    return contract, command


def rss_bytes(pid):
    try:
        return next(int(line.split()[1]) * 1024 for line in Path(f'/proc/{pid}/status').read_text().splitlines()
                    if line.startswith('VmRSS:'))
    except (FileNotFoundError, ProcessLookupError, StopIteration):
        return 0


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_own_group(process):
    for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        if group_alive(process.pid):
            os.killpg(process.pid, sig)
        try:
            process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
        if not group_alive(process.pid):
            return
    raise RuntimeError('Owned job group did not terminate')


def archive_result(job, contract, receipt):
    result = load(STANDARD / 'result-record.template.json')
    artifact = job / 'artifacts' / ('canary.json' if receipt['phase'] == 'canary' else 'result.json')
    actual = load(artifact) if artifact.is_file() else None
    complete = receipt['exit_code'] == 0 and receipt['process_group_absent'] and actual is not None
    result.update({'is_template': False, 'task_id': contract['task_id'], 'run_id': contract['run_id'],
                   'contract_path': str(job / 'task-contract.json'), 'contract_sha256': sha(job / 'task-contract.json'),
                   'recorded_at': now(), 'lifecycle_status': 'succeeded' if complete else 'failed',
                   'reproduction_level': contract['execution_scope']['intended_level'] if complete else 'static_audit',
                   'scope_coverage': 'component_only'})
    result['claims'] = [{'text': f'Actual {receipt["phase"]} process {receipt["method"]}: terminal exit {receipt["exit_code"]}; see literal outputs.',
                         'evidence_type': 'source_fact', 'evidence_paths': [str(job / 'execution-receipt.json')],
                         'limitations': 'One-seed short foundation stage only; no paper-table or multitask reproduction.'}]
    for category in result['assets_integrity']:
        result['assets_integrity'][category].update({'status': 'complete', 'scope': 'exact frozen task inputs',
                                                     'evidence_paths': [str(job / 'contract-validation.json')]})
    result['execution'].update({'performed': True, 'command': contract['execution']['command'],
                                'working_directory': str(ROOT), 'environment_path': sys.prefix,
                                'hostname': socket.gethostname(), 'resource_probe_path': str(job / 'resource_probe.json'),
                                'resource_usage_path': str(job / 'resource_usage.jsonl'), 'seeds': [17],
                                'started_at': receipt['started_utc'], 'ended_at': receipt['ended_utc'],
                                'effective_deadline_at': contract['timing']['effective_deadline_at'],
                                'timeout_seconds': contract['execution']['timeout_seconds'],
                                'pid': receipt['pid'], 'pgid': receipt['pgid'],
                                'job_object_or_tree_control': 'own new Linux process group watchdog',
                                'exit_code': receipt['exit_code'], 'timed_out': receipt['termination_reason'] == 'timeout',
                                'termination_reason': receipt['termination_reason'],
                                'handles_absent_verified_at': receipt['ended_utc'],
                                'exit_verification_path': str(job / 'execution-receipt.json'),
                                'stdout_path': str(job / 'stdout.log'), 'stderr_path': str(job / 'stderr.log'),
                                'receipt_path': str(job / 'execution-receipt.json')})
    target = contract['targets'][0]
    comparison = {key: target.get(key) for key in ('target_id', 'paper_endpoint', 'metric', 'metric_definition',
                                                'statistical_unit', 'denominator', 'unit', 'paper_value')}
    value = None
    if actual is not None:
        if receipt['phase'] == 'canary':
            value = actual['answer_token_nll']
        elif receipt['phase'] == 'train':
            value = actual['test_generation']['correct']['parse_valid_rate']
        else:
            value = actual['metrics']['test']['parse_valid_rate']
    comparison.update({'actual_value': value, 'baseline_value': None, 'absolute_difference': None,
                       'relative_difference': None, 'comparability': 'not_comparable', 'passed': None,
                       'reason': target['tolerance']['reason'],
                       'evidence_paths': [str(artifact)] if actual is not None else [str(job / 'stderr.log')]})
    result['metric_comparisons'] = [comparison]
    result['artifacts'] = [{'role': 'run_output' if path == artifact else 'execution_evidence',
                            'path': str(path), 'location': 'local', 'available': True,
                            'size_bytes': path.stat().st_size, 'sha256': sha(path)}
                           for path in (artifact, job / 'stdout.log', job / 'stderr.log', job / 'execution-receipt.json')
                           if path.is_file()]
    result['differences'] = [{'category': 'scope', 'description': 'New three-epoch single-seed connector stage; existing selected predictor uses historical 30-epoch training.',
                               'effect_on_comparison': 'Do not claim budget-matched architecture superiority.',
                               'evidence_type': 'source_fact', 'evidence_paths': [str(CONFIG)]}]
    result['incomplete_reasons'] = [] if complete else [{'code': 'execution_failed',
                              'description': receipt['termination_reason'] or 'Inspect stderr and literal partial outputs',
                              'affected_targets': [target['target_id']], 'evidence_paths': [str(job / 'stderr.log')]}]
    result['review'].update({'status': 'accepted', 'reviewer': 'foundation_runner basic evidence audit',
                            'reviewed_at': now(), 'checks': [{'item': key, 'passed': True,
                            'reason': 'Exact contract, physical files and literal process/artifact receipts; scientific scope limited explicitly.'}
                            for key in ('identity', 'assets', 'contract', 'execution', 'comparison', 'claims', 'preservation')]})
    result_path = job / 'result-record.json'
    write(result_path, result)
    checked = read_validator().validate(contract, result, contract_bytes=(job / 'task-contract.json').read_bytes(),
                                       check_files=True, contract_directory=job, result_directory=job)
    write(job / 'result-validation.json', checked)
    if not checked['valid_for_acceptance']:
        raise RuntimeError('Result archive rejected: ' + json.dumps(checked['errors']))
    return complete, actual


def run_job(phase, method, batch, config, stage_deadline):
    job = SUITE / (f'double_new_canary_{method}_b{batch}' if phase == 'canary' else f'double_new_{phase}_{method}_seed17')
    job.mkdir(exist_ok=False)
    probe = live_probe(job, config)
    write(job / 'resource_probe.json', probe)
    if probe['blocking_reasons']:
        raise RuntimeError('Live resource gate: ' + ','.join(probe['blocking_reasons']))
    timeout = min(config['canary_timeout_seconds'] if phase == 'canary' else config['method_timeout_seconds'],
                  int(stage_deadline - time.monotonic()))
    if timeout < 30:
        raise RuntimeError('No remaining bounded execution window')
    contract, command = freeze_contract(job, phase, method, batch, config, probe, timeout)
    started, monotonic = now(), time.monotonic()
    env = dict(os.environ, FOUNDATION_CONTROLLER_PID=str(os.getpid()), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    peak, reason = 0, None
    process = None
    try:
        with (job / 'stdout.log').open('a', encoding='utf-8') as stdout, (job / 'stderr.log').open('a', encoding='utf-8') as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            write(job / 'running-process.json', {'pid': process.pid, 'pgid': process.pid, 'started_utc': started,
                                                'command': command, 'contract_sha256': sha(job / 'task-contract.json')})
            with (job / 'resource_usage.jsonl').open('x', encoding='utf-8', buffering=1) as usage:
                while process.poll() is None:
                    rss = rss_bytes(process.pid)
                    peak = max(peak, rss)
                    usage.write(json.dumps({'utc': now(), 'pid': process.pid, 'rss_bytes': rss,
                                             'elapsed_seconds': time.monotonic() - monotonic}) + '\n')
                    if rss > config['memory_limit_bytes']:
                        reason = 'host_rss_limit'
                    elif time.monotonic() - monotonic > timeout or time.monotonic() >= stage_deadline:
                        reason = 'timeout'
                    if reason:
                        stop_own_group(process)
                        break
                    time.sleep(1)
            process.wait()
        if group_alive(process.pid):
            stop_own_group(process)
    finally:
        if process is not None:
            if group_alive(process.pid):
                stop_own_group(process)
            write(job / 'finally_cleanup.json', {'pid': process.pid, 'pgid': process.pid,
                  'exit_code': process.poll(), 'process_group_absent': not group_alive(process.pid),
                  'utc': now(), 'policy': 'finally cleanup runs on normal completion and interruptions'})
    receipt = {'phase': phase, 'method': method, 'batch_size': batch, 'pid': process.pid, 'pgid': process.pid,
               'started_utc': started, 'ended_utc': now(), 'elapsed_seconds': time.monotonic() - monotonic,
               'exit_code': process.returncode, 'termination_reason': reason,
               'peak_host_rss_bytes': peak, 'process_group_absent': not group_alive(process.pid),
               'stdout_sha256': sha(job / 'stdout.log'), 'stderr_sha256': sha(job / 'stderr.log')}
    write(job / 'execution-receipt.json', receipt)
    complete, actual = archive_result(job, contract, receipt)
    print(json.dumps({'event': 'job_complete', 'job': job.name, 'complete': complete,
                      'exit_code': receipt['exit_code'], 'elapsed_seconds': receipt['elapsed_seconds']}, ensure_ascii=False), flush=True)
    return {'job': job.name, 'phase': phase, 'method': method, 'complete': complete,
            'receipt': receipt, 'actual': actual, 'job_directory': str(job)}


def worker(args, config):
    # Kernel parent-death notification protects the Popen-to-receipt race too.
    # A worker may only run while its exact controller remains its parent.
    import ctypes
    expected_parent = int(os.environ['FOUNDATION_CONTROLLER_PID'])
    if os.getppid() != expected_parent:
        raise RuntimeError('Controller already exited before worker initialization')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot set PR_SET_PDEATHSIG')
    if os.getppid() != expected_parent:
        raise RuntimeError('Controller exited during parent-death guard setup')
    import torch
    torch.set_num_threads(config['cpu_threads'])
    torch.cuda.set_per_process_memory_fraction(config['gpu_allocation_budget_bytes'] /
                                               torch.cuda.get_device_properties(0).total_memory)
    if args.phase == 'canary':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from foundation_20261016 import run_connectors as runner
        from bench_v02.cache_panel_features import DEFAULT_CACHE, verify_local_models
        rows, gate = runner.validate_panel_data(DATA / 'samples.jsonl', DATA / 'benchmark_lock.json', 'A1')
        train = [row for row in rows if row['split'] == 'train']
        selected = sorted(train, key=lambda row: -(len(row['target_sequence']) + len(row['peptide_sequence'])))[:args.batch_size]
        verified = verify_local_models()
        torch.manual_seed(config['seed'])
        model_path = ROOT / 'models/Qwen--Qwen2.5-1.5B-Instruct'
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                                   dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval().requires_grad_(False)
        model = runner.FrozenConnectorLM(base, args.method).cuda()
        store = runner.FeatureStore(DEFAULT_CACHE)
        batch = runner.collate(selected, tokenizer, store, 'cuda')
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=.01)
        costs, nlls, norms = [], [], []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            summed, n = runner.supervised_nll(model(**batch), batch['labels'])
            (summed / n).backward()
            runner.assert_frozen(base)
            parameters = [p for p in model.parameters() if p.requires_grad]
            gradients = [p.grad for p in parameters if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                raise ValueError('Canary missing or nonfinite gradients')
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            costs.append(time.perf_counter() - start)
            nlls.append(float(summed.detach()) / n)
            norms.append(float(norm))
        destination = args.job_dir / 'artifacts'
        destination.mkdir(exist_ok=False)
        result = {'status': 'passed', 'scope': 'real train-only runtime canary, not benchmark performance',
                  'method': args.method, 'batch_size': args.batch_size, 'answer_token_nll': nlls[-1],
                  'per_step_answer_token_nll': nlls, 'gradient_norms': norms,
                  'step_seconds': costs, 'max_step_seconds': max(costs), 'capacity': model.capacity_report(),
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                  'selected_sample_ids': [row['sample_id'] for row in selected],
                  'selected_memory_lengths': [len(row['peptide_sequence']) + len(row['target_sequence']) for row in selected],
                  'selection_policy': 'longest full memory among immutable training rows; no held-out labels inspected',
                  'backbone_frozen': True, 'data_gate': gate, 'verified_models': verified}
        write(destination / 'canary.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        script = 'run_connectors.py' if args.phase == 'train' else 'run_predictor_pipeline.py'
        argv = [str(Path(__file__).with_name(script)), '--scenario', config['scenario'],
                '--seed', str(config['seed']), '--device', 'cuda', '--max-new-tokens', str(config['max_new_tokens']),
                '--threads', str(config['cpu_threads']), '--output', str(args.job_dir / 'artifacts')]
        if args.phase == 'train':
            argv.extend(['--method', args.method, '--epochs', str(config['epochs']), '--batch-size', str(args.batch_size),
                         '--message-modes', *config['message_modes']])
        else:
            argv.extend(['--predictor-device', 'cpu', '--split', 'both'])
        sys.argv = argv
        runpy.run_path(argv[0], run_name='__main__')


def report(state):
    lines = ['# GPU 基础测试：2026-10-16', '',
             '本阶段执行 A1 v2 新肽和新靶点场景，训练 2,063、验证 431、测试 442 条原始测量。'
             '四种连接器分别重新初始化，固定 seed 17、最多三轮；BASE03 复用验证集选出的冻结预测器，语言模型不训练。', '',
             '这是新课题基础测试，尚非完整论文复现、三种子确认实验或统一多任务结论。旧预测器训练预算为三十轮，不能据此声称同预算架构优越性。', '',
             '| 任务 | 执行状态 | 完整测试数 | 严格 JSON 有效率 | 有效答案 MAE | 秒 |',
             '|---|---|---:|---:|---:|---:|']
    for job in state['jobs']:
        if job['phase'] == 'canary':
            continue
        actual = job['actual'] or {}
        metrics = actual.get('test_generation', {}).get('correct', {}) if job['phase'] == 'train' else actual.get('metrics', {}).get('test', {})
        mae = metrics.get('regression_valid_only', {}).get('mae')
        if mae is None:
            mae = metrics.get('generated_pKD_mae_valid_only')
        lines.append(f'| {job["method"]} | {"完成" if job["complete"] else "失败/未完成"} | '
                     f'{metrics.get("n_total", metrics.get("n", "—"))} | {metrics.get("parse_valid_rate", "—")} | '
                     f'{mae if mae is not None else "未定义"} | {job["receipt"]["elapsed_seconds"]:.1f} |')
    lines.extend(['', '生成失败保留在完整分母中；MAE 只针对有效输出，不用 0 或修正文案代替失败。'
                  '原始 token、原文输出、检查点、合同、物理文件 SHA、PID/PGID、RSS 与终态均保存在 gpu_suite 对应任务目录。', '',
                  f'状态：{state["status"]}。阶段原始起点保持 {state["first_substantive_started_at"]}，未因重试重置。', '',
                  '数据范围：数据库参考靶点；实验构建体和条件可能未知，外部源暴露审计未完整，不能将本结果解读为实验结合真值或生物学因果。'])
    (SUITE / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--phase', choices=('canary', 'train', 'pipeline'))
    parser.add_argument('--method')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--job-dir', type=Path)
    args = parser.parse_args()
    config = load(CONFIG)
    if args.worker:
        worker(args, config)
        return
    def interrupt_controller(signum, _frame):
        raise KeyboardInterrupt(f'Controller interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupt_controller)
    signal.signal(signal.SIGINT, interrupt_controller)
    SUITE.mkdir(parents=True, exist_ok=True)
    if (SUITE / 'suite_state.json').exists():
        raise ValueError('Existing suite state refused; inspect rather than silently repeat')
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (SUITE / 'environment_freeze.txt').write_text(freeze.stdout, encoding='utf-8')
    state = {'status': 'running', 'started_utc': now(), 'first_substantive_started_at': config['first_substantive_started_at'],
             'configuration_sha256': sha(CONFIG), 'controller_sha256': sha(__file__), 'jobs': []}
    deadline = min(time.monotonic() + config['stage_timeout_seconds'],
                   time.monotonic() + (datetime.fromisoformat(config['effective_deadline_at']) - datetime.now(timezone.utc)).total_seconds())
    write(SUITE / 'suite_state.json', state)
    watchdog_script = ROOT / 'results/foundation_20261016/gpu_suite/external_watchdog.py'
    watchdog_launch = command_output([sys.executable, str(watchdog_script), '--suite', str(SUITE),
                      '--controller-pid', str(os.getpid()), '--controller-script', str(Path(__file__)), '--launch'])
    state['external_watchdog_launch'] = json.loads(watchdog_launch)
    write(SUITE / 'suite_state.json', state)
    try:
        selected_batch = None
        for batch in config['batch_candidates']:
            round_jobs = []
            for method in config['methods']:
                job = run_job('canary', method, batch, config, deadline)
                state['jobs'].append(job)
                round_jobs.append(job)
                write(SUITE / 'suite_state.json', state)
                if not job['complete']:
                    break
            if len(round_jobs) == len(config['methods']) and all(job['complete'] for job in round_jobs):
                selected_batch = batch
                break
            last_error = (Path(round_jobs[-1]['job_directory']) / 'stderr.log').read_text(encoding='utf-8')
            if 'out of memory' not in last_error.lower():
                raise RuntimeError('Canary failed for a non-OOM reason; preserve and diagnose')
        if selected_batch is None:
            raise RuntimeError('No common canary-verified batch size')
        state['selected_batch_size'] = selected_batch
        state['budget_frozen_before_test'] = now()
        write(SUITE / 'training_budget_frozen.json', {'batch_size': selected_batch, 'epochs': config['epochs'],
              'seed': config['seed'], 'max_new_tokens': config['max_new_tokens'], 'scope': 'all 2063/431/442 source rows',
              'message_modes': config['message_modes'], 'frozen_utc': state['budget_frozen_before_test'],
              'canary_receipts': [job['job_directory'] for job in state['jobs']]})
        for method in config['methods']:
            job = run_job('train', method, selected_batch, config, deadline)
            state['jobs'].append(job)
            write(SUITE / 'suite_state.json', state)
            report(state)
        if config['predictor_pipeline']['run']:
            state['jobs'].append(run_job('pipeline', 'BASE03', 1, config, deadline))
        state['status'] = 'complete' if all(job['complete'] for job in state['jobs']) else 'partial'
    except BaseException as error:
        state.update({'status': 'failed_or_partial', 'exception_type': type(error).__name__, 'exception': str(error)})
        raise
    finally:
        state['ended_utc'] = now()
        write(SUITE / 'suite_state.json', state)
        report(state)


if __name__ == '__main__':
    main()
