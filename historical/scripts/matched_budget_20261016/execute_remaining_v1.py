"""Original V1 batch-one continuation after the fixed first two completed cases.

No GPU job is admitted without a frozen protocol, exact input hashes, a fresh
resource gate and a standard non-template contract accepted with --check-files.
The explicit --release-reference records the parent's actual GPU release.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STANDARD = ROOT.parent / '科研工作台/复现/规范'
TASK = 'protein-language-connector-matched-budget-20261016'
PENDING_SIGNAL = None
SOURCE_AUDITOR=Path(__file__).with_name('execute_suite_batched.py')


def source_auditor():
    # Frozen read-only administrative verifier; this module imports no Torch.
    spec=importlib.util.spec_from_file_location('original_V1_source_auditor',SOURCE_AUDITOR)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def admit_original_sources(suite,protocol,protocol_sha,manifest):
    audit=source_auditor()
    return {case_id(case):audit.validate_reuse_job(Path(suite)/case_id(case),case,protocol_sha,manifest)
            for case in trial_matrix(protocol)[:2]}


def verify_original_cache(contract,manifest):
    item=contract['matched_budget_registration']
    path=Path(item['cache_manifest_path'])
    if sha(path)!=item['cache_manifest_sha256']: raise ValueError('Original qualification manifest changed')
    old=load(path)
    for key in ('datasets','projection','initializations'):
        if old[key]!=manifest[key]: raise ValueError('Original qualification messages/initializations changed: '+key)


def admit_original_qualifications(suite,protocol,protocol_sha,manifest,current_environment_freeze):
    admitted={}
    for case in canary_matrix(protocol):
        job=Path(suite)/case_id(case)
        contract=load(job/'task-contract.json');record=load(job/'result-record.json');receipt=load(job/'execution-receipt.json')
        actual=load(job/'artifacts/result.json')
        if load(job/'case.json')!=case: raise ValueError('Original TRAIN canary case changed')
        if (contract['matched_budget_registration']['protocol_sha256']!=protocol_sha
                or contract['execution_scope']['intended_level']!='smoke'
                or record.get('reproduction_level')!='smoke' or record.get('lifecycle_status')!='succeeded'
                or receipt.get('exit_code')!=0 or not receipt.get('process_group_absent') or receipt.get('termination_reason') is not None):
            raise ValueError('Original six TRAIN qualifications lack terminal accepted smoke evidence')
        checked=validator().validate(contract,record,contract_bytes=(job/'task-contract.json').read_bytes(),check_files=True,
            contract_directory=job,result_directory=job)
        if not checked.get('valid_for_acceptance'): raise ValueError('Original TRAIN qualification standard physical archive rejected')
        if (contract['environment']['lockfile_sha256']!=sha(current_environment_freeze)
                or contract['environment']['environment_path']!=sys.prefix):
            raise ValueError('Current actual pip freeze/environment differs from original TRAIN qualifications')
        numerical=load(job/'numerical_policy_actual.json');policy=protocol['numerical_policy']
        for key in ('float32_matmul_precision','cuda_matmul_allow_tf32','cudnn_allow_tf32','cudnn_benchmark',
                    'cudnn_deterministic','deterministic_algorithms','deterministic_warn_only'):
            expected=policy[key];actual_flag=numerical.get(key)
            if type(actual_flag) is not type(expected) or actual_flag!=expected:
                raise ValueError('Original TRAIN numerical flag differs from current protocol: '+key)
        if (numerical.get('applied_before_model_instantiation') is not True
                or any(numerical.get('process_environment',{}).get(key)!=policy[key]
                       for key in ('CUDA_LAUNCH_BLOCKING','TORCH_SHOW_CPP_STACKTRACES','CUBLAS_WORKSPACE_CONFIG'))):
            raise ValueError('Original TRAIN pre-model numerical runtime environment differs')
        if (actual.get('status')!='passed' or actual.get('optimizer_steps')!=1 or actual.get('scientific_trial_count')!=0
                or actual.get('test_examples_evaluated')!=0 or actual.get('sample_split')!='train'
                or actual.get('train_generation_metrics',{}).get('n_total')!=4):
            raise ValueError('Original qualification is not the unchanged four-row TRAIN-only V1 canary')
        runner=[x for x in contract['assets']['code'] if Path(x['path']).name=='run_trial.py']
        if len(runner)!=1 or runner[0]['sha256']!=sha(Path(__file__).with_name('run_trial.py')):
            raise ValueError('Original qualification worker differs from exact V1 source')
        verify_original_cache(contract,manifest)
        files=[job/'task-contract.json',job/'result-record.json',job/'execution-receipt.json',job/'artifacts/result.json',
               job/'resource_probe.json',job/'numerical_policy_actual.json']
        admitted[case_id(case)]={'original_job':str(job),'new_execution_performed':False,'new_pid':None,
            'evidence_pins':{str(path):sha(path) for path in files},'interpretation':'Original exact V1 TRAIN qualification reused, not a new canary or scientific training replicate'}
    return admitted


def verify_reused_evidence(source_plan,qualified):
    for item in list(source_plan.values())+list(qualified.values()):
        for path,digest in item['evidence_pins'].items():
            if sha(path)!=digest: raise ValueError('Original accepted evidence changed: '+path)


def request_stop(signum, _frame):
    # Avoid asynchronously interrupting Popen's fork-to-PID assignment or the
    # terminal archive. The polling loop observes this request within 250 ms.
    global PENDING_SIGNAL
    PENDING_SIGNAL = signum


def runtime_environment(protocol, parent_pid):
    policy=protocol['numerical_policy']
    expected={'CUDA_LAUNCH_BLOCKING':'1','TORCH_SHOW_CPP_STACKTRACES':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    if any(policy.get(key)!=value for key,value in expected.items()):
        raise ValueError('Strict pre-Torch runtime environment differs from protocol')
    return dict(expected,MATCHED_CONTROLLER_PID=str(parent_pid),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')


def apply_numerical_policy(torch, policy):
    """Called after environment setup and before CUDA/model instantiation.

    No exception is caught: unsupported deterministic kernels must fail the
    TRAIN qualification rather than silently choose another numerical policy.
    """
    expected={'base_dtype':'bfloat16','reader_and_message_dtype':'float32','reader_autocast':False,
        'float32_matmul_precision':'highest','cuda_matmul_allow_tf32':False,'cudnn_allow_tf32':False,
        'cudnn_benchmark':False,'cudnn_deterministic':True,'deterministic_algorithms':True,'deterministic_warn_only':False}
    if any(policy.get(key)!=value for key,value in expected.items()):
        raise ValueError('Strict numerical policy differs from frozen specification')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=False)
    return dict(utc=now(),applied_before_model_instantiation=True,applied_before_CUDA_property_probe=True,
        process_environment={key:os.environ.get(key) for key in ('CUDA_LAUNCH_BLOCKING','TORCH_SHOW_CPP_STACKTRACES','CUBLAS_WORKSPACE_CONFIG')},
        float32_matmul_precision=torch.get_float32_matmul_precision(),cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        dtype_enforcement='base bfloat16 and reader/message float32 without autocast verified separately by runner',
        unsupported_kernel_policy='raise, retain failed canary, stop queued trials; no fallback')


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def validator():
    spec = importlib.util.spec_from_file_location('matched_reproduction_validator', STANDARD / 'validate_reproduction.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trial_matrix(protocol):
    """Fixed order; no branching or selection based on observed scores."""
    if tuple(protocol['scenarios']) != ('new_peptide_known_target','new_peptide_new_target') or protocol['seeds'] != [17,42,2026]:
        raise ValueError('Fixed scenario order and three seeds must not change')
    cases = []
    for seed in protocol['seeds']:
        for scenario in protocol['scenarios']:
            for placement in ('early', 'mid'):
                for content in ('real', 'neutral'):
                    cases.append(dict(scenario=scenario, seed=seed, placement=placement,
                                      message_input=content, sender_mode='balanced', gate_init='zero', phase='trial'))
    for extension in ('near_unit', 'global16'):
        for seed in protocol['seeds']:
            for scenario in protocol['scenarios']:
                cases.append(dict(scenario=scenario, seed=seed, placement='mid', message_input='real',
                                  sender_mode='global16' if extension == 'global16' else 'balanced',
                                  gate_init='near_unit' if extension == 'near_unit' else 'zero', phase='trial'))
    if len(cases) != 36 or len({case_id(c) for c in cases}) != 36:
        raise ValueError('Protocol must enumerate exactly 36 unique trials')
    return cases


def canary_matrix(protocol):
    return [dict(scenario=scenario, seed=17, placement=place, message_input='real',
                 sender_mode='balanced', gate_init=gate, phase='canary')
            for scenario in protocol['scenarios'] for place, gate in
            (('early', 'zero'), ('mid', 'zero'), ('mid', 'near_unit'))]


def case_id(case):
    return '_'.join(str(case[k]) for k in ('phase', 'scenario', 'seed', 'placement', 'message_input', 'sender_mode', 'gate_init'))


def verify_inputs(protocol_path, expected_protocol_sha, manifest_path, expected_manifest_sha):
    if sha(protocol_path) != expected_protocol_sha or sha(manifest_path) != expected_manifest_sha:
        raise ValueError('Protocol/cache manifest differs from frozen supervisor admission')
    protocol, manifest = load(protocol_path), load(manifest_path)
    if protocol['status'] not in ('frozen', 'approved_for_execution', 'frozen_for_execution'):
        raise ValueError('Protocol not frozen for execution')
    if manifest.get('purpose') != 'matched_budget_fixed_message_cache':
        raise ValueError('Wrong cache purpose')
    for raw, expected in manifest['source_pins'].items():
        if sha(ROOT / raw) != expected:
            raise ValueError('Frozen source changed: ' + raw)
    for scenario, data in manifest['datasets'].items():
        if sha(ROOT / data['samples_path']) != protocol['scenarios'][scenario]['samples_sha256']:
            raise ValueError('Scenario samples SHA differs from protocol')
        for item in data['layouts'].values():
            if sha(ROOT / item['path']) != item['sha256']:
                raise ValueError('Fixed-message payload changed')
    for item in list(manifest['initializations'].values()) + [manifest['projection']]:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Fixed initial state/projection changed')
    return protocol, manifest


def output(command):
    completed = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
    return completed.stdout.strip()


def live_probe(directory, resources):
    gpu = output(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'])
    processes = output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'])
    parts = [p.strip() for p in gpu.splitlines()[0].split(',')]
    used, total, utilization = map(int, parts[2:5])
    mount = output(['findmnt', '-T', str(directory), '-no', 'TARGET,SOURCE,FSTYPE,OPTIONS'])
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    reasons = []
    if socket.gethostname() != resources['host']: reasons.append('unexpected_hostname')
    if processes: reasons.append('external_compute_process_present')
    if utilization > 15: reasons.append('gpu_utilization_above_15_percent')
    if (total - used) * 1024**2 < resources['torch_GPU_allocation_limit_bytes'] + 256 * 1024**2:
        reasons.append('insufficient_GPU_headroom')
    if available < 6 * 1024**3: reasons.append('insufficient_host_RAM')
    if shutil.disk_usage(directory).free < 10 * 1024**3: reasons.append('insufficient_storage')
    if not mount.startswith('/mnt/d '): reasons.append('unexpected_storage_mount')
    if not str(directory.resolve()).startswith('PROJECT_ROOT/'):
        reasons.append('unexpected_output_location')
    try:
        probe = directory / '.write_probe'
        with probe.open('x') as handle: handle.write('bounded temporary write probe')
        probe.unlink()
    except OSError:
        reasons.append('output_not_writable')
    if str(Path(sys.executable).resolve()) != str(Path(resources['environment']).resolve()):
        reasons.append('unexpected_interpreter')
    return dict(utc=now(), hostname=socket.gethostname(), gpu=gpu, compute_processes=processes,
                mount=mount, mem_available_bytes=available, free_storage_bytes=shutil.disk_usage(directory).free,
                environment=sys.executable, readiness='blocked' if reasons else 'ready_for_exact_bounded_contract',
                blocking_reasons=reasons, scientific_validation_inferred=False)


def build_command(case, job, protocol_path, protocol_sha, manifest_path, manifest_sha, timeout):
    return [sys.executable, str(Path(__file__).resolve()), '--worker', '--case-file', str(job / 'case.json'),
            '--protocol', str(protocol_path), '--protocol-sha256', protocol_sha,
            '--cache-manifest', str(manifest_path), '--cache-manifest-sha256', manifest_sha,
            '--output', str(job / 'artifacts'), '--timeout-seconds', str(timeout)]


def build_contract(case, job, suite, protocol_path, protocol, protocol_sha, manifest_path, manifest_sha,
                   probe, timeout, command, release):
    contract = load(STANDARD / 'task-contract.template.json')
    stamp = now()
    contract.update(is_template=False, task_id=TASK, run_id='v1_continuation__'+job.name, contract_status='frozen', created_at=stamp, frozen_at=stamp)
    pdf = ROOT / 'papers/bench_v02/ppikb/article.pdf'
    contract['identity'].update(title='A data resource of peptide–protein interaction structures and binding affinities for peptide drug discovery',
        doi='10.1038/s41597-026-07962-1', publication_status='published', journal='Scientific Data',
        paper_type='data-resource provenance; new matched-budget method experiment, no author table reproduction',
        pdf=dict(path=str(pdf), sha256=sha(pdf), version='accepted_manuscript', source_url='https://doi.org/10.1038/s41597-026-07962-1'))
    contract['execution_scope'].update(objective='New matched-budget ' + case_id(case),
        authorization_reference=protocol['authorization'], relation_to_prior_run=protocol['timing']['relationship'],
        safety_assessment='allowed', scope_coverage='component_only', intended_level='smoke' if case['phase'] == 'canary' else 'training',
        excluded_claims=['full author system reproduction', 'unobserved confirmatory test', 'biological activity optimization', 'matched total FLOPs'],
        restriction_check=dict(prior_platform_restriction=False, reason='Scalar computation benchmark; no sequence generation or target-specific biological optimization.', evidence_paths=[str(protocol_path)]))
    contract['timing'].update(first_substantive_started_at=protocol['timing']['first_substantive_started_at'],
        earliest_associated_preparation_at=protocol['timing']['earliest_associated_preparation_at'],
        original_deadline_at=protocol['timing']['effective_deadline_at'], effective_deadline_at=protocol['timing']['effective_deadline_at'],
        prior_window_expired=False, timing_evidence_paths=[str(protocol_path)])
    data = load(manifest_path)['datasets'][case['scenario']]
    lock = ROOT / data['lock_path']
    counts = protocol['scenarios'][case['scenario']]
    # Both manifest_path and its exact digest are explicitly paired and checked.
    contract['assets']['datasets'] = [dict(name='Immutable A1 v2 ' + case['scenario'], version='a1_locked_v2',
        source_url='https://ppikb.duanlab.ac/', path=str(ROOT / data['samples_path']), sha256=counts['samples_sha256'],
        manifest_path=str(lock), manifest_sha256=sha(lock), sample_count=sum(counts[k] for k in ('train','validation','test')),
        statistical_unit='database source measurement; correlated source/family components',
        splits={**{k: counts[k] for k in ('train','validation','test')}, 'split_manifest_sha256':sha(lock)},
        integrity='complete', evidence_paths=[str(lock)]),
        dict(name='Fixed message/cache manifest', version='matched_budget_20261016', source_url='local frozen ESM residue cache',
             path=str(manifest_path), sha256=manifest_sha, manifest_path=str(manifest_path), manifest_sha256=manifest_sha,
             integrity='complete', notes='Scenario-specific TRAIN mean; initialization untrained; projection independent of labels.')]
    code = [Path(__file__).resolve(), Path(__file__).with_name('run_trial.py'), Path(__file__).with_name('connectors.py'),
            Path(__file__).with_name('fixed_message.py'), Path(__file__).with_name('prepare_cache.py'),
            Path(__file__).with_name('token_diagnostics.py')]
    contract['assets']['code'] = [dict(role='primary' if i == 0 else 'component', path=str(p), archive_sha256=sha(p), sha256=sha(p), integrity='complete') for i,p in enumerate(code)]
    model = ROOT / 'models/Qwen--Qwen2.5-1.5B-Instruct/model.safetensors'
    contract['assets']['models'] = [dict(name='Qwen2.5-1.5B-Instruct',version='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
        path=str(model),sha256=sha(model),configuration_path=str(model.with_name('config.json')),
        configuration_sha256=sha(model.with_name('config.json')),integrity='complete',coverage='Frozen LM; reader only training')]
    canary = case['phase'] == 'canary'
    contract['targets'] = [dict(target_id='parse_coverage',paper_endpoint='new controlled experiment; no author table',required=True,
        metric='parse_valid_rate',metric_definition='strict finite JSON predictions divided by every designated evaluation row; failures retained',
        statistical_unit='TRAIN runtime sample' if canary else 'held-out database measurement',denominator=4 if canary else counts['test'],
        unit='fraction',paper_value=None,paper_value_source=None,checkpoint_selection='none; TRAIN-only canary' if canary else protocol['training']['checkpoint_selection'],
        repetitions=dict(seed=case['seed'],n_seeds=1,epochs=0 if canary else 3),
        tolerance=dict(kind='not_comparable',absolute=None,relative=None,reason='New prospectively specified experiment, no corresponding author endpoint'))]
    lockfile = suite / 'environment_freeze.txt'
    contract['environment'].update(isolation='existing dedicated WSL venv; no dependency or driver changes',environment_path=sys.prefix,
        lockfile_path=str(lockfile),lockfile_sha256=sha(lockfile),shared_environment_modified=False)
    contract['environment']['process_local_runtime_flags']=runtime_environment(protocol,os.getpid())
    contract['environment']['numerical_policy'].update(parameter_dtype='base bfloat16; reader float32',input_dtype='message float32; token IDs integer',
        autocast=False,float32_matmul_precision='highest',cuda_matmul_allow_tf32=False,cudnn_allow_tf32=False,
        determinism='torch.use_deterministic_algorithms(True,warn_only=False); cudnn benchmarkFalse deterministicTrue',
        applied_before_model_instantiation=True,actual_policy_receipt_path=None,
        notes='Declared mandatory worker setup; actual numerical_policy_actual.json is written before model initialization and archived after execution. Unsupported deterministic kernels fail TRAIN qualification without fallback.')
    resources = protocol['resources']
    contract['resources'].update(expected_hostname=resources['host'],cpu_threads=2,data_loader_workers=0,
        memory_limit_bytes=resources['host_RSS_limit_bytes'],gpu_devices=[0],gpu_memory_budget_bytes=resources['torch_GPU_allocation_limit_bytes'],
        resource_probe_path=str(job / 'resource_probe.json'),resource_probe_at=probe['utc'],start_gate=probe['readiness'],
        budget_basis='fresh idle GPU gate; <=12GiB whole process-group RSS, <=11.5GiB Torch allocator; no preemption')
    contract['execution'].update(command=command,working_directory=str(ROOT),seed_policy='Fixed 17/42/2026 paired initialization; no score-based selection',
        seeds=[case['seed']],timeout_seconds=timeout,process_control=dict(platform='Linux WSL',
        mechanism='new process group, subreaper, parent-death signal with race check; timed group RSS watchdog and finally SIGTERM/SIGKILL cleanup',whole_tree_bounded=True),
        stdout_path=str(job/'stdout.log'),stderr_path=str(job/'stderr.log'))
    contract['matched_budget_registration'] = dict(case=case,protocol_path=str(protocol_path),protocol_sha256=protocol_sha,
        cache_manifest_path=str(manifest_path),cache_manifest_sha256=manifest_sha,GPU_release_reference=release)
    contract['original_V1_continuation']=dict(original_decoder='Unchanged run_trial.py batch1',
        source_cases_path=str(suite/'original_source_receipts.json'),source_cases_sha256=sha(suite/'original_source_receipts.json'),
        qualification_path=str(suite/'reused_qualification_receipts.json'),qualification_sha256=sha(suite/'reused_qualification_receipts.json'),
        skipped_fixed_trial_indices=[0,1],new_training_case_count=34,new_runtime_identity_only=True,
        source_auditor_path=str(SOURCE_AUDITOR),source_auditor_sha256=sha(SOURCE_AUDITOR),
        preserved_failed_candidates='V2/V3 qualifications are separate preserved evidence; no predictions from them are admitted')
    return contract


def group_members(pgid):
    members = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid:
                members.append((int(path.parent.name), fields[0]))
        except (OSError, ValueError, IndexError):
            continue
    return members


def group_rss(pgid):
    total = 0
    for pid, state in group_members(pgid):
        if state == 'Z': continue
        try:
            total += next(int(line.split()[1]) * 1024 for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('VmRSS:'))
        except (OSError, StopIteration):
            pass
    return total


def reap(pgid):
    for pid, state in group_members(pgid):
        if state != 'Z': continue
        try:
            os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, ProcessLookupError):
            pass


def cleanup(process):
    # Popen.wait must reap the direct child before the subreaper collects orphans.
    for sig, seconds in ((signal.SIGTERM, 3), (signal.SIGKILL, 3)):
        if group_members(process.pid):
            try: os.killpg(process.pid, sig)
            except ProcessLookupError: pass
        try: process.wait(timeout=seconds)
        except subprocess.TimeoutExpired: pass
        end = time.monotonic() + seconds
        while group_members(process.pid) and time.monotonic() < end:
            reap(process.pid)
            time.sleep(.02)
        if not group_members(process.pid): break
    reap(process.pid)
    return not group_members(process.pid)


def supervise(command, job, timeout, rss_limit, deadline, env=None, poll_seconds=.25):
    """Always produce terminal receipts, including Python exceptions/signals."""
    started, clock = now(), time.monotonic()
    process, reason, error, peak = None, None, None, 0
    try:
        with (job/'stdout.log').open('a',encoding='utf-8') as out, (job/'stderr.log').open('a',encoding='utf-8') as err:
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
            write(job/'running-process.json',dict(pid=process.pid,pgid=process.pid,started_utc=started,command=command))
            with (job/'resource_usage.jsonl').open('x',encoding='utf-8',buffering=1) as usage:
                while process.poll() is None:
                    rss = group_rss(process.pid)
                    peak=max(peak,rss)
                    usage.write(json.dumps(dict(utc=now(),group_rss_bytes=rss,elapsed_seconds=time.monotonic()-clock))+'\n')
                    if PENDING_SIGNAL is not None: reason='controller_signal_'+str(PENDING_SIGNAL)
                    elif rss > rss_limit: reason='host_process_group_RSS_limit'
                    elif time.monotonic()-clock >= timeout or datetime.now(timezone.utc) >= deadline: reason='timeout'
                    if reason: break
                    time.sleep(poll_seconds)
    except BaseException as exc:
        reason='controller_exception'
        error=dict(type=type(exc).__name__,message=str(exc))
    finally:
        absent = cleanup(process) if process is not None else True
        receipt=dict(started_utc=started,ended_utc=now(),elapsed_seconds=time.monotonic()-clock,
            pid=process.pid if process else None,pgid=process.pid if process else None,
            exit_code=process.returncode if process else None,termination_reason=reason,
            exception=error,performed=process is not None,peak_host_process_group_rss_bytes=peak,
            process_group_absent=absent,stdout_sha256=sha(job/'stdout.log'),stderr_sha256=sha(job/'stderr.log'))
        write(job/'execution-receipt.json',receipt)
    return receipt


def archive(job,contract,receipt):
    artifact=job/'artifacts/result.json'
    try:
        actual=load(artifact) if artifact.is_file() else None
    except (ValueError, OSError) as exc:
        actual=None
        write(job/'artifact-read-error.json',dict(type=type(exc).__name__,message=str(exc)))
    if not isinstance(actual,dict): actual=None
    complete=bool(receipt['performed'] and receipt['exit_code']==0 and receipt['process_group_absent'] and
                  receipt['termination_reason'] is None and actual and actual.get('status') in ('complete','passed'))
    case=contract['matched_budget_registration']['case']
    output_errors=[]
    if complete:
        if case['phase']=='trial':
            if (actual.get('status')!='complete' or actual.get('optimizer_steps')!=1548
                    or actual.get('actual_epochs')!=3 or actual.get('test_evaluation_passes')!=1):
                output_errors.append('Full immutable training/evaluation budget incomplete')
        elif (actual.get('status')!='passed' or actual.get('optimizer_steps')!=1 or actual.get('test_examples_evaluated')!=0
              or actual.get('scientific_trial_count')!=0 or actual.get('sample_split')!='train'):
            output_errors.append('Canary is not the registered TRAIN-only one-update runtime check')
        metrics=actual.get('train_generation_metrics',{}) if case['phase']=='canary' else actual.get('test_generation',{}).get('correct',{})
        expected=contract['targets'][0]['denominator']
        counts=[metrics.get(k) for k in ('n_total','n_valid','n_failed')]
        if (any(not isinstance(x,int) or isinstance(x,bool) or x<0 for x in counts)
                or counts[0]!=expected or counts[1]+counts[2]!=counts[0]):
            output_errors.append('Evaluation counts do not cover the exact registered population')
        else:
            coverage=metrics.get('parse_valid_rate')
            if not isinstance(coverage,(int,float)) or isinstance(coverage,bool) or not math.isfinite(coverage) or abs(coverage-counts[1]/counts[0])>1e-12:
                output_errors.append('Parse coverage does not match literal counts')
            value=metrics.get('full_sample_mae')
            if counts[2] and value is not None:
                output_errors.append('Full MAE cannot be defined when any output fails')
            if not counts[2] and (not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value<0):
                output_errors.append('Complete population requires a finite full MAE')
        complete=not output_errors
    if output_errors: write(job/'output-validation-errors.json',output_errors)
    result=load(STANDARD/'result-record.template.json')
    result.update(is_template=False,task_id=TASK,run_id=contract['run_id'],contract_path=str(job/'task-contract.json'),
        contract_sha256=sha(job/'task-contract.json'),recorded_at=now(),lifecycle_status='succeeded' if complete else 'failed',
        reproduction_level=contract['execution_scope']['intended_level'] if receipt['performed'] else 'registered',scope_coverage='component_only')
    result['claims']=[dict(text='Literal matched-budget execution completed' if complete else 'Attempt incomplete; no successful scientific result claimed',
        evidence_type='actual_run' if complete else 'source_fact',evidence_paths=[str(job/'execution-receipt.json')],
        limitations='One component trial; not full paper or independent unseen confirmatory test.')]
    result['assets_integrity']={key:dict(status='complete',scope='exact frozen contract',evidence_paths=[str(job/'contract-validation.json')]) for key in ('datasets','code','models','environment')}
    result['execution'].update(performed=receipt['performed'],command=contract['execution']['command'],working_directory=str(ROOT),environment_path=sys.prefix,
        hostname=socket.gethostname(),resource_probe_path=str(job/'resource_probe.json'),resource_usage_path=str(job/'resource_usage.jsonl') if (job/'resource_usage.jsonl').exists() else None,
        seeds=[case['seed']],started_at=receipt['started_utc'],ended_at=receipt['ended_utc'],effective_deadline_at=contract['timing']['effective_deadline_at'],
        timeout_seconds=contract['execution']['timeout_seconds'],pid=receipt['pid'],pgid=receipt['pgid'],job_object_or_tree_control='Own process group; subreaper and kernel PDEATHSIG',
        exit_code=receipt['exit_code'],timed_out=receipt['termination_reason']=='timeout',termination_reason=receipt['termination_reason'],
        handles_absent_verified_at=receipt['ended_utc'] if receipt['process_group_absent'] else None,exit_verification_path=str(job/'execution-receipt.json'),
        stdout_path=str(job/'stdout.log'),stderr_path=str(job/'stderr.log'),receipt_path=str(job/'execution-receipt.json'))
    metrics=(actual.get('train_generation_metrics',{}) if case['phase']=='canary' else actual.get('test_generation',{}).get('correct',{})) if actual else {}
    target=contract['targets'][0]
    result['metric_comparisons']=[dict(**{k:target.get(k) for k in ('target_id','paper_endpoint','metric','metric_definition','statistical_unit','denominator','unit','paper_value')},
        actual_value=metrics.get('parse_valid_rate') if complete else None,baseline_value=None,absolute_difference=None,relative_difference=None,
        comparability='not_comparable',passed=None,reason=target['tolerance']['reason'],evidence_paths=[str(artifact if artifact.exists() else job/'stderr.log')])]
    result['artifacts']=[dict(role='literal_run_artifact',path=str(p),location='local',available=True,size_bytes=p.stat().st_size,sha256=sha(p))
                         for p in sorted(job.rglob('*')) if p.is_file() and p.name not in ('result-record.json','result-validation.json')]
    result['differences']=[dict(category='scope',description='New fixed-message depth/content/gate/pooling experiment; original results retained',
        effect_on_comparison='No full paper replication or matched total FLOPs claim',evidence_type='source_fact',evidence_paths=[str(job/'task-contract.json')])]
    result['incomplete_reasons']=[] if complete else [dict(code='execution_failed',description='; '.join(output_errors) or receipt['termination_reason'] or 'Nonzero exit or incomplete literal output',affected_targets=['parse_coverage'],evidence_paths=[str(job/'execution-receipt.json')])]
    result['review'].update(status='accepted',reviewer='mechanical contract/archive audit; scientific review separate',reviewed_at=now(),checks=[])
    write(job/'result-record.json',result)
    validation=validator().validate(contract,result,contract_bytes=(job/'task-contract.json').read_bytes(),check_files=True,contract_directory=job,result_directory=job)
    write(job/'result-validation.json',validation)
    if not validation['valid_for_acceptance']: raise RuntimeError('Terminal archive validation failed: '+json.dumps(validation['errors']))
    return complete


def not_started_receipt(job, reason):
    """An admission failure is archived without inventing a PID/exit code."""
    stamp=now()
    receipt=dict(started_utc=stamp,ended_utc=stamp,elapsed_seconds=0,pid=None,pgid=None,exit_code=None,
                 termination_reason=reason,exception=None,performed=False,peak_host_process_group_rss_bytes=0,
                 process_group_absent=True,stdout_sha256=sha(job/'stdout.log'),stderr_sha256=sha(job/'stderr.log'))
    write(job/'execution-receipt.json',receipt)
    return receipt


def worker(args):
    parent=int(os.environ['MATCHED_CONTROLLER_PID'])
    if os.getppid()!=parent: raise RuntimeError('Controller exited before worker setup')
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGTERM,0,0,0)!=0: raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG failed')
    if os.getppid()!=parent: raise RuntimeError('Controller exited during PDEATHSIG setup')
    protocol,_=verify_inputs(args.protocol,args.protocol_sha256,args.cache_manifest,args.cache_manifest_sha256)
    if 'torch' in sys.modules:
        raise RuntimeError('Torch imported before strict process-local environment setup')
    os.environ.update(runtime_environment(protocol,parent))
    import torch
    policy_receipt=apply_numerical_policy(torch,protocol['numerical_policy'])
    write(args.output.parent/'numerical_policy_actual.json',policy_receipt)
    torch.cuda.set_per_process_memory_fraction(protocol['resources']['torch_GPU_allocation_limit_bytes']/torch.cuda.get_device_properties(0).total_memory)
    case=load(args.case_file)
    script=Path(__file__).with_name('run_trial.py')
    command=[str(script),'--scenario',case['scenario'],'--seed',str(case['seed']),'--placement',case['placement'],
        '--sender-mode',case['sender_mode'],'--message-input',case['message_input'],'--gate-init',case['gate_init'],
        '--cache-manifest',str(args.cache_manifest),'--cache-manifest-sha256',args.cache_manifest_sha256,
        '--protocol',str(args.protocol),'--protocol-sha256',args.protocol_sha256,'--output',str(args.output),
        '--device','cuda','--threads','2','--timeout-seconds',str(args.timeout_seconds)]
    if case['phase']=='canary': command.append('--canary')
    sys.argv=command
    runpy.run_path(str(script),run_name='__main__')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/matched_budget_20261016.frozen.json')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--cache-manifest',type=Path,required=True)
    parser.add_argument('--cache-manifest-sha256')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--release-reference')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--case-file',type=Path)
    parser.add_argument('--timeout-seconds',type=int,default=1800)
    parser.add_argument('--source-suite',type=Path)
    parser.add_argument('--qualification-source-suite',type=Path)
    parser.add_argument('--fresh-canaries',action='store_true')
    args=parser.parse_args()
    if args.worker:
        worker(args);return
    if args.source_suite is None: raise ValueError('Original completed V1 source suite required')
    if os.name!='posix' or not sys.platform.startswith('linux'): raise RuntimeError('Supervisor requires bounded Linux WSL process groups')
    if not args.release_reference: raise ValueError('Explicit parent GPU release reference required')
    if args.output.exists(): raise FileExistsError('Existing suite refused; no automatic restart or overwrite')
    protocol_sha=args.protocol_sha256 or sha(args.protocol)
    manifest_sha=args.cache_manifest_sha256 or sha(args.cache_manifest)
    protocol,manifest=verify_inputs(args.protocol,protocol_sha,args.cache_manifest,manifest_sha)
    source_plan=admit_original_sources(args.source_suite,protocol,protocol_sha,manifest)
    deadline=datetime.fromisoformat(protocol['timing']['effective_deadline_at'])
    if datetime.now(timezone.utc)>=deadline: raise TimeoutError('Original matched-budget deadline expired')
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0: raise OSError(ctypes.get_errno(),'PR_SET_CHILD_SUBREAPER failed')
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    args.output.mkdir(parents=True)
    freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (args.output/'environment_freeze.txt').write_text(freeze.stdout,encoding='utf-8')
    try:
        qualified=admit_original_qualifications(args.qualification_source_suite or args.source_suite,protocol,protocol_sha,manifest,
            args.output/'environment_freeze.txt') if not args.fresh_canaries else {}
    except Exception as exc:
        write(args.output/'qualification_admission_failure.json',dict(utc=now(),new_execution_performed=False,
            first_substantive_started_at=protocol['timing']['first_substantive_started_at'],error_type=type(exc).__name__,error=str(exc)))
        raise
    state=dict(status='running',started_utc=now(),first_substantive_started_at=protocol['timing']['first_substantive_started_at'],
        effective_deadline_at=protocol['timing']['effective_deadline_at'],protocol_sha256=protocol_sha,cache_manifest_sha256=manifest_sha,
        controller_sha256=sha(__file__),GPU_release_reference=args.release_reference,jobs=[])
    state.update(original_completed_source_cases=source_plan,reused_TRAIN_qualifications=qualified,original_decoder='V1 batch1 exact unchanged',source_suite=str(args.source_suite),expected_new_training_trials=34)
    state['source_auditor_sha256']=sha(SOURCE_AUDITOR)
    write(args.output/'original_source_receipts.json',source_plan)
    write(args.output/'reused_qualification_receipts.json',qualified)
    cases=(canary_matrix(protocol) if args.fresh_canaries else [])+trial_matrix(protocol)[2:]
    write(args.output/'fixed_matrix.json',cases)
    try:
        for case in cases:
            if PENDING_SIGNAL is not None: raise KeyboardInterrupt('Supervisor stop requested before next job')
            if sha(__file__) != state['controller_sha256']: raise ValueError('Supervisor code changed after suite freeze')
            verify_inputs(args.protocol,protocol_sha,args.cache_manifest,manifest_sha)
            if sha(SOURCE_AUDITOR)!=state['source_auditor_sha256']: raise ValueError('Original source auditor changed')
            verify_reused_evidence(source_plan,qualified)
            job=args.output/case_id(case);job.mkdir(exist_ok=False)
            write(job/'case.json',case)
            for name in ('stdout.log','stderr.log'): (job/name).touch(exist_ok=False)
            try:
                probe=live_probe(job,protocol['resources'])
            except Exception as exc:
                probe=dict(utc=now(),hostname=socket.gethostname(),readiness='blocked',
                           blocking_reasons=['probe_error: '+type(exc).__name__+': '+str(exc)],scientific_validation_inferred=False)
            write(job/'resource_probe.json',probe)
            remaining=int((deadline-datetime.now(timezone.utc)).total_seconds())-10
            timeout=min(protocol['resources']['canary_timeout_seconds'] if case['phase']=='canary' else protocol['resources']['trial_timeout_seconds'],remaining)
            if timeout<30:
                probe['blocking_reasons'].append('Insufficient remaining original execution window')
                probe['readiness']='blocked';timeout=1
                write(job/'resource_probe.json',probe)
            command=build_command(case,job,args.protocol,protocol_sha,args.cache_manifest,manifest_sha,timeout)
            contract=build_contract(case,job,args.output,args.protocol,protocol,protocol_sha,args.cache_manifest,manifest_sha,probe,timeout,command,args.release_reference)
            write(job/'task-contract.json',contract)
            validation=validator().validate(contract,contract_bytes=(job/'task-contract.json').read_bytes(),check_files=True,contract_directory=job)
            write(job/'contract-validation.json',validation)
            if PENDING_SIGNAL is not None:
                receipt=not_started_receipt(job,'controller_signal_'+str(PENDING_SIGNAL))
                archive(job,contract,receipt)
                state['jobs'].append(dict(case=case,job_directory=str(job),complete=False,receipt=receipt))
                raise KeyboardInterrupt('Supervisor stop requested before spawn')
            if not validation['valid_for_execution'] or probe['blocking_reasons']:
                reason='admission_rejected: '+json.dumps(validation['errors']+validation['execution_blockers']+probe['blocking_reasons'])
                receipt=not_started_receipt(job,reason)
                archive(job,contract,receipt)
                state['jobs'].append(dict(case=case,job_directory=str(job),complete=False,receipt=receipt))
                raise RuntimeError(reason)
            env=dict(os.environ,**runtime_environment(protocol,os.getpid()))
            receipt=supervise(command,job,timeout,protocol['resources']['host_RSS_limit_bytes'],deadline,env)
            complete=archive(job,contract,receipt)
            state['jobs'].append(dict(case=case,job_directory=str(job),complete=complete,receipt=receipt))
            write(args.output/'suite_state.json',state)
            print(json.dumps(dict(event='terminal_job',job=job.name,complete=complete,elapsed_seconds=receipt['elapsed_seconds'])),flush=True)
            if not complete: raise RuntimeError('Trial/canary failed; preserved terminal archive; queued jobs stopped, no automatic retry')
        state['status']='complete'
    except BaseException as exc:
        state.update(status='failed_or_partial',exception=dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        state['ended_utc']=now();write(args.output/'suite_state.json',state)


if __name__=='__main__': main()
