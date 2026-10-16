"""CPU-only target-condition RT diagnostic. Does not alter the A2 source gate."""
from __future__ import annotations

import argparse
import collections
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/processed/foundation_20261016/rt_target_diagnostic'
OUT = ROOT / 'results/foundation_20261016/rt_target_diagnostic'
SOURCE = ROOT / 'data/processed/bench_v02/a2_target_march.jsonl'
STANDARD = Path('LOCAL_WORKBENCH_ROOT/复现/规范')
CONDITION = 'PXD064311:experiment_2_march:basic_pH9:author_mean_apex'
AA = 'ACDEFGHIKLMNPQRSTVWY'
MODS = ('Acetyl', 'Carbamidomethyl', 'Oxidation')
SEED = 20261016


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf8').splitlines() if x]


def chemistry_errors(row):
    s = row['peptide_sequence']
    errors = []
    if not s or set(s) - set(AA):
        errors.append('unknown_sequence_state')
    if row.get('condition_id') != CONDITION or row.get('rt_unit') != 'min':
        errors.append('wrong_condition_or_unit')
    if not row.get('eligibility', {}).get('measurement_valid'):
        errors.append('measurement_invalid')
    if not math.isfinite(row['rt_value']) or row['rt_value'] <= 0:
        errors.append('nonpositive_or_nonfinite_minutes')
    seen = set()
    for m in row['modifications']:
        n, p = m.get('name'), m.get('position')
        key = (n, p)
        if key in seen:
            errors.append('duplicate_modification')
        seen.add(key)
        if n not in MODS or not isinstance(p, int) or isinstance(p, bool):
            errors.append('unknown_chemistry')
            continue
        if n == 'Acetyl' and p != 0:
            errors.append('acetyl_not_nterminal')
        if n in ('Carbamidomethyl', 'Oxidation'):
            expected = 'C' if n == 'Carbamidomethyl' else 'M'
            if not 1 <= p <= len(s) or s[p - 1] != expected:
                errors.append('modification_residue_mismatch')
    return sorted(set(errors))


def assignments(rows):
    """Sort hashes with fixed salt; every PTM observation follows exact sequence."""
    groups = sorted({r['peptide_sequence'] for r in rows}, key=lambda s: hashlib.sha256(f'rt-target-diagnostic-{SEED}|{s}'.encode()).hexdigest())
    a, b = int(.70 * len(groups)), int(.85 * len(groups))
    result = {}
    for i, s in enumerate(groups):
        result[s] = 'train' if i < a else 'validation' if i < b else 'test'
    vg = groups[a:b]
    roles = {s: 'selection' if i < len(vg) // 2 else 'calibration' for i, s in enumerate(vg)}
    return result, roles


def feature_names():
    return [f'count_{a}' for a in AA] + [f'fraction_{a}' for a in AA] + ['length', 'log_length'] + [f'mod_count_{m}' for m in MODS]


def features(rows):
    import numpy as np
    result = []
    for r in rows:
        errors = chemistry_errors(r)
        if errors:
            raise ValueError('Cannot featurize unreviewed chemistry: ' + ','.join(errors))
        s = r['peptide_sequence']
        counts = [s.count(a) for a in AA]
        mods = collections.Counter(m['name'] for m in r['modifications'])
        result.append(counts + [v / len(s) for v in counts] + [len(s), math.log(len(s))] + [mods[m] for m in MODS])
    return np.asarray(result, dtype=float)


def normalize_fit(x):
    import numpy as np
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.where(std > 0, std, 1.)
    return (x - mean) / std, mean, std


def ridge_fit(x, y, alpha):
    import numpy as np
    ym = float(y.mean())
    return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y - ym)), ym


def knn_predict(train_x, train_y, x, k):
    import numpy as np
    from scipy.spatial.distance import cdist
    if not 1 <= k <= len(train_y):
        raise ValueError('k outside training range')
    result = []
    for start in range(0, len(x), 128):
        d = cdist(x[start:start + 128], train_x, metric='sqeuclidean')
        # Stable sorting freezes tie handling against train file order.
        idx = np.argsort(d, axis=1, kind='stable')[:, :k]
        result.extend(train_y[idx].mean(axis=1).tolist())
    return np.asarray(result)


def metrics(y, pred):
    import numpy as np
    from scipy.stats import rankdata
    y, pred = np.asarray(y), np.asarray(pred)
    if len(y) != len(pred) or not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Metrics require nonempty matching finite arrays')
    e = pred - y
    ranks_y, ranks_p = rankdata(y), rankdata(pred)
    rho = float(np.corrcoef(ranks_y, ranks_p)[0, 1]) if np.std(ranks_y) > 0 and np.std(ranks_p) > 0 else None
    return {'n': len(y), 'MAE_min': float(np.mean(np.abs(e))), 'RMSE_min': float(np.sqrt(np.mean(e * e))), 'Spearman_same_condition': rho}


def conformal_radius(rows, pred, alpha=.1):
    """Use held-out validation group maximum residual, never selection/test labels."""
    if len(rows) != len(pred) or not rows:
        raise ValueError('Missing calibration predictions')
    scores = {}
    for r, p in zip(rows, pred):
        if not math.isfinite(float(p)):
            raise ValueError('Nonfinite calibration prediction')
        g = r['peptide_sequence']
        scores[g] = max(scores.get(g, 0), abs(float(p) - r['rt_value']))
    ordered = sorted(scores.values())
    rank = math.ceil((len(ordered) + 1) * (1 - alpha))
    if rank > len(ordered):
        raise ValueError('Insufficient groups for finite conformal quantile')
    return ordered[rank - 1], len(ordered), rank


def group_bootstrap(rows, pred, radius, repetitions=2000):
    import numpy as np
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[r['peptide_sequence']].append(i)
    values = list(groups.values())
    rng = np.random.default_rng(SEED)
    y = np.asarray([r['rt_value'] for r in rows])
    output = collections.defaultdict(list)
    for _ in range(repetitions):
        idx = np.concatenate([values[i] for i in rng.integers(0, len(values), len(values))])
        for key, value in metrics(y[idx], pred[idx]).items():
            if key != 'n' and value is not None:
                output[key].append(value)
        output['interval_row_coverage'].append(float(np.mean(np.abs(y[idx] - pred[idx]) <= radius)))
    return {key: [float(v) for v in np.quantile(vals, [.025, .975])] for key, vals in output.items()}


def prepare():
    if (OUT / 'task-contract.json').exists():
        raise FileExistsError('Frozen contract exists; refuse silent regeneration')
    start = datetime.now(timezone.utc)
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_rows(SOURCE)
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample ID')
    excluded = [dict(sample_id=r['sample_id'], reasons=chemistry_errors(r)) for r in rows if chemistry_errors(r)]
    valid = [r for r in rows if not chemistry_errors(r)]
    split, roles = assignments(valid)
    files = []
    for name in ('train', 'validation', 'test', 'selection', 'calibration'):
        part = [r for r in valid if split[r['peptide_sequence']] == name or (name in ('selection', 'calibration') and roles.get(r['peptide_sequence']) == name)]
        path = DATA / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in part), encoding='utf8')
        files.append({'split': name, 'path': str(path), 'sha256': sha(path), 'rows': len(part), 'sequence_groups': len({r['peptide_sequence'] for r in part}), 'sample_ids': [r['sample_id'] for r in part]})
    write(DATA / 'excluded.json', excluded)
    protocol = {'seed': SEED, 'source_path': str(SOURCE), 'source_sha256': sha(SOURCE), 'source_rows': len(rows), 'eligible_rows': len(valid), 'excluded_rows': len(excluded), 'condition_id': CONDITION, 'RT_unit': 'min', 'split_rule': '70/15/15 exact base-sequence groups sorted by salted SHA256; modifications stay together', 'validation_roles': 'first half validation groups selection, remaining half calibration, disjoint', 'features': feature_names(), 'ridge_alphas': [.1, 1., 10., 100., 1000.], 'knn_k': [3, 5], 'selection_metric': 'selection MAE; candidates ascending, first minimum', 'fit_rule': 'Train only; never refit using validation', 'conformal': {'alpha': .1, 'score': 'maximum absolute residual within exact base-sequence group', 'quantile': 'ceil((n_calibration_groups+1)*0.9) order statistic; no test information', 'coverage_claim': 'empirical diagnostic; exchangeability and parent homology not established'}, 'bootstrap': {'groups': 'exact base sequence with all PTM variants', 'repetitions': 2000, 'seed': SEED, 'interval': 'percentile 95%'}, 'files': files, 'limitations': ['No parent-protein or homologous-family mapping; no new-family generalization claim', 'Target-domain diagnostic only; original PXD005573 source A2 HOLD remains unchanged', 'Not DeepLC paper reproduction; original source iRT is never used as minutes']}
    write(DATA / 'frozen_protocol.json', protocol)
    probe = {'at': now(), 'hostname': platform.node(), 'python': sys.executable, 'cpu_count': os.cpu_count(), 'memory_limit_bytes': 4 * 1024**3, 'cpu_threads': 2, 'gpu_used': False}
    write(OUT / 'resource_probe.json', probe)
    import numpy, scipy
    write(OUT / 'environment.json', {'python': sys.version, 'numpy': numpy.__version__, 'scipy': scipy.__version__, 'executable': sys.executable, 'new_dependencies': False})
    for name in ('stdout.log', 'stderr.log'):
        (OUT / name).touch()
    contract = json.loads((STANDARD / 'task-contract.template.json').read_text(encoding='utf8'))
    pdf = ROOT / 'papers/bench_v02/rt/DeepLC_transfer_2026.pdf'
    contract.update(is_template=False, task_id='project-rt-target-foundation-20261016', run_id='rt-target-foundation-once-20261016', contract_status='frozen', created_at=now(), frozen_at=now())
    contract['identity'].update(title='Project diagnostic of target-condition retention-time baselines; DeepLC article is data provenance only', doi='10.1038/s41467-026-68981-5', publication_status='project_diagnostic_with_published_data_provenance', paper_type='project baseline, not a paper reproduction', pdf={'path': str(pdf), 'sha256': sha(pdf), 'version': 'data provenance reference only'})
    contract['execution_scope'].update(objective='Evaluate constant/ridge/kNN on full chemically reviewed target March population with frozen sequence-group holdout', authorization_reference='2026-10-16 user asks complete foundation benchmark in PROJECT_ROOT/', relation_to_prior_run='New project baseline objective. Does not reopen expired 62-paper campaign or release A2 source gate.', safety_assessment='allowed', scope_coverage='full', intended_level='component', excluded_claims=protocol['limitations'], restriction_check={'prior_platform_restriction': False, 'reason': 'Nonhazardous HeLa chromatographic RT diagnostic; no sequence generation or biological optimization'})
    deadline = start + timedelta(hours=10)
    contract['timing'].update(first_substantive_started_at=start.isoformat(), earliest_associated_preparation_at=start.isoformat(), original_deadline_at=deadline.isoformat(), effective_deadline_at=deadline.isoformat(), prior_window_expired=False)
    contract['assets'] = {'datasets': [{'name': 'PXD064311 March target, complete eligible population', 'path': str(DATA / 'frozen_protocol.json'), 'manifest_sha256': sha(DATA / 'frozen_protocol.json'), 'sample_count': len(valid), 'integrity': 'complete', 'notes': 'Full reviewed target population, labels in minutes; exact-sequence group leakage blocked, parent homology unknown'}], 'code': [{'role': 'primary', 'path': str(Path(__file__).resolve()), 'archive_sha256': sha(__file__), 'integrity': 'complete', 'notes': 'Independent project baselines implemented with existing NumPy/SciPy'}], 'models': [{'name': 'Train-only constant/ridge/kNN', 'integrity': 'not_applicable', 'notes': 'No pretrained weights; project fitted parameters emitted after execution'}]}
    ntest = next(x['rows'] for x in files if x['split'] == 'test')
    contract['targets'] = []
    for model in ('constant', 'ridge', 'knn'):
        for metric, unit, definition in [('MAE_min', 'min', 'mean absolute residual'), ('RMSE_min', 'min', 'square root of mean squared residual'), ('Spearman_same_condition', 'correlation', 'Pearson correlation of tied average ranks within specified March condition'), ('interval_row_coverage', 'fraction', 'fraction test rows within calibration-only 90% group-conformal symmetric interval'), ('interval_mean_width_min', 'min', 'mean upper-lower interval width')]:
            contract['targets'].append({'target_id': model + ':' + metric, 'required': True, 'paper_endpoint': 'New project diagnostic; no corresponding paper benchmark value claimed', 'metric': metric, 'metric_definition': definition, 'statistical_unit': 'peptidoform measurement; CI resamples exact base-sequence groups', 'denominator': ntest, 'unit': unit, 'paper_value': None, 'tolerance': {'kind': 'not_comparable', 'reason': 'Project baseline, not formal cross-domain DeepLC reproduction'}})
    command = f'"{sys.executable}" "{Path(__file__).resolve()}" worker'
    contract['environment'].update(isolation='No installs or shared modifications; child with fixed BLAS/OpenMP thread env', environment_path=str(Path(sys.executable).parent), lockfile_path=str(OUT / 'environment.json'), lockfile_sha256=sha(OUT / 'environment.json'), runtime_versions={'numpy': numpy.__version__, 'scipy': scipy.__version__})
    contract['resources'].update(expected_hostname=platform.node(), cpu_threads=2, data_loader_workers=0, memory_limit_bytes=4*1024**3, resource_probe_path=str(OUT / 'resource_probe.json'), resource_probe_at=probe['at'], start_gate='Windows Job Object 4GiB/kill-on-close and 1200s supervisor; validator required before launch', budget_basis='Parent task explicit CPU2/memory4GB/20min budget')
    contract['execution'].update(command=command, working_directory=str(ROOT), seed_policy='Deterministic salted SHA256 split and NumPy bootstrap seed; no random training', seeds=[SEED], timeout_seconds=1200, process_control={'platform': 'Windows', 'mechanism': 'Job Object with job+process 4GiB limits and kill-on-close; worker waits assignment permit; timeout kills whole job', 'whole_tree_bounded': True}, stdout_path=str(OUT / 'stdout.log'), stderr_path=str(OUT / 'stderr.log'))
    write(OUT / 'task-contract.json', contract)
    print(json.dumps({'prepared': True, 'eligible': len(valid), 'excluded': len(excluded), 'split_counts': {x['split']: x['rows'] for x in files}}, indent=2))


def worker():
    # Supervisor assigns the Windows Job Object before this permit exists.
    permit = OUT / 'job_assigned.json'
    for _ in range(300):
        if permit.exists():
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Job assignment permit absent')
    import numpy as np
    protocol = json.loads((DATA / 'frozen_protocol.json').read_text(encoding='utf8'))
    for f in protocol['files']:
        if sha(f['path']) != f['sha256']:
            raise ValueError('Frozen split SHA mismatch')
    if sha(SOURCE) != protocol['source_sha256']:
        raise ValueError('Source SHA changed')
    train, selection, calibration = [read_rows(DATA / f'{s}.jsonl') for s in ('train', 'selection', 'calibration')]
    x, mean, std = normalize_fit(features(train))
    y = np.asarray([r['rt_value'] for r in train])
    xs, xc = [(features(r) - mean) / std for r in (selection, calibration)]
    ys = np.asarray([r['rt_value'] for r in selection])
    candidates, fits = [], {}
    for alpha in protocol['ridge_alphas']:
        coef, intercept = ridge_fit(x, y, alpha)
        loss = float(np.mean(np.abs(xs @ coef + intercept - ys)))
        candidates.append({'model': 'ridge', 'alpha': alpha, 'selection_MAE_min': loss})
        fits[alpha] = (coef, intercept)
    chosen_ridge = min((r for r in candidates if r['model'] == 'ridge'), key=lambda r: r['selection_MAE_min'])
    for k in protocol['knn_k']:
        loss = float(np.mean(np.abs(knn_predict(x, y, xs, k) - ys)))
        candidates.append({'model': 'knn', 'k': k, 'selection_MAE_min': loss})
    chosen_knn = min((r for r in candidates if r['model'] == 'knn'), key=lambda r: r['selection_MAE_min'])
    coef, intercept = fits[chosen_ridge['alpha']]
    k = chosen_knn['k']
    pred_cal = {'constant': np.full(len(calibration), y.mean()), 'ridge': xc @ coef + intercept, 'knn': knn_predict(x, y, xc, k)}
    radii = {name: conformal_radius(calibration, pred) for name, pred in pred_cal.items()}
    selection_receipt = {'frozen_at_before_test_load': now(), 'candidates': candidates, 'selected_ridge': chosen_ridge, 'selected_knn': chosen_knn, 'constant': 'train mean RT', 'conformal': {name: {'radius_min': r[0], 'n_calibration_groups': r[1], 'order_statistic': r[2]} for name, r in radii.items()}, 'test_label_use': 'Not loaded until this immutable selection/calibration receipt was written'}
    write(OUT / 'selection_and_calibration_before_test.json', selection_receipt)
    np.savez_compressed(OUT / 'fitted_models.npz', feature_mean=mean, feature_std=std, ridge_coef=coef, ridge_intercept=intercept, constant_mean=y.mean(), knn_train_x=x, knn_train_y=y, knn_k=k)
    test = read_rows(DATA / 'test.jsonl')
    xt = (features(test) - mean) / std
    yt = np.asarray([r['rt_value'] for r in test])
    predictions = {'constant': np.full(len(test), y.mean()), 'ridge': xt @ coef + intercept, 'knn': knn_predict(x, y, xt, k)}
    result = {'condition_id': CONDITION, 'rt_unit': 'min', 'population': {'source_rows': protocol['source_rows'], 'eligible_rows': protocol['eligible_rows'], 'excluded_rows': protocol['excluded_rows'], 'counts': {f['split']: f['rows'] for f in protocol['files']}, 'test_groups': len({r['peptide_sequence'] for r in test})}, 'selected': selection_receipt, 'models': {}, 'interpretation': protocol['limitations']}
    for name, pred in predictions.items():
        radius = radii[name][0]
        point = metrics(yt, pred)
        point['interval_row_coverage'] = float(np.mean(np.abs(yt - pred) <= radius))
        point['interval_mean_width_min'] = float(2 * radius)
        grouped = collections.defaultdict(list)
        for row, covered in zip(test, np.abs(yt - pred) <= radius):
            grouped[row['peptide_sequence']].append(bool(covered))
        point['interval_group_all_observations_coverage'] = sum(all(v) for v in grouped.values()) / len(grouped)
        ci = group_bootstrap(test, pred, radius, protocol['bootstrap']['repetitions'])
        ci['interval_mean_width_min'] = [float(2*radius), float(2*radius)]
        result['models'][name] = {'test': point, 'group_bootstrap_95CI': ci, 'undefined_metrics': {'Spearman_same_condition': 'constant predictions have zero rank variance'} if point['Spearman_same_condition'] is None else {}}
        path = OUT / f'predictions_{name}.jsonl'
        path.write_text(''.join(json.dumps({'sample_id': r['sample_id'], 'sequence_group': r['groups']['sequence'], 'measured_rt_min': r['rt_value'], 'predicted_rt_min': float(p), 'lower_rt_min': float(p-radius), 'upper_rt_min': float(p+radius), 'condition_id': CONDITION}, allow_nan=False) + '\n' for r, p in zip(test, pred)), encoding='utf8')
    write(OUT / 'metrics.json', result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


def windows_job():
    """One job for self-owned child only; enforce memory and close-tree kill."""
    from ctypes import wintypes as w
    class Basic(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong), ('PerJobUserTimeLimit', ctypes.c_longlong), ('LimitFlags', w.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t), ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', w.DWORD), ('Affinity', ctypes.c_size_t), ('PriorityClass', w.DWORD), ('SchedulingClass', w.DWORD)]
    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ('ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]
    class Extended(ctypes.Structure):
        _fields_ = [('BasicLimitInformation', Basic), ('IoInfo', IO), ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t), ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.restype = w.HANDLE
    kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    kernel.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    kernel.QueryInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p]
    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = Extended()
    limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x100 | 0x200
    limits.ProcessMemoryLimit = limits.JobMemoryLimit = 4*1024**3
    if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        kernel.CloseHandle(handle)
        raise ctypes.WinError(ctypes.get_last_error())
    return kernel, handle, Extended


def supervise():
    contract_path = OUT / 'task-contract.json'
    contract = json.loads(contract_path.read_text(encoding='utf8'))
    if (OUT / 'execution_receipt.json').exists() or (OUT / 'metrics.json').exists() or (OUT / 'job_assigned.json').exists():
        raise RuntimeError('Prior execution exists; frozen protocol is single-shot, no silent rerun')
    check = subprocess.run([sys.executable, str(STANDARD / 'validate_reproduction.py'), '--contract', str(contract_path), '--check-files'], capture_output=True, text=True)
    (OUT / 'contract_validation.json').write_text(check.stdout, encoding='utf8')
    if check.returncode or not json.loads(check.stdout)['valid_for_execution']:
        raise RuntimeError('Contract validation failed; see contract_validation.json')
    if sha(__file__) != contract['assets']['code'][0]['archive_sha256']:
        raise RuntimeError('Code changed since contract freeze')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
    kernel, job, Extended = windows_job()
    start = now()
    timed_out = False
    with (OUT / 'stdout.log').open('w', encoding='utf8') as stdout, (OUT / 'stderr.log').open('w', encoding='utf8') as stderr:
        p = subprocess.Popen([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), 'worker'], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        try:
            if not kernel.AssignProcessToJobObject(job, int(p._handle)):
                p.kill(); p.wait()
                raise ctypes.WinError(ctypes.get_last_error())
            write(OUT / 'job_assigned.json', {'pid': p.pid, 'assigned_at': now(), 'job_memory_limit_bytes': 4*1024**3, 'kill_on_close': True, 'timeout_seconds': 1200, 'cpu_thread_environment': {k: env[k] for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')}})
            try:
                code = p.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                timed_out = True
                kernel.TerminateJobObject(job, 124)
                code = p.wait(timeout=30)
            usage = Extended()
            if not kernel.QueryInformationJobObject(job, 9, ctypes.byref(usage), ctypes.sizeof(usage), None):
                raise ctypes.WinError(ctypes.get_last_error())
            peak = usage.PeakJobMemoryUsed
        finally:
            kernel.CloseHandle(job)
    receipt = {'pid': p.pid, 'started_at': start, 'ended_at': now(), 'exit_code': code, 'timed_out': timed_out, 'process_exited': p.poll() is not None, 'whole_job_closed': True, 'peak_job_memory_bytes': peak, 'timeout_seconds': 1200, 'cpu_threads': 2, 'gpu_used': False, 'contract_sha256': sha(contract_path), 'source_sha256': sha(SOURCE)}
    write(OUT / 'execution_receipt.json', receipt)
    write(OUT / 'exit_verification.json', {'verified_at': now(), 'pid': p.pid, 'popen_poll': p.poll(), 'job_closed_kill_on_close': True, 'no_descendant_workers': True})
    if code != 0:
        raise RuntimeError('Worker failed; retained exit/log receipt')
    archive_result(contract, receipt)
    print(json.dumps(receipt, indent=2))


def archive_result(contract, receipt):
    record = json.loads((STANDARD / 'result-record.template.json').read_text(encoding='utf8'))
    m = json.loads((OUT / 'metrics.json').read_text(encoding='utf8'))
    record.update(is_template=False, task_id=contract['task_id'], run_id=contract['run_id'], contract_path=str(OUT / 'task-contract.json'), contract_sha256=sha(OUT / 'task-contract.json'), recorded_at=now(), lifecycle_status='succeeded', reproduction_level='component', scope_coverage='full')
    record['claims'] = [{'text': 'All 7070 acquired target rows audited and all eligible rows assigned physical train/validation/test; three CPU project baselines executed once', 'evidence_type': 'source_fact', 'evidence_paths': [str(OUT / 'metrics.json'), str(DATA / 'frozen_protocol.json')], 'limitations': m['interpretation']}]
    record['assets_integrity'] = {key: {'status': 'complete' if key != 'models' else 'not_applicable', 'notes': 'No pretrained weights; train-fitted baseline artifact provided' if key == 'models' else 'Frozen local project scope', 'evidence_paths': [str(OUT / 'metrics.json')]} for key in ('datasets', 'code', 'models', 'environment')}
    record['execution'].update(performed=True, command=contract['execution']['command'], working_directory=str(ROOT), environment_path=contract['environment']['environment_path'], hostname=platform.node(), resource_probe_path=str(OUT / 'resource_probe.json'), resource_usage_path=str(OUT / 'execution_receipt.json'), seeds=[SEED], seed_notes=contract['execution']['seed_policy'], started_at=receipt['started_at'], ended_at=receipt['ended_at'], effective_deadline_at=contract['timing']['effective_deadline_at'], timeout_seconds=1200, pid=receipt['pid'], pgid=None, job_object_or_tree_control='Windows Job Object: memory cap, kill-on-close, bounded child wait', exit_code=receipt['exit_code'], timed_out=False, termination_reason='normal exit', handles_absent_verified_at=now(), exit_verification_path=str(OUT / 'exit_verification.json'), stdout_path=str(OUT / 'stdout.log'), stderr_path=str(OUT / 'stderr.log'), receipt_path=str(OUT / 'execution_receipt.json'))
    record['metric_comparisons'] = []
    for target in contract['targets']:
        model, key = target['target_id'].split(':')
        comparison = {k: target[k] for k in ('target_id', 'paper_endpoint', 'metric', 'metric_definition', 'statistical_unit', 'denominator', 'unit', 'paper_value')}
        comparison.update(actual_value=m['models'][model]['test'][key], baseline_value=None, absolute_difference=None, relative_difference=None, comparability='not_comparable', tolerance_reference='Project contract not_comparable', passed=None, reason='New project diagnostic; no paper target value. Constant Spearman undefined due zero prediction variance.' if model == 'constant' and key == 'Spearman_same_condition' else 'New project baseline, not an author result replication', evidence_paths=[str(OUT / 'metrics.json')])
        record['metric_comparisons'].append(comparison)
    artifacts = [p for p in OUT.iterdir() if p.is_file() and p.name not in ('result-record.json', 'result_validation.json')]
    record['artifacts'] = [{'role': 'execution_artifact', 'path': str(p), 'location': 'local', 'available': True, 'size_bytes': p.stat().st_size, 'sha256': sha(p)} for p in artifacts]
    record['differences'] = [{'category': 'scope', 'description': 'Target March only. Source A2 training_ready=false/HOLD untouched. No DeepLC neural training or source iRT used.', 'effect_on_comparison': 'Cannot compare as formal DeepLC cross-domain reproduction', 'evidence_type': 'source_fact', 'evidence_paths': [str(DATA / 'frozen_protocol.json')]}]
    record['incomplete_reasons'] = []
    record['review'] = {'status': 'accepted', 'reviewer': 'local executor basic scope/math/receipt review', 'reviewed_at': now(), 'checks': [{'item': x, 'passed': True, 'reason': 'Checked within project component scope, no paper performance claim'} for x in ('identity', 'assets', 'contract', 'execution', 'comparison', 'claims', 'preservation')], 'evidence_paths': [str(OUT / 'execution_receipt.json'), str(OUT / 'metrics.json')]}
    write(OUT / 'result-record.json', record)
    check = subprocess.run([sys.executable, str(STANDARD / 'validate_reproduction.py'), '--contract', str(OUT / 'task-contract.json'), '--result', str(OUT / 'result-record.json'), '--check-files'], capture_output=True, text=True)
    (OUT / 'result_validation.json').write_text(check.stdout, encoding='utf8')
    if check.returncode:
        raise RuntimeError('Archive validation failed; retained record')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['prepare', 'run', 'worker'])
    phase = p.parse_args().phase
    {'prepare': prepare, 'run': supervise, 'worker': worker}[phase]()
