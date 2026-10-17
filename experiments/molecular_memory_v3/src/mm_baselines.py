"""Contract-gated, stdlib numeric A1 references; no model/GPU dependencies.

TRAIN/validation use mm_data's separate labelled views. Test inputs contain no
gold; the original locked test gold is parsed only after model states and the
selection receipt have been written. Importing or planning performs no fitting.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import time

from . import mm_data

METHODS = ('train_mean', 'train_median', 'ridge52', 'knn52')
AA = 'ACDEFGHIKLMNPQRSTVWY'
SCHEMA = 'mm_numeric_baseline_request.v1'


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _digest(value):
    return hashlib.sha256(mm_data.canonical(value).encode('utf-8')).hexdigest()


def _relative(root, value, *, existing=True):
    root = Path(root).resolve(strict=True)
    rel = Path(value)
    if PurePosixPath(value).is_absolute() or '..' in rel.parts or '\\' in str(value) or ':' in str(value):
        raise ValueError('request paths must be project-relative POSIX paths')
    result = (root / rel).resolve(strict=existing)
    if not result.is_relative_to(root):
        raise ValueError('request path escapes the project')
    return result


def make_request(scenario, view_manifest_relative, view_manifest_sha256, output_relative,
                 methods=('train_mean', 'train_median'),
                 ridge_alphas=(.1, 1., 10., 100., 1000.), knn_ks=(1, 3, 5, 10, 25)):
    request = {'schema': SCHEMA, 'scenario': scenario,
               'view_manifest_relative': str(view_manifest_relative),
               'view_manifest_sha256': view_manifest_sha256,
               'output_relative': str(output_relative), 'methods': list(methods),
               'ridge_alphas': list(ridge_alphas), 'knn_ks': list(knn_ks)}
    _check_request(request)
    return request


def _check_request(request):
    required = {'schema', 'scenario', 'view_manifest_relative', 'view_manifest_sha256',
                'output_relative', 'methods', 'ridge_alphas', 'knn_ks'}
    if set(request) != required or request['schema'] != SCHEMA or request['scenario'] not in mm_data.SCENARIOS:
        raise ValueError('unregistered numeric request schema/scenario/fields')
    methods = request['methods']
    if (not isinstance(methods, list) or not methods or len(set(methods)) != len(methods)
            or any(m not in METHODS for m in methods)):
        raise ValueError('methods must be unique registered numeric references')
    for key in ('view_manifest_relative', 'output_relative'):
        value = request[key]
        if (not isinstance(value, str) or not value or PurePosixPath(value).is_absolute()
                or '..' in PurePosixPath(value).parts or '\\' in value or ':' in value):
            raise ValueError('request paths must be nonempty project-relative POSIX paths')
    digest = request['view_manifest_sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('view manifest requires an exact SHA256')
    alphas, ks = request['ridge_alphas'], request['knn_ks']
    if (not isinstance(alphas, list) or not alphas or any(not _finite(a) or a <= 0 for a in alphas)
            or len(set(alphas)) != len(alphas)):
        raise ValueError('ridge alpha grid must contain unique finite positive values')
    if (not isinstance(ks, list) or not ks or any(type(k) is not int or k < 1 for k in ks)
            or len(set(ks)) != len(ks)):
        raise ValueError('kNN grid must contain unique positive integers')


def execution_binding(request, project_root):
    """Add this mapping as contract['baseline_run_binding'] before standard review."""
    _check_request(request)
    return {'request_sha256': _digest(request),
            'request_canonical_json': mm_data.canonical(request),
            'runner_sha256': mm_data.file_sha(__file__),
            'data_module_sha256': mm_data.file_sha(mm_data.__file__),
            'output_directory': str(_relative(project_root, request['output_relative'], existing=False))}


def _load_views(request, project_root):
    _check_request(request)
    path = _relative(project_root, request['view_manifest_relative'])
    mm_data.verify_sha(path, request['view_manifest_sha256'])
    manifest = mm_data.strict_loads(path.read_text(encoding='utf-8'))
    scenario = request['scenario']
    if manifest.get('schema') != 'mm_a1_views_v1':
        raise ValueError('wrong prepared-view schema')
    receipt = manifest['scenarios'][scenario]
    if receipt['source_shas'] != mm_data.FROZEN_SHAS[scenario]:
        raise ValueError('prepared views do not bind the frozen scenario sources')
    # Hash-only upstream verification: no mixed-split source labels are parsed.
    for name, expected in receipt['source_shas'].items():
        original = mm_data.registered_path(project_root, mm_data.DATA_RELATIVE / scenario / name)
        mm_data.verify_sha(original, expected)
    splits = {split: mm_data.read_input_view(path.parent, scenario, split)
              for split in ('train', 'validation', 'test')}
    ids = [r['sample_id'] for rows in splits.values() for r in rows]
    if (any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids)
            or any(not _finite(r.get('pkd')) for split in ('train', 'validation') for r in splits[split])):
        raise ValueError('duplicate identity or invalid TRAIN/validation labels')
    if any('pkd' in row for row in splits['test']):
        raise ValueError('test inputs contain gold')
    return splits, receipt


def train_statistics(rows, scenario):
    """Pure TRAIN-only core; suitable for tiny fixtures, not an execution permit."""
    if scenario not in mm_data.SCENARIOS or not rows:
        raise ValueError('known scenario and nonempty TRAIN rows required')
    if any(r.get('split') != 'train' or not _finite(r.get('pkd'))
           or r.get('scenario', scenario) != scenario for r in rows):
        raise ValueError('statistics may consume only this scenario TRAIN labels')
    ids = [r['sample_id'] for r in rows]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError('unique TRAIN sample IDs required')
    values = [float(r['pkd']) for r in rows]
    try:
        total = math.fsum(values)
    except OverflowError as exc:
        raise ValueError('nonfinite TRAIN label sum') from exc
    if not _finite(total):
        raise ValueError('nonfinite TRAIN label sum')
    return {'scenario': scenario, 'train_count': len(values), 'train_sum': total,
            'train_mean': total / len(values), 'train_median': statistics.median(values),
            'train_ids': ids, 'train_ids_sha256': _digest(ids),
            'train_labels_sha256': _digest(values)}


def old52_features(row):
    """Old52 allowlist; no ESM, assay text, ID, split, DOI or answer features."""
    values = []
    for field in ('peptide_sequence', 'target_sequence'):
        seq = row[field]
        if not isinstance(seq, str) or not seq or not set(seq) <= set(AA):
            raise ValueError('old52 requires complete canonical sequences')
        values += [seq.count(aa) / len(seq) for aa in AA] + [math.log1p(len(seq))]
    chem, context = row.get('chemical_state') or {}, row.get('experimental_context') or {}
    for key, categories in [('n_terminal', ('free', 'acetyl', 'unknown')),
                            ('c_terminal', ('free', 'amide', 'unknown'))]:
        value = str(chem.get(key) or 'unknown').lower()
        values += [float(value == c) for c in categories]
    values += [float(context.get(k) is None) for k in ('temperature_kelvin', 'ph', 'buffer')]
    missing = context.get('actual_construct_sequence_missing', context.get('actual_construct_sequence') is None)
    if not isinstance(missing, bool):
        raise ValueError('actual construct missing indicator must be boolean')
    return values + [float(missing)]


def _standardize_fit(x):
    center = [math.fsum(col) / len(x) for col in zip(*x)]
    scale = [max(math.sqrt(math.fsum((row[j] - center[j]) ** 2 for row in x) / len(x)), 1e-6)
             for j in range(len(center))]
    return center, scale


def _transform(x, center, scale):
    return [[(v - m) / s for v, m, s in zip(row, center, scale)] for row in x]


def _ridge(x, y, alpha):
    """Cholesky solve with an unpenalized intercept; no adaptive jitter."""
    a = [row + [1.] for row in x]
    p = len(a[0])
    matrix = [[math.fsum(row[i] * row[j] for row in a) + (alpha if i == j and i < p - 1 else 0.)
               for j in range(p)] for i in range(p)]
    rhs = [math.fsum(row[i] * value for row, value in zip(a, y)) for i in range(p)]
    lower = [[0.] * p for _ in range(p)]
    for i in range(p):
        for j in range(i + 1):
            value = matrix[i][j] - math.fsum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if value <= 0 or not _finite(value):
                    raise ValueError('ridge normal equations are not finite positive definite')
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value / lower[j][j]
    forward = []
    for i in range(p):
        forward.append((rhs[i] - math.fsum(lower[i][j] * forward[j] for j in range(i))) / lower[i][i])
    result = [0.] * p
    for i in range(p - 1, -1, -1):
        result[i] = (forward[i] - math.fsum(lower[j][i] * result[j] for j in range(i + 1, p))) / lower[i][i]
    return result


def fit_reference(train_rows, scenario, method, parameter=None):
    """Pure tiny-fixture API; production fitting goes through run_baselines."""
    stats = train_statistics(train_rows, scenario)
    if method not in METHODS:
        raise ValueError('unknown reference')
    state = {'method': method, 'scenario': scenario, 'train_statistics': stats,
             'channel': 'numeric_predictor', 'generation_performed': False}
    if method in ('train_mean', 'train_median'):
        if parameter is not None:
            raise ValueError('constant references have no hyperparameter')
        state['constant'] = stats[method]
        state['capacity'] = {'stored_prediction_scalars': 1, 'neural_trainable_parameters': 0}
    else:
        x = [old52_features(row) for row in train_rows]
        y = [float(row['pkd']) for row in train_rows]
        center, scale = _standardize_fit(x)
        x = _transform(x, center, scale)
        state.update(center=center, scale=scale, parameter=parameter, feature_schema='old52_no_ESM_v1')
        if method == 'ridge52':
            if not _finite(parameter) or parameter <= 0:
                raise ValueError('positive ridge alpha required')
            state['coefficients'] = _ridge(x, y, parameter)
            state['capacity'] = {'nominal_coefficients_including_intercept': 53,
                                 'statistical_degrees_of_freedom': None}
        else:
            if type(parameter) is not int or parameter < 1:
                raise ValueError('positive integer k required')
            state.update(train_features=x, train_labels=y)
            state['capacity'] = {'stored_train_feature_values': 52 * len(x), 'stored_train_labels': len(y),
                                 'neural_trainable_parameters': 0}
    return state


def predict_reference(state, rows, scenario):
    if state.get('scenario') != scenario or state['train_statistics']['scenario'] != scenario:
        raise ValueError('cross-scenario TRAIN-state reuse is forbidden')
    method = state['method']
    if method in ('train_mean', 'train_median'):
        result = [state['constant']] * len(rows)
    else:
        x = _transform([old52_features(row) for row in rows], state['center'], state['scale'])
        if method == 'ridge52':
            result = [math.fsum(a * b for a, b in zip(row + [1.], state['coefficients'])) for row in x]
        elif method == 'knn52':
            result = []
            for row in x:
                distances = [math.fsum((a - b) ** 2 for a, b in zip(row, train))
                             for train in state['train_features']]
                nearest = sorted(range(len(distances)), key=lambda i: (distances[i], i))[:state['parameter']]
                weights = [1 / (distances[i] + 1e-6) for i in nearest]
                result.append(math.fsum(w * state['train_labels'][i] for w, i in zip(weights, nearest)) / math.fsum(weights))
        else:
            raise ValueError('unknown stored reference')
    if any(not _finite(v) for v in result):
        raise ValueError('nonfinite numeric prediction; no row dropping')
    return result


def _ranks(values):
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for i in ordered[start:end]:
            result[i] = (start + 1 + end) / 2
        start = end
    return result


def numeric_metrics(rows, predictions):
    if not rows or len(rows) != len(predictions) or any(not _finite(v) for v in predictions):
        raise ValueError('every labelled row requires one finite numeric prediction')
    gold = [r['pkd'] for r in rows]
    if any(not _finite(y) for y in gold):
        raise ValueError('finite exact gold required')
    errors = [abs(y - p) for y, p in zip(gold, predictions)]
    if any(not _finite(error) for error in errors):
        raise ValueError('numeric metric overflow; no row dropping')
    x, y = _ranks(gold), _ranks(predictions)
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    xc, yc = [v - mx for v in x], [v - my for v in y]
    denominator = math.hypot(*xc) * math.hypot(*yc)
    by_target = defaultdict(list)
    for row, error in zip(rows, errors):
        by_target[hashlib.sha256(row['target_sequence'].encode('ascii')).hexdigest()].append(error)
    return {'n_total': len(rows), 'n_finite': len(predictions), 'numeric_coverage': 1.,
            'micro_MAE': math.fsum(errors) / len(errors),
            'RMSE': math.hypot(*errors) / math.sqrt(len(errors)),
            'global_Spearman': max(-1., min(1., math.fsum(a * b for a, b in zip(xc, yc)) / denominator)) if denominator else None,
            'exact_target_macro_MAE': statistics.mean(math.fsum(v) / len(v) for v in by_target.values()),
            'n_exact_targets': len(by_target), 'unit': 'pKD',
            'coverage_semantics': 'finite numeric predictor output, not strict language generation'}


def _write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _authorize(request, root, contract_path, validator_path, validator_sha256):
    if not contract_path or not validator_path or not validator_sha256:
        raise ValueError('execution needs a standard task contract and pinned standard validator')
    contract_path, validator_path = Path(contract_path), Path(validator_path)
    mm_data.verify_sha(validator_path, validator_sha256)
    raw = contract_path.read_bytes()
    contract = mm_data.strict_loads(raw.decode('utf-8'))
    if contract.get('baseline_run_binding') != execution_binding(request, root):
        raise ValueError('standard contract does not bind this exact baseline request/code/output')
    spec = importlib.util.spec_from_file_location('mm_baseline_standard_validator', validator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checked = module.validate(contract, contract_bytes=raw, check_files=True,
                              contract_directory=contract_path.parent)
    if checked.get('valid_for_execution') is not True:
        raise ValueError('standard task contract rejected: ' + json.dumps(checked))
    deadline = datetime.fromisoformat(contract['timing']['effective_deadline_at'].replace('Z', '+00:00'))
    if deadline.tzinfo is None:
        raise ValueError('contract deadline must carry timezone')
    timeout = contract['execution']['timeout_seconds']
    if not _finite(timeout) or timeout <= 0:
        raise ValueError('positive contract timeout required')
    return {'contract_sha256': hashlib.sha256(raw).hexdigest(), 'validator_sha256': validator_sha256,
            'standard_validation': checked}, deadline, float(timeout)


def run_baselines(request, project_root, *, execute=False, contract_path=None,
                  validator_path=None, validator_sha256=None):
    """Plan by default. Explicit authorized execution produces new numeric artifacts."""
    root = Path(project_root).resolve(strict=True)
    _check_request(request)
    authorization, deadline, timeout = None, None, None
    if execute:
        authorization, deadline, timeout = _authorize(request, root, contract_path, validator_path, validator_sha256)
    splits, receipt = _load_views(request, root)
    plan = {'request': request, 'request_sha256': _digest(request), 'scenario': request['scenario'],
            'split_counts': {k: len(v) for k, v in splits.items()},
            'channel': 'numeric_predictor', 'generation_performed': False,
            'budget_group': 'unmatched_numeric_reference_not_M0_joint_model',
            'fit_performed': False, 'test_gold_parsed': False}
    if not execute:
        return plan
    start = time.monotonic()

    def check():
        if datetime.now(timezone.utc) >= deadline or time.monotonic() - start > timeout:
            raise TimeoutError('baseline contract deadline/runtime reached')

    check()
    output = _relative(root, request['output_relative'], existing=False)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'request.json', request)
    _write(output / 'authorization.json', authorization)
    selected, states = {}, {}
    try:
        for method in request['methods']:
            check()
            parameters = request['ridge_alphas'] if method == 'ridge52' else request['knn_ks'] if method == 'knn52' else [None]
            trials = []
            for parameter in parameters:
                check()
                state = fit_reference(splits['train'], request['scenario'], method, parameter)
                pred = predict_reference(state, splits['validation'], request['scenario'])
                trials.append((numeric_metrics(splits['validation'], pred)['micro_MAE'], parameter, state))
            best = min(range(len(trials)), key=lambda i: trials[i][0])  # First declared tie.
            _, parameter, state = trials[best]
            _write(output / (method + '.state.json'), state)
            states[method] = state
            selected[method] = {'parameter': parameter,
                                'validation_trials': [{'parameter': p, 'micro_MAE': score} for score, p, _ in trials],
                                'state_sha256': mm_data.file_sha(output / (method + '.state.json'))}
        _write(output / 'selection_before_test.json', {'selection': 'validation_micro_MAE_only_first_declared_tie',
               'scenario': request['scenario'], 'models': selected, 'test_gold_parsed': False,
               'model_family_selected_using_test': False})
        # Only now parse original test gold, after the fixed states/choices exist.
        check()
        gold_path = mm_data.registered_path(root, mm_data.DATA_RELATIVE / request['scenario'] / 'test_gold.jsonl')
        mm_data.verify_sha(gold_path, receipt['source_shas']['test_gold.jsonl'])
        gold = mm_data.read_jsonl(gold_path)
        if [r['sample_id'] for r in gold] != [r['sample_id'] for r in splits['test']]:
            raise ValueError('original test gold ID/order differs from input views')
        test = [{**row, 'pkd': g['pkd']} for row, g in zip(splits['test'], gold)]
        results = {}
        for method, state in states.items():
            check()
            results[method] = {}
            for split, rows in [('validation', splits['validation']), ('test', test)]:
                pred = predict_reference(state, rows, request['scenario'])
                metrics = numeric_metrics(rows, pred)
                with (output / (method + '.' + split + '.predictions.jsonl')).open('x', encoding='utf-8', newline='\n') as stream:
                    for row, value in zip(rows, pred):
                        stream.write(mm_data.canonical({'sample_id': row['sample_id'], 'split': split,
                            'scenario': request['scenario'], 'gold_pkd': row['pkd'], 'prediction_pkd': value,
                            'channel': 'numeric_predictor', 'generation_performed': False}) + '\n')
                results[method][split] = metrics
        _load_views(request, root)  # Reverify data pins at exit.
        check()
        result = {**plan, 'status': 'complete', 'fit_performed': True, 'test_gold_parsed': True,
                  'models': results, 'selection': selected, 'authorization': authorization,
                  'elapsed_seconds': time.monotonic() - start,
                  'artifact_sha256': {p.name: mm_data.file_sha(p) for p in output.iterdir() if p.is_file()}}
        _write(output / 'result.json', result)
        return result
    except Exception as exc:
        _write(output / 'failure.json', {'status': 'failed', 'error': type(exc).__name__, 'message': str(exc),
                                         'partial_artifacts_retained': True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--request', required=True)
    parser.add_argument('--execute', action='store_true', help='requires standard reviewed contract; default plans only')
    parser.add_argument('--contract')
    parser.add_argument('--validator')
    parser.add_argument('--validator-sha256')
    args = parser.parse_args()
    request = mm_data.strict_loads(Path(args.request).read_text(encoding='utf-8'))
    result = run_baselines(request, args.project_root, execute=args.execute, contract_path=args.contract,
                           validator_path=args.validator, validator_sha256=args.validator_sha256)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
