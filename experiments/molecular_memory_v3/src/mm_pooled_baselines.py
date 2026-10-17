"""Fixed pooled-cache numeric references; default prepare only, never model forward.

NumPy is an existing environment dependency. Torch is used only inside the
already-qualified SequenceStore to read weights_only CPU cache tensors.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np

from . import mm_baselines as scalar
from . import mm_data as data
from . import mm_extract as extract

SCHEMA = 'mm_pooled_numeric_baseline_request.v1'
METHODS = ('ridge2700_alpha1000', 'knn2700_k25')
DIMENSION = 2700
FLOOR = 1e-6


def digest(value):
    return hashlib.sha256(data.canonical(value).encode('utf-8')).hexdigest()


def _require(value, message):
    if not value: raise ValueError(message)


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


def _load(path):
    return data.strict_loads(Path(path).read_text(encoding='utf-8-sig'))


def _check_config(config):
    _require(set(config) == {'schema', 'scenarios', 'methods', 'features', 'training', 'output_relative'}, 'unexpected config fields')
    _require(config['schema'] == 'mm_pooled_numeric_baseline_config.v1' and config['scenarios'] == list(data.SCENARIOS)
             and config['methods'] == list(METHODS), 'unregistered baseline schema/scenarios/methods')
    expected_features = {'dimension': DIMENSION, 'pooled_order': ['peptide_H15', 'peptide_H30', 'target_H15', 'target_H30'],
        'slot_reduction': 'unweighted_float64_mean_of_8_saved_slots', 'lengths': 'peptide_and_target_log1p_full_sequence_length',
        'condition_observed_dimension': 10, 'condition_hash_dimension': 128,
        'condition_hash': 'SHA256(slotname+NUL+canonicalJSON); bucket=bigendian_first8_mod128; sign=1_if_byte8_even_else_minus1; sum_observed_slots'}
    expected_training = {'standardization': 'TRAIN_mean_population_SD', 'standard_deviation_floor': FLOOR,
        'ridge_alpha': 1000., 'ridge_intercept': 'unpenalized', 'knn_k': 25,
        'knn_weights': '1/(squared_euclidean_distance+1e-6)', 'tie_break': 'locked_TRAIN_row_order',
        'validation_selection': 'none; fixed_parameters_both_methods_retained'}
    _require(data.canonical(config['features']) == data.canonical(expected_features)
             and data.canonical(config['training']) == data.canonical(expected_training), 'fixed feature/training recipe changed')


def make_request(config_path, project_root, dataset_root, view_root, cache_root, scenario):
    """Resolve explicit roots; no fitting, feature construction or gold parsing."""
    project = Path(project_root).resolve(strict=True)
    config_path = Path(config_path).resolve(strict=True)
    config = _load(config_path)
    _check_config(config)
    _require(scenario in data.SCENARIOS, 'unknown scenario')
    output = scalar._relative(project, config['output_relative'], existing=False) / scenario
    return {'schema': SCHEMA, 'scenario': scenario, 'config_file': str(config_path), 'config_sha256': data.file_sha(config_path),
        'project_root': str(project), 'dataset_root': str(Path(dataset_root).resolve(strict=True)),
        'view_root': str(Path(view_root).resolve(strict=True)), 'cache_root': str(Path(cache_root).resolve(strict=True)),
        'output_directory': str(output)}


def _check_request(request):
    keys = {'schema', 'scenario', 'config_file', 'config_sha256', 'project_root', 'dataset_root', 'view_root', 'cache_root', 'output_directory'}
    _require(set(request) == keys and request['schema'] == SCHEMA and request['scenario'] in data.SCENARIOS, 'invalid pooled request')
    for key in ('config_file', 'project_root', 'dataset_root', 'view_root', 'cache_root', 'output_directory'):
        _require(isinstance(request[key], str) and Path(request[key]).is_absolute(), 'request roots must be explicit absolute paths')
    data.verify_sha(request['config_file'], request['config_sha256'])
    config = _load(request['config_file'])
    _check_config(config)
    expected = scalar._relative(request['project_root'], config['output_relative'], existing=False) / request['scenario']
    _require(str(expected) == request['output_directory'], 'output differs from fixed recipe')


def _view_closure(request):
    root = Path(request['view_root'])
    manifest_path = data.registered_path(root, 'manifest.json')
    manifest = _load(manifest_path)
    _require(manifest.get('schema') == 'mm_a1_views_v1' and manifest.get('condition_slots') == list(data.CONDITION_SLOTS), 'prepared view schema differs')
    _require(set(manifest['scenarios']) == set(data.SCENARIOS), 'both frozen scenarios required')
    closure = {'manifest_sha256': data.file_sha(manifest_path), 'views': {}, 'sources': {}}
    for scenario in data.SCENARIOS:
        receipt = manifest['scenarios'][scenario]
        _require(receipt['source_shas'] == data.FROZEN_SHAS[scenario], 'frozen source pins differ')
        for name, expected in data.FROZEN_SHAS[scenario].items():
            path = data.registered_path(request['dataset_root'], f'{scenario}/{name}')
            data.verify_sha(path, expected)
            closure['sources'][f'{scenario}/{name}'] = {'sha256': expected, 'bytes': path.stat().st_size}
        for split in ('train', 'validation', 'test'):
            item = receipt['views'][split]
            path = data.registered_path(root, item['path'])
            data.verify_sha(path, item['sha256'])
            _require(item['rows'] == data.SPLIT_COUNTS[scenario][split], 'registered view count differs')
            closure['views'][f'{scenario}/{split}'] = {'path': item['path'], 'sha256': item['sha256'], 'rows': item['rows']}
    splits = {split: data.read_input_view(root, request['scenario'], split) for split in ('train', 'validation', 'test')}
    ids = [row['sample_id'] for records in splits.values() for row in records]
    _require(len(set(ids)) == len(ids), 'cross-split duplicate identity')
    return closure, splits


def prepare(request):
    """Read-only binding of full cache/view/source closure; no fit/test-gold parse."""
    _check_request(request)
    closure, splits = _view_closure(request)
    store = extract.SequenceStore(request['cache_root'])
    cache = store.verify()  # Every registered payload is hashed, not just referenced sequences.
    required = {extract.sequence_sha(row[key]) for rows in splits.values() for row in rows for key in ('peptide_sequence', 'target_sequence')}
    _require(required <= set(cache['verified_payloads']), 'cache lacks a required complete sequence')
    modules = (Path(__file__), Path(data.__file__), Path(extract.__file__), Path(scalar.__file__))
    binding = {'kind': 'pooled_numeric_baselines', 'request_sha256': digest(request), 'request_canonical_json': data.canonical(request),
        'code_sha256': {p.name: data.file_sha(p) for p in modules}, 'view_closure_sha256': digest(closure),
        'cache_manifest_sha256': cache['manifest_sha256'], 'cache_payload_closure_sha256': digest(cache['verified_payloads']),
        'cache_payloads_count': len(cache['verified_payloads']), 'config_sha256': request['config_sha256']}
    return {'request': request, 'training_run_binding': binding, 'view_closure': closure, 'cache_closure': cache,
        'split_counts': {k: len(v) for k, v in splits.items()}, 'feature_dimension': DIMENSION,
        'fit_performed': False, 'test_gold_parsed': False, 'generation_performed': False,
        'budget_group': 'same_source_information_unmatched_numeric_reference'}


def condition_features(row):
    """No fitted vocabulary: 10 observed bits then 128 signed slot/value hashes."""
    view = data.context_view(row)
    measurement = {**view['experimental_context'], 'buffer': view['buffer']}
    target = {key: view[key] for key in ('target_sequence_role', 'actual_construct_verified')}
    target['actual_construct_sequence_missing'] = view['missing']['actual_construct_sequence']
    values = [view['chemical_state'][key] for key in data.CHEMICAL_KEYS] + [view['assay'], measurement, target]
    _, observed = data.condition_encoding(row)
    hashed = np.zeros(128, dtype=np.float64)
    for name, value, present in zip(data.CONDITION_SLOTS, values, observed):
        if present:
            code = hashlib.sha256((name + '\0' + data.canonical(value)).encode('utf-8')).digest()
            hashed[int.from_bytes(code[:8], 'big') % 128] += 1. if code[8] % 2 == 0 else -1.
    return np.concatenate((np.asarray(observed, dtype=np.float64), hashed))


def feature_matrix(rows, store, pooled_means=None, check=lambda: None):
    """Unweighted means of saved slots; not an exact original-residue mean."""
    memory = {} if pooled_means is None else pooled_means
    matrix = np.empty((len(rows), DIMENSION), dtype=np.float64)
    for i, row in enumerate(rows):
        check()
        vector = []
        for key in ('peptide_sequence', 'target_sequence'):
            sequence = row[key]
            _require(isinstance(sequence, str) and sequence and set(sequence) <= set(scalar.AA), 'full canonical sequence required')
            identity = extract.sequence_sha(sequence)
            if identity not in memory:
                values = store.pooled(sequence)
                _require(set(values) == {'H15', 'H30'}, 'paired layer payload required')
                means = []
                for layer in ('H15', 'H30'):
                    value = values[layer]
                    if hasattr(value, 'detach'): value = value.detach().float().cpu().numpy()
                    array = np.asarray(value, dtype=np.float64)
                    _require(array.shape == (8, 640) and np.isfinite(array).all(), 'invalid paired pooled tensor')
                    means.append(array.mean(axis=0, dtype=np.float64))
                memory[identity] = np.concatenate(means)
            vector.extend(memory[identity])
        vector += [math.log1p(len(row['peptide_sequence'])), math.log1p(len(row['target_sequence']))]
        matrix[i] = np.concatenate((np.asarray(vector, dtype=np.float64), condition_features(row)))
    _require(np.isfinite(matrix).all(), 'nonfinite feature matrix')
    return matrix


def standardize_fit(x):
    x = np.asarray(x, dtype=np.float64)
    _require(x.ndim == 2 and len(x) > 0 and np.isfinite(x).all(), 'finite nonempty TRAIN feature matrix required')
    center = x.mean(axis=0, dtype=np.float64)
    scale = np.maximum(x.std(axis=0, ddof=0, dtype=np.float64), FLOOR)
    return center, scale


def transform(x, center, scale):
    result = (np.asarray(x, dtype=np.float64) - center) / scale
    _require(np.isfinite(result).all(), 'nonfinite standardized features')
    return result


def ridge_fit(x, y, alpha=1000., formulation='auto'):
    """Equivalent primal/dual solve, centered y, exact unpenalized intercept."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    _require(x.ndim == 2 and y.shape == (len(x),) and len(x) > 0 and np.isfinite(x).all() and np.isfinite(y).all(), 'finite matched TRAIN arrays required')
    _require(type(alpha) in (float, int) and math.isfinite(alpha) and alpha > 0, 'positive ridge alpha required')
    xmean, ymean = x.mean(axis=0), y.mean()
    xc, yc = x - xmean, y - ymean
    mode = ('dual' if len(x) <= x.shape[1] else 'primal') if formulation == 'auto' else formulation
    if mode == 'dual':
        gram = xc @ xc.T
        gram.flat[::len(x) + 1] += alpha
        coefficients = xc.T @ np.linalg.solve(gram, yc)
    elif mode == 'primal':
        gram = xc.T @ xc
        gram.flat[::x.shape[1] + 1] += alpha
        coefficients = np.linalg.solve(gram, xc.T @ yc)
    else: raise ValueError('unknown ridge formulation')
    intercept = float(ymean - xmean @ coefficients)
    _require(np.isfinite(coefficients).all() and math.isfinite(intercept), 'nonfinite ridge state')
    return {'coefficients': coefficients, 'intercept': intercept, 'formulation': mode, 'alpha': float(alpha)}


def ridge_predict(state, x):
    result = np.asarray(x, dtype=np.float64) @ state['coefficients'] + state['intercept']
    _require(np.isfinite(result).all(), 'nonfinite ridge predictions')
    return result


def knn_predict(train_x, train_y, x, k=25, check=lambda: None):
    """Bounded query blocks; stable row-order tie breaking. No evaluation labels."""
    train_x, train_y, x = (np.asarray(value, dtype=np.float64) for value in (train_x, train_y, x))
    _require(train_x.ndim == x.ndim == 2 and train_x.shape[1] == x.shape[1] and train_y.shape == (len(train_x),)
        and len(train_x) > 0 and type(k) is int and k > 0 and all(np.isfinite(a).all() for a in (train_x, train_y, x)), 'invalid finite kNN arrays')
    # BLAS distance identity is fast; clamp roundoff below zero. Stable sorting
    # uses the locked TRAIN input order for exact ties in this numerical path.
    train_norm = np.einsum('ij,ij->i', train_x, train_x)
    result = []
    for start in range(0, len(x), 32):
        check()
        query = x[start:start + 32]
        distances = np.maximum(0., np.einsum('ij,ij->i', query, query)[:, None] + train_norm[None, :] - 2 * query @ train_x.T)
        _require(np.isfinite(distances).all(), 'kNN distance overflow')
        nearest = np.argsort(distances, axis=1, kind='stable')[:, :min(k, len(train_x))]
        weights = 1 / (np.take_along_axis(distances, nearest, axis=1) + 1e-6)
        result.extend((weights * train_y[nearest]).sum(axis=1) / weights.sum(axis=1))
    result = np.asarray(result, dtype=np.float64)
    _require(np.isfinite(result).all(), 'nonfinite kNN predictions')
    return result


def _authorize(prepared, contract_path, standard_root, validator_sha256):
    _require(contract_path and standard_root and validator_sha256, 'execution requires exact standard contract and validator SHA')
    contract_path = Path(contract_path).resolve(strict=True)
    standard = Path(standard_root).resolve(strict=True)
    validator = standard / 'validate_reproduction.py'
    data.verify_sha(validator, validator_sha256)
    raw, contract = contract_path.read_bytes(), _load(contract_path)
    _require(contract.get('training_run_binding') == prepared['training_run_binding'], 'contract does not bind exact pooled request/code/cache/views')
    spec = importlib.util.spec_from_file_location('mm_pooled_canonical_validator', validator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checked = module.validate(contract, contract_bytes=raw, check_files=True, contract_directory=contract_path.parent)
    _require(checked.get('valid_for_execution') is True and not checked.get('files_skipped'), 'canonical standard did not authorize complete local execution')
    deadline = datetime.fromisoformat(contract['timing']['effective_deadline_at'].replace('Z', '+00:00'))
    _require(deadline.tzinfo is not None and scalar._finite(contract['execution']['timeout_seconds']) and contract['execution']['timeout_seconds'] > 0, 'invalid bounded timing')
    return {'contract_sha256': hashlib.sha256(raw).hexdigest(), 'validator_sha256': validator_sha256, 'standard_validation': checked}, deadline, contract['execution']['timeout_seconds']


def run(request, *, execute=False, contract_path=None, standard_root=None, validator_sha256=None):
    began = time.monotonic()
    prepared = prepare(request)
    if not execute: return prepared
    authorization, deadline, timeout = _authorize(prepared, contract_path, standard_root, validator_sha256)
    def check():
        if datetime.now(timezone.utc) >= deadline or time.monotonic() - began >= timeout:
            raise TimeoutError('pooled baseline contract deadline/runtime reached')
    check()
    output = Path(request['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'request.json', request)
    _write(output / 'authorization.json', authorization)
    _write(output / 'input_closure.json', prepared)
    try:
        _, splits = _view_closure(request)
        stats = scalar.train_statistics(splits['train'], request['scenario'])
        store, means = extract.SequenceStore(request['cache_root']), {}
        x = feature_matrix(splits['train'], store, means, check)
        center, scale = standardize_fit(x)
        z = transform(x, center, scale)
        y = np.asarray([row['pkd'] for row in splits['train']], dtype=np.float64)
        check()
        ridge = ridge_fit(z, y, alpha=1000.)
        # Fixed parameters: both states are saved, no validation selection.
        np.savez(output / 'standardizer.npz', center=center, scale=scale)
        np.savez(output / 'ridge2700_alpha1000.state.npz', coefficients=ridge['coefficients'], intercept=np.asarray(ridge['intercept']))
        np.savez(output / 'knn2700_k25.state.npz', train_features=z, train_labels=y)
        state_meta = {'scenario': request['scenario'], 'TRAIN_statistics': stats, 'methods': list(METHODS),
            'selection': 'fixed_parameters_no_validation_or_test_selection', 'feature_dimension': DIMENSION,
            'ridge': {'alpha': 1000., 'formulation': ridge['formulation'], 'nominal_coefficients_including_intercept': DIMENSION + 1,
                      'statistical_degrees_of_freedom': None, 'training_target': 'original TRAIN pKD; no target standardization'},
            'knn': {'k': 25, 'stored_TRAIN_feature_values': DIMENSION * len(y), 'stored_TRAIN_labels': len(y)},
            'standard_deviation_floor': FLOOR, 'test_gold_parsed': False,
            'state_sha256': {p.name: data.file_sha(p) for p in output.glob('*.npz')}}
        _write(output / 'selected_states_before_test.json', state_meta)
        check()
        # Only after saved state hashes/selection receipt exist is test gold parsed.
        gold_path = data.registered_path(request['dataset_root'], request['scenario'] + '/test_gold.jsonl')
        data.verify_sha(gold_path, data.FROZEN_SHAS[request['scenario']]['test_gold.jsonl'])
        gold = data.read_jsonl(gold_path)
        _require([row['sample_id'] for row in gold] == [row['sample_id'] for row in splits['test']], 'test gold/input identity/order differs')
        splits['test'] = [{**row, 'pkd': g['pkd']} for row, g in zip(splits['test'], gold)]
        metrics = {name: {} for name in METHODS}
        for split in ('validation', 'test'):
            check()
            features = transform(feature_matrix(splits[split], store, means, check), center, scale)
            predictions = {METHODS[0]: ridge_predict(ridge, features), METHODS[1]: knn_predict(z, y, features, k=25, check=check)}
            for method, values in predictions.items():
                metrics[method][split] = scalar.numeric_metrics(splits[split], [float(v) for v in values])
                with (output / (method + '.' + split + '.predictions.jsonl')).open('x', encoding='utf-8') as stream:
                    for row, value in zip(splits[split], values):
                        stream.write(data.canonical({'sample_id': row['sample_id'], 'scenario': request['scenario'], 'split': split,
                            'gold_pkd': row['pkd'], 'prediction_pkd': float(value), 'channel': 'numeric_predictor', 'generation_performed': False}) + '\n')
        final = prepare(request)
        _require(final['training_run_binding'] == prepared['training_run_binding'], 'data/cache/source closure changed during execution')
        check()
        result = {'schema': 'mm_pooled_numeric_baseline_result.v1', 'status': 'complete', 'formal_eligible': True,
            'scenario': request['scenario'], 'request_sha256': digest(request), 'training_run_binding': prepared['training_run_binding'],
            'authorization': authorization, 'split_counts': prepared['split_counts'], 'n_expected_test': len(splits['test']),
            'fit_performed': True, 'test_gold_parsed': True, 'generation_performed': False, 'channel': 'numeric_predictor',
            'budget_group': prepared['budget_group'], 'models': metrics, 'train_state_metadata': state_meta,
            'numpy_version': np.__version__, 'elapsed_seconds': time.monotonic() - began,
            'pair_supervision': 'unavailable_not_constructed', 'mechanism_C1_or_conditional_probe': 'not_measured',
            'artifact_sha256': {p.name: data.file_sha(p) for p in output.iterdir() if p.is_file()}}
        _write(output / 'result.json', result)
        return result
    except Exception as exc:
        _write(output / 'failure.json', {'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc), 'partial_artifacts_retained': True})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', nargs='?', choices=('prepare', 'execute'), default='prepare')
    p.add_argument('--config', required=True)
    p.add_argument('--project-root', required=True)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--view-root', required=True)
    p.add_argument('--cache-root', required=True)
    p.add_argument('--scenario', required=True, choices=data.SCENARIOS)
    p.add_argument('--print-binding', action='store_true')
    p.add_argument('--contract')
    p.add_argument('--standard-root')
    p.add_argument('--validator-sha256')
    args = p.parse_args()
    _require(not (args.print_binding and args.phase == 'execute'), 'print-binding is prepare only')
    request = make_request(args.config, args.project_root, args.dataset_root, args.view_root, args.cache_root, args.scenario)
    result = run(request, execute=args.phase == 'execute', contract_path=args.contract,
                 standard_root=args.standard_root, validator_sha256=args.validator_sha256)
    if args.print_binding:
        result = {key: result[key] for key in ('request', 'training_run_binding', 'split_counts', 'fit_performed', 'test_gold_parsed', 'generation_performed')}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
