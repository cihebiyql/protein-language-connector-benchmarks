"""Strict, model-free A1 evaluation. Uses only Python's standard library.

CLI output is exclusive. Frozen source data and existing results are read-only.
No model, tokenizer, tensor library, network or training code is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

SCOPE_ID = 'molecular_memory_v3_20261017'
SCENARIOS = ('new_peptide_known_target', 'new_peptide_new_target')
ARMS = ('M0', 'M1', 'M2', 'M3', 'A_L', 'A_Q')
SEEDS = (17, 42, 2026)
COUNTS = {SCENARIOS[0]: {'train': 2062, 'validation': 147, 'test': 152},
          SCENARIOS[1]: {'train': 2063, 'validation': 431, 'test': 442}}
DATA_SHA256 = {
    SCENARIOS[0]: {
        'samples.jsonl': 'd1d9ce44db4aeefcc829f5ce7b50b3e288ad03a28af9d15bb9f1fd550a6b657e',
        'test_gold.jsonl': '6f50e85443d1056ef35d509b5229dfb0bc364e266da2aebfa3fa78dcdb07c07b',
        'manifest.json': 'f90f94866baa34f05af2b5d9416c653847fe4beb0cebbd4614592399e12c190e',
        'benchmark_lock.json': 'bf15cc6dfc1a62c3b97a3201ff5961167f0ac0d7afcf7872c46aa1ff57e65249'},
    SCENARIOS[1]: {
        'samples.jsonl': '7ff5a67203efaa94db29fad1f7d2774b5f5e5f8a9155fc47a090e0f2a3baee71',
        'test_gold.jsonl': '388e50ac4776bd122dc3fb60a688ee715173a8703c890ac7d0ad38eaab82065f',
        'manifest.json': '508aa3835d652aab4d115a61fa90bc6aa157e3c195e035803fb808b0cbed88c5',
        'benchmark_lock.json': '329ebdf75e792eabe80d2edf423c53990ea31551fae5112e20b463c30b27b90f'}}
CONDITION_SLOTS = (
    'c_terminal', 'chemical_modification', 'cyclization_method', 'explicitly_representable',
    'n_terminal', 'residue_modification', 'topology', 'assay', 'measurement_context', 'target_context')
DEFAULT_DATA = Path(__file__).resolve().parents[3] / 'data/processed/bench_v02/a1_locked_v2'
ARITHMETIC_TOLERANCE = 1e-12  # Reconstruction only; never selection/superiority tolerance.


class EvaluationError(ValueError):
    """An explicit contract failure; never silently drop an offending row."""


def _require(condition, reason):
    if not condition:
        raise EvaluationError(reason)


def finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _keys(value, required, optional=()):
    _require(isinstance(value, dict), 'object_required')
    _require(set(required) <= set(value) <= set(required) | set(optional), 'schema_keys_mismatch')


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def ordered_ids_sha(rows):
    return canonical_sha([row['sample_id'] for row in rows])


def _object_pairs(pairs):
    _require(len({k for k, _ in pairs}) == len(pairs), 'duplicate_json_key')
    return dict(pairs)


def _reject_constant(_value):
    raise EvaluationError('nonfinite_json_constant')


def strict_loads(text):
    try:
        return json.loads(text, object_pairs_hook=_object_pairs, parse_constant=_reject_constant)
    except (ValueError, TypeError, RecursionError, OverflowError) as exc:
        raise EvaluationError('invalid_strict_json') from exc


def read_json(path):
    value = strict_loads(Path(path).read_text(encoding='utf-8-sig'))
    _finite_json_values(value)
    return value


def _finite_json_values(value):
    # json.loads converts a legal exponent such as1e309 into inf without calling
    # parse_constant. Reject this in artifacts, while parse_answer keeps the old
    # parser's exact nonfinite_or_nonnumeric_pKD reason for raw answer text.
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif type(item) in (int, float):
            _require(finite(item), 'nonfinite_JSON_artifact_number')


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            _require(bool(line.strip()), 'blank_jsonl_record')
            value = strict_loads(line)
            _finite_json_values(value)
            _require(isinstance(value, dict), 'jsonl_object_required')
            rows.append(value)
    return rows


def parse_answer(text):
    """Original three reason labels and strict one-field finite-pKD semantics."""
    try:
        value = strict_loads(text)
    except EvaluationError:
        return {'valid': False, 'value': None, 'reason': 'invalid_json_or_nonfinite'}
    if not isinstance(value, dict) or set(value) != {'pKD'}:
        return {'valid': False, 'value': None, 'reason': 'wrong_schema'}
    if not finite(value['pKD']):
        return {'valid': False, 'value': None, 'reason': 'nonfinite_or_nonnumeric_pKD'}
    return {'valid': True, 'value': float(value['pKD']), 'reason': None}


def _mean(values):
    if not values:
        return None
    try:
        return math.fsum(values) / len(values)
    except OverflowError:
        return math.fsum(v/len(values) for v in values)


def average_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[start]] == values[order[end]]:
            end += 1
        rank = (start + end + 1) / 2
        for position in order[start:end]:
            ranks[position] = rank
        start = end
    return ranks


def spearman(gold, predictions):
    _require(len(gold) == len(predictions) and all(finite(v) for v in [*gold, *predictions]),
             'invalid_spearman_vectors')
    if len(gold) < 2:
        return None
    x, y = average_ranks(gold), average_ranks(predictions)
    mx, my = _mean(x), _mean(y)
    x, y = [v - mx for v in x], [v - my for v in y]
    denominator = math.hypot(*x) * math.hypot(*y)
    return max(-1., min(1., math.fsum(a*b for a, b in zip(x, y))/denominator)) if denominator else None


def regression(rows, values):
    _require(len(rows) == len(values) and all(finite(v) for v in values), 'incomplete_numeric_vector')
    gold = [r.get('pkd') for r in rows]
    _require(all(finite(v) for v in gold), 'invalid_gold')
    errors = [abs(y-p) for y, p in zip(gold, values)]
    _require(all(finite(v) for v in errors), 'metric_numeric_overflow')
    grouped = defaultdict(list)
    for row, error in zip(rows, errors):
        target = row.get('groups', {}).get('target_sequence_sha256')
        _require(isinstance(target, str) and bool(target), 'target_group_missing')
        grouped[target].append(error)
    try:
        groups = [{'target_sequence_sha256': key, 'n': len(v), 'mae': _mean(v)} for key, v in grouped.items()]
        scale = max(errors, default=0.)
        score = {'n': len(rows), 'mae': _mean(errors),
            'rmse': scale*math.sqrt(_mean([(v/scale)**2 for v in errors])) if scale else 0. if rows else None,
            'spearman': spearman(gold, values), 'n_targets': len(groups),
            'target_macro_mae': _mean([g['mae'] for g in groups]), 'targets': groups}
    except OverflowError as exc:
        raise EvaluationError('metric_numeric_overflow') from exc
    _require(all(v is None or finite(v) for k, v in score.items() if k != 'targets'), 'metric_numeric_overflow')
    return score


def _align(rows, predictions, partial=False):
    expected = [r['sample_id'] for r in rows]
    _require(len(set(expected)) == len(expected), 'duplicate_gold_ID')
    _require(all(isinstance(p, dict) for p in predictions), 'prediction_object_required')
    actual = [p.get('sample_id') for p in predictions]
    _require(all(isinstance(v, str) and bool(v) for v in actual), 'invalid_prediction_ID')
    _require(len(set(actual)) == len(actual), 'duplicate_prediction_ID')
    _require(actual == (expected[:len(actual)] if partial else expected), 'prediction_ID_order_or_coverage')
    for row, prediction in zip(rows, predictions):
        _require(prediction.get('split') == row.get('split'), 'prediction_split_mismatch')
        _require(finite(prediction.get('gold_pkd')) and prediction['gold_pkd'] == row['pkd'], 'prediction_gold_mismatch')


_ROW_PROVENANCE = ('checkpoint_sha256', 'protocol_sha256', 'run_id')


def generation_metrics(rows, predictions):
    _align(rows, predictions)
    valid_rows, values, failures = [], [], Counter()
    for row, prediction in zip(rows, predictions):
        _keys(prediction, ('sample_id', 'split', 'gold_pkd', 'raw_text'),
              ('parse', 'raw_generated_token_ids', 'first_eos_length', 'generation_error', *_ROW_PROVENANCE))
        text = prediction['raw_text']
        error = prediction.get('generation_error')
        _require(error is None or isinstance(error, str) and bool(error), 'invalid_generation_error')
        _require(isinstance(text, str) or text is None and error is not None, 'raw_text_required')
        if 'raw_generated_token_ids' in prediction:
            ids = prediction['raw_generated_token_ids']
            _require(isinstance(ids, list) and all(_integer(v) for v in ids) and len(ids) <= 24, 'invalid_generated_token_ids')
            length = prediction.get('first_eos_length')
            _require(length is None or _integer(length) and length <= len(ids), 'invalid_EOS_length')
        parsed = parse_answer(text)
        if 'parse' in prediction:
            _keys(prediction['parse'], ('valid', 'value', 'reason'))
            saved = prediction['parse']
            _require(type(saved['valid']) is bool and saved == parsed
                and (saved['value'] is None or finite(saved['value'])), 'stored_parse_mismatch')
        _require(error is None or not parsed['valid'], 'generation_error_with_valid_answer')
        if parsed['valid']:
            valid_rows.append(row); values.append(parsed['value'])
        else:
            failures[parsed['reason']] += 1
    metric = regression(valid_rows, values)
    complete = len(values) == len(rows)
    return {'n_total': len(rows), 'n_valid': len(values), 'n_failed': len(rows)-len(values),
        'parse_valid_rate': len(values)/len(rows) if rows else None, 'failure_counts': dict(failures),
        'full_sample_mae': metric['mae'] if complete else None,
        'full_sample_mae_status': 'defined' if complete and rows else 'undefined_no_rows' if not rows else 'undefined_parse_failures',
        'valid_only_mae': metric['mae'], 'valid_only_denominator': len(values),
        'regression_valid_only': metric, 'prediction_clipping': False, 'hand_corrected_outputs': False}


def numeric_metrics(rows, predictions):
    _align(rows, predictions)
    valid_rows, values, failures = [], [], Counter()
    for row, prediction in zip(rows, predictions):
        _keys(prediction, ('sample_id', 'split', 'gold_pkd', 'prediction_pkd', 'error'), _ROW_PROVENANCE)
        error = prediction['error']
        _require(error is None or isinstance(error, str) and bool(error), 'invalid_numeric_error')
        value = prediction['prediction_pkd']
        if error is None:
            _require(finite(value), 'numeric_failure_must_be_explicit_null')
            valid_rows.append(row); values.append(float(value))
        else:
            _require(value is None, 'numeric_error_with_value')
            failures[error] += 1
    complete = len(values) == len(rows)
    metric = regression(valid_rows, values)
    return {'n_total': len(rows), 'n_finite': len(values), 'n_failed': len(rows)-len(values),
        'coverage': len(values)/len(rows) if rows else None, 'failure_counts': dict(failures),
        'full_population': metric if complete else None,
        'status': 'defined' if complete and rows else 'undefined_no_rows' if not rows else 'undefined_numeric_failures',
        'finite_only_diagnostic': metric, 'finite_only_denominator': len(values), 'units': 'original_pKD'}


def select_checkpoint(history):
    _require(isinstance(history, list) and bool(history), 'empty_selection_history')
    _require(all(isinstance(h, dict) for h in history), 'history_object_required')
    epochs = [h.get('epoch') for h in history]
    _require(all(_integer(e, 1) for e in epochs) and epochs == sorted(set(epochs)), 'invalid_epoch_sequence')
    candidates = []
    for h in history:
        loss, count = h.get('validation_answer_loss_sum'), h.get('validation_answer_token_count')
        _require(finite(loss) and loss >= 0 and _integer(count, 1), 'invalid_full_validation_NLL')
        score = loss/count
        _require(finite(score), 'invalid_full_validation_NLL')
        if 'validation_answer_token_nll' in h:
            _require(finite(h['validation_answer_token_nll']) and math.isclose(score, h['validation_answer_token_nll'],
                rel_tol=ARITHMETIC_TOLERANCE, abs_tol=ARITHMETIC_TOLERANCE), 'stored_validation_NLL_mismatch')
        candidates.append((score, h['epoch']))
    score, epoch = min(candidates)  # Exact score comparison; earliest epoch on exact tie.
    return {'best_epoch': epoch, 'validation_answer_token_nll': score,
            'criterion': 'minimum_full_validation_answer_token_NLL_earliest_exact_tie'}


def seed_statistics(per_seed):
    normalized = {str(k): v for k, v in per_seed.items()}
    _require(len(normalized) == len(per_seed) and set(normalized) == {str(s) for s in SEEDS}, 'seed_set_mismatch')
    _require(all(v is None or finite(v) for v in normalized.values()), 'invalid_seed_metric')
    missing = [s for s in SEEDS if normalized[str(s)] is None]
    values = [normalized[str(s)] for s in SEEDS]
    complete = not missing
    return {'per_seed': normalized, 'k': 3-len(missing), 'expected_k': 3, 'coverage': f'{3-len(missing)}/3',
        'mean': statistics.mean(values) if complete else None,
        'sample_SD': statistics.stdev(values) if complete else None, 'missing_invalid_seeds': missing}


def c1_metrics(rows, records):
    _require(all(isinstance(r, dict) for r in records), 'C1_object_required')
    _require([r.get('sample_id') for r in records] == [r['sample_id'] for r in rows], 'C1_coverage_mismatch')
    differences = {'off': [], 'train_mean': []}
    numeric_effects = []
    mean_present = ['train_mean' in item for item in records]
    _require(not any(mean_present) or all(mean_present), 'partial_C1_mean_coverage')
    numeric_present = ['numeric_correct_pkd' in item or 'numeric_off_pkd' in item for item in records]
    _require(not any(numeric_present) or all(numeric_present), 'partial_C1_numeric_coverage')
    for row, item in zip(rows, records):
        _keys(item, ('sample_id', 'answer_tokens', 'correct', 'off'),
              ('train_mean', 'numeric_correct_pkd', 'numeric_off_pkd', 'checkpoint_sha256',
               'neutral_checkpoint_sha256', 'neutral_statistic_sha256', 'neutral_fit_split'))
        n = item['answer_tokens']
        _require(_integer(n, 1), 'invalid_C1_token_count')
        means = {}
        for mode in ('correct', 'off', 'train_mean') if 'train_mean' in item else ('correct', 'off'):
            value = item[mode]
            _keys(value, ('loss_sum', 'token_count'))
            _require(finite(value['loss_sum']) and value['loss_sum'] >= 0 and value['token_count'] == n
                and _integer(value['token_count'], 1), 'C1_denominator_mismatch')
            means[mode] = value['loss_sum']/n
        if 'train_mean' in item:
            _require(item.get('neutral_fit_split') == 'train' and _sha(item.get('neutral_statistic_sha256')),
                     'neutral_not_TRAIN_only')
            _require(item.get('neutral_checkpoint_sha256') == item.get('checkpoint_sha256')
                and _sha(item.get('checkpoint_sha256')), 'neutral_wrong_theta')
        for mode in differences:
            if mode in means:
                differences[mode].append(means[mode]-means['correct'])
        if any(numeric_present):
            _require(finite(item.get('numeric_correct_pkd')) and finite(item.get('numeric_off_pkd')), 'C1_numeric_nonfinite')
            numeric_effects.append(abs(row['pkd']-item['numeric_off_pkd'])-abs(row['pkd']-item['numeric_correct_pkd']))
    return {'n_test': len(rows), 'G_off': _mean(differences['off']), 'G_train_mean': _mean(differences['train_mean']),
        'train_mean_status': 'defined' if differences['train_mean'] else 'not_run', 'uniform_status': 'not_run',
        'numeric_off_minus_on_MAE': _mean(numeric_effects),
        'numeric_off_status': 'defined' if numeric_effects else 'not_run',
        'units': 'nats_per_answer_token', 'aggregation': 'mean_of_per_row_mean_NLL_differences',
        'scope': 'fixed_theta_message_distribution_sensitivity', 'n_biological_donors': 0,
        'n_strict_comparable_pairs': 0, 'P': None, 'H': None, 'P_H_status': 'undefined_zero_independent_donors'}


def missing_condition_counts(rows):
    result = {}
    for field in ('temperature_kelvin', 'ph', 'buffer', 'actual_construct_sequence', 'assay'):
        missing = sum(r.get('experimental_context', {}).get(field) in (None, '') for r in rows)
        result[field] = {'n_missing': missing, 'n_observed': len(rows)-missing,
            'comparison_status': 'undefined_empty_stratum' if missing in (0, len(rows)) else 'descriptive_strata_available'}
    return result


def load_locked_scenario(data_root, scenario):
    _require(scenario in SCENARIOS, 'unknown_scenario')
    base = Path(data_root)/scenario
    for name, expected in DATA_SHA256[scenario].items():
        _require((base/name).is_file() and file_sha(base/name) == expected, 'frozen_data_SHA_mismatch')
    rows, gold = read_jsonl(base/'samples.jsonl'), read_jsonl(base/'test_gold.jsonl')
    _require(Counter(r.get('split') for r in rows) == COUNTS[scenario], 'frozen_split_counts_mismatch')
    _require(len({r.get('sample_id') for r in rows}) == len(rows), 'duplicate_gold_ID')
    split = {s: [r for r in rows if r['split'] == s] for s in ('train', 'validation', 'test')}
    _require([r['sample_id'] for r in gold] == [r['sample_id'] for r in split['test']], 'gold_ID_order_mismatch')
    _require(all(finite(r.get('pkd')) and r['pkd'] == g.get('pkd') for r, g in zip(split['test'], gold)), 'gold_value_mismatch')
    return split


def validate_protocol(protocol, formal=False):
    _require(isinstance(protocol, dict), 'protocol_object_required')
    _require(type(protocol.get('schema_version')) is int and protocol['schema_version'] == 1
        and protocol.get('scope_id') == SCOPE_ID, 'protocol_identity_mismatch')
    _require(protocol.get('scenarios') == list(SCENARIOS) and protocol.get('arms') == list(ARMS)
        and protocol.get('seeds') == list(SEEDS), 'protocol_matrix_mismatch')
    _require(protocol.get('data_sha256') == DATA_SHA256 and protocol.get('split_counts') == COUNTS,
        'protocol_frozen_data_mismatch')
    _require(protocol.get('condition_slots') == list(CONDITION_SLOTS), 'condition_allowlist_mismatch')
    generation = protocol.get('generation')
    _keys(generation, ('max_new_tokens', 'batch_size', 'sampling'))
    _require(type(generation['max_new_tokens']) is int and generation['max_new_tokens'] == 24
        and _integer(generation['batch_size'], 1) and generation['sampling'] is False, 'generation_contract_mismatch')
    _require(protocol.get('checkpoint_selector') == 'full_validation_answer_token_nll_earliest_exact_tie',
             'checkpoint_selector_mismatch')
    budget = protocol.get('training_budget', {})
    _require(isinstance(budget, dict), 'budget_object_required')
    if formal:
        _require(protocol.get('evaluator_source_sha256') == file_sha(__file__), 'formal_evaluator_source_SHA_mismatch')
        _require(budget.get('status') == 'FROZEN' and _sha(budget.get('training_budget_sha256')),
                 'formal_budget_UNFROZEN')
        _require(isinstance(budget.get('training'), dict)
            and canonical_sha(budget['training']) == budget['training_budget_sha256'], 'formal_budget_signature_invalid')
        _require(_integer(budget['training'].get('generation_batch_size'), 1)
            and budget['training']['generation_batch_size'] == generation['batch_size'], 'formal_generation_batch_mismatch')
        _require(_sha(budget.get('config_sha256')) and isinstance(budget.get('config_path'), str),
                 'formal_config_pin_missing')
        _require(set(budget.get('expected_training', {})) == set(SCENARIOS), 'formal_expected_budget_missing')
        for value in budget['expected_training'].values():
            _keys(value, ('epochs_completed', 'optimizer_steps', 'examples_seen'))
            _require(all(_integer(v, 1) for v in value.values()), 'formal_expected_budget_invalid')
    return True


def _artifact(root, ref):
    _keys(ref, ('path', 'sha256'))
    _require(isinstance(ref['path'], str) and _sha(ref['sha256']), 'invalid_artifact_ref')
    candidate = root/ref['path']
    resolved = candidate.resolve()
    _require(resolved.is_relative_to(root) and not any(p.is_symlink() for p in [candidate, *candidate.parents]
        if p.is_relative_to(root)), 'artifact_path_escape')
    _require(resolved.is_file() and file_sha(resolved) == ref['sha256'], 'artifact_SHA_mismatch')
    return resolved


def _row_provenance(records, run, checkpoint_sha):
    for row in records:
        for key, expected in (('run_id', run['run_id']), ('protocol_sha256', run['protocol_sha256']),
                              ('checkpoint_sha256', checkpoint_sha)):
            if key in row:
                _require(row[key] == expected, 'row_provenance_mismatch')


def evaluate_run(run_manifest_path, protocol_path, data_root=DEFAULT_DATA):
    manifest_path, protocol_path = Path(run_manifest_path), Path(protocol_path)
    root = manifest_path.resolve().parent
    run, protocol = read_json(manifest_path), read_json(protocol_path)
    _keys(run, ('schema_version', 'scope_id', 'run_id', 'phase', 'status', 'scenario', 'arm', 'seed',
        'protocol_sha256', 'training_budget_sha256', 'data_sha256', 'training', 'test_labels_used_for_fit', 'artifacts'),
        ('identity', 'execution_phase', 'runtime', 'model_policy', 'provenance', 'source_fit',
         'evaluation', 'worker_completion_is_controller_acceptance', 'deadline', 'created_utc', 'elapsed_seconds'))
    _require(type(run['schema_version']) is int and run['schema_version'] == 1 and run['scope_id'] == SCOPE_ID, 'run_identity_mismatch')
    _require(isinstance(run['run_id'], str) and bool(run['run_id']), 'run_id_missing')
    _require(run['scenario'] in SCENARIOS and run['arm'] in ARMS and type(run['seed']) is int
        and run['seed'] in SEEDS, 'unregistered_run_tuple')
    _require(run['phase'] in ('development', 'formal'), 'invalid_run_phase')
    _require(run['status'] in ('complete', 'partial', 'training_failed', 'evaluation_failed'), 'invalid_run_status')
    _require(run['test_labels_used_for_fit'] is False, 'test_fit_leakage')
    _require(_sha(run['training_budget_sha256']), 'budget_signature_missing')
    _require(_sha(run['protocol_sha256']), 'config_SHA_missing')
    validate_protocol(protocol, formal=run['phase'] == 'formal')
    config_sha = protocol['training_budget'].get('config_sha256')
    if config_sha is not None:
        _require(run['protocol_sha256'] == config_sha, 'protocol_config_SHA_mismatch')
        config_path = _artifact(protocol_path.resolve().parent.parent,
            {'path': protocol['training_budget']['config_path'], 'sha256': config_sha})
        config = read_json(config_path)
        _require(isinstance(config, dict) and isinstance(config.get('training'), dict)
            and canonical_sha(config['training']) == run['training_budget_sha256'], 'actual_config_training_signature_mismatch')
        _require(_integer(config['training'].get('generation_batch_size'), 1)
            and config['training']['generation_batch_size'] == protocol['generation']['batch_size'],
            'actual_config_generation_batch_mismatch')
    if run['phase'] == 'formal':
        _require(run['training_budget_sha256'] == protocol['training_budget']['training_budget_sha256'], 'budget_signature_mismatch')
    _require(run['data_sha256'] == DATA_SHA256[run['scenario']], 'run_data_pins_mismatch')
    split = load_locked_scenario(data_root, run['scenario'])
    _keys(run['artifacts'], ('history', 'selection', 'checkpoint', 'lm_predictions', 'numeric_predictions', 'c1'))
    paths = {key: None if value is None else _artifact(root, value) for key, value in run['artifacts'].items()}
    tests = split['test']
    base = {'schema_version': 1, 'scope_id': SCOPE_ID, 'run_id': run['run_id'], 'phase': run['phase'],
        'scenario': run['scenario'], 'arm': run['arm'], 'seed': run['seed'],
        'run_manifest_sha256': file_sha(manifest_path), 'protocol_sha256': run['protocol_sha256'],
        'evaluation_protocol_sha256': file_sha(protocol_path),
        'training_budget_sha256': run['training_budget_sha256'], 'data_sha256': DATA_SHA256[run['scenario']],
        'evaluator_source_sha256': file_sha(__file__),
        'n_expected_test': len(tests), 'condition_coverage': missing_condition_counts(tests),
        'historical_results_are_new_budget_matched_baselines': False}
    if run['status'] != 'complete':
        coverage = {}
        for key in ('lm_predictions', 'numeric_predictions'):
            rows = read_jsonl(paths[key]) if paths[key] else []
            _align(tests, rows, partial=True)
            coverage[key] = {'n_written': len(rows), 'n_missing': len(tests)-len(rows),
                             'missing_ids': [r['sample_id'] for r in tests[len(rows):]]}
        return {**base, 'status': run['status'], 'formal_eligible': False, 'partial_coverage': coverage,
                'LM': None, 'numeric': None, 'C1': None}
    _require(all(paths[k] for k in ('history', 'selection', 'checkpoint', 'lm_predictions', 'numeric_predictions')),
             'complete_artifact_missing')
    _keys(run['training'], ('epochs_completed', 'optimizer_steps', 'examples_seen'))
    _require(all(_integer(v, 1) for v in run['training'].values()), 'invalid_training_counts')
    if run['phase'] == 'formal':
        _require(run['training'] == protocol['training_budget']['expected_training'][run['scenario']], 'training_budget_not_completed')
    history = read_json(paths['history'])
    selected = select_checkpoint(history)
    _require(len(history) == run['training']['epochs_completed'] and [h['epoch'] for h in history] == list(range(1, len(history)+1)),
             'incomplete_training_history')
    for h in history:
        _require(h.get('validation_rows') == len(split['validation'])
            and h.get('validation_ordered_ids_sha256') == ordered_ids_sha(split['validation']), 'validation_not_full_locked_population')
    selection = read_json(paths['selection'])
    _keys(selection, ('best_epoch', 'validation_answer_token_nll', 'checkpoint_sha256',
        'test_labels_used_for_selection', 'written_before_test'))
    _require(selection['test_labels_used_for_selection'] is False and selection['written_before_test'] is True,
             'selection_not_before_test')
    _require(type(selection['best_epoch']) is int and selection['best_epoch'] == selected['best_epoch']
        and finite(selection['validation_answer_token_nll'])
        and math.isclose(selection['validation_answer_token_nll'], selected['validation_answer_token_nll'],
                        rel_tol=ARITHMETIC_TOLERANCE, abs_tol=ARITHMETIC_TOLERANCE), 'checkpoint_selection_mismatch')
    checkpoint_sha = file_sha(paths['checkpoint'])
    _require(selection['checkpoint_sha256'] == checkpoint_sha, 'selected_checkpoint_SHA_mismatch')
    lm_rows, numeric_rows = read_jsonl(paths['lm_predictions']), read_jsonl(paths['numeric_predictions'])
    _row_provenance(lm_rows+numeric_rows, run, checkpoint_sha)
    c1 = None
    if paths['c1']:
        c1_rows = read_jsonl(paths['c1'])
        for row in c1_rows:
            _require(row.get('checkpoint_sha256') == checkpoint_sha, 'C1_same_theta_unbound')
            if run['phase'] == 'formal' and protocol.get('C1_plan', {}).get('numeric_on_off'):
                _require('numeric_correct_pkd' in row and 'numeric_off_pkd' in row, 'formal_C1_numeric_missing')
            if run['phase'] == 'formal' and protocol.get('C1_plan', {}).get('train_mean') == 'not_run':
                _require('train_mean' not in row, 'unregistered_C1_mean')
        c1 = c1_metrics(tests, c1_rows)
    elif protocol.get('C1_required_for_formal') and run['phase'] == 'formal':
        raise EvaluationError('formal_C1_missing')
    return {**base, 'status': 'complete', 'formal_eligible': run['phase'] == 'formal',
        'selection': selected, 'checkpoint_sha256': checkpoint_sha,
        'LM': generation_metrics(tests, lm_rows), 'numeric': numeric_metrics(tests, numeric_rows),
        'C1': c1, 'C1_status': 'not_run' if c1 is None else 'defined_distribution_sensitivity'}


def summarize_runs(results, protocol_sha256):
    """Preserve the full36 registry; development/partial runs never fill formal scores."""
    _require(_sha(protocol_sha256), 'summary_protocol_SHA_missing')
    registered = {}
    evaluation_hashes = {r.get('evaluation_protocol_sha256') for r in results}
    _require(len(evaluation_hashes) <= 1 and all(_sha(v) for v in evaluation_hashes), 'summary_evaluation_protocol_mismatch')
    for result in results:
        key = (result.get('scenario'), result.get('arm'), result.get('seed'))
        _require(key[0] in SCENARIOS and key[1] in ARMS and type(key[2]) is int and key[2] in SEEDS,
                 'summary_unregistered_tuple')
        _require(key not in registered, 'duplicate_summary_tuple')
        _require(result.get('scope_id') == SCOPE_ID and result.get('protocol_sha256') == protocol_sha256,
                 'summary_scope_or_protocol_mismatch')
        registered[key] = result
    def value(case, metric):
        if not case or case.get('phase') != 'formal' or case.get('formal_eligible') is not True or case.get('status') != 'complete':
            return None
        if metric == 'LM_full_sample_mae':
            _require(isinstance(case.get('LM'), dict), 'summary_LM_missing')
            number = case['LM'].get('full_sample_mae')
        elif metric.startswith('C1_'):
            c1 = case.get('C1')
            number = c1.get(metric.removeprefix('C1_')) if isinstance(c1, dict) else None
        else:
            _require(isinstance(case.get('numeric'), dict), 'summary_numeric_missing')
            full = case['numeric'].get('full_population')
            number = full.get(metric.removeprefix('numeric_')) if full else None
        _require(number is None or finite(number), 'summary_nonfinite_metric')
        return number
    metrics = ('LM_full_sample_mae', 'numeric_mae', 'numeric_rmse', 'numeric_spearman', 'numeric_target_macro_mae',
               'C1_G_off', 'C1_numeric_off_minus_on_MAE', 'C1_G_train_mean')
    configurations, contrasts, cases = [], [], []
    for scenario in SCENARIOS:
        for arm in ARMS:
            for seed in SEEDS:
                case = registered.get((scenario, arm, seed))
                cases.append({'scenario': scenario, 'arm': arm, 'seed': seed,
                    'status': case.get('status') if case else 'missing',
                    'phase': case.get('phase') if case else None,
                    'formal_eligible': case.get('formal_eligible') is True if case else False,
                    'LM_full_sample_mae': value(case, 'LM_full_sample_mae'),
                    'numeric_full_mae': value(case, 'numeric_mae')})
            configurations.append({'scenario': scenario, 'arm': arm,
                'metrics': {metric: seed_statistics({str(seed): value(registered.get((scenario, arm, seed)), metric)
                    for seed in SEEDS}) for metric in metrics}})
            if arm != 'M0':
                for metric in ('LM_full_sample_mae', 'numeric_mae'):
                    values = {}
                    for seed in SEEDS:
                        baseline = value(registered.get((scenario, 'M0', seed)), metric)
                        candidate = value(registered.get((scenario, arm, seed)), metric)
                        values[str(seed)] = baseline-candidate if baseline is not None and candidate is not None else None
                    contrasts.append({'scenario': scenario, 'arm': arm, 'reference': 'M0', 'metric': metric,
                        'direction': 'positive_is_lower_error_than_M0', **seed_statistics(values)})
    return {'schema_version': 1, 'scope_id': SCOPE_ID, 'protocol_sha256': protocol_sha256,
        'n_expected': 36, 'n_records_provided': len(results), 'cases': cases,
        'n_formal_complete': sum(c['formal_eligible'] and c['status'] == 'complete' for c in cases),
        'n_formal_LM_full_MAE_defined': sum(c['LM_full_sample_mae'] is not None for c in cases),
        'n_formal_numeric_full_MAE_defined': sum(c['numeric_full_mae'] is not None for c in cases),
        'configurations': configurations, 'registered_M0_contrasts': contrasts,
        'old30_historical_results_counted': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--run-manifest', type=Path, required=True)
    evaluate.add_argument('--protocol', type=Path, required=True)
    evaluate.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    evaluate.add_argument('--output', type=Path, required=True)
    validate = sub.add_parser('validate-protocol')
    validate.add_argument('--protocol', type=Path, required=True)
    validate.add_argument('--development', action='store_true', help='Inspect UNFROZEN development protocol only')
    data = sub.add_parser('validate-data')
    data.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    summary = sub.add_parser('summarize')
    summary.add_argument('--run-manifests', type=Path, nargs='+', required=True)
    summary.add_argument('--protocol', type=Path, required=True)
    summary.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    summary.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'validate-protocol':
            validate_protocol(read_json(args.protocol), formal=not args.development)
            result = {'status': 'valid_development_protocol' if args.development else 'valid_frozen_protocol'}
        elif args.command == 'validate-data':
            result = {'status': 'frozen_data_valid', 'scenarios': {s: {k: len(v) for k, v in load_locked_scenario(args.data_root, s).items()} for s in SCENARIOS}}
        elif args.command == 'evaluate':
            result = evaluate_run(args.run_manifest, args.protocol, args.data_root)
        else:
            runs = [evaluate_run(p, args.protocol, args.data_root) for p in args.run_manifests]
            result = summarize_runs(runs, runs[0]['protocol_sha256'])
        if args.command in ('evaluate', 'summarize'):
            with args.output.open('x', encoding='utf8') as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
        print(json.dumps(result if args.command not in ('evaluate', 'summarize') else
            {'status': result.get('status', 'summarized'), 'output': str(args.output)}, ensure_ascii=False, allow_nan=False))
        return 0
    except (EvaluationError, OSError, OverflowError) as exc:
        print(json.dumps({'status': 'invalid_input', 'error_type': type(exc).__name__,
                          'reason': str(exc) if isinstance(exc, EvaluationError) else 'file_IO_or_numeric_error'}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
