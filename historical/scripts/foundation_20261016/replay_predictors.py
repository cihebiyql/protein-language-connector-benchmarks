"""CPU replay of the 18 frozen v0.2 predictor checkpoints, plus calibration diagnostics.

No training, checkpoint selection or test-label fitting occurs here. Validation
was already used for checkpoint selection; its empirical residual quantiles
therefore are descriptive calibration, not independent conformal guarantees.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import platform
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from bench_v02.contracts import sha256
from bench_v02.metrics import regression
from bench_v02.panel_execution import validate_panel_data
from bench_v02.run_simple_baselines import run as run_simple

OUTPUT = ROOT / 'results/foundation_20261016/replay_predictors'
OLD_A1 = ROOT / 'results/bench_v02/panel_baselines_20261015_v2'
OLD_B1 = ROOT / 'results/bench_v02/panel_baselines_20261015'
SCENARIOS = ('new_peptide_known_target', 'new_peptide_new_target')
MEMORY_CAP_BYTES = 6 * 1024 ** 3


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numerically_equal(a, b):
    """Strict structural identity with portable float64 arithmetic tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(numerically_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(numerically_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
    return a == b


def compare_control_selection(fitted, previous):
    require(fitted['selected_parameter'] == previous['selected_parameter'], 'control model selection differs')
    require(numerically_equal(fitted['test'], previous['test']), 'selected control test metrics differ')
    trials, old_trials = fitted['validation_trials'], previous['validation_trials']
    require([t['parameter'] for t in trials] == [t['parameter'] for t in old_trials], 'control tuning grid changed')
    selected = fitted['selected_parameter']
    require(numerically_equal(next(t for t in trials if t['parameter'] == selected),
                              next(t for t in old_trials if t['parameter'] == selected)),
            'selected control validation metric differs')
    return [{'parameter': a['parameter'], 'prior_validation_mae': b['validation_mae'],
             'current_validation_mae': a['validation_mae'],
             'absolute_difference': abs(a['validation_mae'] - b['validation_mae'])}
            for a, b in zip(trials, old_trials) if not numerically_equal(a, b)]


def vector(values, name):
    result = np.asarray(values, dtype=float)
    require(result.ndim == 1 and len(result) > 0 and np.isfinite(result).all(),
            f'{name} must be a nonempty finite vector')
    return result


def calibrate(rows, predictions, coverage):
    """Only validation rows can determine the empirical residual threshold."""
    require(isinstance(coverage, (int, float)) and not isinstance(coverage, bool)
            and math.isfinite(coverage) and 0 < coverage < 1, 'coverage must be in (0, 1)')
    require(bool(rows) and all(r.get('split') == 'validation' for r in rows),
            'calibration requires exclusively nonempty validation rows')
    ids = [r['sample_id'] for r in rows]
    require(len(set(ids)) == len(ids), 'duplicate calibration sample IDs')
    labels = vector([r['pkd'] for r in rows], 'validation labels')
    predicted = vector(predictions, 'validation predictions')
    require(len(labels) == len(predicted), 'calibration lengths differ')
    residuals = np.abs(labels - predicted)
    # Explicit order statistic avoids version-dependent quantile interpolation.
    rank = min(len(residuals), math.ceil(coverage * len(residuals)))
    return {'nominal_coverage': float(coverage), 'threshold': float(np.sort(residuals)[rank - 1]),
            'validation_n': len(rows), 'order_statistic_1based': rank,
            'fit_split': 'validation', 'calibration_sample_ids': ids,
            'quantile_rule': 'ceil(n * nominal_coverage), clipped to n; empirical absolute residual',
            'scope': 'descriptive_only_validation_previously_used_for_checkpoint_selection'}


def interval_evaluation(rows, predictions, calibration):
    require(bool(rows) and all(r.get('split') == 'test' for r in rows),
            'interval evaluation requires exclusively nonempty test rows')
    require(calibration.get('fit_split') == 'validation', 'invalid calibration split')
    ids = [r['sample_id'] for r in rows]
    require(len(ids) == len(set(ids)), 'duplicate test sample IDs')
    require(not set(ids) & set(calibration['calibration_sample_ids']), 'calibration/test ID overlap')
    threshold = float(calibration['threshold'])
    require(math.isfinite(threshold) and threshold >= 0, 'invalid residual threshold')
    labels, predicted = vector([r['pkd'] for r in rows], 'test labels'), vector(predictions, 'test predictions')
    require(len(labels) == len(predicted), 'interval evaluation lengths differ')
    return {'nominal_coverage': calibration['nominal_coverage'], 'test_n': len(rows),
            'test_empirical_coverage': float(np.mean(np.abs(labels - predicted) <= threshold)),
            'interval_width_pKD': 2 * threshold, 'test_mae_pKD': float(np.mean(np.abs(labels - predicted))),
            'threshold_fit_split': 'validation', 'guarantee_under_family_shift': False,
            'independent_calibration_split': False}


def observed_memory():
    if sys.platform == 'linux':
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        require(peak <= MEMORY_CAP_BYTES, 'process exceeded registered 6 GiB resident-memory budget')
        return int(peak)
    return None


def resolve_old(relative):
    text = str(relative).replace('\\', '/')
    if '/results/' in text:
        return ROOT / 'results' / text.split('/results/', 1)[1]
    return ROOT / text


def pinned_file(path, expected, pins):
    path = Path(path)
    actual = sha256(path)
    require(actual == expected, f'pinned file changed: {path}')
    pins[str(path.relative_to(ROOT))] = actual


def save_predictions(path, rows, scores, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, sample_ids=np.array([r['sample_id'] for r in rows]), scores=scores, **extra)


def replay(output):
    import torch
    from bench_v02.cache_panel_features import DEFAULT_CACHE, read_jsonl, verify_local_models
    from bench_v02.train_panel_baselines import (A1Predictor, B1Aligner, ContextEncoder,
                                               FeatureStore, a1_metrics, b1_metrics, deduplicate_b1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / 'summary.json').exists(), 'refusing to overwrite completed replay')
    contract = {'scope': 'published_project_checkpoint_replay_not_new_training',
                'source': 'local v0.2 panel predictor checkpoints; not reproduced published-system scores',
                'device': 'cpu', 'torch_threads': 2, 'wall_timeout_seconds': 1800,
                'resident_memory_cap_bytes': MEMORY_CAP_BYTES, 'expected_checkpoint_runs': 18,
                'expected_deterministic_control_fits': 6, 'test_label_fitting': False,
                'calibration_scope': 'validation-selected checkpoint; empirical validation residual thresholds only',
                'calibration_coverages': [.90, .95], 'pid': os.getpid(),
                'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'python_version': sys.version, 'numpy_version': np.__version__,
                'torch_version': torch.__version__, 'platform': platform.platform(),
                'script_sha256': sha256(__file__)}
    write_json(output / 'run_contract.json', contract)
    print(json.dumps({'event': 'started', **contract}), flush=True)
    pins = {}
    # Independently trusted prior audit pins the actual run results and reports.
    final = read_json(OLD_A1 / 'final_verification.json')
    for relative, expected in final['artifact_sha256'].items():
        pinned_file(ROOT / relative, expected, pins)
    review = read_json(OLD_A1 / 'independent_review.json')
    require(review['status'] == 'passed' and len(review['runs']) == 18, 'prior full independent review required')
    model_verification = verify_local_models()
    store = FeatureStore(DEFAULT_CACHE)
    cache_hash = sha256(store.manifest_path)
    pins[str(store.manifest_path.relative_to(ROOT))] = cache_hash
    locks, data = {}, {}
    for scenario in SCENARIOS:
        source = ROOT / 'data/processed/bench_v02/a1_locked_v2' / scenario
        data[scenario], locks[scenario] = validate_panel_data(source / 'samples.jsonl', source / 'benchmark_lock.json', 'A1')
    source = ROOT / 'data/processed/bench_v02/b1_homology_locked'
    b1_rows, locks['unseen_classes'] = validate_panel_data(source / 'b1_samples.jsonl', source / 'benchmark_lock.json', 'B1')
    data['unseen_classes'] = deduplicate_b1(b1_rows)
    mapping = {r['candidate_id']: r for r in read_jsonl(source / 'b1_classes_main.jsonl')}
    candidates = [{**r, 'class_id': mapping[r['candidate_id']]['class_id'],
                   'split': mapping[r['candidate_id']]['split']} for r in read_jsonl(source / 'b1_candidates_main.jsonl')]
    for receipt in locks.values():
        require(receipt.get('status') in ('frozen', 'frozen_restricted_scope'), 'unfrozen panel')
    require(all(r['panel_id'] in ('A1', 'B1') for r in review['runs']), 'unexpected panel')
    summaries = []
    for audited in review['runs']:
        name, panel, scenario = audited['run_id'], audited['panel_id'], audited['scenario']
        directory = resolve_old(audited['result_path']).parent
        result = read_json(directory / 'result.json')
        pinned_file(directory / 'result.json', audited['result_sha256'], pins)
        for filename, expected in result['artifact_sha256'].items():
            pinned_file(directory / filename, expected, pins)
        require(result['feature_cache_manifest_sha256'] == cache_hash, 'cache manifest differs from original training')
        current_training = ROOT / 'scripts/bench_v02/train_panel_baselines.py'
        training_source = current_training
        equivalent_definitions = None
        if result['script_sha256'] != sha256(current_training):
            # B1 predates the A1 v2 CLI-default/split revision. Its exact source
            # snapshot is retained; verify every reusable definition before
            # evaluating with the current imports (only main may differ).
            training_source = OLD_B1 / 'deprecated_a1_split_v1/train_panel_baselines_v1.py'
            pinned_file(training_source, result['script_sha256'], pins)
            def definitions(path):
                tree = ast.parse(path.read_text(encoding='utf-8'))
                return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name != 'main'}
            previous, current = definitions(training_source), definitions(current_training)
            require(previous == current, 'frozen model/training definitions differ from checkpoint source')
            equivalent_definitions = len(previous)
        else:
            pinned_file(training_source, result['script_sha256'], pins)
        require(result['cache_script_sha256'] == sha256(ROOT / 'scripts/bench_v02/cache_panel_features.py'), 'cache implementation changed')
        pinned_file(directory.parent / 'preregistered_protocol.json', result['preregistered_protocol_sha256'], pins)
        pinned_file(directory.parent / 'architecture_training_preregistration.json', result['architecture_registration_sha256'], pins)
        receipt = locks[scenario]
        for field in ('benchmark_lock_sha256', 'samples_sha256'):
            require(receipt[field] == result['data_gate'][field], f'original training data {field} changed')
        for kind in ('proteins', 'texts'):
            for key, expected in result['feature_receipts'][kind].items():
                require(store.manifest[kind][key]['sha256'] == expected, 'training feature receipt changed')
        saved = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
        registration = read_json(directory / 'run_preregistration.json')
        require(saved['registration'] == registration, 'checkpoint registration mismatch')
        require(saved['best_epoch'] == result['best_epoch'], 'checkpoint epoch mismatch')
        rows = data[scenario]
        training = [r for r in rows if r['split'] == 'train']
        validation = [r for r in rows if r['split'] == 'validation']
        test = [r for r in rows if r['split'] == 'test']
        history = read_json(directory / 'history.json')
        require(len(history) == 30 and result['actual_epochs'] == 30, 'not full registered training budget')
        require(result['optimizer_steps'] == 30 * math.ceil(len(training) / 64), 'optimizer step count changed')
        key = 'mae' if panel == 'A1' else 'class_macro_mrr'
        best = (min if panel == 'A1' else max)(history, key=lambda r: r['validation'][key])
        require(best['epoch'] == saved['best_epoch'], 'earliest validation selection mismatch')
        if panel == 'A1':
            context = ContextEncoder(training)
            require(saved['context_statistics'] == context.statistics, 'context statistics not train-only')
            require(saved['label_center'] == float(np.mean([r['pkd'] for r in training])), 'label center not train-only')
            require(saved['label_scale'] == max(float(np.std([r['pkd'] for r in training])), 1e-5), 'label scale not train-only')
            model = A1Predictor(result['architecture'] == 'BASE01').eval()
            descriptions = None
        else:
            require(saved['context_statistics'] is None and saved['label_center'] is None and saved['label_scale'] is None,
                    'unexpected B1 regression normalization')
            model = B1Aligner(result['architecture'] == 'BASE02').eval()
            descriptions = store.descriptions([c['description'] for c in candidates], 'cpu')
        model.load_state_dict(saved['model'], strict=True)

        def predict(selected):
            pieces = []
            with torch.inference_mode():
                for begin in range(0, len(selected), 8):
                    batch = selected[begin:begin + 8]
                    if panel == 'A1':
                        p, pm = store.batch([r['peptide_sequence'] for r in batch], 'cpu')
                        t, tm = store.batch([r['target_sequence'] for r in batch], 'cpu')
                        c = torch.stack([context(r) for r in batch])
                        values = model(p, pm, t, tm, c)
                    else:
                        residues, mask = store.batch([r['segment_sequence'] for r in batch], 'cpu')
                        values = model(residues, mask, descriptions)
                    pieces.append(values.numpy())
                    observed_memory()
                    # Keep bounded resident feature storage; each fresh load rechecks its hash.
                    if sum(v.numel() * v.element_size() for v in store.memory.values()) > 384 * 1024 ** 2:
                        store.memory.clear()
            values = np.concatenate(pieces)
            return values * saved['label_scale'] + saved['label_center'] if panel == 'A1' else values

        actual = predict(test)
        with np.load(directory / 'test_scores.npz', allow_pickle=False) as expected:
            require(expected['sample_ids'].tolist() == [r['sample_id'] for r in test], 'test query identities differ')
            tolerance = {'rtol': 1e-5 if panel == 'A1' else 1e-4, 'atol': 1e-4}
            require(np.allclose(actual, expected['scores'], **tolerance), 'full checkpoint scores fail CPU replay tolerance')
            max_error = float(np.max(np.abs(actual - expected['scores'])))
            if panel == 'A1':
                require(np.array_equal(expected['labels'], np.array([r['pkd'] for r in test])), 'test labels changed')
            else:
                require(expected['candidate_ids'].tolist() == [c['candidate_id'] for c in candidates], 'candidate order changed')
                require(expected['candidate_class_ids'].tolist() == [c['class_id'] for c in candidates], 'candidate classes changed')
                require(np.array_equal(np.argsort(-actual, axis=1, kind='stable'),
                                       np.argsort(-expected['scores'], axis=1, kind='stable')), 'B1 rankings differ')
        run_dir = output / name
        metrics = a1_metrics(test, actual) if panel == 'A1' else b1_metrics(test, actual, candidates)
        row = {'run_id': name, 'panel': panel, 'scenario': scenario, 'architecture': result['architecture'],
               'seed': result['seed'], 'test_n': len(test), 'status': 'passed', 'device': 'cpu',
               'source_result_sha256': audited['result_sha256'], 'checkpoint_sha256': sha256(directory / 'best.pt'),
               'original_training_source_sha256': result['script_sha256'],
               'training_definition_AST_equivalence_count': equivalent_definitions,
               'max_abs_cpu_saved_score_difference': max_error, 'tolerance': tolerance,
               'original_frozen_data_and_train_statistics_verified': True, 'test_metrics': metrics}
        extras = {'labels': np.array([r['pkd'] for r in test])} if panel == 'A1' else {
            'candidate_ids': np.array([c['candidate_id'] for c in candidates]),
            'candidate_class_ids': np.array([c['class_id'] for c in candidates])}
        save_predictions(run_dir / 'test_predictions.npz', test, actual, **extras)
        if panel == 'A1':
            val_scores = predict(validation)
            save_predictions(run_dir / 'validation_predictions.npz', validation, val_scores,
                             labels=np.array([r['pkd'] for r in validation]))
            calibration = [calibrate(validation, val_scores, coverage) for coverage in (.90, .95)]
            row['reliability'] = [interval_evaluation(test, actual, c) for c in calibration]
            write_json(run_dir / 'calibration_thresholds.json', calibration)
            require(abs(float(np.abs(np.array([r['pkd'] for r in validation]) - val_scores).mean()) - result['best_validation_metric']) < 1e-4,
                    'validation score replay differs from selected checkpoint')
        else:
            row['all_rankings_exactly_equal'] = True
        write_json(run_dir / 'replay.json', row)
        summaries.append(row)
        write_json(output / 'progress.json', {'completed': len(summaries), 'runs': summaries})
        print(json.dumps({'event': 'checkpoint_replayed', 'run_id': name, 'max_error': max_error,
                          'elapsed_seconds': time.perf_counter() - started, 'peak_rss_bytes': observed_memory()}), flush=True)
        del model, saved
        gc.collect()
    controls = []
    for scenario, old_name in zip(SCENARIOS, ('A1_known_BASE00', 'A1_double_new_BASE00')):
        source = ROOT / 'data/processed/bench_v02/a1_locked_v2' / scenario
        directory = output / ('BASE00_' + scenario)
        old = OLD_A1 / old_name
        old_result = read_json(old / 'results.json')
        prior_control = next(c for c in review['summary']['control_baselines'] if c['scenario'] == scenario)
        pinned_file(old / 'results.json', prior_control['results_sha256'], pins)
        pinned_file(old / 'test_predictions.jsonl', prior_control['predictions_sha256'], pins)
        for filename, expected in old_result['artifact_sha256'].items():
            pinned_file(old / filename, expected, pins)
        fitted = run_simple(source / 'samples.jsonl', source / 'benchmark_lock.json', directory)
        original_predictions = read_jsonl(old / 'test_predictions.jsonl')
        fitted_predictions = read_jsonl(directory / 'test_predictions.jsonl')
        require(numerically_equal(original_predictions, fitted_predictions), 'refitted control predictions differ beyond 1e-10')
        for model_name, metrics in fitted['models'].items():
            previous = old_result['models'][model_name]
            trial_differences = compare_control_selection(metrics, previous)
            max_difference = max(abs(a['predictions'][model_name] - b['predictions'][model_name])
                                 for a, b in zip(original_predictions, fitted_predictions))
            controls.append({'scenario': scenario, 'model': model_name,
                             'mae': metrics['test']['mae'], 'selected_parameter': metrics['selected_parameter'],
                             'all_test_predictions_numerically_match_prior': True,
                             'max_abs_control_prediction_difference': max_difference,
                             'nonselected_validation_trial_differences': trial_differences,
                             'tolerance': {'rtol': 1e-10, 'atol': 1e-10}})
    # Preserve old receipts, scores, weights, source scripts, reports and cache manifest.
    for relative, expected in pins.items():
        require(sha256(ROOT / relative) == expected, f'prior input modified during replay: {relative}')
    require(len(summaries) == 18 and len(controls) == 6, 'incomplete registered replay suite')
    report = {'status': 'passed', 'scope': contract['scope'], 'actual_replayed_checkpoints': 18,
              'deterministic_control_fits': 6, 'calibrated_A1_checkpoints': 12,
              'runs': summaries, 'controls': controls, 'panel_gates': locks,
              'models_physically_verified': model_verification,
              'used_feature_hashes': {'proteins': store.checked_proteins, 'texts': store.checked_texts},
              'original_input_pins_unchanged': pins, 'elapsed_seconds': time.perf_counter() - started,
              'peak_rss_bytes': observed_memory(), 'exit_code': 0, 'pid': os.getpid(),
              'limitations': ['No new neural training; no published-system reproduction claim.',
                             'Validation selected the checkpoint and is reused for descriptive calibration.',
                             'No finite-sample coverage guarantee under grouped/family distribution shift.',
                             'Three seeds are independent fitted checkpoints, not a predictive-uncertainty ensemble.',
                             'A1 construct/condition limits and B1 two unseen classes remain unchanged.']}
    write_json(output / 'summary.json', report)
    write_json(output / 'exit.json', {'exit_code': 0, 'pid': os.getpid(), 'elapsed_seconds': report['elapsed_seconds'],
                                     'peak_rss_bytes': report['peak_rss_bytes']})
    print(json.dumps({'event': 'complete', 'replayed': 18, 'fits': 6, 'elapsed_seconds': report['elapsed_seconds']}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if hasattr(signal, 'SIGALRM'):
        def timeout(_signal, _frame):
            raise TimeoutError('registered replay wall-time budget of 1800 seconds exceeded')
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(1800)
    try:
        replay(args.output)
    except Exception as exc:
        if args.output.is_dir():
            write_json(args.output / 'exit.json', {'exit_code': 1, 'pid': os.getpid(),
                                                 'exception_type': type(exc).__name__, 'error': str(exc)})
        raise
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


if __name__ == '__main__':
    main()
