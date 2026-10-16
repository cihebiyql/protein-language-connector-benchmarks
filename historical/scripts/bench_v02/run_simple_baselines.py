"""Train-only A1 composition/chemistry controls on an independently locked panel."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_v02.contracts import ROOT, sha256
from bench_v02.metrics import regression
from bench_v02.panel_execution import validate_panel_data

AA = 'ACDEFGHIKLMNPQRSTVWY'


def sequence_features(sequence):
    length = len(sequence)
    return [sequence.count(aa) / length for aa in AA] + [np.log1p(length)]


def features(row):
    chemistry = row['chemical_state']
    context = row['experimental_context']
    # Unknown is a distinct indicator. No DOI, target ID, label or split is input.
    n_terminal = str(chemistry.get('n_terminal') or 'unknown').lower()
    c_terminal = str(chemistry.get('c_terminal') or 'unknown').lower()
    chemistry_values = [float(n_terminal == value) for value in ('free', 'acetyl', 'unknown')]
    chemistry_values += [float(c_terminal == value) for value in ('free', 'amide', 'unknown')]
    missing_context = [float(context.get(key) is None)
                       for key in ('temperature_kelvin', 'ph', 'buffer', 'actual_construct_sequence')]
    return sequence_features(row['peptide_sequence']) + sequence_features(row['target_sequence']) + chemistry_values + missing_context


def ridge_fit(x, y, alpha):
    augmented = np.column_stack([x, np.ones(len(x))])
    penalty = np.eye(augmented.shape[1]) * alpha
    penalty[-1, -1] = 0
    return np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ y)


def knn_predict(train_x, train_y, x, k):
    output = []
    for begin in range(0, len(x), 128):
        query = x[begin:begin + 128]
        distances = np.maximum(0, (query ** 2).sum(1)[:, None] + (train_x ** 2).sum(1)[None, :] - 2 * query @ train_x.T)
        # Stable ties use the locked train row order, never evaluation labels.
        nearest = np.argsort(distances, axis=1, kind='stable')[:, :min(k, len(train_x))]
        weights = 1 / (np.take_along_axis(distances, nearest, axis=1) + 1e-6)
        output.extend((weights * train_y[nearest]).sum(1) / weights.sum(1))
    return np.asarray(output)


def run(samples, lock, output):
    started = time.perf_counter()
    rows, receipt = validate_panel_data(samples, lock, 'A1')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'results.json').exists():
        raise ValueError('refusing to overwrite completed baseline results')
    x = np.asarray([features(row) for row in rows], dtype=float)
    y = np.asarray([row['pkd'] for row in rows])
    indices = {split: np.asarray([i for i, row in enumerate(rows) if row['split'] == split])
               for split in ('train', 'validation', 'test')}
    train, val, test = (indices[s] for s in ('train', 'validation', 'test'))
    center, scale = x[train].mean(0), x[train].std(0).clip(1e-6)
    x = (x - center) / scale
    selected, predictions = {}, {}
    for name in ('train_mean', 'composition_chemistry_ridge', 'composition_chemistry_knn'):
        trials = []
        parameters = [None] if name == 'train_mean' else ([.1, 1., 10., 100., 1000.] if name.endswith('ridge') else [1, 3, 5, 10, 25])
        for parameter in parameters:
            if name == 'train_mean':
                state = np.asarray([y[train].mean()])
                p = np.repeat(state[0], len(val))
            elif name.endswith('ridge'):
                state = ridge_fit(x[train], y[train], parameter)
                p = np.column_stack([x[val], np.ones(len(val))]) @ state
            else:
                state = None
                p = knn_predict(x[train], y[train], x[val], parameter)
            trials.append({'parameter': parameter, 'validation_mae': float(np.abs(y[val] - p).mean()), 'state': state})
        best = min(trials, key=lambda trial: trial['validation_mae'])
        if name == 'train_mean':
            pred = np.repeat(best['state'][0], len(test))
        elif name.endswith('ridge'):
            pred = np.column_stack([x[test], np.ones(len(test))]) @ best['state']
        else:
            pred = knn_predict(x[train], y[train], x[test], best['parameter'])
        predictions[name] = pred
        target_groups = [rows[i]['groups']['target_family'] for i in test]
        selected[name] = {'selected_parameter': best['parameter'],
                          'validation_trials': [{k: v for k, v in t.items() if k != 'state'} for t in trials],
                          'test': regression(y[test], pred, target_groups)}
        np.savez(output / f'{name}.state.npz', center=center, scale=scale,
                 state=best['state'] if best['state'] is not None else np.asarray([]),
                 train_features=x[train] if name.endswith('knn') else np.asarray([]),
                 train_labels=y[train] if name.endswith('knn') else np.asarray([]))
    with (output / 'test_predictions.jsonl').open('w', encoding='utf-8') as handle:
        for position, index in enumerate(test):
            handle.write(json.dumps({'sample_id': rows[index]['sample_id'], 'pkd': float(y[index]),
                                     'predictions': {name: float(p[position]) for name, p in predictions.items()}}) + '\n')
    result = {'benchmark_version': '0.2', 'panel_id': 'A1', 'baseline_id': 'BASE-00',
              'channel': 'predictor', 'role': 'controlled_predictor', 'execution_receipt': receipt,
              'models': selected, 'selection': 'validation_MAE_only_no_test_refit',
              'seeds': 'deterministic_no_stochastic_training', 'full_target_sequence_composition': True,
              'within_condition_pair_metrics': None, 'strict_comparable_pairs_available': False,
              'elapsed_seconds': time.perf_counter() - started,
              'script_sha256': sha256(__file__),
              'artifact_sha256': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'n_test': len(test),
                      'mae': {name: value['test']['mae'] for name, value in selected.items()}}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.samples, args.lock, args.output)
