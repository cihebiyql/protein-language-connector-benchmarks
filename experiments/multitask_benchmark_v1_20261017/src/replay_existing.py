"""CPU-only historical replay of four non-pKD panels; no training or GPU imports.

Run from any directory: python <this file> --root <project root>.
Input files stay read-only. A separate manifest records byte hashes of every
loaded input/scorer and each derived output. Unknown pairs remain unknown.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

SEEDS = (17, 42, 2026)


class IntegrityError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def align(gold, predicted, key='sample_id'):
    """Require an exact bijection before reordering; never silently intersect IDs."""
    expected = [r[key] for r in gold]
    actual = [r[key] for r in predicted]
    require(len(expected) == len(set(expected)), 'duplicate gold ID')
    require(len(actual) == len(set(actual)), 'duplicate prediction ID')
    require(set(actual) == set(expected), 'prediction ID coverage mismatch')
    indexed = {r[key]: r for r in predicted}
    return [indexed[k] for k in expected]


def ranking_score(gold, predicted, candidates, positive_key, rank_key, claimed_positive_key=None):
    require(len(candidates) == len(set(candidates)), 'duplicate candidate ID')
    candidates = set(candidates)
    predicted = align(gold, predicted)
    positives, rankings = [], []
    for row, pred in zip(gold, predicted):
        positive = row[positive_key]
        require(bool(positive) and set(positive) <= candidates, 'invalid gold positive/candidate map')
        ranked = pred[rank_key]
        require(len(ranked) == len(candidates) and len(ranked) == len(set(ranked))
                and set(ranked) == candidates, 'incomplete or duplicate candidate ranking')
        if claimed_positive_key:
            require(set(pred[claimed_positive_key]) == set(positive), 'prediction gold/label map mismatch')
        positives.append(positive)
        rankings.append(ranked)
    return METRICS.known_positive_retrieval(rankings, positives)


def compare(saved, calculated, keys):
    differences = {}
    for k in keys:
        require(k in saved, f'saved metric missing: {k}')
        a, b = saved[k], calculated[k]
        require((a is None) == (b is None), f'undefined metric mismatch: {k}')
        if a is not None:
            differences[k] = float(b - a)
            require(np.isfinite(a) and abs(b - a) < 1e-10, f'saved metric mismatch: {k}')
    return {'matched': True, 'absolute_tolerance': 1e-10, 'recomputed_minus_saved': differences}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Ledger:
    def __init__(self, root):
        self.root = root
        self.entries = {}

    def pin(self, path, role='input'):
        path = Path(path)
        relative = path.resolve().relative_to(self.root).as_posix()
        if relative not in self.entries:
            self.entries[relative] = {'path': relative, 'sha256': sha(path), 'bytes': path.stat().st_size, 'roles': []}
        if role not in self.entries[relative]['roles']:
            self.entries[relative]['roles'].append(role)
        return path

    def json(self, path, role='input'):
        return json.loads(self.pin(path, role).read_text(encoding='utf-8'))

    def rows(self, path, role='input'):
        return [json.loads(line) for line in self.pin(path, role).read_text(encoding='utf-8').splitlines() if line.strip()]


def elm(root, ledger):
    base = root / 'data/processed/bench_v02/b1_homology_locked'
    lock = ledger.json(base / 'benchmark_lock.json', 'frozen_lock')
    # Deliberately use this panel's native homology lock, not the later generic B1 contract.
    for relative, digest in lock['pins'].items():
        p = root / relative
        require(sha(ledger.pin(p, 'frozen_input')) == digest, f'ELM frozen input changed: {relative}')
    rows = ledger.rows(base / 'b1_samples.jsonl', 'gold')
    gold = [r for r in rows if r['split'] == 'test']
    test_gold = ledger.rows(base / 'test_gold.jsonl', 'gold')
    require(align(gold, test_gold) == gold, 'ELM test gold changed')
    classes = ledger.rows(base / 'b1_classes_main.jsonl', 'label_map')
    catalog = ledger.rows(base / 'b1_candidates_main.jsonl', 'candidate_catalog')
    candidate_ids = [r['candidate_id'] for r in catalog]
    class_ids = [r['class_id'] for r in classes]
    require([r['candidate_id'] for r in classes] == candidate_ids, 'ELM class/candidate map order changed')
    require(len(rows) == 588 and len(gold) == 76 and len(catalog) == len(set(class_ids)) == len(set(candidate_ids)) == 31,
            'ELM frozen population changed')
    mapping = dict(zip(candidate_ids, class_ids))
    for row in rows:
        require({mapping[k] for k in row['known_positive_candidate_ids']} == set(row['known_positive_class_ids']),
                'ELM gold class/candidate map mismatch')
    output = {'scope': 'historical_panel_specific_predictor; class-limited unseen function',
              'population': {'all': 588, 'train': 378, 'validation': 134, 'test': 76,
                             'candidates': 31, 'unseen_test_classes': 2},
              'limitations': lock['limitations'], 'unannotated_is_negative': False, 'runs': []}
    for model in ('BASE02', 'MEANALIGN'):
        for seed in SEEDS:
            run = root / f'results/foundation_20261016/replay_predictors/B1_unseen_classes_{model}_seed{seed}'
            path, receipt = run / 'test_predictions.npz', run / 'replay.json'
            item = {'model': model, 'seed': seed}
            if not path.exists() or not receipt.exists():
                output['runs'].append({**item, 'status': 'unavailable', 'reason': 'missing complete saved prediction/receipt'})
                continue
            meta = ledger.json(receipt, 'prior_replay_receipt')
            source = root / f'results/bench_v02/panel_baselines_20261015/B1_unseen_classes_{model}_seed{seed}/result.json'
            original = ledger.json(source, 'original_result')
            require(meta['status'] == 'passed' and meta['architecture'] == model and meta['seed'] == seed,
                    'ELM replay identity mismatch')
            require(meta['source_result_sha256'] == sha(source), 'ELM source result changed')
            require(original['data_gate']['benchmark_lock_sha256'] == sha(base / 'benchmark_lock.json'),
                    'ELM training lock mismatch')
            with np.load(ledger.pin(path, 'original_prediction'), allow_pickle=False) as data:
                require(data['candidate_ids'].tolist() == candidate_ids and data['candidate_class_ids'].tolist() == class_ids,
                        'ELM prediction candidate/label map mismatch')
                scores = data['scores']
                require(scores.shape == (76, 31) and data['sample_ids'].shape == (76,)
                        and np.isfinite(scores).all(), 'ELM invalid scores or ID length')
                order = np.argsort(-scores, axis=1, kind='stable')
                pred = [{'sample_id': sid, 'ranking': [class_ids[i] for i in rank]}
                        for sid, rank in zip(data['sample_ids'].tolist(), order)]
            score = ranking_score(gold, pred, class_ids, 'known_positive_class_ids', 'ranking')
            checked = ('mrr', 'recall_at_1', 'recall_at_3', 'class_macro_mrr', 'class_macro_recall_at_3')
            output['runs'].append({**item, 'status': 'complete', 'metrics': score,
                                   'replay_check': compare(meta['test_metrics'], score, checked),
                                   'original_check': compare(original['test_metrics'], score, checked)})
    return output


def rt(root, ledger):
    base = root / 'data/processed/foundation_20261016/rt_target_diagnostic'
    source = root / 'results/foundation_20261016/rt_target_diagnostic'
    protocol = ledger.json(base / 'frozen_protocol.json', 'frozen_lock')
    require(sha(base / 'frozen_protocol.json') == '8a15cce4d1daf5d7e7ddb2aa1f85750975a3b7a60ad4bca36d535d81962520ed',
            'RT frozen protocol changed')
    original = root / 'data/processed/bench_v02/a2_target_march.jsonl'
    require(sha(ledger.pin(original, 'original_data')) == protocol['source_sha256'], 'RT original source changed')
    parts = {}
    for f in protocol['files']:
        path = base / (f['split'] + '.jsonl')
        require(sha(ledger.pin(path, 'gold')) == f['sha256'], 'RT frozen split changed')
        parts[f['split']] = ledger.rows(path)
        require(len(parts[f['split']]) == f['rows'], 'RT split row count changed')
    counts = {k: len(v) for k, v in parts.items()}
    require(counts == {'train': 4952, 'validation': 1055, 'test': 1063, 'selection': 525, 'calibration': 530},
            'RT boundary changed')
    require(align(parts['validation'], parts['selection'] + parts['calibration']) == parts['validation'],
            'RT validation role membership changed')
    group_sets = {k: {r['groups']['sequence'] for r in v} for k, v in parts.items()}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test'), ('selection', 'calibration')]:
        require(not group_sets[a] & group_sets[b], 'RT sequence group leakage')
    receipt = ledger.json(source / 'selection_and_calibration_before_test.json', 'selection_and_calibration_receipt')
    old = ledger.json(source / 'metrics.json', 'original_result')
    output = {'scope': 'historical_same_condition_RT_diagnostic; not cross_condition_transfer',
              'population': counts, 'unit': 'minutes', 'condition_id': protocol['condition_id'],
              'calibration_coverage_target': 0.9, 'test_label_tuning': False,
              'limitations': old['interpretation'], 'runs': []}
    for model in ('constant', 'ridge', 'knn'):
        path = source / f'predictions_{model}.jsonl'
        item = {'model': model, 'seed': protocol['seed'], 'seed_role': 'frozen group split; deterministic estimator'}
        if not path.exists():
            output['runs'].append({**item, 'status': 'unavailable', 'reason': 'missing saved predictions'})
            continue
        predicted = align(parts['test'], ledger.rows(path, 'original_prediction'))
        y, p, lo, hi = [], [], [], []
        covered_groups = defaultdict(list)
        radius = receipt['conformal'][model]['radius_min']
        for gold, pred in zip(parts['test'], predicted):
            require(pred['condition_id'] == gold['condition_id'] == protocol['condition_id']
                    and gold['rt_unit'] == 'min', 'RT condition or unit mismatch')
            require(pred['measured_rt_min'] == gold['rt_value'] and pred['sequence_group'] == gold['groups']['sequence'],
                    'RT prediction gold/group mismatch')
            numbers = [pred[k] for k in ('predicted_rt_min', 'lower_rt_min', 'upper_rt_min')]
            require(np.isfinite(numbers).all() and numbers[1] <= numbers[0] <= numbers[2], 'RT invalid interval/prediction')
            require(abs(numbers[0] - numbers[1] - radius) < 1e-10
                    and abs(numbers[2] - numbers[0] - radius) < 1e-10, 'RT saved calibration radius changed')
            y.append(gold['rt_value']); p.append(numbers[0]); lo.append(numbers[1]); hi.append(numbers[2])
            covered_groups[gold['groups']['sequence']].append(numbers[1] <= gold['rt_value'] <= numbers[2])
        basic = METRICS.regression(y, p)
        y, p, lo, hi = (np.asarray(v) for v in (y, p, lo, hi))
        score = {'n': len(y), 'MAE_min': basic['mae'], 'RMSE_min': float(np.sqrt(np.mean((y-p)**2))),
                 'Spearman_same_condition': basic['spearman'],
                 'interval_row_coverage': float(np.mean((lo <= y) & (y <= hi))),
                 'interval_mean_width_min': float(np.mean(hi-lo)),
                 'interval_group_all_observations_coverage': float(np.mean([all(v) for v in covered_groups.values()]))}
        output['runs'].append({**item, 'status': 'complete', 'metrics': score,
                               'original_check': compare(old['models'][model]['test'], score, score.keys())})
    return output


def benz(root, ledger):
    rows = ledger.rows(root / 'data/processed/benz_samples.jsonl', 'gold')
    gold = [r for r in rows if r['split'] == 'test']
    targets = [r['target_id'] for r in ledger.json(root / 'data/processed/benz_targets.json', 'candidate_catalog')]
    require(len(targets) == 34 and len(gold) == 387, 'Benz frozen population changed')
    for name, digest in REAL.ORIGINAL_HASHES.items():
        require(sha(ledger.pin(root / 'data/processed' / name, 'frozen_input')) == digest, 'Benz original changed')
    output = {'scope': 'historical_Benz_engineering_pilot_known_positive_ranking',
              'population': {'train': 1914, 'validation': 393, 'test': 387, 'candidates': 34},
              'unannotated_is_negative': False, 'classification_auroc': None,
              'limitations': ['Documented positives only; unknown pairings cannot be used as negative labels.',
                              'Historical pilot; no current V4 architecture or multitask claim.'], 'runs': []}
    path = root / 'results/b0_predictions.jsonl'
    b0 = ledger.rows(path, 'original_prediction') if path.exists() else []
    require(all(r.get('method') in ('train_frequency', 'sequence_1nn')
                and r.get('split') in ('validation', 'test') for r in b0),
            'Benz B0 unknown method or split cannot be silently dropped')
    old = ledger.json(root / 'results/b0_metrics.json', 'original_result')
    for model in ('train_frequency', 'sequence_1nn'):
        selected = [r for r in b0 if r['method'] == model and r['split'] == 'test']
        if not selected:
            output['runs'].append({'model': model, 'seed': None, 'status': 'unavailable'})
            continue
        score = ranking_score(gold, selected, targets, 'targets', 'ranking', 'targets')
        saved = old['test']['methods'][model]
        score['known_positive_recall_at_3'] = score['recall_at_3']
        output['runs'].append({'model': model, 'seed': None, 'status': 'complete', 'metrics': score,
                               'original_check': compare(saved, score, ('mrr', 'known_positive_recall_at_3'))})
    path = root / 'results/b1_test_predictions.jsonl'
    if path.exists():
        score = ranking_score(gold, ledger.rows(path, 'original_prediction'), targets, 'targets',
                              'ranked_target_ids', 'known_positive_target_ids')
        meta = ledger.json(root / 'results/b1_pilot.json', 'original_result')
        require(meta['input_sha256'] == sha(root / 'data/processed/benz_samples.jsonl'), 'Benz pilot input changed')
        score['known_positive_recall_at_3'] = score['recall_at_3']
        output['runs'].append({'model': 'B1_ESM_frozen_MLP_pilot', 'seed': meta.get('seed'),
                               'status': 'complete', 'metrics': score,
                               'original_check': compare(meta['scores']['test'], score, ('mrr', 'known_positive_recall_at_3'))})
    else:
        output['runs'].append({'model': 'B1_ESM_frozen_MLP_pilot', 'seed': None, 'status': 'unavailable'})
    return output


def direction_score(gold, predicted, mode):
    predicted = align(gold, predicted, 'pair_id')
    labels, guesses, groups = [], [], []
    for row, pred in zip(gold, predicted):
        require(pred['label'] == row['label'] and pred['target_id'] == row['target_id']
                and pred['group_id'] == row['group_id'], 'direction gold/target/group mismatch')
        raw = pred['logits'][mode]
        require(len(raw) == 2 and np.isfinite(raw).all(), 'invalid direction logits')
        guess = 'AB'[int(np.argmax(raw))]
        require(pred['predictions'][mode] == guess, 'direction label/logit map mismatch')
        labels.append(row['label']); guesses.append(guess); groups.append(row['group_id'])
    return METRICS.pair_direction(labels, guesses, groups)


def kliche(root, ledger):
    ctx = REAL.inputs(root)
    for name in (*REAL.ORIGINAL_HASHES, *REAL.T3_FILES, 't3_target_catalog.json'):
        ledger.pin(root / 'data/processed' / name, 'gold_or_frozen_input')
    output = {'scope': 'historical_Kliche_direction_small_sample_diagnostic',
              'limitations': ['D2 test 9 pairs; clean-after-Benz-pretraining subset 5 pairs.',
                              'G1 source-consistent 14 pairs have weak directional supervision, not quantitative KD.',
                              'Target exposure in D2 training is reported per subset; no universal unseen-target claim.'],
              'runs': []}
    subsets = ('d2_test', 'd2_test_clean_after_t1_pretraining', 'g1_source_consistent_direction')
    for model in REAL.METHODS:
        for seed in SEEDS:
            mlp = model == 'MLP'
            base = root / 'results/real_run_20261015'
            bridge = root / 'results/real_t3_20261015'
            meta, arrays = REAL.load_t3(model, seed, base, bridge, ctx)
            for subset in subsets:
                rows = ctx['subsets'][subset]
                item = {'model': model, 'seed': seed, 'subset': subset, 'n': len(rows),
                        'target_seen_in_d2_train': sum(ctx['seen_targets'][r['target_id']] for r in rows),
                        'target_unseen_in_d2_train': sum(not ctx['seen_targets'][r['target_id']] for r in rows)}
                if meta['status'] != 'complete':
                    output['runs'].append({**item, 'status': 'unavailable', 'reason': 'missing complete saved run'})
                    continue
                ledger.pin(meta['result'], 'original_result'); ledger.pin(meta['scores'], 'original_prediction_logits')
                path = base / f't3_mlp_seed{seed}.{subset}.predictions.jsonl' if mlp else bridge / f'{model}_seed{seed}/{subset}.predictions.jsonl'
                if not path.exists():
                    output['runs'].append({**item, 'status': 'unavailable', 'reason': 'missing original row predictions'})
                    continue
                predicted = ledger.rows(path, 'original_prediction')
                modes = {'correct': 'correct', 'off': 'closed_raw_zero' if mlp else 'off', 'swapped_messages': 'swapped_messages'}
                scores = {mode: direction_score(rows, predicted, raw_mode) for mode, raw_mode in modes.items()}
                for mode in modes:
                    require(abs(scores[mode]['accuracy'] - float(arrays[subset][mode].mean())) < 1e-10,
                            'Kliche JSONL/NPZ scorer mismatch')
                scores['reversed_input_and_label_accuracy'] = float(arrays[subset]['reversed_input_and_label'].mean())
                scores['prediction_reversal_rate'] = float(arrays[subset]['prediction_reversal_rate'].mean())
                output['runs'].append({**item, 'status': 'complete', 'metrics': scores,
                                      'original_check': {'matched': True, 'method': 'existing frozen-run scorer + independent JSONL logits'}})
    return output


def initialize(root, ledger):
    global METRICS, REAL
    metrics = ledger.pin(root / 'scripts/bench_v02/metrics.py', 'scorer_source')
    real = ledger.pin(root / 'scripts/summarize_real_results.py', 'scorer_source')
    METRICS = load_module(metrics, 'historical_replay_metrics')
    REAL = load_module(real, 'historical_replay_real')
    ledger.pin(Path(__file__), 'runner_source')
    test = Path(__file__).resolve().parents[1] / 'tests/test_replay_existing.py'
    if test.exists():
        ledger.pin(test, 'regression_test_source')


def run(root, output):
    root = root.resolve()
    ledger = Ledger(root)
    initialize(root, ledger)
    panels = {}
    for name, evaluator in [('ELM_unseen_function', elm), ('March_RT', rt), ('Benz_known_positive', benz), ('Kliche_direction', kliche)]:
        try:
            panels[name] = evaluator(root, ledger)
            statuses = [r['status'] for r in panels[name]['runs']]
            panels[name]['status'] = 'complete' if all(s == 'complete' for s in statuses) else 'partial'
        except (ValueError, KeyError, FileNotFoundError) as exc:
            # Preserve an explicit panel failure, never turn malformed predictions into low score or silent omission.
            panels[name] = {'status': 'invalid', 'error': str(exc).replace(str(root), '<project>'), 'runs': []}
    result = {'schema': 'non_pkd_historical_replay.v1', 'execution': 'CPU_only_existing_predictions',
              'historical_replay': True, 'current_V4_M0_M3_multitask_training': False,
              'new_training': False, 'frozen_test_changed': False, 'panels': panels}
    output.mkdir(parents=True, exist_ok=True)
    score = output / 'scores.json'
    score.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    ledger.pin(score, 'derived_output')
    table = output / 'metrics.csv'
    with table.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['panel', 'model', 'seed', 'subset', 'mode', 'metric', 'value', 'n', 'scope'])
        writer.writeheader()
        for panel_name, panel in panels.items():
            for item in panel.get('runs', []):
                if item['status'] != 'complete':
                    continue
                metric_groups = item['metrics']
                if panel_name != 'Kliche_direction':
                    metric_groups = {'standard': metric_groups}
                for mode, metrics in metric_groups.items():
                    if not isinstance(metrics, dict):
                        metrics = {mode: metrics}
                    for metric, value in metrics.items():
                        if isinstance(value, (float, int)) or value is None:
                            writer.writerow({'panel': panel_name, 'model': item['model'], 'seed': item['seed'],
                                             'subset': item.get('subset', 'test'), 'mode': mode, 'metric': metric,
                                             'value': value, 'n': metrics.get('n', item.get('n', '')),
                                             'scope': panel['scope']})
    ledger.pin(table, 'derived_output')
    manifest = {'schema': 'historical_replay_manifest.v1', 'paths_relative_to': 'project_root',
                'inputs_read_only': True, 'files': sorted(ledger.entries.values(), key=lambda r: r['path'])}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or Path(__file__).resolve().parents[1] / 'results/existing_replay_v1'
    result = run(args.root, output)
    statuses = {k: v['status'] for k, v in result['panels'].items()}
    print(json.dumps({'historical_replay': True, 'panels': statuses}, ensure_ascii=False))
    if any(v != 'complete' for v in statuses.values()):
        sys.exit(2)


if __name__ == '__main__':
    main()
