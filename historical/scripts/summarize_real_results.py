"""Recompute fixed-budget real-data comparisons from sample-level logits.

Missing runs are pending; malformed/stale artifacts are invalid. Neither is
silently dropped from a three-seed comparison. No model or GPU is loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 42, 2026)
METHODS = ('MLP', 'L30-J0', 'L15-J14')
ORIGINAL_HASHES = {
    'benz_samples.jsonl': '57038156925fb5d772824716eef68cf6ce55d288175913e5d86a095007c9e8da',
    'benz_split_manifest.json': '530cd6db379132b353cb1eff6776552232efc1c0e77013eae072cbe716ce492a',
    'benz_targets.json': 'ce93a9376f8d5c7f4bff4138b6d67b83e86e28014be6338aae361e6ba8e8689f',
}
T3_DERIVED_FILE = 't3_g1_source_consistent_direction_pairs.jsonl'
T3_FILES = ('t3_d2_pairs.jsonl', 't3_g1_direction_pairs.jsonl', 't3_split_manifest.json', T3_DERIVED_FILE)


class IntegrityError(ValueError):
    """An existing artifact violates the fixed comparison contract."""


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(root):
    data = root / 'data/processed'
    for name, expected in ORIGINAL_HASHES.items():
        require(sha256(data / name) == expected, f'Original Benz input changed: {name}')
    t1 = read_rows(data / 'benz_samples.jsonl')
    targets = [r['target_id'] for r in read_json(data / 'benz_targets.json')]
    require(len(targets) == len(set(targets)) == 34, 'T1 candidate catalog changed')
    require({s: sum(r['split'] == s for r in t1) for s in ('train', 'validation', 'test')}
            == {'train': 1914, 'validation': 393, 'test': 387}, 'T1 split sizes changed')
    d2 = read_rows(data / T3_FILES[0])
    g1 = read_rows(data / T3_FILES[1])
    manifest = read_json(data / T3_FILES[2])
    require(len(d2) == 59 and manifest['d2_pairs'] == 59, 'Expected final QC59 D2 data')
    require({s: sum(r['split'] == s for r in d2) for s in ('train', 'val', 'test')}
            == manifest['d2_splits'] == {'train': 34, 'val': 16, 'test': 9}, 'D2 splits changed')
    test = [r for r in d2 if r['split'] == 'test']
    clean = [r for r in test if r['clean_after_t1_pretraining']]
    consistent = [r for r in g1 if r['sequence_source_consistent']]
    require(read_rows(data / T3_DERIVED_FILE) == consistent,
            'Source-consistent G1 file differs from the predeclared full-G1 subset')
    require(len(clean) == 5 and len(g1) == 17 and len(consistent) == 14, 'T3 subset sizes changed')
    for name, rows, key in [('T1', t1, 'sample_id'), ('D2', d2, 'pair_id'), ('G1', g1, 'pair_id')]:
        require(len({r[key] for r in rows}) == len(rows), f'Duplicate {name} IDs')
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        require(not {r['group_id'] for r in d2 if r['split'] == a}
                & {r['group_id'] for r in d2 if r['split'] == b}, 'D2 cross-split groups')
    catalog = read_json(data / 't3_target_catalog.json')
    seen = {r['target_id']: r['seen_in_d2_train'] for r in catalog['entries']}
    require(set(seen) >= {r['target_id'] for r in d2 + g1}, 'Unknown T3 target')
    require({k for k, v in seen.items() if v} == {r['target_id'] for r in d2 if r['split'] == 'train'},
            'Target training-exposure catalog stale')
    return {'t1': [r for r in t1 if r['split'] == 'test'], 'targets': targets,
            'subsets': {'d2_test': test, 'd2_test_clean_after_t1_pretraining': clean,
                        'g1_direction': g1, 'g1_source_consistent_direction': consistent},
            'seen_targets': seen, 't3_hashes': {name: sha256(data / name) for name in T3_FILES},
            'target_catalog_sha256': sha256(data / 't3_target_catalog.json')}


def group_draws(rows, seed=2026, n_boot=2000):
    require(bool(rows), 'Cannot bootstrap an empty subset')
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row['group_id'], []).append(i)
    members = [np.asarray(groups[key], dtype=int) for key in sorted(groups)]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[g] for g in rng.integers(len(members), size=len(members))])
            for _ in range(n_boot)]


def interval(values, rows, labels=None, n_boot=2000):
    """CI of one sample-level scalar, optionally class balanced after resampling."""
    values = np.asarray(values, dtype=float)
    require(values.shape == (len(rows),) and np.isfinite(values).all(), 'Invalid sample-level values')
    classes = np.unique(labels) if labels is not None else None

    def score(indices):
        if labels is None:
            return values[indices].mean()
        if set(np.unique(labels[indices])) != set(classes):
            return None
        return np.mean([values[indices][labels[indices] == c].mean() for c in classes])

    boot = [score(index) for index in group_draws(rows, n_boot=n_boot)]
    boot = [v for v in boot if v is not None]
    return {'estimate': float(score(np.arange(len(rows)))),
            'ci95': np.quantile(boot, [.025, .975]).tolist() if boot else None,
            'n_samples': len(rows), 'n_groups': len({r['group_id'] for r in rows}),
            'bootstrap_draws': n_boot, 'valid_draws': len(boot), 'bootstrap_seed': 2026,
            'unit': 'group_id; retain all paired rows/pockets; sample-weighted',
            'seed_rule': 'average three seed-specific sample metrics before group resampling'}


def describe(values):
    values = np.asarray(values, dtype=float)
    require(values.shape == (3,) and np.isfinite(values).all(), 'Expected exactly three finite seed metrics')
    return {'mean': float(values.mean()), 'sd': float(values.std(ddof=1)),
            'by_seed': dict(zip(map(str, SEEDS), values.tolist()))}


def ranking_arrays(scores, rows, targets):
    scores = np.asarray(scores)
    require(scores.shape == (len(rows), len(targets)) and np.isfinite(scores).all(), 'Invalid T1 logits')
    order = np.argsort(-scores, axis=1, kind='stable')
    recall, reciprocal = [], []
    for ranked, row in zip(order, rows):
        positives = {targets.index(t) for t in row['targets']}
        require(bool(positives), 'Missing documented positives')
        recall.append(len(positives.intersection(ranked[:3])) / len(positives))
        reciprocal.append(1 / next(i for i, t in enumerate(ranked, 1) if t in positives))
    return {'recall_at_3': np.asarray(recall), 'mrr': np.asarray(reciprocal)}


def expected_donors(rows):
    donors = []
    for row in rows:
        eligible = [i for i, other in enumerate(rows) if other['group_id'] != row['group_id']
                    and abs(len(other['sequence']) - len(row['sequence'])) <= 2
                    and not set(row['targets']) & set(other['targets'])]
        eligible.sort(key=lambda i: (hashlib.sha256(
            f"42|{row['sample_id']}|{rows[i]['sample_id']}".encode()).digest(), rows[i]['sample_id']))
        require(len(eligible) >= 10, 'Insufficient matched donors')
        donors.append(eligible[:10])
    return np.asarray(donors)


def check_run(result, seed, steps, task, ctx):
    config = result.get('config', result)
    require(config['seed'] == seed, f'{task} seed mismatch')
    require(result['optimizer_steps'] == steps and config['epochs'] == 30, f'{task} budget mismatch')
    history = result['history']
    require(len(history) == 30 and [r['epoch'] for r in history] == list(range(1, 31)), 'Incomplete history')
    require(all(np.isfinite(r['train_loss']) for r in history), 'Nonfinite training history')
    require(result['best_epoch'] in range(1, 31), 'Invalid selected epoch')
    selection_metric = 'mrr' if task == 'T1' else 'balanced_accuracy'
    validation = [row['validation'][selection_metric] for row in history]
    require(np.isfinite(validation).all(), 'Nonfinite validation history')
    require(result['best_epoch'] == int(np.argmax(validation)) + 1,
            'Selected checkpoint is not earliest validation maximum')
    require(all(row.get('optimizer_steps', (i + 1) * (steps // 30)) == (i + 1) * (steps // 30)
                for i, row in enumerate(history)), 'Per-epoch update schedule mismatch')
    if task == 'T1':
        require(config['input_sha256'] == ORIGINAL_HASHES['benz_samples.jsonl'], 'T1 stale input hash')
        require(config['effective_batch_size'] == 64, 'T1 effective batch mismatch')
    else:
        expected_hashes = ctx['t3_hashes']
        recorded_hashes = config['inputs_sha256']
        # Some bridge runs derive the 14-pair panel from the full 17-pair input;
        # newer MLP runs additionally hash the equivalent materialized panel.
        required = set(expected_hashes) - {T3_DERIVED_FILE}
        require(required <= set(recorded_hashes),
                f'T3 incomplete data hashes: missing {sorted(required - set(recorded_hashes))}')
        require(set(recorded_hashes) <= set(expected_hashes),
                f'T3 unrecognized data hashes: {sorted(set(recorded_hashes) - set(expected_hashes))}')
        for name, digest in recorded_hashes.items():
            require(digest == expected_hashes[name], f'T3 stale data hash: {name}')
        require(config['target_catalog_sha256'] == ctx['target_catalog_sha256'], 'T3 stale target catalog')
        require(config['d2_batch_size'] == 32 and config['d1_d2_samples_per_epoch'] == [34, 34],
                'T3 replay/sample budget mismatch')
    return config


def aligned(data, key, rows, row_key):
    require(key in data, f'Missing array {key}')
    require(data[key].tolist() == [r[row_key] for r in rows], f'Incorrect sample order: {key}')


def load_t1(method, seed, base, ctx):
    mlp = method == 'MLP'
    result_path = base / f'mlp_seed{seed}.json' if mlp else base / f'{method}_seed{seed}/result.json'
    score_path = base / f'mlp_seed{seed}.scores.npz' if mlp else result_path.parent / 'test_scores.npz'
    needed = [result_path, score_path]
    if not mlp:
        needed.append(base / 'bridge_donors.json')
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        return {'status': 'pending', 'missing': missing}, None
    result = read_json(result_path)
    config = check_run(result, seed, 900, 'T1', ctx)
    rows, targets = ctx['t1'], ctx['targets']
    if not mlp:
        require(config['configuration'] == method and list(config['alias_map']) == targets,
                'T1 bridge configuration/candidate order mismatch')
    donors = expected_donors(rows)
    with np.load(score_path, allow_pickle=False) as data:
        aligned(data, 'sample_ids', rows, 'sample_id')
        if mlp:
            require(data['target_ids'].tolist() == targets, 'MLP target order mismatch')
            require(np.array_equal(data['donor_indices'], donors), 'MLP donor plan mismatch')
        else:
            plan = read_json(base / 'bridge_donors.json')
            require(plan['seed'] == 42 and plan['test_sample_ids'] == [r['sample_id'] for r in rows]
                    and np.array_equal(plan['donor_indices'], donors), 'Bridge donor plan mismatch')
        arrays = {mode: ranking_arrays(data[key], rows, targets)
                  for mode, key in [('correct', 'correct'), ('off', 'closed_raw_zero' if mlp else 'off')]}
        mismatch = data['mismatched']
        require(mismatch.shape == (len(rows), 10, len(targets)), 'Expected ten donor logits per recipient')
        draws = [ranking_arrays(mismatch[:, i], rows, targets) for i in range(10)]
        arrays['mismatched'] = {k: np.mean([draw[k] for draw in draws], axis=0) for k in draws[0]}
    return {'status': 'complete', 'result': str(result_path), 'scores': str(score_path),
            'optimizer_steps': 900, 'best_epoch': result['best_epoch'], 'loss': config['loss'],
            'sha256': {'result': sha256(result_path), 'scores': sha256(score_path)}}, arrays


def load_t3(method, seed, base, bridge_base, ctx):
    mlp = method == 'MLP'
    result_path = base / f't3_mlp_seed{seed}.json' if mlp else bridge_base / f'{method}_seed{seed}/result.json'
    score_path = base / f't3_mlp_seed{seed}.scores.npz' if mlp else result_path.parent / 'test_scores.npz'
    missing = [str(p) for p in (result_path, score_path) if not p.exists()]
    if missing:
        return {'status': 'pending', 'missing': missing}, None
    result = read_json(result_path)
    config = check_run(result, seed, 60, 'T3', ctx)
    require(mlp or config.get('configuration', config.get('config')) == method, 'T3 bridge configuration mismatch')
    arrays = {}
    with np.load(score_path, allow_pickle=False) as data:
        for subset, rows in ctx['subsets'].items():
            # The source-consistent panel is a predeclared source-only subset of G1.
            source = subset if f'{subset}__pair_ids' in data else 'g1_direction' if subset == 'g1_source_consistent_direction' else subset
            source_rows = ctx['subsets'][source]
            aligned(data, f'{source}__pair_ids', source_rows, 'pair_id')
            labels = np.asarray([0 if r['label'] == 'A' else 1 for r in source_rows])
            if not mlp:
                require(np.array_equal(data[f'{source}__labels'], labels), 'T3 label array mismatch')
                aligned(data, f'{source}__group_ids', source_rows, 'group_id')
            index = np.asarray([next(i for i, r in enumerate(source_rows) if r['pair_id'] == row['pair_id']) for row in rows])
            predictions = {}
            for mode, key in [('correct', 'correct'), ('off', 'closed_raw_zero' if mlp else 'off'),
                              ('swapped_messages', 'swapped_messages')]:
                logits = data[f'{source}__{key}']
                require(logits.shape == (len(source_rows), 2) and np.isfinite(logits).all(), 'Invalid T3 logits')
                predictions[mode] = logits.argmax(-1)[index]
            truth = labels[index]
            arrays[subset] = {mode: (pred == truth).astype(float) for mode, pred in predictions.items()}
            arrays[subset]['reversed_input_and_label'] = (predictions['swapped_messages'] == 1 - truth).astype(float)
            arrays[subset]['prediction_reversal_rate'] = (predictions['correct'] != predictions['swapped_messages']).astype(float)
    return {'status': 'complete', 'result': str(result_path), 'scores': str(score_path),
            'optimizer_steps': 60, 'best_epoch': result['best_epoch'], 'loss': config['loss'],
            'sha256': {'result': sha256(result_path), 'scores': sha256(score_path)}}, arrays


def summarize_t1(runs, rows):
    summary = {'methods': {}, 'paired_comparisons': {}}
    means = {}
    for method in METHODS:
        means[method] = {}
        summary['methods'][method] = {}
        for mode in ('correct', 'off', 'mismatched'):
            summary['methods'][method][mode] = {}
            means[method][mode] = {}
            for metric in ('recall_at_3', 'mrr'):
                stack = np.stack([runs[method][s][mode][metric] for s in SEEDS])
                means[method][mode][metric] = stack.mean(0)
                summary['methods'][method][mode][metric] = {
                    **describe(stack.mean(1)), 'group_ci': interval(stack.mean(0), rows)}
    for actual, reference in [('L15-J14', 'L30-J0'), ('L30-J0', 'MLP'), ('L15-J14', 'MLP')]:
        summary['paired_comparisons'][f'{actual} - {reference}'] = {
            metric: interval(means[actual]['correct'][metric] - means[reference]['correct'][metric], rows)
            for metric in ('recall_at_3', 'mrr')}
    summary['message_effects'] = {
        method: {f'correct - {mode}': {
            metric: interval(means[method]['correct'][metric] - means[method][mode][metric], rows)
            for metric in ('recall_at_3', 'mrr')} for mode in ('off', 'mismatched')} for method in METHODS}
    return summary


def summarize_t3(runs, ctx):
    summary = {}
    for subset, full_rows in ctx['subsets'].items():
        panels = {'all': list(range(len(full_rows)))}
        for name, seen in [('target_seen_in_d2_train', True), ('target_unseen_in_d2_train', False)]:
            selected = [i for i, r in enumerate(full_rows) if ctx['seen_targets'][r['target_id']] == seen]
            if selected:
                panels[name] = selected
        summary[subset] = {}
        for panel, selected in panels.items():
            rows = [full_rows[i] for i in selected]
            labels = np.asarray([r['label'] for r in rows])
            classes = np.unique(labels)
            result = {'n': len(rows), 'n_groups': len({r['group_id'] for r in rows}),
                      'class_counts': {c: int(sum(labels == c)) for c in classes}, 'methods': {}}
            means = {}
            for method in METHODS:
                means[method] = {}
                result['methods'][method] = {}
                for mode in ('correct', 'off', 'swapped_messages', 'reversed_input_and_label', 'prediction_reversal_rate'):
                    stack = np.stack([runs[method][s][subset][mode][selected] for s in SEEDS])
                    means[method][mode] = stack.mean(0)
                    measured = {'accuracy': {**describe(stack.mean(1)), 'group_ci': interval(stack.mean(0), rows)}}
                    if mode == 'prediction_reversal_rate':
                        measured = {'rate': measured['accuracy']}
                    else:
                        balanced = np.mean([stack[:, labels == c].mean(1) for c in classes], axis=0)
                        measured['balanced_accuracy'] = {**describe(balanced),
                                                        'group_ci': interval(stack.mean(0), rows, labels)}
                    result['methods'][method][mode] = measured
            result['paired_comparisons'] = {}
            for actual, reference in [('L15-J14', 'L30-J0'), ('L30-J0', 'MLP'), ('L15-J14', 'MLP')]:
                difference = means[actual]['correct'] - means[reference]['correct']
                result['paired_comparisons'][f'{actual} - {reference}'] = {
                    'accuracy': interval(difference, rows), 'balanced_accuracy': interval(difference, rows, labels)}
            result['message_effects'] = {method: {f'correct - {mode}': {
                'accuracy': interval(means[method]['correct'] - means[method][mode], rows),
                'balanced_accuracy': interval(means[method]['correct'] - means[method][mode], rows, labels)}
                for mode in ('off', 'swapped_messages')} for method in METHODS}
            summary[subset][panel] = result
    return summary


def build_summary(root, run_dir=None, bridge_t3_dir=None):
    ctx = inputs(root)
    base = run_dir or root / 'results/real_run_20261015'
    bridge_base = bridge_t3_dir or root / 'results/real_t3_20261015'
    result = {'schema_version': 1, 'seeds': list(SEEDS), 'input_hashes': ORIGINAL_HASHES,
              't3_input_hashes': ctx['t3_hashes'], 'target_catalog_sha256': ctx['target_catalog_sha256'],
              'budgets': {'T1': {'steps': 900, 'epochs': 30, 'effective_batch_size': 64},
                          'T3': {'steps': 60, 'epochs': 30, 'd1_d2_per_epoch': [34, 34]}},
              'status': 'complete', 'tasks': {}}
    for task in ('T1', 'T3'):
        ledger, runs = {}, {}
        for method in METHODS:
            ledger[method], runs[method] = {}, {}
            for seed in SEEDS:
                try:
                    meta, arrays = (load_t1(method, seed, base, ctx) if task == 'T1'
                                    else load_t3(method, seed, base, bridge_base, ctx))
                except (IntegrityError, KeyError, ValueError, OSError) as error:
                    meta, arrays = {'status': 'invalid', 'error': str(error)}, None
                ledger[method][str(seed)] = meta
                if arrays is not None:
                    runs[method][seed] = arrays
        states = {record['status'] for seeds in ledger.values() for record in seeds.values()}
        status = 'invalid' if 'invalid' in states else 'pending' if 'pending' in states else 'complete'
        result['tasks'][task] = {'status': status, 'runs': ledger}
        if status == 'complete':
            result['tasks'][task]['comparison'] = summarize_t1(runs, ctx['t1']) if task == 'T1' else summarize_t3(runs, ctx)
        if status != 'complete':
            result['status'] = 'invalid' if status == 'invalid' or result['status'] == 'invalid' else 'pending'
    return result


def pretty(value):
    return f"{value['mean']:.4f} ± {value['sd']:.4f}"


def pretty_ci(value):
    lo, hi = value['ci95']
    return f"{value['estimate']:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def render_report(result):
    lines = ['# 真实数据训练与消息干预结果', '', f"整体状态：**{result['status']}**。三方法均要求种子 17、42、2026 完整通过校验后才生成任务比较。", '',
             'T1 沿用原始 Benz 划分：训练 1,914、验证 393、测试 387；每方法每种子 30 轮、900 次优化更新，有效批量 64。T3 使用输入库双端覆盖 QC 后的 59 对：训练 34、验证 16、测试 9；每方法每种子 60 次更新，每轮 D2 与 D1 回放各 34 条。', '',
             '表内数值为相同测试集上的三个种子均值 ± 样本标准差（ddof=1）。所有指标从保存的逐样本 logits 重算。组置信区间先在每个样本内平均供体干预结果和种子指标，再按 group_id 抽样 2,000 次（随机种子 2026），保留同组所有记录及重复口袋。区间表示固定模型运行下测试组抽样差异；种子离散度另外报告。', '']
    for task, payload in result['tasks'].items():
        lines += [f'## {task}', '', f"状态：**{payload['status']}**。", '']
        if payload['status'] != 'complete':
            for method, seeds in payload['runs'].items():
                for seed, meta in seeds.items():
                    detail = meta.get('error', '; '.join(meta.get('missing', [])))
                    lines.append(f"- {method} / {seed}：{meta['status']}" + (f'；{detail}' if detail else ''))
            lines += ['', '完整三种子比较待运行与校验齐全后生成。', '']
            continue
        comp = payload['comparison']
        if task == 'T1':
            lines += ['| 方法 | 消息 | Recall@3 | MRR |', '|---|---|---:|---:|']
            for method in METHODS:
                for mode in ('correct', 'off', 'mismatched'):
                    vals = comp['methods'][method][mode]
                    lines.append(f"| {method} | {mode} | {pretty(vals['recall_at_3'])} | {pretty(vals['mrr'])} |")
            lines += ['', '错配条件为每个接收样本 10 个固定、不同组、长度差 ≤2、已知靶点互斥的供体。先逐供体计算排名指标，再在接收样本内平均。', '',
                      '| 正确消息条件的配对差值 | Recall@3 差值 [95% CI] | MRR 差值 [95% CI] |', '|---|---:|---:|']
            for name, values in comp['paired_comparisons'].items():
                lines.append(f"| {name} | {pretty_ci(values['recall_at_3'])} | {pretty_ci(values['mrr'])} |")
            lines += ['', '| 方法 / 干预差值 | Recall@3 差值 [95% CI] | MRR 差值 [95% CI] |', '|---|---:|---:|']
            for method, effects in comp['message_effects'].items():
                for mode, values in effects.items():
                    lines.append(f"| {method} / {mode} | {pretty_ci(values['recall_at_3'])} | {pretty_ci(values['mrr'])} |")
        else:
            for subset, panels in comp.items():
                full = panels['all']
                lines += ['', f"### {subset}（{full['n']} 对 / {full['n_groups']} 个母体组）", '',
                          '| 方法 | 消息 | accuracy | balanced accuracy |', '|---|---|---:|---:|']
                for method in METHODS:
                    for mode in ('correct', 'off', 'swapped_messages', 'reversed_input_and_label'):
                        values = full['methods'][method][mode]
                        lines.append(f"| {method} | {mode} | {pretty(values['accuracy'])} | {pretty(values['balanced_accuracy'])} |")
                lines += ['', '| 方法 | 预测反转率 |', '|---|---:|']
                for method in METHODS:
                    lines.append(f"| {method} | {pretty(full['methods'][method]['prediction_reversal_rate']['rate'])} |")
                lines += ['', '| 正确消息条件的配对差值 | accuracy 差值 [95% CI] | balanced accuracy 差值 [95% CI] |', '|---|---:|---:|']
                for name, values in full['paired_comparisons'].items():
                    lines.append(f"| {name} | {pretty_ci(values['accuracy'])} | {pretty_ci(values['balanced_accuracy'])} |")
                lines += ['', '| 方法 / 干预差值 | accuracy 差值 [95% CI] | balanced accuracy 差值 [95% CI] |', '|---|---:|---:|']
                for method, effects in full['message_effects'].items():
                    for mode, values in effects.items():
                        lines.append(f"| {method} / {mode} | {pretty_ci(values['accuracy'])} | {pretty_ci(values['balanced_accuracy'])} |")
                for panel in ('target_seen_in_d2_train', 'target_unseen_in_d2_train'):
                    if panel in panels:
                        measured = panels[panel]
                        lines += ['', f"{panel}：{measured['n']} 对 / {measured['n_groups']} 组。正确消息 accuracy / balanced accuracy：" + '；'.join(
                            f"{method} {pretty(measured['methods'][method]['correct']['accuracy'])} / {pretty(measured['methods'][method]['correct']['balanced_accuracy'])}" for method in METHODS) + '。']
        lines.append('')
    lines += ['## 解释范围与可复核材料', '',
              'L30-J0 与 L15-J14 同时改变 ESM 读取层和 Qwen 写入位置。它们的差值对应联合配置差异；内部写入的单独贡献需要 L30-J14 与 L15-J0 配套消融。消息干预差值检验当前输入消息的作用。', '',
              '三方法匹配训练样本、次序、轮数与更新次数。桥使用全词表单答案 token 交叉熵，MLP 使用候选类别交叉熵，计算量、参数化及损失归一化范围各异。MLP 的 off 将原始池化特征置零后沿用训练均值/方差；桥的 off 关闭写入消息。', '',
              'T3 D2 标签衡量筛选计数方向；p 值与具体配对的映射尚未验证。G1 使用真实亲和力测量的可分辨方向，主面板 17 对，来源序列一致性敏感性面板 14 对。D2 clean 面板 5 对排除了 T1 预训练已见序列/保守近邻暴露。反转面板交换 A/B 消息并同步反转参考标签；swap 面板保留原参考标签。', '',
              '小样本子集按母体/近序列组计算 CI；balanced accuracy 的 bootstrap 丢弃缺失原面板类别的抽样，machine JSON 记录有效次数。只有一个类别的分层仍列出 balanced accuracy，此时等于该类召回，不能反映双类平衡表现。', '',
              '机器可读结果：`results/real_run_20261015/final_comparison.json`。各运行的路径、输入与结果 SHA-256、选中轮次、训练预算及逐方法各条件置信区间见该文件。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--t3-bridge-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path, help='Write the final human-readable report only when explicitly requested')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    result = build_summary(args.root, args.run_dir, args.t3_bridge_dir)
    output = args.output or args.root / 'results/real_run_20261015/final_comparison.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(result), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'tasks': {k: v['status'] for k, v in result['tasks'].items()},
                      'output': str(output)}, ensure_ascii=False))
    if result['status'] == 'invalid' or (args.require_complete and result['status'] != 'complete'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
