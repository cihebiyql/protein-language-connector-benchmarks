"""Fixed-budget, three-seed CPU B1 comparator with held-out interventions."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from real_data import ROOT, epoch_order, load_t1, make_donors, paired_group_bootstrap, ranking_metrics


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def train_one(seed, outdir, rows, target_ids, raw_features, donors, fingerprint):
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    splits = {split: [i for i, row in enumerate(rows) if row['split'] == split]
              for split in ('train', 'validation', 'test')}
    test_rows = [rows[i] for i in splits['test']]
    target_map = {target: i for i, target in enumerate(target_ids)}
    target = torch.zeros(len(rows), len(target_ids))
    for i, row in enumerate(rows):
        target[i, [target_map[t] for t in row['targets']]] = 1 / len(row['targets'])
    center = raw_features[splits['train']].mean(0)
    scale = raw_features[splits['train']].std(0).clamp_min(1e-5)
    features = (raw_features - center) / scale
    head = nn.Sequential(nn.Linear(640, 256), nn.GELU(), nn.Dropout(.1), nn.Linear(256, len(target_ids)))
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=.01)
    history, best_mrr, best_epoch, best_state = [], -1., None, None
    steps = 0
    for epoch in range(1, 31):
        head.train()
        order = epoch_order(splits['train'], seed, epoch)
        total = 0.
        for batch in order.split(64):
            optimizer.zero_grad(set_to_none=True)
            logits = head(features[batch])
            loss = -(target[batch] * logits.log_softmax(-1)).sum(-1).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            steps += 1
            total += float(loss.detach()) * len(batch)
        head.eval()
        with torch.no_grad():
            val, _ = ranking_metrics(head(features[splits['validation']]).numpy(),
                                     [rows[i] for i in splits['validation']], target_ids)
        history.append({'epoch': epoch, 'train_loss': total / len(order), 'validation': val})
        if val['mrr'] > best_mrr:
            best_mrr, best_epoch = val['mrr'], epoch
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        print(json.dumps({'seed': seed, **history[-1]}), flush=True)
    assert steps == 30 * ((len(splits['train']) + 63) // 64)
    head.load_state_dict(best_state)
    head.eval()
    test_indices = torch.as_tensor(splits['test'])
    with torch.no_grad():
        correct = head(features[test_indices]).numpy()
        closed = head((torch.zeros_like(raw_features[test_indices]) - center) / scale).numpy()
        train_mean = head(torch.zeros_like(features[test_indices])).numpy()
        # Donor indexing uses test-local positions; labels stay with the recipient.
        mismatched = np.stack([head(features[test_indices[torch.as_tensor(donors)[:, d]]]).numpy()
                               for d in range(10)], axis=1)
    scores = {'correct': correct, 'closed_raw_zero': closed, 'closed_train_mean': train_mean}
    summary, predictions_by_mode = {}, {}
    for mode, logits in scores.items():
        summary[mode], predictions = ranking_metrics(logits, test_rows, target_ids)
        predictions_by_mode[mode] = predictions
    mismatch_metrics = []
    mismatch_predictions = []
    for d in range(10):
        measured, predictions = ranking_metrics(mismatched[:, d], test_rows, target_ids)
        mismatch_metrics.append(measured)
        mismatch_predictions.append(predictions)
    summary['mismatched_10_donor_mean'] = {
        'n': len(test_rows), 'n_donors_per_sample': 10,
        'known_positive_recall_at_3': float(np.mean([m['known_positive_recall_at_3'] for m in mismatch_metrics])),
        'mrr': float(np.mean([m['mrr'] for m in mismatch_metrics]))}
    paired = {}
    for field in ('recall_at_3', 'reciprocal_rank'):
        actual = np.array([p[field] for p in predictions_by_mode['correct']])
        comparisons = {mode: np.array([p[field] for p in pred])
                       for mode, pred in predictions_by_mode.items() if mode != 'correct'}
        comparisons['mismatched_10_donor_mean'] = np.mean(
            [[p[field] for p in pred] for pred in mismatch_predictions], axis=0)
        paired[field] = {mode: paired_group_bootstrap(actual, reference, test_rows)
                         for mode, reference in comparisons.items()}
    records = []
    for i, row in enumerate(test_rows):
        records.append({'sample_id': row['sample_id'], 'group_id': row['group_id'],
                        'known_positive_target_ids': row['targets'],
                        'modes': {mode: pred[i] for mode, pred in predictions_by_mode.items()},
                        'donor_sample_ids': [test_rows[j]['sample_id'] for j in donors[i]],
                        'mismatched_draws': [pred[i] for pred in mismatch_predictions],
                        'mismatched_mean_recall_at_3': float(np.mean([p[i]['recall_at_3'] for p in mismatch_predictions])),
                        'mismatched_mean_reciprocal_rank': float(np.mean([p[i]['reciprocal_rank'] for p in mismatch_predictions]))})
    prefix = outdir / f'mlp_seed{seed}'
    prefix.with_suffix('.predictions.jsonl').write_text(
        '\n'.join(json.dumps(record) for record in records) + '\n', encoding='utf-8')
    np.savez_compressed(prefix.with_suffix('.scores.npz'), correct=correct, closed_raw_zero=closed,
                        closed_train_mean=train_mean, mismatched=mismatched,
                        sample_ids=np.asarray([r['sample_id'] for r in test_rows]),
                        target_ids=np.asarray(target_ids), donor_indices=np.asarray(donors))
    torch.save({'head': best_state, 'center': center, 'scale': scale, 'target_ids': target_ids,
                'seed': seed, 'best_epoch': best_epoch, 'input_sha256': fingerprint},
               prefix.with_suffix('.checkpoint.pt'))
    result = {'model': 'frozen ESM2-150M layer30 residue mean + 640/256/34 MLP',
              'seed': seed, 'epochs': 30, 'optimizer_steps': steps, 'effective_batch_size': 64,
              'optimizer': {'type': 'AdamW', 'lr': 3e-4, 'weight_decay': .01, 'betas': [.9, .999]},
              'loss': 'equal-sample known-positive soft-label cross entropy',
              'order': 'CPU torch.Generator(seed + epoch), torch.randperm; epoch=1..30',
              'selection': 'earliest highest validation MRR across all 30 epochs; no early stopping',
              'best_epoch': best_epoch, 'best_validation_mrr': best_mrr,
              'split_sizes': {k: len(v) for k, v in splits.items()},
              'input_sha256': fingerprint, 'scores': summary, 'paired_intervention_cis': paired,
              'interventions': {'closed_raw_zero': 'raw pooled ESM feature set to zero before train-only standardization',
                                'closed_train_mean': 'raw pooled ESM feature replaced by training mean; sensitivity check',
                                'mismatched': '10 distinct same-test donor pooled features, different group, length +/-2, disjoint documented targets; per-draw metric then mean'},
              'trainable_parameters': sum(p.numel() for p in head.parameters()),
              'seconds': time.perf_counter() - started, 'history': history,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_json(prefix.with_suffix('.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 42, 2026])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/real_run_20261015')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    rows, target_ids = load_t1()
    fingerprint = hashlib.sha256((ROOT / 'data/processed/benz_samples.jsonl').read_bytes()).hexdigest()
    cache = torch.load(ROOT / 'data/cache/benz_esm150m.pt', map_location='cpu', weights_only=True)
    assert cache['input_sha256'] == fingerprint
    assert cache['sample_ids'] == [row['sample_id'] for row in rows]
    raw_features = cache['features_30'].float()
    assert raw_features.shape == (len(rows), 640) and torch.isfinite(raw_features).all()
    test_rows = [r for r in rows if r['split'] == 'test']
    donors = make_donors(test_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / 'mlp_donors.json', {'seed': 42, 'test_sample_ids': [r['sample_id'] for r in test_rows],
                                             'donor_indices': donors})
    results = [train_one(seed, args.output, rows, target_ids, raw_features, donors, fingerprint)
               for seed in args.seeds]
    save_json(args.output / 'mlp_summary.json', {
        'seeds': args.seeds, 'device': 'cpu', 'torch_version': torch.__version__,
        'threads': args.threads, 'input_sha256': fingerprint,
        'runs': [{k: run[k] for k in ('seed', 'best_epoch', 'scores', 'seconds', 'optimizer_steps')}
                 for run in results],
        'across_seed_mean': {mode: {metric: float(np.mean([run['scores'][mode][metric] for run in results]))
                                    for metric in ('known_positive_recall_at_3', 'mrr')}
                             for mode in results[0]['scores']},
        'inference_scope': 'fixed Benz held-out groups; seed spread is descriptive, not three independent test sets'})


if __name__ == '__main__':
    main()
