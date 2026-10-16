"""Shared fixed-data, ordering, and evaluation rules for the real-data comparison."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_t1():
    """Read the audited Benz split without sorting or creating a new split."""
    rows = [json.loads(line) for line in (ROOT / 'data/processed/benz_samples.jsonl')
            .read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = [row for row in rows if row['split'] in ('train', 'validation', 'test')]
    target_ids = [row['target_id'] for row in json.loads(
        (ROOT / 'data/processed/benz_targets.json').read_text(encoding='utf-8'))]
    assert len(set(target_ids)) == len(target_ids)
    assert len({row['sample_id'] for row in rows}) == len(rows)
    groups = {split: {r['group_id'] for r in rows if r['split'] == split}
              for split in ('train', 'validation', 'test')}
    assert all(groups.values())
    assert all(not groups[a] & groups[b] for a, b in
               [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
    assert all(row['sequence'] and row['targets'] and set(row['targets']) <= set(target_ids)
               for row in rows)
    return rows, target_ids


def ranking_metrics(scores, rows, target_ids):
    """Known-positive metrics; absent interactions are never treated as negatives."""
    scores = np.asarray(scores)
    if scores.shape != (len(rows), len(target_ids)) or not np.isfinite(scores).all():
        raise ValueError('Scores must be finite with shape [samples, targets]')
    rankings = np.argsort(-scores, axis=1, kind='stable')
    recalls, reciprocals, predictions = [], [], []
    for order, row in zip(rankings, rows):
        truth = set(row['targets'])
        if not truth or not truth <= set(target_ids):
            raise ValueError('Each sample needs known positives in the target vocabulary')
        ranked = [target_ids[i] for i in order]
        recalls.append(len(truth.intersection(ranked[:3])) / len(truth))
        reciprocals.append(1 / next(rank for rank, target in enumerate(ranked, 1) if target in truth))
        predictions.append({'sample_id': row['sample_id'],
                            'known_positive_target_ids': row['targets'],
                            'ranked_target_ids': ranked, 'recall_at_3': recalls[-1],
                            'reciprocal_rank': reciprocals[-1]})
    return {'n': len(rows), 'known_positive_recall_at_3': float(np.mean(recalls)),
            'mrr': float(np.mean(reciprocals))}, predictions


def epoch_order(train_indices, seed, epoch):
    """CPU-generated order shared by CPU and GPU runs; epochs start at one."""
    indices = torch.as_tensor(train_indices, dtype=torch.long, device='cpu')
    return indices[torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed + epoch))]


def make_donors(test_rows, seed=42, n=10):
    """Distinct hash-stable donors from the same test set, with strict matching.

    Different groups, length within two residues, and disjoint documented targets
    are required. Disjoint documentation does not imply biological nonbinding.
    """
    if n < 1 or any(row.get('split') != 'test' for row in test_rows):
        raise ValueError('Donors require n >= 1 and test-only rows')
    if len({row['sample_id'] for row in test_rows}) != len(test_rows):
        raise ValueError('Test sample IDs must be unique')
    result = []
    for row in test_rows:
        candidates = [i for i, donor in enumerate(test_rows)
                      if donor['group_id'] != row['group_id']
                      and abs(len(donor['sequence']) - len(row['sequence'])) <= 2
                      and not set(row['targets']).intersection(donor['targets'])]
        candidates.sort(key=lambda i: (hashlib.sha256(
            f"{seed}|{row['sample_id']}|{test_rows[i]['sample_id']}".encode()).digest(),
            test_rows[i]['sample_id']))
        if len(candidates) < n:
            raise ValueError(f"{row['sample_id']} has {len(candidates)} eligible donors; need {n}")
        result.append(candidates[:n])
    return result


def paired_group_bootstrap(actual, reference, rows, seed=2026, n_boot=2000):
    """Percentile CI for the paired sample-mean difference, resampling groups.

    Arrays have one scalar per test sample; repeated mismatch draws must first
    be averaged within sample. Duplicate group draws retain all group members.
    """
    difference = np.asarray(actual, dtype=float) - np.asarray(reference, dtype=float)
    if difference.shape != (len(rows),) or not np.isfinite(difference).all():
        raise ValueError('Expected one finite paired difference per row')
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row['group_id'], []).append(i)
    members = [np.asarray(groups[group], dtype=int) for group in sorted(groups)]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        selected = rng.integers(len(members), size=len(members))
        boot[b] = difference[np.concatenate([members[i] for i in selected])].mean()
    return {'mean_difference': float(difference.mean()), 'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'n_samples': len(rows), 'n_groups': len(members), 'bootstrap_draws': n_boot,
            'bootstrap_seed': seed, 'unit': 'test group_id; sample-weighted mean'}
