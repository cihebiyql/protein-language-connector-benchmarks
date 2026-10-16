"""T1-initialized paired MLP, D1 replay, and held-out D2/G1 interventions."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from real_data import ROOT, epoch_order, load_t1


def read_rows(name):
    return [json.loads(line) for line in (ROOT / 'data/processed' / name)
            .read_text(encoding='utf-8').splitlines() if line.strip()]


def flip_training_pair(seed, epoch, pair_id):
    return bool(hashlib.sha256(f'{seed}|{epoch}|{pair_id}'.encode()).digest()[0] % 2)


def order_fingerprints(d1_rows, d2_rows, seeds=(17, 42, 2026)):
    """Portable JSON-encoded order hashes, including truncated D1 replay order."""
    d1_indices = [i for i, row in enumerate(d1_rows) if row['split'] == 'train']
    d2_indices = [i for i, row in enumerate(d2_rows) if row['split'] == 'train']
    result = []
    for seed in seeds:
        for epoch in range(1, 31):
            orders = {'t1_train': epoch_order(d1_indices, seed, epoch),
                      't3_d2_train': epoch_order(d2_indices, seed, epoch),
                      't3_d1_replay': epoch_order(d1_indices, seed + 100000, epoch)[:len(d2_indices)]}
            for name, order in orders.items():
                result.append({'stream': name, 'seed': seed, 'epoch': epoch, 'n': len(order),
                               'sha256': hashlib.sha256(json.dumps(order.tolist(), separators=(',', ':')).encode()).hexdigest()})
    return {'torch_version': torch.__version__, 'encoding': 'utf8 compact JSON integer array',
            'd2_sha256': hashlib.sha256((ROOT / 'data/processed/t3_d2_pairs.jsonl').read_bytes()).hexdigest(),
            'fingerprints': result}


def pair_metrics(predictions, labels):
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    if predictions.shape != labels.shape or not len(labels):
        raise ValueError('Expected nonempty, aligned predictions and labels')
    return {'n': len(labels), 'accuracy': float(np.mean(predictions == labels)),
            'balanced_accuracy': float(np.mean([np.mean(predictions[labels == c] == c)
                                               for c in np.unique(labels)])),
            'class_counts': {str(c): int(np.sum(labels == c)) for c in np.unique(labels)}}


def grouped_metric_cis(predictions, labels, rows, seed=2026, n_boot=2000):
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row['group_id'], []).append(i)
    members = [np.asarray(groups[k]) for k in sorted(groups)]
    rng, accuracy, balanced = np.random.default_rng(seed), [], []
    original_classes = set(labels)
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(len(members), size=len(members))])
        result = pair_metrics(predictions[idx], labels[idx])
        accuracy.append(result['accuracy'])
        if set(labels[idx]) == original_classes:
            balanced.append(result['balanced_accuracy'])
    return {'accuracy_ci95': np.quantile(accuracy, [.025, .975]).tolist(),
            'balanced_accuracy_ci95': np.quantile(balanced, [.025, .975]).tolist() if balanced else None,
            'balanced_accuracy_valid_draws': len(balanced), 'bootstrap_draws': n_boot,
            'bootstrap_seed': seed, 'n_groups': len(members),
            'group_unit': 'group_id (entire parent / near-sequence component; shared pockets retained together)',
            'balanced_ci_rule': 'omit bootstrap draws missing a class present in the original evaluated subset'}


class PairedMLP(nn.Module):
    def __init__(self, t1_checkpoint, target_count):
        super().__init__()
        self.front = nn.Sequential(nn.Linear(640, 256), nn.GELU(), nn.Dropout(.1))
        self.t1_head = nn.Linear(256, 34)
        state = t1_checkpoint['head']
        self.front[0].load_state_dict({'weight': state['0.weight'], 'bias': state['0.bias']})
        self.t1_head.load_state_dict({'weight': state['3.weight'], 'bias': state['3.bias']})
        self.pair_head = nn.Sequential(nn.Linear(4 * 256 + target_count, 256), nn.GELU(),
                                       nn.Dropout(.1), nn.Linear(256, 2))

    def forward(self, a, b, target):
        ha, hb = self.front(a), self.front(b)
        return self.pair_head(torch.cat((ha, hb, ha - hb, (ha - hb).abs(), target), -1))

    def d1_logits(self, features):
        return self.t1_head(self.front(features))


def pooled_cache():
    cache = torch.load(ROOT / 'data/cache/real_residues.pt', weights_only=True, map_location='cpu')
    assert cache['bos_eos_padding_removed']
    mask = cache['mask'].float()
    assert torch.all(mask.sum(1) > 0)
    # Cast before reduction, preserving the residue-mean rule used in T1.
    pooled = (cache['features30'].float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    return {seq: pooled[i] for i, seq in enumerate(cache['sequences'])}


def pair_tensors(rows, pooled, center, scale, target_map):
    a = torch.stack([pooled[r['sequence_a']] for r in rows])
    b = torch.stack([pooled[r['sequence_b']] for r in rows])
    targets = F.one_hot(torch.tensor([target_map[r['target_id']] for r in rows]), len(target_map)).float()
    labels = torch.tensor([0 if r['label'] == 'A' else 1 for r in rows])
    return (a - center) / scale, (b - center) / scale, targets, labels


def evaluate(model, rows, pooled, center, scale, target_map):
    a, b, targets, labels = pair_tensors(rows, pooled, center, scale, target_map)
    off = -center.expand_as(a) / scale
    model.eval()
    with torch.no_grad():
        logits = {'correct': model(a, b, targets).numpy(),
                  'closed_raw_zero': model(off, off, targets).numpy(),
                  'swapped_messages': model(b, a, targets).numpy()}
    predictions = {mode: values.argmax(-1) for mode, values in logits.items()}
    labels_np = labels.numpy()
    metrics = {mode: {**pair_metrics(pred, labels_np),
                      **grouped_metric_cis(pred, labels_np, rows)} for mode, pred in predictions.items()}
    metrics['reversed_input_and_label'] = {
        **pair_metrics(predictions['swapped_messages'], 1 - labels_np),
        **grouped_metric_cis(predictions['swapped_messages'], 1 - labels_np, rows)}
    metrics['prediction_reversal_rate'] = float(np.mean(predictions['correct'] != predictions['swapped_messages']))
    metrics['n_distinct_series'] = len({r.get('series_id', r['group_id']) for r in rows})
    records = []
    for i, row in enumerate(rows):
        records.append({'pair_id': row['pair_id'], 'group_id': row['group_id'],
                        'series_id': row.get('series_id'), 'target_id': row['target_id'],
                        'target_index': target_map[row['target_id']], 'label': row['label'],
                        'clean_after_t1_pretraining': row.get('clean_after_t1_pretraining', row.get('clean_external')),
                        'logits': {mode: values[i].tolist() for mode, values in logits.items()},
                        'predictions': {mode: 'AB'[pred[i]] for mode, pred in predictions.items()},
                        'prediction_reverses': bool(predictions['correct'][i] != predictions['swapped_messages'][i])})
    return metrics, records, logits


def train_seed(seed, d2, g1, catalog, pooled, d1_rows, d1_targets, d1_features, outdir):
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    initial_path = outdir / f'mlp_seed{seed}.checkpoint.pt'
    checkpoint = torch.load(initial_path, weights_only=True, map_location='cpu')
    assert checkpoint['input_sha256'] == hashlib.sha256(
        (ROOT / 'data/processed/benz_samples.jsonl').read_bytes()).hexdigest()
    center, scale = checkpoint['center'], checkpoint['scale']
    target_map = {entry['target_id']: entry['index'] for entry in catalog['entries']}
    model = PairedMLP(checkpoint, len(target_map))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    a, b, target, labels = pair_tensors(d2, pooled, center, scale, target_map)
    split = {s: [i for i, row in enumerate(d2) if row['split'] == s] for s in ('train', 'val', 'test')}
    d1_train = [i for i, r in enumerate(d1_rows) if r['split'] == 'train']
    d1_target_map = {v: i for i, v in enumerate(d1_targets)}
    d1_labels = torch.zeros(len(d1_rows), len(d1_targets))
    for i, row in enumerate(d1_rows):
        d1_labels[i, [d1_target_map[t] for t in row['targets']]] = 1 / len(row['targets'])
    d1_features = (d1_features - center) / scale
    history, best, best_epoch, best_state, steps = [], -1., None, None, 0
    for epoch in range(1, 31):
        model.train()
        order = epoch_order(split['train'], seed, epoch)
        replay = epoch_order(d1_train, seed + 100000, epoch)[:len(order)]
        offset, total, d1_total, d2_total = 0, 0., 0., 0.
        for indices in order.split(32):
            n = len(indices)
            replay_indices = replay[offset:offset + n]
            offset += n
            flips = torch.tensor([flip_training_pair(seed, epoch, d2[i]['pair_id']) for i in indices])
            aa = torch.where(flips[:, None], b[indices], a[indices])
            bb = torch.where(flips[:, None], a[indices], b[indices])
            yy = torch.where(flips, 1 - labels[indices], labels[indices])
            optimizer.zero_grad(set_to_none=True)
            pair_loss = F.cross_entropy(model(aa, bb, target[indices]), yy)
            replay_logits = model.d1_logits(d1_features[replay_indices])
            replay_loss = -(d1_labels[replay_indices] * replay_logits.log_softmax(-1)).sum(-1).mean()
            loss = .5 * (pair_loss + replay_loss)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite T3 training loss')
            loss.backward()
            optimizer.step()
            steps += 1
            total += loss.item() * n
            d1_total += replay_loss.item() * n
            d2_total += pair_loss.item() * n
        model.eval()
        with torch.no_grad():
            val_pred = model(a[split['val']], b[split['val']], target[split['val']]).argmax(-1).numpy()
        val = pair_metrics(val_pred, labels[split['val']].numpy())
        history.append({'epoch': epoch, 'train_loss': total / len(order),
                        'd1_replay_loss': d1_total / len(order), 'd2_loss': d2_total / len(order),
                        'validation': val})
        if val['balanced_accuracy'] > best:
            best, best_epoch = val['balanced_accuracy'], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(json.dumps({'seed': seed, **history[-1]}), flush=True)
    assert steps == 60
    model.load_state_dict(best_state)
    d2_test = [r for r in d2 if r['split'] == 'test']
    subsets = {'d2_test': d2_test,
               'd2_test_clean_after_t1_pretraining': [r for r in d2_test if r['clean_after_t1_pretraining']],
               'g1_direction': g1,
               'g1_source_consistent_direction': [r for r in g1 if r['sequence_source_consistent']]}
    assert len(subsets['g1_source_consistent_direction']) == 14
    result, records, all_logits = {}, {}, {}
    for name, rows in subsets.items():
        result[name], records[name], logits = evaluate(model, rows, pooled, center, scale, target_map)
        all_logits.update({f'{name}__{mode}': values for mode, values in logits.items()})
        all_logits[f'{name}__pair_ids'] = np.array([r['pair_id'] for r in rows])
    prefix = outdir / f't3_mlp_seed{seed}'
    for name, predictions in records.items():
        (outdir / f't3_mlp_seed{seed}.{name}.predictions.jsonl').write_text(
            '\n'.join(json.dumps(r) for r in predictions) + '\n', encoding='utf-8')
    np.savez_compressed(prefix.with_suffix('.scores.npz'), **all_logits)
    torch.save({'model': best_state, 'center': center, 'scale': scale, 'target_catalog': catalog,
                'seed': seed, 'best_epoch': best_epoch, 't1_initialization': str(initial_path),
                't1_checkpoint_sha256': hashlib.sha256(initial_path.read_bytes()).hexdigest()},
               prefix.with_suffix('.checkpoint.pt'))
    report = {'seed': seed, 'epochs': 30, 'optimizer_steps': steps, 'd2_batch_size': 32,
              'd1_d2_samples_per_epoch': [len(split['train']), len(split['train'])],
              'total_sample_presentations': 30 * 2 * len(split['train']),
              'loss': '0.5 * (mean D1 known-positive CE + mean D2 A/B CE)',
              'optimizer': {'name': 'AdamW', 'lr': 3e-4, 'weight_decay': .01, 'betas': [.9, .999]},
              'selection': 'earliest highest D2 validation balanced accuracy; all 30 epochs; no test selection',
              'best_epoch': best_epoch, 'best_validation_balanced_accuracy': best,
              'flip': 'D2 train only: SHA256(seed|epoch|pair_id) first digest byte % 2; exchange messages and label',
              'order': f"D2 CPU randperm Generator(seed+epoch); replay Generator(seed+100000+epoch), first {len(split['train'])}",
              'initialization': 'same-seed validation-selected T1 MLP; front and T1 head inherited; new pair head',
              'architecture': f'front 640->256 GELU dropout .1; T1 head 256->34; pair [hA,hB,diff,absdiff,target{len(target_map)} onehot]->256 GELU dropout .1->2',
              'normalization': 'fixed inherited T1 train mean/std; no D2/G1 numerical preprocessing fit',
              'closed_definition': 'both raw mean-pooled residue features zeroed before inherited normalization; target unchanged',
              'swap_definition': 'exchange A/B messages only, fixed target and original reference label; reversal evaluated with complementary reference separately',
              'target_identity': 'fixed measurement catalog; opaque onehot, no raw gene/domain semantics; target columns absent from D2 train receive no target-specific gradient (AdamW weight decay still applies)',
              'target_training_exposure': {
                  name: {'rows_with_target_absent_from_d2_train': sum(
                      not catalog['entries'][target_map[r['target_id']]]['seen_in_d2_train'] for r in subset),
                         'n': len(subset)} for name, subset in subsets.items()},
              'data_scope': {
                  key: json.loads((ROOT / 'data/processed/t3_split_manifest.json').read_text(encoding='utf-8'))[key]
                  for key in ('d2_label_strength', 'input_library_qc', 'input_coverage_excluded_count')},
              'target_catalog_sha256': hashlib.sha256((ROOT / 'data/processed/t3_target_catalog.json').read_bytes()).hexdigest(),
              'inputs_sha256': {name: hashlib.sha256((ROOT / 'data/processed' / name).read_bytes()).hexdigest()
                                for name in ('t3_d2_pairs.jsonl', 't3_g1_direction_pairs.jsonl',
                                             't3_g1_source_consistent_direction_pairs.jsonl', 't3_split_manifest.json')},
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'trainable_parameters': sum(p.numel() for p in model.parameters()),
              'scores': result, 'history': history, 'seconds': time.perf_counter() - started}
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[17, 42, 2026])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/real_run_20261015')
    parser.add_argument('--order-fingerprints', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    d2, g1 = read_rows('t3_d2_pairs.jsonl'), read_rows('t3_g1_direction_pairs.jsonl')
    manifest = json.loads((ROOT / 'data/processed/t3_split_manifest.json').read_text(encoding='utf-8'))
    expected_counts = manifest['d2_splits']
    assert len(d2) == manifest['d2_pairs'] and len(g1) == 17
    assert {s: sum(r['split'] == s for r in d2) for s in ('train', 'val', 'test')} == expected_counts
    assert expected_counts == {'train': 34, 'val': 16, 'test': 9}, 'Wait for the dual-endpoint-coverage QC data freeze'
    assert manifest['input_library_qc'] == 'both EV1 peptide_coverage_counts > 0'
    assert all(r['input_library_coverage_passed']
               and r['source_evidence']['ev1_wt_input_library_coverage'] > 0
               and r['source_evidence']['ev1_mut_input_library_coverage'] > 0 for r in d2)
    assert len([r for r in g1 if r['sequence_source_consistent']]) == manifest['g1_source_consistent_scored'] == 14
    assert not {r['gene'] for r in d2} & {r['gene'] for r in read_rows('t3_g1_records.jsonl')}
    catalog = json.loads((ROOT / 'data/processed/t3_target_catalog.json').read_text(encoding='utf-8'))
    d1_rows, d1_targets = load_t1()
    if args.order_fingerprints:
        args.order_fingerprints.write_text(json.dumps(order_fingerprints(d1_rows, d2), indent=2) + '\n', encoding='utf-8')
        return
    cache = torch.load(ROOT / 'data/cache/benz_esm150m.pt', weights_only=True, map_location='cpu')
    assert cache['sample_ids'] == [r['sample_id'] for r in d1_rows]
    assert cache['input_sha256'] == hashlib.sha256(
        (ROOT / 'data/processed/benz_samples.jsonl').read_bytes()).hexdigest()
    pooled = pooled_cache()
    reports = [train_seed(seed, d2, g1, catalog, pooled, d1_rows, d1_targets,
                          cache['features_30'].float(), args.output) for seed in args.seeds]
    summary = {'seeds': args.seeds, 'device': 'cpu', 'torch_version': torch.__version__, 'threads': args.threads,
               'target_training_exposure': reports[0]['target_training_exposure'], 'runs': [
        {k: r[k] for k in ('seed', 'best_epoch', 'optimizer_steps', 'seconds', 'scores')} for r in reports],
        'across_seed_mean': {subset: {mode: {metric: float(np.mean([
            r['scores'][subset][mode][metric] for r in reports])) for metric in ('accuracy', 'balanced_accuracy')}
            for mode in ('correct', 'closed_raw_zero', 'swapped_messages', 'reversed_input_and_label')}
            for subset in reports[0]['scores']},
        'scope': 'descriptive three-seed means on the same small test sets, not three independent experiments'}
    (args.output / 't3_mlp_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
