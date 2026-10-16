"""Stage-B real-pair bridge training with matched D1 replay and interventions.

The module's data, scheduling, packing and weighting helpers run on CPU for
independent verification. Model construction is restricted to main().
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from cache_real_residues import ROOT, build
from real_bridge_model import FrozenBridgeScorer
from real_data import epoch_order, load_t1, paired_group_bootstrap
from run_t3_mlp import flip_training_pair, grouped_metric_cis, pair_metrics

OUT = ROOT / 'results/real_t3_20261015'
STAGE_A = ROOT / 'results/real_run_20261015'
AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_pairs(rows, target_map):
    if not rows or len({r['pair_id'] for r in rows}) != len(rows):
        raise ValueError('Expected nonempty unique biological pair IDs')
    for row in rows:
        if row['label'] not in ('A', 'B') or row['target_id'] not in target_map:
            raise ValueError('Unknown pair label or target identity')
        for key in ('sequence_a', 'sequence_b'):
            sequence = row[key]
            if not sequence or len(sequence) > 64 or not set(sequence) <= AMINO_ACIDS:
                raise ValueError('Each peptide must contain 1..64 standard residues')


def training_batches(rows, d1_train_indices, seed, epoch, batch_size=32):
    """Identical D2 examples, flips and D1 replay choices for bridge and MLP."""
    train = [i for i, row in enumerate(rows) if row['split'] == 'train']
    if not train or len(d1_train_indices) < len(train):
        raise ValueError('Need nonempty D2 train and enough distinct D1 replay samples')
    order = epoch_order(train, seed, epoch).tolist()
    replay = epoch_order(d1_train_indices, seed + 100000, epoch).tolist()[:len(order)]
    return [{'pairs': order[start:start + batch_size], 'replay': replay[start:start + batch_size],
             'flips': [flip_training_pair(seed, epoch, rows[i]['pair_id'])
                       for i in order[start:start + batch_size]]}
            for start in range(0, len(order), batch_size)]


def pack_pairs(rows, cache, layer):
    """Use two fixed 64-slot blocks, preserving masks and longer real G1 peptides."""
    sequence_map = {sequence: i for i, sequence in enumerate(cache['sequences'])}
    indices_a = [sequence_map[r['sequence_a']] for r in rows]
    indices_b = [sequence_map[r['sequence_b']] for r in rows]
    features = cache[f'features{layer}']
    masks = cache['mask']
    for row, ia, ib in zip(rows, indices_a, indices_b):
        for key, index in (('sequence_a', ia), ('sequence_b', ib)):
            expected = torch.arange(64) < len(row[key])
            if not torch.equal(masks[index].bool().cpu(), expected):
                raise ValueError('Cached residue mask does not match the full peptide')
    return (torch.cat((features[indices_a], features[indices_b]), dim=1),
            torch.cat((masks[indices_a], masks[indices_b]), dim=1))


def swap_messages(features, masks):
    if features.shape[1] != 128 or masks.shape[1] != 128:
        raise ValueError('Pair messages require two 64-slot blocks')
    return (torch.cat((features[:, 64:], features[:, :64]), dim=1),
            torch.cat((masks[:, 64:], masks[:, :64]), dim=1))


def apply_training_flips(features, masks, labels, flips):
    swapped_features, swapped_masks = swap_messages(features, masks)
    flips = torch.as_tensor(flips, dtype=torch.bool, device=features.device)
    return (torch.where(flips[:, None, None], swapped_features, features),
            torch.where(flips[:, None], swapped_masks, masks),
            torch.where(flips, 1 - labels, labels))


def pair_prompts(model, rows, target_map):
    # Only catalog aliases enter the language model. Raw target fields are audit
    # metadata and do not enter the prompt through this path.
    encoded = [model.encode('t3', 128, target_ids=target_map[row['target_id']]) for row in rows]
    starts = {start for _, start in encoded}
    if len(starts) != 1:
        raise ValueError('All target prompts must share the molecular slot offset')
    return [ids for ids, _ in encoded], starts.pop()


def accumulate_task(model, features, masks, prompts, start, positives, denominator, microbatch):
    """Add sample-summed loss / total mixed-batch size, including partial tails."""
    if denominator < len(features) or len(features) != len(prompts) or len(features) != len(positives):
        raise ValueError('Misaligned task batch or invalid mixed-batch denominator')
    total = 0.
    for begin in range(0, len(features), microbatch):
        end = begin + microbatch
        logits = model(features[begin:end], masks[begin:end], prompts[begin:end], start, full_vocab=True)
        losses = model.soft_positive_loss(logits, positives[begin:end])
        if not torch.isfinite(losses).all():
            raise ValueError('Nonfinite stage-B loss')
        (losses.sum() / denominator).backward()
        total += losses.detach().sum().item()
    return total


def score_pairs(model, features, masks, prompts, start, mode, batch_size):
    values = []
    with torch.no_grad():
        for begin in range(0, len(features), batch_size):
            f, m = features[begin:begin + batch_size], masks[begin:begin + batch_size]
            if mode == 'swapped_messages':
                f, m = swap_messages(f, m)
            scores = model(f, m, prompts[begin:begin + batch_size], start,
                           message_mode='off' if mode == 'off' else 'correct')
            values.append(scores[:, :2].cpu().numpy())
    return np.concatenate(values)


def pair_donors(rows, seed=42, maximum=10):
    """Within-panel, same-target, other-group, other-label pair donors.

    Small panels may have zero eligible donors. Keep their coverage explicit;
    the mandatory off and paired-swap tests still evaluate the full panel.
    """
    if maximum < 1 or any(row['split'] not in ('test', 'external') for row in rows):
        raise ValueError('Pair donors must be held-out test/external rows with a positive maximum')
    result = []
    for row in rows:
        candidates = [i for i, donor in enumerate(rows)
                      if donor['pair_id'] != row['pair_id'] and donor['group_id'] != row['group_id']
                      and donor['target_id'] == row['target_id'] and donor['label'] != row['label']
                      and donor['split'] == row['split']
                      and all(abs(len(donor[key]) - len(row[key])) <= 2
                              for key in ('sequence_a', 'sequence_b'))]
        candidates.sort(key=lambda i: (hashlib.sha256(
            f"{seed}|{row['pair_id']}|{rows[i]['pair_id']}".encode()).digest(), rows[i]['pair_id']))
        result.append({'indices': candidates[:maximum], 'eligible': len(candidates)})
    return result


def evaluate_panel(model, rows, features, masks, target_map, eval_batch):
    prompts, start = pair_prompts(model, rows, target_map)
    labels = np.array([0 if r['label'] == 'A' else 1 for r in rows])
    logits = {mode: score_pairs(model, features, masks, prompts, start, mode, eval_batch)
              for mode in ('correct', 'off', 'swapped_messages')}
    predictions = {mode: values.argmax(-1) for mode, values in logits.items()}
    metrics = {}
    for mode, pred in predictions.items():
        confusion = [[int(np.sum((labels == a) & (pred == b))) for b in (0, 1)] for a in (0, 1)]
        metrics[mode] = {**pair_metrics(pred, labels), **grouped_metric_cis(pred, labels, rows),
                         'confusion_matrix_true_rows_A_B': confusion}
    metrics['reversed_input_and_label'] = {
        **pair_metrics(predictions['swapped_messages'], 1 - labels),
        **grouped_metric_cis(predictions['swapped_messages'], 1 - labels, rows)}
    metrics['prediction_reversal_rate'] = float(np.mean(predictions['correct'] != predictions['swapped_messages']))
    metrics['n_distinct_series'] = len({r.get('series_id', r['group_id']) for r in rows})
    correct_flags = (predictions['correct'] == labels).astype(float)
    metrics['message_effects'] = {
        mode: paired_group_bootstrap(correct_flags, (predictions[mode] == labels).astype(float),
                                     rows, seed=2026, n_boot=2000)
        for mode in ('off', 'swapped_messages')}

    donors = pair_donors(rows)
    donor_records = [[] for _ in rows]
    covered, mismatch_accuracy = [], []
    for i, donor in enumerate(donors):
        indices = donor['indices']
        if not indices:
            continue
        donor_scores = score_pairs(model, features[indices], masks[indices], [prompts[i]] * len(indices),
                                  start, 'correct', eval_batch)
        donor_predictions = donor_scores.argmax(-1)
        covered.append(i)
        mismatch_accuracy.append(float(np.mean(donor_predictions == labels[i])))
        donor_records[i] = [{'donor_pair_id': rows[j]['pair_id'], 'logits': score.tolist(),
                             'prediction': 'AB'[int(pred)]}
                            for j, score, pred in zip(indices, donor_scores, donor_predictions)]
    metrics['mismatched_eligible_pairs'] = len(covered)
    metrics['mismatched_coverage'] = {
        'n_total_pairs': len(rows), 'n_covered_pairs': len(covered), 'requested_maximum_per_pair': 10,
        'eligible_counts': [d['eligible'] for d in donors],
        'used_counts': [len(d['indices']) for d in donors],
        'matching': 'same panel/split and target; different group and documented direction; each peptide length +/-2',
        'seed': 42, 'insufficient_rule': 'use all available distinct donors; zero-donor pairs excluded from this condition only'}
    if covered:
        selected_labels = labels[covered]
        mismatch_accuracy = np.array(mismatch_accuracy)
        metrics['mismatched_donor_mean'] = {
            'n': len(covered), 'accuracy': float(mismatch_accuracy.mean()),
            'balanced_accuracy': float(np.mean([mismatch_accuracy[selected_labels == c].mean()
                                                for c in np.unique(selected_labels)])),
            'correct_on_same_subset_accuracy': float(correct_flags[covered].mean()),
            'message_effect': paired_group_bootstrap(correct_flags[covered], mismatch_accuracy,
                [rows[i] for i in covered], seed=2026, n_boot=2000)}
    records = []
    for i, row in enumerate(rows):
        records.append({'pair_id': row['pair_id'], 'group_id': row['group_id'],
                        'series_id': row.get('series_id'), 'target_id': row['target_id'],
                        'opaque_target_id': target_map[row['target_id']], 'label': row['label'],
                        'clean_after_t1_pretraining': row.get('clean_after_t1_pretraining', row.get('clean_external')),
                        'sequence_source_consistent': row.get('sequence_source_consistent'),
                        'logits': {mode: values[i].tolist() for mode, values in logits.items()},
                        'predictions': {mode: 'AB'[int(pred[i])] for mode, pred in predictions.items()},
                        'prediction_reverses': bool(predictions['correct'][i] != predictions['swapped_messages'][i]),
                        'eligible_donors': donors[i]['eligible'], 'mismatched_draws': donor_records[i]})
    return metrics, records, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=['L30-J0', 'L15-J14'], required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 42, 2026])
    parser.add_argument('--microbatch', type=int, default=2)
    parser.add_argument('--eval-batch', type=int, default=4)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--stage-a', type=Path, default=STAGE_A)
    args = parser.parse_args()
    if args.microbatch not in (2, 4, 8) or args.eval_batch < 1:
        raise ValueError('Use a verified microbatch and a positive evaluation batch')
    torch.set_num_threads(4)
    d2_path = ROOT / 'data/processed/t3_d2_pairs.jsonl'
    g1_path = ROOT / 'data/processed/t3_g1_direction_pairs.jsonl'
    catalog_path = ROOT / 'data/processed/t3_target_catalog.json'
    d2, g1 = read_rows(d2_path), read_rows(g1_path)
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    target_map = {entry['target_id']: entry['opaque_id'] for entry in catalog['entries']}
    if len(target_map) != len(catalog['entries']) or len(set(target_map.values())) != len(target_map):
        raise ValueError('Target catalog identities and aliases must be unique')
    if any(entry['opaque_id'] != f"D{entry['index']:03d}" for entry in catalog['entries']):
        raise ValueError('Target aliases must match the shared identity-only catalog indices')
    validate_pairs(d2, target_map)
    validate_pairs(g1, target_map)
    split = {s: [i for i, row in enumerate(d2) if row['split'] == s] for s in ('train', 'val', 'test')}
    if not all(split.values()):
        raise ValueError('D2 must have nonempty fixed train/val/test sets')
    group_sets = [{d2[i]['group_id'] for i in split[s]} for s in ('train', 'val', 'test')]
    if any(group_sets[a] & group_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('D2 biological groups cross splits')
    d1_rows, d1_targets = load_t1()
    d1_train = [i for i, row in enumerate(d1_rows) if row['split'] == 'train']
    d1_tmap = {target: i for i, target in enumerate(d1_targets)}
    d1_positives = [[d1_tmap[t] for t in row['targets']] for row in d1_rows]
    expected_steps = 30 * math.ceil(len(split['train']) / 32)
    initial_paths = {seed: args.stage_a / f'{args.config}_seed{seed}' / 'best.pt' for seed in args.seeds}
    for path in initial_paths.values():
        if not path.is_file() or not (path.parent / 'result.json').is_file():
            raise FileNotFoundError(f'Stage A must complete and save its selected adapter before stage B: {path}')

    cache = build()
    if not cache.get('bos_eos_padding_removed') or cache.get('model_revision') != 'a695f6045e2e32885fa60af20c13cb35398ce30c':
        raise ValueError('Residue cache encoder revision or special-token removal differs')
    layer, insertion = (30, 0) if args.config == 'L30-J0' else (15, 14)
    d2_features, d2_masks = (x.cuda() for x in pack_pairs(d2, cache, layer))
    d2_labels = torch.tensor([0 if row['label'] == 'A' else 1 for row in d2], device='cuda')
    test_rows = [d2[i] for i in split['test']]
    panels = {'d2_test': test_rows,
              'd2_test_clean_after_t1_pretraining': [r for r in test_rows if r['clean_after_t1_pretraining']],
              'g1_direction': g1}
    if any('sequence_source_consistent' in row for row in g1):
        panels['g1_source_consistent_direction'] = [r for r in g1 if r.get('sequence_source_consistent')]
    panels = {name: rows for name, rows in panels.items() if rows}
    panel_tensors = {name: tuple(x.cuda() for x in pack_pairs(rows, cache, layer)) for name, rows in panels.items()}
    seqmap = {sequence: i for i, sequence in enumerate(cache['sequences'])}
    d1_indices = [seqmap[row['sequence']] for row in d1_rows]
    d1_features = cache[f'features{layer}'][d1_indices, :16].cuda()
    d1_masks = cache['mask'][d1_indices, :16].cuda()
    del cache
    model = FrozenBridgeScorer(insertion)
    d2_prompts, d2_start = pair_prompts(model, d2, target_map)
    d1_prompt, d1_start = model.encode('t1', 16)
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for seed in args.seeds:
        run = args.output / f'{args.config}_seed{seed}'
        run.mkdir(exist_ok=True)
        initial_path = initial_paths[seed]
        config = {'task': 'T3 real-pair stage B with equal-count D1 replay', 'configuration': args.config,
                  'seed': seed, 'epochs': 30, 'd2_batch_size': 32, 'microbatch_size': args.microbatch,
                  'd1_d2_samples_per_epoch': [len(split['train']), len(split['train'])],
                  'expected_optimizer_steps': expected_steps, 'effective_mixed_batch': '32 D2 + 32 D1, equal partial tails',
                  'optimizer': {'name': 'AdamW', 'lr': 3e-4, 'weight_decay': .01, 'betas': [.9, .999]},
                  'selection': 'earliest highest D2 validation balanced accuracy across all 30 epochs',
                  'loss': 'sample-weighted 0.5*(mean D2 full-vocabulary A/B-token CE + mean D1 full-vocabulary known-positive CE)',
                  'flip': 'D2 train only SHA256(seed|epoch|pair_id) first byte % 2; swap messages and label',
                  'order': 'D2 CPU Generator(seed+epoch); D1 Generator(seed+100000+epoch) first N_D2_train',
                  'molecular_slots': {'t3': 128, 'a': 64, 'b': 64, 't1_replay': 16},
                  'target_identity': 'predeclared opaque measurement-object aliases only; no raw target names or domain semantics',
                  'target_catalog_sha256': file_hash(catalog_path),
                  'inputs_sha256': {p.name: file_hash(p) for p in
                      (d2_path, g1_path, ROOT / 'data/processed/t3_split_manifest.json')},
                  'input_sha256': file_hash(ROOT / 'data/processed/benz_samples.jsonl'),
                  't1_checkpoint_sha256': file_hash(initial_path), 't1_initialization': str(initial_path),
                  't1_result_sha256': file_hash(initial_path.parent / 'result.json'),
                  'script_sha256': file_hash(Path(__file__)),
                  'model_script_sha256': file_hash(ROOT / 'scripts/real_bridge_model.py'),
                  'metric_helper_sha256': file_hash(ROOT / 'scripts/run_t3_mlp.py'),
                  'trainable_parameters': sum(p.numel() for p in model.adapter.parameters()),
                  'split_sizes': {name: len(indices) for name, indices in split.items()},
                  'alias_token_ids': model.alias_token_ids[:2], 'answer_labels': ['A', 'B']}
        if (run / 'result.json').exists():
            previous = json.loads((run / 'result.json').read_text(encoding='utf-8'))
            if previous['config'] != config:
                raise ValueError(f'Completed run has a different configuration: {run}')
            reports.append(previous)
            continue
        write(run / 'config.json', config)
        (run / 'train_t3_bridge.executed.py').write_bytes(Path(__file__).read_bytes())
        (run / 'real_bridge_model.executed.py').write_bytes((ROOT / 'scripts/real_bridge_model.py').read_bytes())
        checkpoint = torch.load(initial_path, map_location='cuda', weights_only=True)
        stage_a_result = json.loads((initial_path.parent / 'result.json').read_text(encoding='utf-8'))
        if checkpoint['config']['seed'] != seed or checkpoint['config']['configuration'] != args.config:
            raise ValueError('Stage-A checkpoint seed or joint configuration differs')
        if checkpoint['epoch'] != stage_a_result['best_epoch'] or checkpoint['config']['input_sha256'] != config['input_sha256']:
            raise ValueError('Stage-A checkpoint selection or frozen T1 inputs differ')
        model.adapter.load_state_dict(checkpoint['adapter'])
        torch.manual_seed(seed)
        optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=3e-4, weight_decay=.01)
        history, best, best_epoch, steps = [], -1., None, 0
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1, 31):
            d1_loss_sum = d2_loss_sum = 0.
            for batch in training_batches(d2, d1_train, seed, epoch):
                pair_indices, replay_indices = batch['pairs'], batch['replay']
                f, m, labels = apply_training_flips(d2_features[pair_indices], d2_masks[pair_indices],
                                                   d2_labels[pair_indices], batch['flips'])
                denominator = len(pair_indices) + len(replay_indices)
                optimizer.zero_grad(set_to_none=True)
                d2_loss_sum += accumulate_task(model, f, m, [d2_prompts[i] for i in pair_indices], d2_start,
                                              [[int(label)] for label in labels.tolist()], denominator, args.microbatch)
                d1_loss_sum += accumulate_task(model, d1_features[replay_indices], d1_masks[replay_indices],
                                              [d1_prompt] * len(replay_indices), d1_start,
                                              [d1_positives[i] for i in replay_indices], denominator, args.microbatch)
                if any(p.grad is not None for p in model.qwen.parameters()):
                    raise ValueError('Frozen Qwen acquired gradients')
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.adapter.parameters()):
                    raise ValueError('Adapter gradients missing or nonfinite')
                optimizer.step()
                steps += 1
            validation_logits = score_pairs(model, d2_features[split['val']], d2_masks[split['val']],
                                             [d2_prompts[i] for i in split['val']], d2_start, 'correct', args.eval_batch)
            validation = pair_metrics(validation_logits.argmax(-1), d2_labels[split['val']].cpu().numpy())
            ntrain = len(split['train'])
            entry = {'epoch': epoch, 'optimizer_steps': steps,
                     'train_loss': (d1_loss_sum + d2_loss_sum) / (2 * ntrain),
                     'd1_replay_loss': d1_loss_sum / ntrain, 'd2_loss': d2_loss_sum / ntrain,
                     'validation': validation}
            history.append(entry)
            if validation['balanced_accuracy'] > best:
                best, best_epoch = validation['balanced_accuracy'], epoch
                torch.save({'adapter': model.adapter.state_dict(), 'epoch': epoch, 'config': config}, run / 'best.pt')
            write(run / 'history.json', history)
            print(json.dumps({'config': args.config, 'seed': seed, **entry}), flush=True)
        if steps != expected_steps:
            raise ValueError('Stage-B optimizer budget differs from the registered schedule')
        selected = torch.load(run / 'best.pt', map_location='cuda', weights_only=True)
        model.adapter.load_state_dict(selected['adapter'])
        scores, arrays = {}, {}
        for name, rows in panels.items():
            scores[name], predictions, logits = evaluate_panel(model, rows, *panel_tensors[name], target_map, args.eval_batch)
            (run / f'{name}.predictions.jsonl').write_text(
                '\n'.join(json.dumps(row) for row in predictions) + '\n', encoding='utf-8')
            arrays.update({f'{name}__{mode}': values for mode, values in logits.items()})
            arrays[f'{name}__pair_ids'] = np.array([r['pair_id'] for r in rows])
            arrays[f'{name}__labels'] = np.array([0 if r['label'] == 'A' else 1 for r in rows])
            arrays[f'{name}__group_ids'] = np.array([r['group_id'] for r in rows])
        np.savez_compressed(run / 'test_scores.npz', **arrays)
        report = {'config': config, 'best_epoch': best_epoch, 'optimizer_steps': steps,
                  'best_validation_balanced_accuracy': best, 'history': history, 'scores': scores,
                  'seconds': time.perf_counter() - started,
                  'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                  'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}
        write(run / 'result.json', report)
        reports.append(report)
        print(json.dumps({'completed_t3': args.config, 'seed': seed, 'best_epoch': best_epoch}), flush=True)
    write(args.output / f'{args.config}_summary.json', {'configuration': args.config, 'runs': reports})


if __name__ == '__main__':
    main()
