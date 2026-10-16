"""C1 source-content and C2 prompt-state diagnostics on frozen A1 connectors.

No optimizer, no generated sequence, and no backbone updates. C1 uses answer
teacher forcing only for a separately reported token-NLL sensitivity diagnostic.
C2 states receive prompts only; train/validation/test retain their physical lock.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_v02.cache_panel_features import DEFAULT_CACHE, file_sha, verify_local_models
from bench_v02.panel_execution import validate_panel_data
from bench_v02.run_simple_baselines import features as baseline_features
from bench_v02.train_panel_baselines import ContextEncoder, FeatureStore
from foundation_20261016.connectors import FrozenConnectorLM
from foundation_20261016.run_connectors import (ROOT, SCENARIOS, collate, context_view,
    load_connector_state, select_splits, write_json)

SPLITS = ('train', 'validation', 'test')
ALPHAS = (.1, 1., 10., 100., 1000.)


def now():
    return datetime.now(timezone.utc).isoformat()


def last_active(mask):
    if mask.ndim != 2 or not torch.all(mask.bool().any(1)):
        raise ValueError('Each prompt needs an active token')
    return torch.arange(mask.shape[1], device=mask.device).expand_as(mask).masked_fill(~mask.bool(), -1).max(1).values


def plain_forward(model, batch):
    """Exact intervention-off path, including when every parameter is frozen."""
    mask = batch['attention_mask']
    return model.base_lm(input_ids=batch['input_ids'], attention_mask=mask,
                         position_ids=(mask.cumsum(-1) - 1).clamp_min(0),
                         use_cache=False, return_dict=True)


@contextmanager
def captured_prompt_states(model, batch):
    """Install hooks only for one forward and clean up even after exceptions.

    Receiver hook is installed at base forward entry, after the connector's
    internal injection hook. Thus insertion into the final layer is supported.
    """
    if 'labels' in batch:
        raise ValueError('Prompt-state capture refuses answer labels')
    values, handles = {}, []
    indices = last_active(batch['attention_mask'])
    def receiver(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        positions = indices + (model.num_memory_tokens if model.method != 'internal_cross_attention' else 0)
        values['receiver'] = hidden[torch.arange(len(indices), device=indices.device), positions].detach().float().cpu().numpy()
    def install(_module, _args):
        handles.append(model.base_lm.model.layers[-1].register_forward_hook(receiver))
    handles.append(model.base_lm.register_forward_pre_hook(install))
    def message(_module, _args, output):
        if model.method == 'internal_cross_attention':
            value = output[torch.arange(len(indices), device=indices.device), indices]
        else:
            value = output.mean(1)
        values['message'] = value.detach().float().cpu().numpy()
    if model.method == 'constant':
        values['message'] = model.constant_tokens.mean(0).detach().float().cpu().numpy()[None].repeat(len(indices), axis=0)
    else:
        module = model.projection if model.method == 'input_mlp' else model.output_projection
        handles.append(module.register_forward_hook(message))
    try:
        yield values
        if set(values) != {'message', 'receiver'}:
            raise ValueError('Missing prompt-state hook outputs')
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def prompt_states(model, batch):
    if 'labels' in batch:
        raise ValueError('Gold-answer tokens cannot enter C2 states')
    with captured_prompt_states(model, batch) as captured:
        model(**batch, message_mode='correct')
    return captured


def donor_key(row):
    # Target and input matching use physical sequence/context, never IDs or gold.
    return (row['target_sequence'], len(row['peptide_sequence']),
            json.dumps(context_view(row), sort_keys=True, separators=(',', ':'), allow_nan=False))


def freeze_donors(rows, seed=42):
    """Independent donor choices within exact target/length/observed-input strata."""
    if any(not isinstance(row.get('group_id'), str) or not row['group_id'] for row in rows):
        raise ValueError('Independent donor eligibility requires the physical locked group_id')
    rng = np.random.default_rng(seed)
    grouped = {}
    for i, row in enumerate(rows):
        grouped.setdefault(donor_key(row), []).append(i)
    result = []
    for i, row in enumerate(rows):
        eligible = [j for j in grouped[donor_key(row)] if rows[j]['peptide_sequence'] != row['peptide_sequence']
                    and rows[j]['group_id'] != row['group_id']]
        donor = int(rng.choice(eligible)) if eligible else None
        result.append({'sample_id': row['sample_id'], 'source_index': i, 'donor_index': donor,
                       'donor_sample_id': rows[donor]['sample_id'] if donor is not None else None,
                       'eligible_donors': len(eligible),
                       'status': 'matched' if donor is not None else 'no_distinct_peptide_in_exact_input_stratum'})
    return result


def answer_nll(output, labels):
    losses = F.cross_entropy(output.logits[:, :-1].float().transpose(1, 2), labels[:, 1:],
                             ignore_index=-100, reduction='none')
    mask = labels[:, 1:] != -100
    counts = mask.sum(1)
    if not torch.all(counts > 0):
        raise ValueError('Every diagnostic needs actual answer tokens')
    return (losses.sum(1) / counts).detach().cpu().numpy(), counts.cpu().numpy()


def communication_algebra(correct, off, mismatch):
    # S = negative mean answer-token NLL on the SAME original gold answer.
    result = {'G': float(off - correct), 'P': float(mismatch - correct),
              'H': float(mismatch - off)}
    result['identity_residual_P_minus_G_minus_H'] = result['P'] - result['G'] - result['H']
    if abs(result['identity_residual_P_minus_G_minus_H']) > 1e-8:
        raise ValueError('Communication score algebra failed')
    return result


def fit_scaler(x):
    return x.mean(0), np.maximum(x.std(0), 1e-6)


def fixed_projection(train, all_values, dimension=128, seed=42):
    """Inductive random projection; all fitted statistics use train rows only."""
    if train.ndim != 2 or all_values.ndim != 2 or train.shape[1] != all_values.shape[1]:
        raise ValueError('Projection expects matching feature dimensions')
    center, scale = fit_scaler(train)
    matrix = np.random.default_rng(seed).standard_normal((train.shape[1], dimension)) / math.sqrt(dimension)
    return ((all_values - center) / scale) @ matrix, {'center': center, 'scale': scale, 'projection': matrix}


def ridge_fit(x, y, alpha):
    if alpha < 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Ridge needs finite data and nonnegative alpha')
    augmented = np.column_stack((x, np.ones(len(x))))
    penalty = np.eye(augmented.shape[1]) * alpha
    penalty[-1, -1] = 0
    gram, rhs = augmented.T @ augmented + penalty, augmented.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def regression(y, prediction):
    error = np.asarray(y) - np.asarray(prediction)
    return {'n': len(error), 'mae': float(np.abs(error).mean()),
            'mse': float(np.square(error).mean()), 'rmse': float(np.sqrt(np.square(error).mean()))}


def grouped_bootstrap(y, baseline, candidate, groups, seed=42, repeats=2000):
    unique = sorted(set(groups))
    units = [np.flatnonzero(np.asarray(groups) == group) for group in unique]
    if len(units) < 2:
        return {'status': 'undefined_fewer_than_two_target_groups', 'n_groups': len(units)}
    rng, differences = np.random.default_rng(seed), []
    for _ in range(repeats):
        indices = np.concatenate([units[i] for i in rng.integers(0, len(units), len(units))])
        differences.append(float(np.mean((y[indices] - baseline[indices]) ** 2 - (y[indices] - candidate[indices]) ** 2)))
    return {'status': 'computed', 'n_groups': len(units), 'repeats': repeats,
            'unit': 'physical locked group_id; every row in a sampled locked group retained',
            'delta_mse_95_percentile_interval': np.quantile(differences, [.025, .975]).tolist()}


def conditioned_baseline(rows, training_indices):
    """Composition/chemistry plus the same allowed context seen by the LM.

    Numeric context statistics use TRAIN only. Fixed context hashes require no
    held-out vocabulary fitting. IDs, labels, DOI and raw metadata are excluded.
    """
    training = [rows[int(i)] for i in training_indices]
    if not training:
        raise ValueError('Context baseline requires nonempty TRAIN rows')
    encoder = ContextEncoder(training)
    existing = np.asarray([baseline_features(row) for row in rows], dtype=np.float64)
    context = np.asarray([encoder(row).numpy() for row in rows], dtype=np.float64)
    values = np.column_stack((existing, context))
    return values, {'existing_composition_chemistry_dimensions': existing.shape[1],
                    'allowed_context_dimensions': ContextEncoder.dimension,
                    'baseline_dimensions': values.shape[1],
                    'context_numeric_statistics_TRAIN_only': encoder.statistics,
                    'context_encoding': 'same allowlist as frozen predictor ContextEncoder; fixed categorical trigram hash plus TRAIN-standardized numeric fields and missingness'}


def probe_panel(rows, representations, output, dimension=128, seed=42):
    output = Path(output)
    index = {split: np.asarray([i for i, row in enumerate(rows) if row['split'] == split]) for split in SPLITS}
    train, val, test = (index[s] for s in SPLITS)
    if any(not len(index[split]) for split in SPLITS):
        raise ValueError('Probes need nonempty train, validation, and test')
    b, baseline_receipt = conditioned_baseline(rows, train)
    y = np.asarray([row['pkd'] for row in rows], dtype=np.float64)
    inputs = {'baseline_zero128': np.column_stack((b, np.zeros((len(b), dimension))))}
    states, results, predictions = {}, {}, {}
    for name, values in representations.items():
        projected, state = fixed_projection(values[train], values, dimension, seed)
        inputs[name] = np.column_stack((b, projected))
        states.update({name + '_' + key: value for key, value in state.items()})
    for name, x in inputs.items():
        center, scale = fit_scaler(x[train])
        x = (x - center) / scale
        trials = []
        for alpha in ALPHAS:
            weights = ridge_fit(x[train], y[train], alpha)
            p = np.column_stack((x[val], np.ones(len(val)))) @ weights
            trials.append({'alpha': alpha, 'validation_mse': regression(y[val], p)['mse'], 'weights': weights})
        selected = min(trials, key=lambda trial: trial['validation_mse'])
        p = np.column_stack((x[test], np.ones(len(test)))) @ selected['weights']
        results[name] = {'selected_alpha': selected['alpha'], 'input_dimension': x.shape[1],
                         'test': regression(y[test], p),
                         'validation_trials': [{k: v for k, v in t.items() if k != 'weights'} for t in trials]}
        states.update({name + '_ridge_center': center, name + '_ridge_scale': scale,
                       name + '_ridge_weights': selected['weights']})
        predictions[name] = p
    groups = [rows[i]['group_id'] for i in test]
    if any(not isinstance(group, str) or not group for group in groups):
        raise ValueError('Bootstrap requires physical locked group_id')
    for name in representations:
        results[name]['delta_MSE_vs_B'] = results['baseline_zero128']['test']['mse'] - results[name]['test']['mse']
        results[name]['grouped_bootstrap'] = grouped_bootstrap(y[test], predictions['baseline_zero128'], predictions[name], groups)
    gap = results['message']['test']['mse'] - results['raw']['test']['mse']
    np.savez_compressed(output / 'probe_states.npz', **states)
    np.savez_compressed(output / 'test_probe_predictions.npz', y=y[test], **predictions)
    return {'protocol_revision': 3, 'models': results, 'conditioning_baseline': baseline_receipt,
            'raw_message_MSE_gap_message_minus_raw': gap,
            'selection': 'minimum validation MSE from same five alphas, earliest tie; TRAIN fit only',
            'projection': {'method': 'fixed Gaussian random projection after TRAIN-only standardization',
                           'dimension_each_representation': dimension, 'seed': seed,
                           'inductive_constraint': 'connector representations depend on a supervised TRAIN-selected checkpoint; projection sees no validation/test labels'},
            'metric_units': {'mae': 'pKD', 'rmse': 'pKD', 'mse': 'pKD_squared', 'delta_MSE_vs_B': 'pKD_squared'},
            'interpretation': 'held-out pKD recoverability conditional on composition/chemistry and full allowed experimental context baseline; error reductions are not bits or biological causality'}


def resource_check(start, timeout_seconds, memory_bytes):
    if time.monotonic() - start > timeout_seconds:
        raise TimeoutError('Frozen diagnostic wall-clock cap exceeded')
    status = Path('/proc/self/status')
    if status.exists():
        rss = next(int(line.split()[1]) * 1024 for line in status.read_text().splitlines() if line.startswith('VmRSS:'))
        if rss > memory_bytes:
            raise MemoryError('Diagnostic RSS cap exceeded')


def extract_states(model, tokenizer, store, rows, device, batch_size, check):
    arrays = {name: [] for name in ('raw', 'message', 'receiver')}
    for start in range(0, len(rows), batch_size):
        check()
        selected = rows[start:start + batch_size]
        batch = collate(selected, tokenizer, store, device, supervised=False)
        captured = prompt_states(model, batch)
        arrays['raw'].extend([torch.cat((store.residues(row['peptide_sequence']).float().mean(0),
                                        store.residues(row['target_sequence']).float().mean(0))).numpy() for row in selected])
        for name in ('message', 'receiver'):
            arrays[name].extend(captured[name])
        if start % (batch_size * 25) == 0:
            print(json.dumps({'event': 'prompt_states', 'completed': min(start + batch_size, len(rows)), 'total': len(rows)}), flush=True)
    return {name: np.asarray(value, dtype=np.float32) for name, value in arrays.items()}


@torch.no_grad()
def c1_diagnostics(model, tokenizer, store, test_rows, donor_map, device, check):
    records = []
    for row, mapping in zip(test_rows, donor_map):
        check()
        batch = collate([row], tokenizer, store, device, supervised=True)
        labels = batch.pop('labels')
        correct, tokens = answer_nll(model(**batch), labels)
        off, _ = answer_nll(plain_forward(model, batch), labels)
        record = {'sample_id': row['sample_id'], 'answer_tokens': int(tokens[0]),
                  'correct_original_gold_mean_token_nll': float(correct[0]),
                  'off_original_gold_mean_token_nll': float(off[0]), 'donor_status': mapping['status']}
        if mapping['donor_index'] is not None:
            donor = test_rows[mapping['donor_index']]
            donor_batch = collate([donor], tokenizer, store, device, supervised=True)
            for key in ('memory', 'memory_mask', 'memory_type_ids'):
                batch[key] = donor_batch[key]
            mismatch, _ = answer_nll(model(**batch), labels)
            donor_labels = donor_batch.pop('labels')
            donor_gold, donor_tokens = answer_nll(model(**donor_batch), donor_labels)
            record.update({'mismatched_original_gold_mean_token_nll': float(mismatch[0]),
                           'donor_actual_gold_mean_token_nll': float(donor_gold[0]),
                           'donor_answer_tokens': int(donor_tokens[0]),
                           'algebra': communication_algebra(float(correct[0]), float(off[0]), float(mismatch[0]))})
        records.append(record)
    matched = [r for r in records if 'algebra' in r]
    return {'records': records, 'n_test': len(records), 'n_matched': len(matched),
            'G_all_test': float(np.mean([r['off_original_gold_mean_token_nll'] - r['correct_original_gold_mean_token_nll'] for r in records])),
            'P_H_status': 'computed_on_independent_donor_subset' if matched else 'undefined_zero_independent_donors',
            'units': 'nats_per_answer_token',
            'means_all_test': {key: float(np.mean([r[key] for r in records])) for key in
                              ('correct_original_gold_mean_token_nll', 'off_original_gold_mean_token_nll')},
            'algebra_matched_mean': {key: float(np.mean([r['algebra'][key] for r in matched]))
                                    for key in ('G', 'P', 'H', 'identity_residual_P_minus_G_minus_H')} if matched else None,
            'scope': 'teacher-forced answer-token sensitivity; source changes are computational inputs, experimental conditions may be unknown',
            'mismatch_interpretation': 'score against original gold measures source-content sensitivity; actual donor gold scored separately',
            'score_definition': 'S = -per-sample mean answer-token NLL, natural-log units; G=correct-off, P=correct-mismatch, H=off-mismatch; P=G+H'}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--scenario', choices=SCENARIOS, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=ROOT / 'data/processed/bench_v02/a1_locked_v2')
    p.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--timeout-seconds', type=int, default=840)
    p.add_argument('--dimension', type=int, default=128)
    p.add_argument('--skip-c1', action='store_true')
    p.add_argument('--skip-c2', action='store_true')
    return p


def main():
    args = parser().parse_args()
    if args.skip_c1 and args.skip_c2:
        raise ValueError('At least one diagnostic is required')
    if args.dimension != 128:
        raise ValueError('Foundation probes preregister exactly 128 extra dimensions')
    if min(args.batch_size, args.threads, args.timeout_seconds, args.dimension) < 1:
        raise ValueError('Positive resource and projection dimensions required')
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=False)
    if hasattr(signal, 'SIGALRM'):
        def timeout(_signum, _frame):
            raise TimeoutError('Diagnostic wall-clock cap reached')
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(args.timeout_seconds)
    try:
        if args.device == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError('CUDA unavailable')
            torch.cuda.set_per_process_memory_fraction(11.5 * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory)
            torch.cuda.reset_peak_memory_stats()
        path = args.data_root / args.scenario
        rows, gate = validate_panel_data(path / 'samples.jsonl', path / 'benchmark_lock.json', 'A1')
        splits, selection = select_splits(rows, {})
        ordered = [row for split in SPLITS for row in splits[split]]
        checkpoint_sha = file_sha(args.checkpoint)
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        registration = saved['registration']
        config = registration['configuration']
        if config['scenario'] != args.scenario or registration.get('scope') != 'full_locked_A1_panel':
            raise ValueError('Only a same-scenario full locked A1 checkpoint is accepted')
        if saved.get('selection_eligible') is False or saved.get('best_epoch') is None:
            raise ValueError('Partial/unselected checkpoint refused')
        source_manifest = args.checkpoint.parent / 'input_manifest.json'
        if file_sha(source_manifest) != registration['input_manifest_sha256']:
            raise ValueError('Checkpoint source manifest changed')
        parent_manifest = json.loads(source_manifest.read_text(encoding='utf-8'))
        if gate['samples_sha256'] != parent_manifest['data_gate']['samples_sha256'] or gate['benchmark_lock_sha256'] != parent_manifest['data_gate']['benchmark_lock_sha256']:
            raise ValueError('Checkpoint and diagnostic physical data differ')
        for script_path, digest in parent_manifest['script_sha256'].items():
            if file_sha(script_path) != digest:
                raise ValueError('Checkpoint training code changed: ' + script_path)
        model_receipt = verify_local_models()
        store = FeatureStore(args.cache)
        if file_sha(store.manifest_path) != parent_manifest['feature_manifest_sha256']:
            raise ValueError('Checkpoint and diagnostic ESM cache differ')
        donors = freeze_donors(splits['test'])
        write_json(args.output / 'donor_map.json', {'seed': 42, 'pool': 'same fixed test split', 'map': donors,
            'matching': 'exact target sequence, peptide length, complete allowlisted chemistry/context including missingness; distinct peptide sequence AND distinct physical locked group_id',
            'selection_uses_labels': False, 'unknown_conditions': 'missingness matched; actual equality of unobserved biology is unknown'})
        prereg = {'created_utc': now(), 'scenario': args.scenario, 'method': config['method'],
            'checkpoint_sha256': checkpoint_sha, 'checkpoint_selected_epoch': saved['best_epoch'],
            'checkpoint_registration': registration, 'data_gate': gate, 'split_selection': selection,
            'model_receipt': model_receipt, 'feature_manifest_sha256': file_sha(store.manifest_path),
            'donor_map_sha256': file_sha(args.output / 'donor_map.json'),
            'script_sha256': file_sha(__file__), 'device': args.device,
            'source_code_sha256': {str(p): file_sha(p) for p in (Path(__file__),
                Path(__file__).with_name('connectors.py'), Path(__file__).with_name('run_connectors.py'),
                ROOT / 'scripts/bench_v02/run_simple_baselines.py',
                ROOT / 'scripts/bench_v02/train_panel_baselines.py')},
            'resource_cap': {'wall_seconds': args.timeout_seconds, 'rss_bytes': 12 * 1024 ** 3, 'gpu_torch_bytes': 11.5 * 1024 ** 3},
            'protocol_revision': 3, 'probe_dimension': args.dimension, 'probe_alphas': ALPHAS,
            'conditioning_baseline': '52 composition/chemistry/missingness plus148 existing ContextEncoder dimensions; all numeric statistics TRAIN only; each model adds128 dimensions',
            'probe_selection': 'validation MSE, TRAIN-only fits', 'state_input': 'prompt-only; no gold answer tokens',
            'nll_input': 'teacher-forced real gold in independent C1 diagnostic only'}
        write_json(args.output / 'preregistration.json', prereg)
        write_json(args.output / 'sample_order.json', [{'sample_id': row['sample_id'], 'split': row['split'],
            'target_sequence_sha256': row['groups']['target_sequence_sha256'], 'group_id': row['group_id']} for row in ordered])
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_path = ROOT / 'models/Qwen--Qwen2.5-1.5B-Instruct'
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16 if args.device == 'cuda' else torch.float32, attn_implementation='sdpa').to(args.device)
        model = FrozenConnectorLM(base, config['method'], memory_dim=config['memory_dim'],
            num_memory_tokens=config['num_memory_tokens'], insertion_layer=config['insertion_layer']).to(args.device)
        load_connector_state(model, saved['connector_state'])
        model.requires_grad_(False).eval()
        check = lambda: resource_check(started, args.timeout_seconds, 12 * 1024 ** 3)
        result = {'status': 'complete', 'method': config['method'], 'scenario': args.scenario,
                  'sample_count': {split: len(splits[split]) for split in SPLITS}, 'backbone_updated': False}
        if not args.skip_c2:
            arrays = extract_states(model, tokenizer, store, ordered, args.device, args.batch_size, check)
            np.savez_compressed(args.output / 'prompt_states.npz', **arrays)
            result['C2'] = probe_panel(ordered, arrays, args.output, args.dimension)
        if not args.skip_c1:
            result['C1'] = c1_diagnostics(model, tokenizer, store, splits['test'], donors, args.device, check)
        if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
            raise ValueError('Diagnostic model freeze invariant broken')
        if file_sha(args.checkpoint) != checkpoint_sha or file_sha(__file__) != prereg['script_sha256']:
            raise ValueError('Executing script or checkpoint changed')
        for script_path, digest in prereg['source_code_sha256'].items():
            if file_sha(script_path) != digest:
                raise ValueError('Diagnostic dependency changed: ' + script_path)
        for name, expected in [('samples.jsonl', gate['samples_sha256']), ('benchmark_lock.json', gate['benchmark_lock_sha256'])]:
            if file_sha(path / name) != expected:
                raise ValueError('Frozen data changed')
        result.update({'finished_utc': now(), 'elapsed_seconds': time.monotonic() - started,
            'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0,
            'feature_receipts': store.checked_proteins,
            'artifact_sha256': {p.name: file_sha(p) for p in args.output.iterdir() if p.is_file()}})
        write_json(args.output / 'result.json', result)
        print(json.dumps({'status': 'complete', 'path': str(args.output), 'seconds': result['elapsed_seconds']}), flush=True)
    except BaseException as error:
        write_json(args.output / 'failure.json', {'status': 'failed', 'type': type(error).__name__, 'error': str(error), 'utc': now()})
        raise
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


if __name__ == '__main__':
    main()
