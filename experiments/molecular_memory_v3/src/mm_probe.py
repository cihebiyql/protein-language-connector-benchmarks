"""Preregistered conditional ridge diagnostic; imports never load a model.

Only extract/fit with an exact standard-contract binding may execute real work.
readiness is hash/schema inspection and never deserializes model weights.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'src'

from . import mm_data as data

PROTOCOL = {
    'schema': 'mm_conditional_probe_v1', 'status': 'preregistered',
    'scenarios': list(data.SCENARIOS), 'variant': 'M3', 'seed': 17,
    'checkpoint': 'selected.pt', 'batch_size': 4,
    'representation': 'numeric_head_input_finalnorm_prompt_end_only',
    'base_order': ['p15', 'p30', 't15', 't30'], 'base_raw_dim': 2560,
    'base_projection': {'dim': 64, 'seed': 0, 'rng': 'PCG64', 'scale': '1/sqrt(2560)'},
    'receiver_raw_dim': 1536,
    'receiver_projection': {'dim': 64, 'seed': 1, 'rng': 'PCG64', 'scale': '1/sqrt(1536)'},
    'condition_hash': {'dim': 16, 'algorithm': 'sha256(slot_name:NUL:condition_id); first8_big_mod16; byte8_parity_sign; sum/sqrt(10); observed_only'},
    'observed_dim': 10, 'length_features': 'log1p(peptide_length),log1p(target_length)',
    'base_dim': 92, 'joint_dim': 156,
    'ridge_lambda': 1000.0, 'ridge_intercept': 'unpenalized',
    'standardization': 'TRAIN_only_population_std_floor_1e-8',
    'targets': 'original_precision_TRAIN_pkd_only',
    'comparison': ['[B;R64]', '[B;zero64]'],
    'selection': 'none_fixed_lambda_no_validation_or_test_tuning',
    'delta': 'baseline_error_minus_conditioned_error_positive_is_gain',
    'interpretation': 'single_seed_diagnostic_not_mutual_information_not_three_seed_superiority',
    'source': 'https://aclanthology.org/2021.emnlp-main.122/',
}


def digest(value):
    return hashlib.sha256(data.canonical(value).encode('utf8')).hexdigest()


def read_json(path):
    return data.strict_loads(Path(path).read_text(encoding='utf8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load_protocol(path):
    value = read_json(path)
    if value != PROTOCOL:
        raise ValueError('probe protocol differs from preregistered fixed diagnostic')
    return value


def _context(args):
    """Readiness: hash-only checkpoint inspection; no torch import or gold parse."""
    load_protocol(args.probe_protocol)
    config = read_json(args.config)
    if (config.get('schema_version') != 'mm-v3.protocol.v1' or config.get('status') != 'frozen'
            or config.get('budget_signature') != digest(config.get('training'))
            or args.scenario not in PROTOCOL['scenarios'] or args.scenario not in config.get('scenarios', [])
            or 'M3' not in config.get('variants', []) or 17 not in config.get('seeds', [])):
        raise ValueError('probe requires a frozen training config admitting M3 seed17')
    if (config.get('deadline') != '2026-10-17T10:29:40+08:00'
            or config['training'].get('checkpoint_selection') != 'min_full_validation_answer_token_nll_earliest_tie'):
        raise ValueError('probe cannot change the selected-theta or deadline policy')
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    if checkpoint.name != 'selected.pt':
        raise ValueError('probe must use selected.pt, never another epoch/seed')
    fit = read_json(checkpoint.parent / 'run_manifest.json')
    identity = {'scenario': args.scenario, 'variant': 'M3', 'seed': 17,
                'protocol_sha256': data.file_sha(args.config),
                'training_budget_sha256': digest(config['training'])}
    if fit.get('status') != 'complete' or fit.get('execution_phase') != 'fit' or fit.get('identity') != identity:
        raise ValueError('probe source must be the exact completed M3 seed17 fit identity')
    selection = read_json(checkpoint.parent / 'selection.json')
    checkpoint_sha = data.file_sha(checkpoint)
    if selection.get('checkpoint_sha256') != checkpoint_sha:
        raise ValueError('selected checkpoint SHA mismatch')
    if fit.get('artifacts', {}).get('checkpoint', {}).get('sha256') != checkpoint_sha:
        raise ValueError('fit manifest does not bind selected checkpoint')
    views = Path(args.view_root).resolve(strict=True)
    vm = read_json(views / 'manifest.json')
    receipt = vm['scenarios'][args.scenario]
    if (vm.get('schema') != 'mm_a1_views_v1' or vm.get('condition_slots') != list(data.CONDITION_SLOTS)
            or receipt['source_shas'] != data.FROZEN_SHAS[args.scenario]):
        raise ValueError('probe views differ from frozen data')
    if fit.get('provenance', {}).get('view_manifest_sha256') != data.file_sha(views / 'manifest.json'):
        raise ValueError('probe view manifest differs from fitted checkpoint views')
    cache = Path(args.cache_root).resolve(strict=True)
    cache_sha = data.file_sha(cache / 'manifest.json')
    if fit.get('provenance', {}).get('cache_before_run', {}).get('manifest_sha256') != cache_sha:
        raise ValueError('probe cache differs from fitted checkpoint cache')
    for split in ('train', 'validation', 'test'):
        item = receipt['views'][split]
        data.verify_sha(data.registered_path(views, item['path']), item['sha256'])
    code = {name: data.file_sha(Path(__file__).parent / name)
            for name in ('mm_probe.py', 'mm_model.py', 'mm_data.py', 'mm_extract.py', 'mm_train.py', 'mm_baselines.py')}
    fitted_code = fit.get('provenance', {}).get('code_sha256', {})
    if any(fitted_code.get(name) != code[name]
           for name in ('mm_model.py', 'mm_data.py', 'mm_extract.py', 'mm_train.py')):
        raise ValueError('core model/data/extraction/training code differs from fitted checkpoint provenance')
    stage = getattr(args, 'planned_stage', 'extract') if args.command == 'readiness' else args.command
    binding = {'stage': stage, 'scenario': args.scenario, 'variant': 'M3', 'seed': 17,
               'probe_protocol_sha256': data.file_sha(args.probe_protocol),
               'config_sha256': data.file_sha(args.config), 'checkpoint_sha256': checkpoint_sha,
               'fit_manifest_sha256': data.file_sha(checkpoint.parent / 'run_manifest.json'),
               'selection_sha256': data.file_sha(checkpoint.parent / 'selection.json'),
               'view_manifest_sha256': data.file_sha(views / 'manifest.json'),
               'dataset_sha256': receipt['source_shas'], 'cache_manifest_sha256': cache_sha,
               'code_sha256': code, 'output': str(Path(args.output).resolve()),
               'project_root': str(Path(args.project_root).resolve()),
               'view_root': str(views), 'cache_root': str(cache),
               'device': args.device if stage == 'extract' else 'cpu'}
    if stage == 'fit':
        feature_root = Path(args.features).resolve(strict=True)
        binding['feature_manifest_sha256'] = data.file_sha(feature_root / 'manifest.json')
        binding['features'] = str(feature_root)
    return config, identity, binding


def execution_binding(args):
    return {'kind': 'conditional_probe', 'probe': _context(args)[2]}


def authorize(args, binding):
    if not args.contract or not args.validator or not args.validator_sha256:
        raise ValueError('execution requires standard contract and SHA-pinned validator')
    data.verify_sha(args.validator, args.validator_sha256)
    raw = Path(args.contract).read_bytes()
    contract = data.strict_loads(raw.decode('utf8'))
    if contract.get('training_run_binding') != {'kind': 'conditional_probe', 'probe': binding}:
        raise ValueError('contract does not bind this exact probe stage/assets/output')
    spec = importlib.util.spec_from_file_location('mm_probe_standard_validator', args.validator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checked = module.validate(contract, contract_bytes=raw, check_files=True,
                              contract_directory=Path(args.contract).parent)
    if checked.get('valid_for_execution') is not True:
        raise ValueError('standard validator rejected probe execution')
    deadline = datetime.fromisoformat(contract['timing']['effective_deadline_at'].replace('Z', '+00:00'))
    timeout = contract['execution']['timeout_seconds']
    if deadline.tzinfo is None or type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('invalid bounded probe contract')
    started = time.monotonic()
    def check():
        if datetime.now(timezone.utc) >= deadline or time.monotonic() - started >= timeout:
            raise TimeoutError('probe deadline/runtime exhausted')
    check()
    return check, {'contract_sha256': hashlib.sha256(raw).hexdigest(),
                   'validator_sha256': args.validator_sha256, 'validation': checked}


def projections():
    import numpy as np
    return {name: (np.random.Generator(np.random.PCG64(seed)).standard_normal((width, 64))
                   / math.sqrt(width)).astype(np.float32)
            for name, width, seed in (('B', 2560, 0), ('R', 1536, 1))}


def base_features(row, store, matrices):
    import numpy as np
    means = []
    for field in ('peptide_sequence', 'target_sequence'):
        values = store.pooled(row[field])
        for layer in ('H15', 'H30'):
            value = values[layer].detach().float().cpu().numpy()
            if value.shape != (8, 640) or not np.isfinite(value).all():
                raise ValueError('invalid frozen pooled feature')
            means.append(value.mean(axis=0, dtype=np.float32))
    ids, observed = data.condition_encoding(row)
    hashed = np.zeros(16, dtype=np.float32)
    for slot, condition, present in zip(data.CONDITION_SLOTS, ids, observed):
        if present:
            raw = hashlib.sha256((slot + '\0' + str(condition)).encode('utf8')).digest()
            hashed[int.from_bytes(raw[:8], 'big') % 16] += (1 if raw[8] % 2 == 0 else -1) / math.sqrt(10)
    result = np.concatenate((np.concatenate(means) @ matrices['B'], hashed,
        np.asarray(observed, dtype=np.float32), np.asarray([
            math.log1p(len(row['peptide_sequence'])), math.log1p(len(row['target_sequence']))], dtype=np.float32)))
    if result.shape != (92,) or not np.isfinite(result).all():
        raise ValueError('base feature dimension/finite mismatch')
    return result.astype(np.float32)


def capture_prompt(model, batch, width=1536):
    """Capture the numeric head INPUT, never its scalar output or label suffix."""
    import torch
    if 'labels' in batch:
        raise ValueError('probe extraction accepts prompt-only batches')
    if not torch.all(batch['prompt_end_index'] == batch['attention_mask'].shape[1] - 1):
        raise ValueError('prompt-only rows must end at their final active position')
    captured = []
    def capture(_module, inputs):
        if len(inputs) != 1:
            raise ValueError('unexpected numeric head signature')
        captured.append(inputs[0].detach().float().cpu().clone())
    was_training = model.training
    handle = model.numeric_head.register_forward_pre_hook(capture)
    try:
        model.eval()
        with torch.no_grad():
            model(**batch)
        if len(captured) != 1 or captured[0].shape != (batch['input_ids'].shape[0], width):
            raise ValueError('expected one finalnorm prompt representation per row')
        if not torch.isfinite(captured[0]).all():
            raise ValueError('nonfinite prompt representation')
        return captured[0]
    finally:
        handle.remove()
        model.train(was_training)


def extract(args, config, identity, binding, check, authorization):
    import numpy as np
    import torch
    from .mm_train import load_components, verify_model_assets, verify_checkpoint_identity, configure, assert_model_policy
    from .mm_extract import SequenceStore
    check()
    verify_model_assets(Path(args.project_root), config)
    store = SequenceStore(args.cache_root)
    store.verify()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    verify_checkpoint_identity(payload, identity)
    runtime = configure(17)
    model, tokenizer = load_components(Path(args.project_root), {**config, '_variant': 'M3'}, args.device)
    model.load_trainable_state_dict(payload['trainable_state'])
    policy = assert_model_policy(model, args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    matrices = projections()
    np.savez(output / 'projections.npz', **matrices)
    manifest = {'schema': 'mm_conditional_probe_features_v1', 'status': 'partial',
                'identity': identity, 'binding': binding, 'authorization': authorization,
                'protocol': PROTOCOL, 'projection_sha256': data.file_sha(output / 'projections.npz'),
                'runtime': runtime, 'model_policy': policy, 'device': args.device,
                'matrix_raw_sha256': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in matrices.items()},
                'splits': {}, 'pkd_used_for_features': False, 'numeric_head_output_used': False}
    for split in ('train', 'validation', 'test'):
        # Discard labels before collating; extraction never reads row['pkd'].
        rows = [data.input_view(row, labelled=False) for row in data.read_input_view(args.view_root, args.scenario, split)]
        bases, receivers = [], []
        for start in range(0, len(rows), 4):
            check()
            selected = rows[start:start + 4]
            batch = data.collate(selected, tokenizer, store, device=args.device, supervised=False)
            receivers.append(capture_prompt(model, batch).numpy())
            bases.extend(base_features(row, store, matrices) for row in selected)
        b, receiver = np.stack(bases), np.concatenate(receivers)
        np.savez(output / f'{split}.npz', B=b, R=receiver)
        ids = [row['sample_id'] for row in rows]
        manifest['splits'][split] = {'path': f'{split}.npz', 'sha256': data.file_sha(output / f'{split}.npz'),
            'rows': len(rows), 'ordered_ids': ids, 'ordered_ids_sha256': digest(ids), 'B_shape': list(b.shape),
            'R_shape': list(receiver.shape), 'dtype': 'float32'}
        write_new(output / f'{split}_receipt.json', manifest['splits'][split])
    check()
    manifest['status'] = 'complete'
    manifest['cache_receipt'] = store.provenance()
    write_new(output / 'manifest.json', manifest)
    return {'status': 'complete', 'stage': 'extract', 'manifest_sha256': data.file_sha(output / 'manifest.json')}


def fit_pair(train_b, train_r, y):
    """Pure CPU fixture API. Production execution additionally requires contract."""
    import numpy as np
    b, r, y = (np.asarray(v, dtype=np.float64) for v in (train_b, train_r, y))
    if b.ndim != 2 or r.ndim != 2 or b.shape[0] != r.shape[0] or y.shape != (b.shape[0],) or len(y) < 2:
        raise ValueError('probe TRAIN shape mismatch')
    if b.shape[1] != 92 or r.shape[1] != 64 or not all(np.isfinite(v).all() for v in (b, r, y)):
        raise ValueError('probe finite 92+64 TRAIN features required')
    states = {}
    for name, representation in (('conditioned', r), ('baseline', np.zeros_like(r))):
        x = np.concatenate((b, representation), axis=1)
        mean, std = x.mean(0), np.maximum(x.std(0), 1e-8)
        z = (x - mean) / std
        augmented = np.concatenate((z, np.ones((len(y), 1))), axis=1)
        penalty = np.eye(157) * 1000.; penalty[-1, -1] = 0
        coefficients = np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ y)
        if not np.isfinite(coefficients).all():
            raise ValueError('nonfinite ridge solution')
        singular = np.linalg.svd(z, compute_uv=False)
        states[name] = {'mean': mean, 'std': std, 'coefficients': coefficients,
            'nominal_width': 156, 'nominal_coefficients_with_intercept': 157,
            'nonconstant_train_columns': int((x.std(0) > 0).sum()),
            'centered_train_rank': int(np.linalg.matrix_rank(z)), 'ridge_lambda': 1000.}
        states[name]['effective_ridge_degrees_of_freedom_with_intercept'] = float(1 + (singular**2 / (singular**2 + 1000.)).sum())
        states[name]['receiver_coefficient_max_abs'] = float(np.abs(coefficients[92:156]).max())
    return states


def predict(state, b, r, arm):
    import numpy as np
    if arm not in ('baseline', 'conditioned'):
        raise ValueError('unknown probe arm')
    b, r = np.asarray(b, dtype=np.float64), np.asarray(r, dtype=np.float64)
    if b.ndim != 2 or r.shape != (len(b), 64) or b.shape[1] != 92:
        raise ValueError('probe prediction shape mismatch')
    x = np.concatenate((b, r if arm == 'conditioned' else np.zeros_like(r)), axis=1)
    values = ((x - state['mean']) / state['std']) @ state['coefficients'][:-1] + state['coefficients'][-1]
    if not np.isfinite(values).all():
        raise ValueError('nonfinite probe prediction')
    return values


def metrics(rows, predictions):
    from .mm_baselines import numeric_metrics
    values = [float(v) for v in predictions]
    result = numeric_metrics(rows, values)
    result['MSE'] = math.fsum((float(row['pkd']) - p) ** 2 for row, p in zip(rows, values)) / len(rows)
    result['Spearman_status'] = 'undefined_constant_or_singleton' if result['global_Spearman'] is None else 'defined'
    return result


def fit(args, config, identity, binding, check, authorization):
    import numpy as np
    root = Path(args.features)
    manifest = read_json(root / 'manifest.json')
    expected = {k: v for k, v in binding.items() if k not in ('stage', 'output', 'device', 'features', 'feature_manifest_sha256')}
    extracted = {k: v for k, v in manifest.get('binding', {}).items() if k not in ('stage', 'output', 'device')}
    if (manifest.get('status') != 'complete' or manifest.get('schema') != 'mm_conditional_probe_features_v1'
            or manifest.get('identity') != identity or manifest.get('protocol') != PROTOCOL or extracted != expected):
        raise ValueError('probe features not bound to exact checkpoint/data/code/protocol')
    data.verify_sha(root / 'projections.npz', manifest['projection_sha256'])
    with np.load(root / 'projections.npz', allow_pickle=False) as archive:
        matrices = {k: archive[k].copy() for k in ('B', 'R')}
    actual = projections()
    if any(not np.array_equal(matrices[k], actual[k]) for k in actual):
        raise ValueError('fixed random projection changed')
    rows, features = {}, {}
    for split in ('train', 'validation', 'test'):
        check()
        rows[split] = data.read_input_view(args.view_root, args.scenario, split)
        receipt = manifest['splits'][split]
        if receipt['ordered_ids'] != [row['sample_id'] for row in rows[split]]:
            raise ValueError('probe row identity/order mismatch')
        path = data.registered_path(root, receipt['path'])
        data.verify_sha(path, receipt['sha256'])
        with np.load(path, allow_pickle=False) as archive:
            b, receiver = archive['B'].copy(), archive['R'].copy()
        if (b.shape != (len(rows[split]), 92) or receiver.shape != (len(rows[split]), 1536)
                or b.dtype != np.float32 or receiver.dtype != np.float32
                or not np.isfinite(b).all() or not np.isfinite(receiver).all()):
            raise ValueError('probe stored feature shape/dtype/finite mismatch')
        features[split] = b, receiver @ matrices['R']
    states = fit_pair(*features['train'], [row['pkd'] for row in rows['train']])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    # Freeze both trained probe states and all predictions before opening TEST gold.
    state_json = {arm: {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in state.items()}
                  for arm, state in states.items()}
    write_new(output / 'probe_states.json', state_json)
    predictions = {}
    for split, (b, receiver) in features.items():
        check()
        predictions[split] = {arm: predict(state, b, receiver, arm).tolist() for arm, state in states.items()}
        with (output / f'{split}_predictions.jsonl').open('x', encoding='utf8') as stream:
            for i, row in enumerate(rows[split]):
                stream.write(data.canonical({'sample_id': row['sample_id'], 'split': split,
                    'baseline_pkd': predictions[split]['baseline'][i],
                    'conditioned_pkd': predictions[split]['conditioned'][i]}) + '\n')
    gold_path = data.registered_path(args.project_root, data.DATA_RELATIVE / args.scenario / 'test_gold.jsonl')
    data.verify_sha(gold_path, data.FROZEN_SHAS[args.scenario]['test_gold.jsonl'])
    gold = data.read_jsonl(gold_path)
    if [row['sample_id'] for row in gold] != [row['sample_id'] for row in rows['test']]:
        raise ValueError('TEST gold identity/order mismatch')
    rows['test'] = [{**row, 'pkd': g['pkd']} for row, g in zip(rows['test'], gold)]
    result = {'schema': 'mm_conditional_probe_result_v1', 'status': 'complete', 'binding': binding,
              'authorization': authorization, 'protocol': PROTOCOL,
              'fit_only_train_labels': True, 'states_frozen_before_test_gold': True,
              'checkpoint_selection': 'original_selected_theta_only_no_probe_selection',
              'not_mutual_information': True, 'single_seed_diagnostic': True, 'splits': {}}
    for split in rows:
        scores = {arm: metrics(rows[split], values) for arm, values in predictions[split].items()}
        scores['delta_baseline_minus_conditioned'] = {key: scores['baseline'][key] - scores['conditioned'][key]
            for key in ('MSE', 'micro_MAE', 'exact_target_macro_MAE')}
        result['splits'][split] = scores
    result['artifacts'] = {p.name: data.file_sha(p) for p in output.iterdir() if p.is_file()}
    check()
    write_new(output / 'result.json', result)
    return {'status': 'complete', 'stage': 'fit', 'result_sha256': data.file_sha(output / 'result.json')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('readiness', 'extract', 'fit'))
    parser.add_argument('--probe-protocol', required=True)
    names = ('config', 'checkpoint', 'scenario', 'view-root', 'cache-root', 'project-root', 'output')
    for name in names:
        parser.add_argument('--' + name)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--planned-stage', choices=('extract', 'fit'), default='extract')
    parser.add_argument('--features')
    for name in ('contract', 'validator', 'validator-sha256'):
        parser.add_argument('--' + name)
    args = parser.parse_args(argv)
    missing = [name for name in names if getattr(args, name.replace('-', '_')) is None]
    if (args.command == 'fit' or args.command == 'readiness' and args.planned_stage == 'fit') and not args.features:
        missing.append('features')
    load_protocol(args.probe_protocol)
    if args.command == 'readiness' and missing:
        print(json.dumps({'status': 'waiting_selected_fit_assets', 'protocol_valid': True,
            'probe_protocol_sha256': data.file_sha(args.probe_protocol), 'missing_arguments': missing,
            'model_loaded': False, 'fit_performed': False}))
        return 3
    if missing:
        parser.error('required execution arguments: ' + ', '.join(missing))
    try:
        config, identity, binding = _context(args)
    except (ValueError, OSError, KeyError) as exc:
        if args.command != 'readiness':
            raise
        print(json.dumps({'status': 'not_ready', 'reason': str(exc), 'model_loaded': False, 'fit_performed': False}))
        return 3
    if args.command == 'readiness':
        result = {'status': 'ready_assets_only_not_execution_permission', 'binding_for_requested_stage': binding,
                  'identity': identity, 'model_loaded': False, 'fit_performed': False,
                  'note': 'Use execution_binding with extract/fit stage when freezing the standard contract.'}
    else:
        check, authorization = authorize(args, binding)
        result = (extract if args.command == 'extract' else fit)(args, config, identity, binding, check, authorization)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
