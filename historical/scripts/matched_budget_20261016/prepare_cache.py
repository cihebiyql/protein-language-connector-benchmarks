"""CPU preparation of frozen messages and shared, untrained reader initializations.

No backbone is loaded or executed. The input is the already hash-verified ESM
residue cache. Scenario-specific neutral messages fit TRAIN inputs only, with no
labels. Test membership, labels, prompts, and metrics are never changed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_v02.cache_panel_features import DEFAULT_CACHE, file_sha
from bench_v02.panel_execution import validate_panel_data
from bench_v02.train_panel_baselines import FeatureStore
from foundation_20261016.run_connectors import write_json, utc_now

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ('new_peptide_known_target', 'new_peptide_new_target')
EXPECTED_COUNTS = {SCENARIOS[0]: {'train': 2062, 'validation': 147, 'test': 152},
                   SCENARIOS[1]: {'train': 2063, 'validation': 431, 'test': 442}}
SEEDS = (17, 42, 2026)
LEGACY_SHA256 = '88da80f9c75116625c2ec8601d4d5685882c4d899f693371338179f36e905da5'
LEGACY_SOURCE = ROOT / 'scripts/foundation_20261016/run_connectors.py'


def verify_legacy_source():
    if file_sha(LEGACY_SOURCE) != LEGACY_SHA256:
        raise ValueError('Frozen legacy prompt/label/parser/metric implementation changed')


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def verify_pins(pins):
    for name, expected in pins.items():
        path = ROOT / name
        if not path.is_file() or file_sha(path) != expected:
            raise ValueError(f'Frozen source fingerprint mismatch: {name}')


def build_messages(rows, store, sender, sender_mode):
    """Pure input-only extraction, preserving physical source row order."""
    messages, train_indices = [], []
    checked = {}
    for index, row in enumerate(rows):
        value = sender.encode(store.residues(row['peptide_sequence']).float(),
                              store.residues(row['target_sequence']).float(), sender_mode=sender_mode)
        if value.shape != (16, 128) or not torch.isfinite(value).all():
            raise ValueError('Fixed sender must emit finite float32 [16,128] messages')
        messages.append(value.detach().cpu().float())
        if row['split'] == 'train':
            train_indices.append(index)
        checked.update(getattr(store, 'checked_proteins', {}))
        # A bound on resident source features; the immutable on-disk cache stays intact.
        memory = getattr(store, 'memory', {})
        if sum(x.numel() * x.element_size() for x in memory.values()) > 256 * 1024**2:
            memory.clear()
    if not train_indices or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Nonempty TRAIN and unique source sample IDs required')
    from matched_budget_20261016.fixed_message import neutral_message
    values = torch.stack(messages)
    neutral = neutral_message(values[train_indices])
    return {'schema_version': 1, 'sample_ids': [r['sample_id'] for r in rows],
            'splits': [r['split'] for r in rows], 'messages': values, 'neutral': neutral,
            'neutral_fit_train_sample_ids': [rows[i]['sample_id'] for i in train_indices],
            'neutral_fit_uses_labels': False, 'sender_mode': sender_mode}, checked


def save_initializations(output):
    """One random state per seed; variants differ solely in alpha=0 versus 3."""
    from matched_budget_20261016.connectors import MatchedReader
    records = {}
    for seed in SEEDS:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            reader = MatchedReader(hidden_dim=1536, memory_dim=128, gate_init='zero')
        zero = {k: v.detach().cpu().clone() for k, v in reader.state_dict().items()}
        if 'alpha' not in zero or zero['alpha'].numel() != 1:
            raise ValueError('Exactly one shared scalar gate alpha required')
        near = {k: v.clone() for k, v in zero.items()}
        near['alpha'].fill_(3.)
        if sum(p.numel() for p in reader.parameters() if p.requires_grad) != 464001:
            raise ValueError('Expected exactly 464001 active trainable reader parameters')
        for mode, state in (('zero', zero), ('near_unit', near)):
            path = output / f'initial_reader_seed{seed}_{mode}.pt'
            torch.save({'reader_state': state, 'seed': seed, 'gate_init': mode,
                        'untrained': True, 'source_labels_used': False}, path)
            records[f'{seed}:{mode}'] = {'path': relative(path), 'sha256': file_sha(path),
                                        'tensor_sha256': {k: tensor_sha(v) for k, v in state.items()},
                                        'trainable_parameters': 464001, 'untrained': True}
        if any(not torch.equal(v, near[k]) for k, v in zero.items() if k != 'alpha'):
            raise ValueError('Variant initialization differs outside gate alpha')
    return records


def prepare(output, feature_cache=DEFAULT_CACHE):
    verify_legacy_source()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    from matched_budget_20261016.fixed_message import FixedMessageSender
    sender = FixedMessageSender(seed=314159, source_dim=640, memory_dim=128, slots_per_molecule=8)
    if any(p.requires_grad for p in sender.parameters()):
        raise ValueError('Fixed sender cannot have trainable parameters')
    source_files = [Path(__file__), Path(__file__).with_name('run_trial.py'),
                    Path(__file__).with_name('connectors.py'), Path(__file__).with_name('fixed_message.py'),
                    Path(__file__).with_name('token_diagnostics.py'),
                    LEGACY_SOURCE, ROOT / 'scripts/foundation_20261016/connectors.py',
                    ROOT / 'scripts/foundation_20261016/information_diagnostics.py',
                    ROOT / 'results/foundation_20261016/information_diagnostics/new_peptide_known_target_donor_map_preflight_v2.json',
                    ROOT / 'results/foundation_20261016/information_diagnostics/new_peptide_new_target_donor_map_preflight_v2.json',
                    ROOT / 'scripts/bench_v02/train_panel_baselines.py',
                    ROOT / 'scripts/bench_v02/cache_panel_features.py',
                    ROOT / 'scripts/bench_v02/panel_execution.py', ROOT / 'scripts/bench_v02/metrics.py',
                    ROOT / 'scripts/bench_v02/contracts.py', ROOT / 'sources/model_manifest.json',
                    Path(feature_cache) / 'manifest.json']
    pins = {relative(p): file_sha(p) for p in source_files}
    store = FeatureStore(feature_cache)
    datasets, all_receipts = {}, {}
    for scenario in SCENARIOS:
        base = ROOT / 'data/processed/bench_v02/a1_locked_v2' / scenario
        samples, lock_path = base / 'samples.jsonl', base / 'benchmark_lock.json'
        rows, gate = validate_panel_data(samples, lock_path, 'A1')
        if gate['split_counts'] != EXPECTED_COUNTS[scenario]:
            raise ValueError('Unexpected frozen full-scenario counts')
        lock = json.loads(lock_path.read_text(encoding='utf-8'))
        pins.update(lock['pins'])
        for p in (samples, lock_path, base / 'test_gold.jsonl'):
            pins[relative(p)] = file_sha(p)
        datasets[scenario] = {'data_gate': gate, 'samples_path': relative(samples),
                              'lock_path': relative(lock_path), 'layouts': {}}
        for mode in ('balanced', 'global16'):
            payload, checked = build_messages(rows, store, sender, mode)
            all_receipts.update(checked)
            payload.update({'scenario': scenario, 'samples_sha256': file_sha(samples),
                            'projection_sha256': tensor_sha(sender.projection)})
            path = output / f'{scenario}_{mode}.pt'
            torch.save(payload, path)
            datasets[scenario]['layouts'][mode] = {'path': relative(path), 'sha256': file_sha(path),
                                                  'shape': list(payload['messages'].shape),
                                                  'neutral_sha256': tensor_sha(payload['neutral']),
                                                  'neutral_fit_train_count': len(payload['neutral_fit_train_sample_ids'])}
    initializations = save_initializations(output)
    projection_path = output / 'fixed_projection.pt'
    torch.save({'projection': sender.projection.detach().cpu().clone(), 'seed': 314159,
                'labels_used': False, 'data_used_for_projection': False}, projection_path)
    # FeatureStore verified every used payload before extraction. Preserve those
    # receipts without forcing every future training run to reread unused raw tensors.
    raw_receipts = {key: {'path': relative(Path(feature_cache) / 'proteins' / f'{key}.pt'), 'sha256': value}
                    for key, value in all_receipts.items()}
    verify_pins(pins)
    manifest = {'schema_version': 1, 'purpose': 'matched_budget_fixed_message_cache', 'created_utc': utc_now(),
                'scope_started_at': '2026-10-16T13:20:00+08:00',
                'scope_deadline_at': '2026-10-16T23:20:00+08:00',
                'relation_to_foundation': 'New user-authorized multiple-seed matched-budget objective; historical foundation execution contracts are not rerun',
                'source_pins': pins, 'legacy_semantics_sha256': LEGACY_SHA256,
                'feature_manifest_path': relative(store.manifest_path),
                'feature_manifest_sha256': file_sha(store.manifest_path), 'verified_raw_feature_receipts': raw_receipts,
                'projection': {'path': relative(projection_path), 'sha256': file_sha(projection_path),
                               'tensor_sha256': tensor_sha(sender.projection), 'seed': 314159,
                               'shape': list(sender.projection.shape), 'data_or_labels_used': False},
                'sender_trainable_parameters': 0, 'message_shape_per_example': [16, 128],
                'neutral_policy': 'Per-scenario and per-layout TRAIN-only mean at every slot; no labels; no cross-scenario training weights',
                'datasets': datasets, 'initializations': initializations,
                'training_or_model_inference_performed': False}
    write_json(output / 'manifest.json', manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--feature-cache', type=Path, default=DEFAULT_CACHE)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    if args.threads not in (1, 2):
        raise ValueError('CPU preparation permits one or two threads')
    torch.set_num_threads(args.threads)
    result = prepare(args.output, args.feature_cache)
    print(json.dumps({'status': 'prepared', 'manifest': str(args.output / 'manifest.json'),
                      'manifest_sha256': file_sha(args.output / 'manifest.json'),
                      'scenarios': list(result['datasets'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
