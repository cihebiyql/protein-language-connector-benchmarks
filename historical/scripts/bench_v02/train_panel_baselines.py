"""Fresh, panel-specific BASE-01/02 predictors with validation-only selection."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_v02.cache_panel_features import DEFAULT_CACHE, ESM_REVISION, QWEN_REVISION, content_key, file_sha, protein_file, read_jsonl, text_file
from bench_v02.metrics import grouped_regression, known_positive_retrieval, regression
from bench_v02.panel_execution import validate_panel_data

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'results/bench_v02/panel_baselines_20261015'
SEEDS = (17, 42, 2026)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def masked_mean(values, mask):
    return (values * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class ContextEncoder:
    """Allowed chemistry/context only; numeric fits use training rows only."""
    bins = 128
    dimension = 2 + 6 + 12 + 128

    def __init__(self, train_rows):
        self.statistics = {}
        for key in ('temperature_kelvin', 'ph'):
            values = [r.get('experimental_context', {}).get(key) for r in train_rows]
            values = [float(v) for v in values if _finite(v)]
            self.statistics[key] = {'mean': float(np.mean(values)) if values else 0.,
                                    'scale': max(float(np.std(values)), 1e-5) if values else 1.,
                                    'n_observed_training_rows': len(values)}

    def __call__(self, row):
        context, chemical = row.get('experimental_context', {}), row.get('chemical_state', {})
        numeric = []
        for key in ('temperature_kelvin', 'ph'):
            value, stat = context.get(key), self.statistics[key]
            numeric.append((float(value) - stat['mean']) / stat['scale'] if _finite(value) else 0.)
        missing = [float(context.get(k) is None or context.get(k) == '')
                   for k in ('temperature_kelvin', 'ph', 'buffer', 'assay', 'actual_construct_sequence')]
        missing.append(float(bool(context.get('actual_construct_verified'))))
        categories = []
        for key, choices in [('n_terminal', ('free', 'acetyl')),
                             ('c_terminal', ('free', 'amide')),
                             ('topology', ('linear', 'cyclic')),
                             ('residue_modification', ('no', 'yes'))]:
            text = str(chemical.get(key) or '').strip().lower()
            categories.extend([float(text == choices[0]), float(text == choices[1]),
                               float(text not in choices)])
        # Fixed input hashing has no fitted test vocabulary. No IDs, DOI, labels,
        # publication or rawmetadata fields can enter this representation.
        allowed = {'chemical_state': chemical, 'assay': context.get('assay'),
                   'buffer': context.get('buffer'), 'target_sequence_role': row.get('target_sequence_role')}
        text = json.dumps(allowed, sort_keys=True, ensure_ascii=False).lower()
        hashed = np.zeros(self.bins, dtype=np.float32)
        for i in range(max(len(text) - 2, 1)):
            gram = text[i:i + 3]
            hashed[int.from_bytes(hashlib.sha256(gram.encode()).digest()[:4], 'little') % self.bins] += 1
        hashed /= max(float(np.linalg.norm(hashed)), 1.)
        return torch.tensor(numeric + missing + categories + hashed.tolist(), dtype=torch.float32)


class FeatureStore:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest_path = self.path / 'manifest.json'
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        if not self.manifest.get('verified_local_models'):
            raise ValueError('Feature cache must physically verify pinned local model assets')
        if self.manifest['model_revisions'] != {'ESM2': ESM_REVISION, 'Qwen': QWEN_REVISION}:
            raise ValueError('Unexpected frozen backbone revisions')
        self.memory = {}
        self.checked_proteins, self.checked_texts = {}, {}

    def residues(self, sequence):
        key = content_key(sequence)
        if key not in self.memory:
            path = protein_file(self.path, sequence)
            receipt = self.manifest['proteins'].get(key)
            if not receipt or file_sha(path) != receipt['sha256']:
                raise ValueError('Missing or changed full-sequence frozen features')
            value = torch.load(path, weights_only=True, map_location='cpu')
            assert value['sequence'] == sequence and value['model_revision'] == ESM_REVISION
            assert value['residues30'].shape == (len(sequence), 640)
            self.memory[key] = value['residues30']
            self.checked_proteins[key] = receipt['sha256']
        return self.memory[key]

    def batch(self, sequences, device):
        values = [self.residues(s) for s in sequences]
        masks = [torch.ones(len(s), dtype=torch.bool) for s in sequences]
        return pad_sequence(values, batch_first=True).to(device).float(), pad_sequence(masks, batch_first=True).to(device)

    def descriptions(self, texts, device):
        result = []
        for text in texts:
            key, path = content_key(text), text_file(self.path, text)
            receipt = self.manifest['texts'].get(key)
            if not receipt or file_sha(path) != receipt['sha256']:
                raise ValueError('Missing or changed frozen description features')
            value = torch.load(path, weights_only=True, map_location='cpu')
            assert value['text'] == text and value['model_revision'] == QWEN_REVISION
            result.append(value['mean'])
            self.checked_texts[key] = receipt['sha256']
        return torch.stack(result).to(device)


class A1Predictor(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.peptide = nn.Sequential(nn.LayerNorm(640), nn.Linear(640, 128), nn.GELU())
        self.target = nn.Sequential(nn.LayerNorm(640), nn.Linear(640, 128), nn.GELU())
        self.context = nn.Sequential(nn.Linear(ContextEncoder.dimension, 64), nn.GELU())
        if attention:
            self.peptide_query = nn.Linear(128 + 64, 128)
            self.target_query = nn.Linear(128 + 64, 128)
        self.head = nn.Sequential(nn.Linear(4 * 128 + 64 + 2, 256), nn.GELU(), nn.Dropout(.1), nn.Linear(256, 1))

    def forward(self, peptide, peptide_mask, target, target_mask, context):
        p, t, c = self.peptide(peptide), self.target(target), self.context(context)
        pm, tm = masked_mean(p, peptide_mask), masked_mean(t, target_mask)
        if self.attention:
            pq = self.peptide_query(torch.cat((tm, c), -1))
            tq = self.target_query(torch.cat((pm, c), -1))
            pa = ((p * pq.unsqueeze(1)).sum(-1) / math.sqrt(128)).masked_fill(~peptide_mask, -torch.inf).softmax(-1)
            ta = ((t * tq.unsqueeze(1)).sum(-1) / math.sqrt(128)).masked_fill(~target_mask, -torch.inf).softmax(-1)
            pm, tm = (p * pa.unsqueeze(-1)).sum(1), (t * ta.unsqueeze(-1)).sum(1)
        lengths = torch.stack((peptide_mask.sum(1).float().log1p(), target_mask.sum(1).float().log1p()), -1)
        return self.head(torch.cat((pm, tm, pm * tm, (pm - tm).abs(), c, lengths), -1)).squeeze(-1)


class B1Aligner(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.sequence = nn.Sequential(nn.LayerNorm(640), nn.Linear(640, 256), nn.GELU(), nn.Dropout(.1))
        self.text = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, 256), nn.GELU(), nn.Dropout(.1))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(.07)))

    def forward(self, residues, mask, descriptions):
        seq, text = self.sequence(residues), F.normalize(self.text(descriptions), dim=-1)
        temperature = self.log_temperature.exp().clamp(.01, 1.)
        if self.attention:
            attention = torch.einsum('bld,cd->bcl', F.normalize(seq, dim=-1), text) / temperature
            weights = attention.masked_fill(~mask[:, None, :], -torch.inf).softmax(-1)
            pooled = F.normalize(torch.einsum('bcl,bld->bcd', weights, seq), dim=-1)
            return torch.einsum('bcd,cd->bc', pooled, text) / temperature
        pooled = F.normalize(masked_mean(seq, mask), dim=-1)
        return pooled @ text.T / temperature


def deduplicate_b1(rows):
    """One local-sequence query per split, with all documented positives."""
    grouped = {}
    for row in rows:
        key = (row['split'], row['segment_sequence'])
        if key not in grouped:
            grouped[key] = {**row, 'member_sample_ids': [], 'known_positive_class_ids': set(),
                            'known_positive_candidate_ids': set()}
        merged = grouped[key]
        merged['member_sample_ids'].append(row['sample_id'])
        merged['known_positive_class_ids'].update(row.get('known_positive_class_ids', [row['class_id']]))
        merged['known_positive_candidate_ids'].update(row['known_positive_candidate_ids'])
    result = []
    for row in grouped.values():
        row['known_positive_class_ids'] = sorted(row['known_positive_class_ids'])
        row['known_positive_candidate_ids'] = sorted(row['known_positive_candidate_ids'])
        result.append(row)
    return result


def multi_positive_loss(scores, positive_mask):
    if scores.shape != positive_mask.shape or not positive_mask.any(1).all():
        raise ValueError('Each training query requires all documented positive candidates')
    # Uniform positive CE: no documented shared-sequence annotation is a negative.
    weights = positive_mask.float() / positive_mask.sum(1, keepdim=True)
    return -(weights * scores.log_softmax(-1)).sum(-1)


def a1_metrics(rows, predictions):
    values = [r['pkd'] for r in rows]
    targets = [r['groups']['target_sequence_sha256'] for r in rows]
    score = regression(values, predictions, targets)
    comparable = [i for i, r in enumerate(rows) if r['experimental_context'].get('comparable_condition_group')]
    score['within_target_condition'] = grouped_regression(
        [values[i] for i in comparable], [float(predictions[i]) for i in comparable],
        [rows[i]['experimental_context']['comparable_condition_group'] for i in comparable],
        [targets[i] for i in comparable]) if comparable else {'status': 'undefined_no_verified_comparable_conditions', 'n': 0}
    score['experimental_pair_direction'] = {'status': 'undefined_no_verified_same_condition_pair_panel'}
    return score


def b1_metrics(rows, scores, candidates):
    order = np.argsort(-scores, axis=1, kind='stable')
    rankings = [[candidates[i]['class_id'] for i in ranking] for ranking in order]
    return known_positive_retrieval(rankings, [r['known_positive_class_ids'] for r in rows])


def _batches(order, size):
    return [order[i:i + size] for i in range(0, len(order), size)]


def train_run(panel, scenario, rows, receipt, candidates, store, architecture, seed, args):
    name = f'{panel}_{scenario}_{architecture}_seed{seed}'
    directory = args.output / name
    if (directory / 'result.json').exists():
        raise ValueError(f'Refuse to overwrite completed fresh run: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'validation']
    test = [r for r in rows if r['split'] == 'test']
    context = ContextEncoder(train) if panel == 'A1' else None
    if panel == 'A1':
        y_center, y_scale = float(np.mean([r['pkd'] for r in train])), max(float(np.std([r['pkd'] for r in train])), 1e-5)
        model = A1Predictor(architecture == 'BASE01').to(args.device)
        descriptions, train_candidate_indices = None, None
    else:
        y_center, y_scale = None, None
        model = B1Aligner(architecture == 'BASE02').to(args.device)
        descriptions = store.descriptions([c['description'] for c in candidates], args.device)
        train_candidate_indices = [i for i, c in enumerate(candidates) if c['split'] == 'train']
        train_candidate_ids = {candidates[i]['candidate_id'] for i in train_candidate_indices}
        if any(not set(r['known_positive_candidate_ids']) <= train_candidate_ids for r in train):
            raise ValueError('Training positives cross class holdout; do not fit held-out class labels')
    registration = {'benchmark_version': '0.2', 'panel_id': panel, 'scenario': scenario,
                    'role': 'controlled_predictor', 'channel': 'predictor', 'architecture': architecture,
                    'seed': seed, 'epochs': 30, 'effective_batch_size': 64, 'micro_batch_size': args.micro_batch,
                    'optimizer': {'name': 'AdamW', 'lr': .0003, 'weight_decay': .01, 'betas': [.9, .999]},
                    'early_stopping': False, 'selection': 'earliest best validation MAE' if panel == 'A1' else 'earliest best validation class-macro MRR',
                    'training_units': 'source assay measurements' if panel == 'A1' else 'distinct exact local sequence per split, union of all documented positives',
                    'split_counts_after_unit_aggregation': {'train': len(train), 'validation': len(validation), 'test': len(test)},
                    'data_gate': receipt, 'initialization': 'fresh seeded weights; no legacy trained checkpoint loaded',
                    'feature_cache_manifest_sha256': file_sha(store.manifest_path),
                    'script_sha256': file_sha(__file__), 'cache_script_sha256': file_sha(Path(__file__).with_name('cache_panel_features.py')),
                    'preregistered_protocol_sha256': file_sha(args.output / 'preregistered_protocol.json'),
                    'architecture_registration_sha256': file_sha(args.output / 'architecture_training_preregistration.json'),
                    'foundation_revisions': {'ESM2': ESM_REVISION, 'Qwen': QWEN_REVISION},
                    'normalization': {'label_training_mean': y_center, 'label_training_scale': y_scale,
                                      'context_numeric_train_statistics': context.statistics if context else None,
                                      'residues': 'learned LayerNorm; no test-fitted statistics'},
                    'text_pool': 'frozen Qwen attention-mask mean of full rule-free description' if panel == 'B1' else None,
                    'unannotated_is_negative': False,
                    'b1_loss_scope': 'train classes only; positive-normalized CE over every documented same-sequence train positive; alternatives are contrastive unconfirmed classes' if panel == 'B1' else None,
                    'trainable_parameters': sum(p.numel() for p in model.parameters())}
    registration_path = directory / 'run_preregistration.json'
    if registration_path.exists() and json.loads(registration_path.read_text()) != registration:
        raise ValueError('Interrupted run registration changed; choose a fresh run directory')
    write_json(registration_path, registration)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    history, best, best_state, best_epoch, steps = [], (math.inf if panel == 'A1' else -math.inf), None, None, 0
    started = time.perf_counter()
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    def forward(batch):
        if panel == 'A1':
            p, pm = store.batch([r['peptide_sequence'] for r in batch], args.device)
            t, tm = store.batch([r['target_sequence'] for r in batch], args.device)
            c = torch.stack([context(r) for r in batch]).to(args.device)
            return model(p, pm, t, tm, c)
        residues, mask = store.batch([r['segment_sequence'] for r in batch], args.device)
        return model(residues, mask, descriptions)

    def predict(selected):
        model.eval()
        with torch.inference_mode():
            values = torch.cat([forward(selected[i:i + args.micro_batch]).cpu()
                                for i in range(0, len(selected), args.micro_batch)]).numpy()
        return values * y_scale + y_center if panel == 'A1' else values

    for epoch in range(1, 31):
        model.train()
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        train_loss = 0.
        for effective in _batches(order, 64):
            optimizer.zero_grad(set_to_none=True)
            for micro in _batches(effective, args.micro_batch):
                batch = [train[i] for i in micro]
                scores = forward(batch)
                if panel == 'A1':
                    labels = torch.tensor([(r['pkd'] - y_center) / y_scale for r in batch], device=args.device)
                    losses = (scores - labels).square()
                else:
                    candidate_ids = [candidates[i]['candidate_id'] for i in train_candidate_indices]
                    positives = torch.tensor([[c in r['known_positive_candidate_ids'] for c in candidate_ids]
                                               for r in batch], device=args.device)
                    losses = multi_positive_loss(scores[:, train_candidate_indices], positives)
                if not torch.isfinite(losses).all():
                    raise ValueError('Nonfinite training loss')
                loss = losses.sum() / len(effective)
                loss.backward()
                train_loss += float(losses.detach().sum())
            optimizer.step()
            steps += 1
        validation_scores = predict(validation)
        metric = a1_metrics(validation, validation_scores) if panel == 'A1' else b1_metrics(validation, validation_scores, candidates)
        value = metric['mae'] if panel == 'A1' else metric['class_macro_mrr']
        history.append({'epoch': epoch, 'train_loss': train_loss / len(train), 'validation': metric})
        if (value < best) if panel == 'A1' else (value > best):
            best, best_epoch = value, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        write_json(directory / 'history.json', history)
        print(json.dumps({'run': name, 'epoch': epoch, 'train_loss': train_loss / len(train),
                          'validation_selection_metric': value, 'best_epoch': best_epoch}), flush=True)
    assert steps == 30 * math.ceil(len(train) / 64)
    model.load_state_dict(best_state)
    checkpoint_path = directory / 'best.pt'
    torch.save({'model': best_state, 'registration': registration, 'best_epoch': best_epoch,
                'label_center': y_center, 'label_scale': y_scale,
                'context_statistics': context.statistics if context else None}, checkpoint_path)
    checkpoint_sha = file_sha(checkpoint_path)
    test_scores = predict(test)
    metrics = a1_metrics(test, test_scores) if panel == 'A1' else b1_metrics(test, test_scores, candidates)
    predictions = []
    for i, row in enumerate(test):
        base = {'sample_id': row['sample_id'], 'panel_id': panel, 'channel': 'predictor',
                'checkpoint_sha256': checkpoint_sha, 'group_id': row.get('group_id', row['groups'].get('parent_homology'))}
        if panel == 'A1':
            base['prediction'] = {'pKD': float(test_scores[i])}
        else:
            ranking = np.argsort(-test_scores[i], kind='stable')
            base['prediction'] = {'ranked_candidate_ids': [candidates[j]['candidate_id'] for j in ranking]}
            base['member_sample_ids'] = row['member_sample_ids']
            base['known_positive_candidate_ids'] = row['known_positive_candidate_ids']
        predictions.append(base)
    (directory / 'test_predictions.jsonl').write_text('\n'.join(json.dumps(p) for p in predictions) + '\n', encoding='utf-8')
    np.savez_compressed(directory / 'test_scores.npz', scores=test_scores,
                        sample_ids=np.array([r['sample_id'] for r in test]),
                        candidate_ids=np.array([c['candidate_id'] for c in candidates]) if candidates else np.array([]),
                        candidate_class_ids=np.array([c['class_id'] for c in candidates]) if candidates else np.array([]),
                        labels=np.array([r['pkd'] for r in test]) if panel == 'A1' else np.array([]))
    result = {**registration, 'status': 'complete', 'checkpoint_path': str(checkpoint_path),
              'checkpoint_sha256': checkpoint_sha, 'best_epoch': best_epoch,
              'best_validation_metric': best, 'optimizer_steps': steps, 'actual_epochs': 30,
              'training_seconds': time.perf_counter() - started, 'test_metrics': metrics,
              'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0,
              'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved() if args.device == 'cuda' else 0,
              'feature_receipts': {'proteins': dict(store.checked_proteins), 'texts': dict(store.checked_texts)},
              'artifact_sha256': {p.name: file_sha(p) for p in directory.iterdir() if p.is_file()},
              'formal_multitask_connector_result': False}
    write_json(directory / 'result.json', result)
    return {'run_id': name, 'result_path': str(directory / 'result.json'),
            'result_sha256': file_sha(directory / 'result.json'), 'test_metrics': metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', choices=['A1', 'B1', 'all'], default='all')
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--a1-root', type=Path, default=ROOT / 'data/processed/bench_v02/a1_locked')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--micro-batch', type=int, default=8)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    args = parser.parse_args()
    if tuple(args.seeds) != SEEDS or not 1 <= args.micro_batch <= 64:
        raise ValueError('Registered three seeds and micro-batch <=64 required')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    store = FeatureStore(args.cache)
    jobs = []
    # B1 executes first so independently ready panels are not blocked by A1.
    if args.panel in ('B1', 'all'):
        path = ROOT / 'data/processed/bench_v02/b1_homology_locked'
        rows, receipt = validate_panel_data(path / 'b1_samples.jsonl', path / 'benchmark_lock.json', 'B1')
        mapping = {r['candidate_id']: r for r in read_jsonl(path / 'b1_classes_main.jsonl')}
        candidates = [{**r, 'class_id': mapping[r['candidate_id']]['class_id'],
                       'split': mapping[r['candidate_id']]['split']}
                      for r in read_jsonl(path / 'b1_candidates_main.jsonl')]
        rows = deduplicate_b1(rows)
        jobs.append(('B1', 'unseen_classes', rows, receipt, candidates, ('BASE02', 'MEANALIGN')))
    if args.panel in ('A1', 'all'):
        for scenario in ('new_peptide_known_target', 'new_peptide_new_target'):
            path = args.a1_root / scenario
            rows, receipt = validate_panel_data(path / 'samples.jsonl', path / 'benchmark_lock.json', 'A1')
            jobs.append(('A1', scenario, rows, receipt, [], ('BASE01', 'MEANPOOL')))
    summary_path = args.output / 'neural_run_summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {
        'benchmark_version': '0.2', 'scope': 'independently locked controlled panel predictors', 'runs': []}
    for panel, scenario, rows, receipt, candidates, models in jobs:
        for architecture in models:
            for seed in SEEDS:
                name = f'{panel}_{scenario}_{architecture}_seed{seed}'
                existing = args.output / name / 'result.json'
                if existing.exists():
                    if any(r['run_id'] == name and r['result_sha256'] == file_sha(existing) for r in summary['runs']):
                        continue
                    raise ValueError('Unindexed completed artifact; inspect rather than overwrite')
                result = train_run(panel, scenario, rows, receipt, candidates, store, architecture, seed, args)
                summary['runs'].append(result)
                write_json(summary_path, summary)


if __name__ == '__main__':
    main()
