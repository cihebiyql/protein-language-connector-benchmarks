"""Offline A1-v2 connector training and genuine held-out text generation.

All protein residues enter through the pinned ESM30 cache. The language model
receives only an anonymous question and allowlisted experimental context. This
runner does not assert whole-paper replication or a completed multitask study.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_v02.cache_panel_features import (
    DEFAULT_CACHE, ESM_REVISION, QWEN_REVISION, file_sha, verify_local_models,
)
from bench_v02.panel_execution import validate_panel_data
from bench_v02.train_panel_baselines import FeatureStore, a1_metrics
from foundation_20261016.connectors import FrozenConnectorLM

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / 'data/processed/bench_v02/a1_locked_v2'
METHODS = ('input_mlp', 'qformer_style', 'internal_cross_attention', 'constant')
SCENARIOS = ('new_peptide_known_target', 'new_peptide_new_target')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def context_view(row):
    """Same information allowlist as ContextEncoder, in readable values.

    The predictor standardizes numeric fields using training data and hashes
    categorical strings. This LM instead receives their true input values.
    Neither representation receives construct sequence text, IDs, or answers.
    """
    chemical = row.get('chemical_state') or {}
    context = row.get('experimental_context') or {}
    result = {'chemical_state': dict(chemical),
              'experimental_context': {key: float(context[key]) if _finite(context.get(key)) else None
                                       for key in ('temperature_kelvin', 'ph')},
              'missing': {key: context.get(key) in (None, '')
                          for key in ('temperature_kelvin', 'ph', 'assay', 'buffer', 'actual_construct_sequence')},
              'actual_construct_verified': bool(context.get('actual_construct_verified')),
              'assay': context.get('assay'), 'buffer': context.get('buffer'),
              'target_sequence_role': row.get('target_sequence_role')}
    return result


def prompt_text(row, tokenizer):
    user = ('Predict experimental binding affinity from the supplied peptide and target residue memories. '
            'Memory type 0 is the peptide; memory type 1 is the full target reference protein. '
            'The reference protein can differ from the unverified experimental construct. '
            'The endpoint is pKD = -log10(KD in molar). Return exactly one JSON object '
            'with one finite numeric field pKD, without explanations or additional fields.\n'
            'Allowed chemistry and experimental context:\n'
            + json.dumps(context_view(row), sort_keys=True, separators=(',', ':'), allow_nan=False))
    messages = [{'role': 'system', 'content': 'You estimate protein-peptide affinity from residue representations.'},
                {'role': 'user', 'content': user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def answer_text(row):
    if not _finite(row.get('pkd')):
        raise ValueError('A1 training requires a finite real pKD label')
    # Two decimals are the preregistered serialization precision, not a readout.
    # Real held-out labels retain their original precision when scoring.
    return '{"pKD":' + format(float(row['pkd']), '.2f') + '}'


def parse_answer(text):
    """Exact schema; reject duplicates, prose, nonfinite values, and booleans."""
    def object_hook(pairs):
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError('duplicate_key')
        return dict(pairs)

    try:
        value = json.loads(text, object_pairs_hook=object_hook,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite')))
    except (ValueError, TypeError):
        return {'valid': False, 'value': None, 'reason': 'invalid_json_or_nonfinite'}
    if not isinstance(value, dict) or set(value) != {'pKD'}:
        return {'valid': False, 'value': None, 'reason': 'wrong_schema'}
    if not _finite(value['pKD']):
        return {'valid': False, 'value': None, 'reason': 'nonfinite_or_nonnumeric_pKD'}
    # pKD has no universal bounded range. Do not clip predictions or derive a
    # permitted range from held-out labels; any finite real pKD is admissible.
    return {'valid': True, 'value': float(value['pKD']), 'reason': None}


def generation_metrics(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError('Every held-out sample must have a prediction record')
    if any(row['sample_id'] != prediction['sample_id'] for row, prediction in zip(rows, predictions)):
        raise ValueError('Prediction/sample order or identities differ')
    valid = [i for i, prediction in enumerate(predictions) if prediction['parse']['valid']]
    selected_rows = [rows[i] for i in valid]
    selected_values = [predictions[i]['parse']['value'] for i in valid]
    return {'n_total': len(rows), 'n_valid': len(valid), 'n_failed': len(rows) - len(valid),
            'parse_valid_rate': len(valid) / len(rows) if rows else None,
            'failure_counts': dict(Counter(prediction['parse']['reason'] for prediction in predictions
                                           if not prediction['parse']['valid'])),
            'regression_valid_only': a1_metrics(selected_rows, selected_values),
            'regression_denominator': 'only strictly parseable finite JSON answers; failures reported above',
            'full_sample_mae': a1_metrics(selected_rows, selected_values)['mae'] if len(valid) == len(rows) else None,
            'full_sample_mae_status': 'undefined_if_any_generation_fails' if len(valid) != len(rows)
                                      else 'see_regression_valid_only',
            'prediction_clipping': False, 'hand_corrected_outputs': False}


def select_splits(rows, limits):
    result, manifest = {}, {}
    for split in ('train', 'validation', 'test'):
        full = [row for row in rows if row['split'] == split]
        limit = limits.get(split)
        if limit is not None and limit < 1:
            raise ValueError('Explicit subset limits must be positive')
        selected = full if limit is None else full[:limit]
        if not selected:
            raise ValueError(f'Empty {split} split')
        result[split] = selected
        manifest[split] = {'available_rows': len(full), 'selected_rows': len(selected),
                           'requested_limit': limit, 'sample_ids': [row['sample_id'] for row in selected]}
    return result, manifest


def collate(rows, tokenizer, store, device, supervised=True):
    ids, targets, memories, memory_types = [], [], [], []
    for row in rows:
        prefix = tokenizer.encode(prompt_text(row, tokenizer), add_special_tokens=False)
        suffix = tokenizer.encode(answer_text(row), add_special_tokens=False) + [tokenizer.eos_token_id] if supervised else []
        if not prefix or (supervised and len(suffix) < 2):
            raise ValueError('Empty prompt or answer tokenization')
        ids.append(torch.tensor(prefix + suffix, dtype=torch.long))
        targets.append(torch.tensor([-100] * len(prefix) + suffix, dtype=torch.long))
        peptide = store.residues(row['peptide_sequence']).float()
        target = store.residues(row['target_sequence']).float()
        if peptide.shape != (len(row['peptide_sequence']), 640) or target.shape != (len(row['target_sequence']), 640):
            raise ValueError('ESM cache does not contain all peptide and target residues')
        memory = torch.cat((peptide, target))
        if not torch.isfinite(memory).all():
            raise ValueError('Nonfinite ESM residue memory')
        memories.append(memory)
        memory_types.append(torch.cat((torch.zeros(len(peptide), dtype=torch.long),
                                       torch.ones(len(target), dtype=torch.long))))
    # Left padding makes batched greedy decoding start at every real last token.
    max_length = max(map(len, ids))
    input_ids = torch.full((len(rows), max_length), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    attention = torch.zeros_like(input_ids)
    for i, (tokens, answer) in enumerate(zip(ids, targets)):
        input_ids[i, -len(tokens):] = tokens
        labels[i, -len(tokens):] = answer
        attention[i, -len(tokens):] = 1
    memory_mask = pad_sequence([torch.ones(len(value), dtype=torch.bool) for value in memories], batch_first=True)
    result = {'input_ids': input_ids, 'attention_mask': attention,
              'memory': pad_sequence(memories, batch_first=True), 'memory_mask': memory_mask,
              'memory_type_ids': pad_sequence(memory_types, batch_first=True)}
    if supervised:
        result['labels'] = labels
    return {key: value.to(device) for key, value in result.items()}


def supervised_nll(output, labels):
    logits = output.logits[:, :-1].float()
    shifted = labels[:, 1:]
    if logits.shape[:2] != shifted.shape:
        raise ValueError('Connector output must align to original text tokens')
    count = int((shifted != -100).sum())
    if not count:
        raise ValueError('No supervised answer tokens')
    summed = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), shifted.reshape(-1),
                             ignore_index=-100, reduction='sum')
    if not torch.isfinite(summed):
        raise ValueError('Nonfinite supervised NLL')
    return summed, count


def assert_frozen(base_lm):
    if any(parameter.requires_grad for parameter in base_lm.parameters()):
        raise ValueError('Language model backbone is trainable')
    if any(parameter.grad is not None for parameter in base_lm.parameters()):
        raise ValueError('Frozen backbone accumulated gradients')


def connector_state(model):
    # Saving the 1.5B frozen backbone repeatedly is unnecessary. Physical model
    # assets are pinned separately, and every learned tensor is saved here.
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items() if key in names}


def load_connector_state(model, state):
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state) != expected:
        raise ValueError('Checkpoint trainable tensor names differ')
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state:
                parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


def save_checkpoint(path, payload):
    temporary = path.with_suffix(path.suffix + '.partial')
    torch.save(payload, temporary)
    temporary.replace(path)


def validation_nll(model, rows, tokenizer, store, device, batch_size):
    model.eval()
    total, count = 0., 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = collate(rows[start:start + batch_size], tokenizer, store, device)
            summed, n = supervised_nll(model(**batch), batch['labels'])
            total += float(summed)
            count += n
    return total / count


def generated_predictions(model, rows, tokenizer, store, args, mode, handle=None):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            selected = rows[start:start + args.batch_size]
            batch = collate(selected, tokenizer, store, args.device, supervised=False)
            tokens = model.greedy_generate(**batch, max_new_tokens=args.max_new_tokens,
                                           eos_token_id=tokenizer.eos_token_id,
                                           pad_token_id=tokenizer.pad_token_id, message_mode=mode)
            if tokens.ndim != 2 or len(tokens) != len(selected):
                raise ValueError('greedy_generate must return only generated IDs, one row per sample')
            for row, ids in zip(selected, tokens):
                raw_ids = ids.detach().cpu().tolist()
                # Retain every emitted ID, including special terminators. The
                # parser sees ordinary decoded content, never hand-edited text.
                text = tokenizer.decode(raw_ids, skip_special_tokens=True)
                record = {'sample_id': row['sample_id'], 'group_id': row['group_id'],
                          'message_mode': mode, 'raw_generated_token_ids': raw_ids,
                          'raw_text': text, 'parse': parse_answer(text), 'gold_pkd': float(row['pkd']),
                          'target_sequence_sha256': row['groups']['target_sequence_sha256'],
                          'decoding': 'unconstrained greedy autoregressive generation'}
                result.append(record)
                if handle is not None:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                    handle.flush()
    return result


def train_and_evaluate(model, tokenizer, splits, store, args, registration, log):
    """Injectable core; fixture tests cannot be mistaken for real benchmark runs."""
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError('Connector contains no trainable parameters')
    assert_frozen(model.base_lm)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    best, best_epoch, history, steps = math.inf, None, [], 0
    checkpoint = args.output / 'best_connector.pt'
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = list(range(len(splits['train'])))
        random.Random(args.seed + epoch).shuffle(indices)
        train_sum, train_tokens = 0., 0
        for start in range(0, len(indices), args.batch_size):
            selected = [splits['train'][i] for i in indices[start:start + args.batch_size]]
            batch = collate(selected, tokenizer, store, args.device)
            optimizer.zero_grad(set_to_none=True)
            summed, count = supervised_nll(model(**batch), batch['labels'])
            (summed / count).backward()
            gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
            if not gradients or any(not torch.isfinite(value).all() for value in gradients):
                raise ValueError('Missing or nonfinite connector gradients')
            norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm, error_if_nonfinite=True)
            assert_frozen(model.base_lm)
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in trainable):
                raise ValueError('Nonfinite connector parameters after optimizer update')
            steps += 1
            train_sum += float(summed.detach())
            train_tokens += count
            log({'event': 'optimizer_step', 'epoch': epoch, 'step': steps,
                 'answer_token_nll': float(summed.detach()) / count, 'gradient_norm': float(norm)})
            if steps % getattr(args, 'checkpoint_every_steps', 100) == 0:
                save_checkpoint(args.output / 'latest_partial_connector.pt',
                                {'connector_state': connector_state(model), 'epoch': epoch,
                                 'optimizer_steps': steps, 'registration': registration,
                                 'selection_eligible': False,
                                 'purpose': 'interruption audit only; never selected without validation'})
        score = validation_nll(model, splits['validation'], tokenizer, store, args.device, args.batch_size)
        if not math.isfinite(score):
            raise ValueError('Nonfinite validation NLL')
        if score < best:
            best, best_epoch = score, epoch
            save_checkpoint(checkpoint, {'connector_state': connector_state(model), 'best_epoch': best_epoch,
                                        'validation_answer_token_nll': best, 'registration': registration,
                                        'normalization': 'none; raw numeric pKD text labels'})
        item = {'epoch': epoch, 'train_answer_token_nll': train_sum / train_tokens,
                'validation_answer_token_nll': score, 'best_epoch': best_epoch, 'optimizer_steps': steps}
        history.append(item)
        write_json(args.output / 'history.json', history)
        log({'event': 'epoch_complete', **item})
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    load_connector_state(model, saved['connector_state'])
    assert_frozen(model.base_lm)
    results = {}
    for mode in args.message_modes:
        path = args.output / f'test_predictions_{mode}.jsonl'
        with path.open('x', encoding='utf-8') as handle:
            predictions = generated_predictions(model, splits['test'], tokenizer, store, args, mode, handle)
        results[mode] = generation_metrics(splits['test'], predictions)
        log({'event': 'test_generation_complete', 'mode': mode, 'metrics': results[mode]})
    return {'status': 'complete', 'best_epoch': best_epoch, 'selection': 'lowest validation answer-token NLL; earliest tie',
            'best_validation_answer_token_nll': best, 'optimizer_steps': steps, 'actual_epochs': len(history),
            'training_and_generation_seconds': time.perf_counter() - started,
            'checkpoint_path': str(checkpoint), 'checkpoint_sha256': file_sha(checkpoint),
            'test_generation': results, 'backbone_gradients': 'none_verified',
            'synthetic_results_counted': False, 'formal_multitask_connector_result': False}


def process_record(args):
    versions = {}
    for name in ('torch', 'transformers', 'numpy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    record = {'pid': os.getpid(), 'parent_pid': os.getppid(), 'hostname': socket.gethostname(),
              'platform': platform.platform(), 'python': sys.version, 'executable': sys.executable,
              'started_utc': utc_now(), 'device': args.device, 'torch_threads': torch.get_num_threads(),
              'versions': versions, 'launch_argv': sys.argv,
              'process_group_id': os.getpgid(0) if hasattr(os, 'getpgid') else None,
              'thread_environment': {key: os.environ.get(key) for key in
                                     ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}}
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        record['gpu'] = {'name': torch.cuda.get_device_name(), 'cuda_version': torch.version.cuda,
                         'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                         'free_memory_bytes_at_start': torch.cuda.mem_get_info()[0]}
    return record


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    result.add_argument('--scenario', choices=SCENARIOS, required=True)
    result.add_argument('--method', choices=METHODS, required=True)
    result.add_argument('--output', type=Path, required=True, help='New directory; existing paths are refused')
    result.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    result.add_argument('--seed', type=int, default=17)
    result.add_argument('--epochs', type=int, default=3)
    result.add_argument('--train-limit', type=int)
    result.add_argument('--validation-limit', type=int)
    result.add_argument('--test-limit', type=int)
    result.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    result.add_argument('--batch-size', type=int, default=1)
    result.add_argument('--max-new-tokens', type=int, default=32)
    result.add_argument('--learning-rate', type=float, default=3e-4)
    result.add_argument('--weight-decay', type=float, default=.01)
    result.add_argument('--max-grad-norm', type=float, default=1.)
    result.add_argument('--threads', type=int, default=2)
    result.add_argument('--checkpoint-every-steps', type=int, default=100)
    result.add_argument('--memory-dim', type=int, default=128)
    result.add_argument('--num-memory-tokens', type=int, default=16)
    result.add_argument('--insertion-layer', type=int, default=14)
    result.add_argument('--message-modes', nargs='+', choices=('correct', 'off'), default=['correct'])
    return result


def main():
    args = parser().parse_args()
    if min(args.epochs, args.batch_size, args.threads, args.memory_dim, args.num_memory_tokens,
           args.checkpoint_every_steps) < 1:
        raise ValueError('Epochs, batches, threads, and connector capacities must be positive')
    if not 1 <= args.max_new_tokens <= 256 or args.learning_rate <= 0 or args.max_grad_norm <= 0 or args.weight_decay < 0:
        raise ValueError('Invalid decoding or optimizer configuration')
    if len(set(args.message_modes)) != len(args.message_modes):
        raise ValueError('Duplicate message modes')
    # New run only. Never resume or mutate a historical result/lock implicitly.
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    process = process_record(args)
    write_json(args.output / 'process.json', process)
    raw_log = (args.output / 'run.jsonl').open('x', encoding='utf-8', buffering=1)

    def log(value):
        entry = {'utc': utc_now(), **value}
        line = json.dumps(entry, ensure_ascii=False, allow_nan=False)
        raw_log.write(line + '\n')
        raw_log.flush()
        print(line, flush=True)

    try:
        path = args.data_root / args.scenario
        # An explicit override is recorded, but the obsolete A1 lock cannot
        # accidentally become the default scientific training input.
        if args.data_root.resolve() == (ROOT / 'data/processed/bench_v02/a1_locked').resolve():
            raise ValueError('Obsolete A1 lock refused; use independently audited a1_locked_v2')
        rows, gate = validate_panel_data(path / 'samples.jsonl', path / 'benchmark_lock.json', 'A1')
        lock = json.loads((path / 'benchmark_lock.json').read_text(encoding='utf-8'))
        if lock.get('scenario') != args.scenario:
            raise ValueError('Selected scenario differs from the physical lock')
        splits, split_manifest = select_splits(rows, {key: getattr(args, key + '_limit')
                                                     for key in ('train', 'validation', 'test')})
        model_receipt = verify_local_models()
        store = FeatureStore(args.cache)
        for split in splits.values():
            for row in split:
                store.residues(row['peptide_sequence'])
                store.residues(row['target_sequence'])
        snapshot = {'schema_version': 1, 'panel': 'A1', 'scenario': args.scenario,
                    'data_root': str(args.data_root.resolve()), 'data_gate': gate,
                    'split_selection': split_manifest,
                    'scope': 'explicit_subset_pilot' if any(value['selected_rows'] < value['available_rows']
                                                           for value in split_manifest.values()) else 'full_locked_A1_panel',
                    'subset_rule': 'first N rows in immutable source order; epoch order seeded independently',
                    'feature_manifest_sha256': file_sha(store.manifest_path),
                    'feature_receipts': {'proteins': dict(store.checked_proteins)},
                    'model_receipt': model_receipt,
                    'script_sha256': {str(script): file_sha(script) for script in
                                      (Path(__file__), Path(__file__).with_name('connectors.py'),
                                       ROOT / 'scripts/bench_v02/train_panel_baselines.py',
                                       ROOT / 'scripts/bench_v02/panel_execution.py',
                                       ROOT / 'scripts/bench_v02/metrics.py')},
                    'label_normalization': 'none; no scaler fitted',
                    'input_rule': 'all cached ESM30 peptide and full-target residues; anonymous allowlisted context; no sequence text',
                    'prompt_context_fields': context_view(splits['train'][0]),
                    'answer_serialization': 'one JSON numeric pKD field rounded to two decimals plus EOS; original precision retained for scoring'}
        train_targets = {row['target_sequence'] for row in splits['train']}
        snapshot['selected_known_target_coverage'] = {
            split: {'n': len(splits[split]),
                    'n_target_sequence_seen_in_selected_train': sum(row['target_sequence'] in train_targets
                                                                    for row in splits[split])}
            for split in ('validation', 'test')}
        write_json(args.output / 'input_manifest.json', snapshot)
        # An audit copy freezes row membership without duplicating source PDFs,
        # repository assets or bulky frozen ESM tensors.
        for split, selected in splits.items():
            with (args.output / f'{split}_input_snapshot.jsonl').open('x', encoding='utf-8') as handle:
                for row in selected:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_path = ROOT / 'models/Qwen--Qwen2.5-1.5B-Instruct'
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                                   dtype=torch.bfloat16 if args.device == 'cuda' else torch.float32,
                                                   attn_implementation='sdpa').to(args.device)
        base.requires_grad_(False).eval()
        if getattr(base, 'is_gradient_checkpointing', False):
            raise ValueError('Internal hook training requires backbone gradient checkpointing disabled')
        model = FrozenConnectorLM(base, args.method, memory_dim=args.memory_dim,
                                  num_memory_tokens=args.num_memory_tokens,
                                  insertion_layer=args.insertion_layer).to(args.device)
        max_text_tokens = max(len(tokenizer.encode(prompt_text(row, tokenizer), add_special_tokens=False))
                              + len(tokenizer.encode(answer_text(row), add_special_tokens=False)) + 1
                              for split in ('train', 'validation') for row in splits[split])
        max_prompt_tokens = max(len(tokenizer.encode(prompt_text(row, tokenizer), add_special_tokens=False))
                                for selected in splits.values() for row in selected)
        positional_budget = max(max_text_tokens, max_prompt_tokens + args.max_new_tokens) + (
            args.num_memory_tokens if args.method != 'internal_cross_attention' else 0)
        if positional_budget > base.config.max_position_embeddings:
            raise ValueError('Prompt plus memory and generation exceed context capacity; no silent truncation')
        if any(not name.startswith('base_lm.') and not parameter.requires_grad
               for name, parameter in model.named_parameters()):
            log({'event': 'non_backbone_frozen_parameters_present'})
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        registration = {'schema_version': 1, 'benchmark_version': 'foundation_20261016',
                        'panel': 'A1', 'role': 'frozen_lm_connector_generation',
                        'configuration': config, 'capacity': model.capacity_report(),
                        'input_manifest_sha256': file_sha(args.output / 'input_manifest.json'),
                        'foundation_revisions': {'ESM2': ESM_REVISION, 'Qwen': QWEN_REVISION},
                        'selection': 'lowest validation answer-token NLL only; earliest tie',
                        'test_label_usage': 'evaluation after checkpoint selection only',
                        'test_input_snapshots_sha256': {split: file_sha(args.output / f'{split}_input_snapshot.jsonl')
                                                       for split in splits},
                        'train_order': 'Python Random(seed + epoch).shuffle immutable source indices',
                        'normalization': 'none', 'scope': snapshot['scope'],
                        'max_supervised_text_tokens': max_text_tokens,
                        'conservative_position_budget': positional_budget,
                        'selected_known_target_coverage': snapshot['selected_known_target_coverage'],
                        'matched_allowed_context': True,
                        'context_representation_difference': 'Same allowlisted chemical_state, assay/buffer, target_sequence_role, pH/temperature, context missingness and actual_construct_verified inputs. Existing predictor hashes strings and uses train-only numeric standardization; LM receives readable raw values.',
                        'formal_multitask_connector_result': False}
        write_json(args.output / 'run_preregistration.json', registration)
        log({'event': 'training_start', 'scope': snapshot['scope'], 'capacity': registration['capacity']})
        outcome = train_and_evaluate(model, tokenizer, splits, store, args, registration, log)
        # Source assets remain immutable throughout this run; verify them again.
        if file_sha(path / 'samples.jsonl') != gate['samples_sha256'] or file_sha(path / 'benchmark_lock.json') != gate['benchmark_lock_sha256']:
            raise ValueError('Immutable source data changed while the run was executing')
        for script, fingerprint in snapshot['script_sha256'].items():
            if file_sha(script) != fingerprint:
                raise ValueError(f'Executing script changed mid-run: {script}')
        outcome.update({'registration': registration, 'finished_utc': utc_now(),
                        'process': process, 'data_gate': gate,
                        'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0,
                        'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved() if args.device == 'cuda' else 0})
        log({'event': 'run_complete', 'best_epoch': outcome['best_epoch']})
        raw_log.close()
        outcome['artifact_sha256'] = {item.name: file_sha(item) for item in args.output.iterdir() if item.is_file()}
        write_json(args.output / 'result.json', outcome)
    except BaseException as error:
        log({'event': 'run_failed', 'error_type': type(error).__name__, 'error': str(error)})
        write_json(args.output / 'failure.json', {'status': 'failed', 'utc': utc_now(),
                                                'error_type': type(error).__name__, 'error': str(error),
                                                'partial_artifacts_preserved': True})
        raise
    finally:
        raw_log.close()


if __name__ == '__main__':
    main()
