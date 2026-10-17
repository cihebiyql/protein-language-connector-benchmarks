"""Real multitask fits, test-blind checkpoint selection, and separate readout/LM evidence.

Only this new scope is written. The canary loads TRAIN exclusively; formal evaluation
is opened after a selected CPU checkpoint and its freeze receipt have been committed.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F

SCOPE_ID = 'multitask_connector_v5_20261017'
TASK_KEYS = {'meltome': 'melting_temperature_c', 'rt': 'retention_time_min', 'elm': 'function'}
REVISIONS = {'qwen25_1p5b': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
             'qwen3_4b_2507': 'cdbee75f17c01a7cc42f958dc650907174af0554'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest_json(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def write_jsonl(path, rows):
    Path(path).write_text(''.join(canonical(row) + '\n' for row in rows), encoding='utf-8')


def artifact(path, root):
    return {'path': Path(path).relative_to(root).as_posix(), 'sha256': file_sha(path)}


def plain(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def configure(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return {'seed': seed, 'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
            'cpu_threads': torch.get_num_threads(), 'deterministic_algorithms': True, 'tf32': False,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}


@dataclass(frozen=True)
class NumericStats:
    mean: float
    std: float
    count: int
    source_split: str = 'train'

    @classmethod
    def fit(cls, rows):
        if not rows or any(row['split'] != 'train' for row in rows):
            raise ValueError('numeric standardization requires TRAIN only')
        values = [row['target'] for row in rows]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ValueError('finite original TRAIN targets required')
        mean = math.fsum(values) / len(values)
        std = max(math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values)), 1e-8)
        return cls(mean, std, len(values))

    def targets(self, rows, device):
        return torch.tensor([(row['target'] - self.mean) / self.std for row in rows], dtype=torch.float32, device=device)

    def inverse(self, value):
        return float(value) * self.std + self.mean


def batch_dimensions(training):
    dimensions = tuple(training.get(name, default) for name, default in
                       (('micro_batch_size', 1), ('grad_accumulation', 4), ('effective_batch_size', 4)))
    if any(type(value) is not int or value < 1 for value in dimensions) or dimensions[0] * dimensions[1] != dimensions[2]:
        raise ValueError('micro batch times accumulation must equal effective batch')
    return dimensions


def check_protocol(config, task, method, backbone, seed, phase, max_updates):
    if task not in TASK_KEYS or method not in ('ESM', 'M0', 'M3') or seed not in (17, 42, 2026):
        raise ValueError('outside the fixed task/method/seed matrix')
    if method != 'ESM' and backbone not in REVISIONS:
        raise ValueError('unrecognized backbone')
    training = config['training']
    signature = digest_json(training)
    if phase == 'formal' and (config.get('status') != 'frozen' or config.get('training_signature') != signature):
        raise ValueError('formal fits require frozen config and exact training signature')
    if phase == 'train-canary' and max_updates != 4:
        raise ValueError('TRAIN canary must use exactly four updates')
    if phase == 'formal' and max_updates is not None:
        raise ValueError('formal fits cannot be shortened with max-updates')
    if batch_dimensions(training) != (1, 4, 4):
        raise ValueError('this scope fixes micro1 accumulation4 effective4')
    if type(training['epochs']) is not int or training['epochs'] < 1:
        raise ValueError('positive complete epoch budget required')
    for key, expected in (('lr', .0003), ('weight_decay', .01), ('max_grad_norm', 1.)):
        if float(training.get(key, expected)) != expected:
            raise ValueError('prospective optimizer contract differs: ' + key)
    return signature


def prompt_text(row, tokenizer):
    task = row['task']
    instruction = {
        'meltome': 'Predict melting temperature in degrees Celsius from the supplied residue memory.',
        'rt': 'Predict chromatographic retention time in minutes from the supplied residue memory.',
        'elm': 'Describe the molecular function represented by the supplied residue memory.'}[task]
    requested = 'one finite number' if task != 'elm' else 'one complete function description string'
    # The input envelope is deliberately constructed rather than serializing source rows.
    user = (instruction + ' Return exactly one JSON object with only the field "' + TASK_KEYS[task]
            + '" containing ' + requested + '. Do not include explanations.\nSafe experimental conditions: '
            + canonical(row['condition_values']))
    kwargs = {'enable_thinking': False} if getattr(tokenizer, '_mt_backbone_id', '') == 'qwen3_4b_2507' else {}
    return tokenizer.apply_chat_template([
        {'role': 'system', 'content': 'Use the supplied residue and condition memories to answer the requested molecular task.'},
        {'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True, **kwargs)


def answer_text(row, catalog):
    if row['split'] != 'train':
        raise ValueError('supervised answers can be built from TRAIN only')
    if row['task'] != 'elm':
        value = row['target']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('finite TRAIN label required')
        return '{"' + TASK_KEYS[row['task']] + '":' + format(value, '.4f') + '}'
    positives = set(row['positive_ids'])
    selected = [entry for entry in catalog if entry['candidate_id'] in positives]
    if not selected or len(selected) != len(positives):
        raise ValueError('every known positive must occur in the frozen catalog')
    # Catalog-order first positive is fixed across seeds/methods/backbones; no class ID enters tokens.
    return canonical({'function': selected[0]['description']})


class Collator:
    def __init__(self, store, tokenizer=None, device='cpu', catalog=None):
        self.store, self.tokenizer, self.device = store, tokenizer, device
        self.catalog = catalog or []

    def __call__(self, rows, supervised=False):
        from .mt_data import condition_tensors_values
        if not rows or (supervised and any(row['split'] != 'train' for row in rows)):
            raise ValueError('empty batch or attempted non-TRAIN supervision')
        memories, conditions, observations = {'p15': [], 'p30': []}, [], []
        for row in rows:
            feature = self.store.get(row['sequence'])
            for key in memories:
                value = torch.as_tensor(feature[key], dtype=torch.float32)
                if value.shape != (8, 640) or not torch.isfinite(value).all():
                    raise ValueError('invalid ESM pooled cache')
                memories[key].append(value)
            ids, observed = condition_tensors_values(row)
            conditions.append(ids)
            observations.append(observed)
        memory_inputs = {key: torch.stack(value).to(self.device) for key, value in memories.items()}
        memory_inputs.update(condition_ids=torch.tensor(conditions, dtype=torch.long, device=self.device),
                             observed=torch.tensor(observations, dtype=torch.bool, device=self.device))
        if self.tokenizer is None:
            return {'memory_inputs': memory_inputs}
        prefixes = [self.tokenizer.encode(prompt_text(row, self.tokenizer), add_special_tokens=False) for row in rows]
        suffixes = [self.tokenizer.encode(answer_text(row, self.catalog), add_special_tokens=False)
                    + [self.tokenizer.eos_token_id] if supervised else [] for row in rows]
        if any(not prefix for prefix in prefixes):
            raise ValueError('empty prompt tokenization')
        width = max(len(p) + len(s) for p, s in zip(prefixes, suffixes))
        ids = torch.full((len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        mask, labels, ends = torch.zeros_like(ids), torch.full_like(ids, -100), []
        for index, (prefix, suffix) in enumerate(zip(prefixes, suffixes)):
            offset = width - len(prefix) - len(suffix)
            ids[index, offset:] = torch.tensor(prefix + suffix, device=self.device)
            mask[index, offset:] = 1
            ends.append(offset + len(prefix) - 1)
            if supervised:
                labels[index, offset + len(prefix):] = torch.tensor(suffix, device=self.device)
        result = {'input_ids': ids, 'attention_mask': mask,
                  'prompt_end_index': torch.tensor(ends, dtype=torch.long, device=self.device), 'memory_inputs': memory_inputs}
        if supervised:
            result['labels'] = labels
        return result


def supervised_ce(logits, labels):
    shifted = labels[:, 1:].contiguous()
    count = int((shifted != -100).sum().item())
    if not count:
        raise ValueError('answer contains no supervised token')
    loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), shifted.reshape(-1),
                           ignore_index=-100, reduction='sum')
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite answer token CE')
    return loss, count


def ranking_loss_sum(logits, rows, candidate_ids):
    if logits.shape != (len(rows), len(candidate_ids)) or len(candidate_ids) != 31:
        raise ValueError('ranking must score all 31 semantic descriptions')
    if not torch.isfinite(logits).all():
        raise FloatingPointError('nonfinite ranking logits')
    index = {candidate: position for position, candidate in enumerate(candidate_ids)}
    losses = []
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    for row, log_probs in zip(rows, log_probabilities):
        positives = row['positive_ids']
        if not positives or len(set(positives)) != len(positives) or any(value not in index for value in positives):
            raise ValueError('known positives must be unique catalog members')
        losses.append(-log_probs[[index[value] for value in positives]].mean())
    return torch.stack(losses).sum()


@contextmanager
def inference(model):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


def forward(model, batch, off=False):
    inputs = {key: value for key, value in batch.items() if key != 'labels'}
    if off:
        inputs['off'] = True
    return model(**inputs)


def readout_predictions(model, rows, collator, task, stats, candidate_ids, batch_size=8, off=False):
    predictions = []
    with inference(model):
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            result = forward(model, collator(selected, supervised=False), off=off)
            if task == 'elm':
                scores = result['ranking_logits'].detach().float().cpu()
                if scores.shape != (len(selected), 31) or not torch.isfinite(scores).all():
                    raise FloatingPointError('invalid full-catalog readout')
                for row, values in zip(selected, scores.tolist()):
                    ranked = sorted(range(len(candidate_ids)), key=lambda i: (-values[i], i))
                    predictions.append({'sample_id': row['sample_id'], 'ranking': [candidate_ids[i] for i in ranked],
                                        'candidate_scores': values})
            else:
                values = result['numeric_z'].detach().float().cpu().reshape(-1).tolist()
                if len(values) != len(selected) or not all(map(math.isfinite, values)):
                    raise FloatingPointError('nonfinite numeric readout')
                predictions.extend({'sample_id': row['sample_id'], 'prediction': stats.inverse(value)}
                                   for row, value in zip(selected, values))
    return predictions


def selection_value(task, rows, predictions, candidate_ids):
    from .mt_metrics import score_regression, score_ranking
    expected = 'selection' if task == 'rt' else 'validation'
    if not rows or any(row['split'] != expected for row in rows):
        raise ValueError('wrong checkpoint selection split')
    metric = score_ranking(rows, predictions, candidate_ids) if task == 'elm' else score_regression(rows, predictions)
    name = {'meltome': 'spearman', 'elm': 'class_macro_mrr', 'rt': 'mae'}[task]
    value = metric[name]
    if value is None or not math.isfinite(value):
        raise ValueError('checkpoint selection metric is undefined')
    return {'name': name, 'value': float(value), 'maximize': task != 'rt', 'rows': len(rows),
            'split': expected, 'ordered_ids_sha256': digest_json([row['sample_id'] for row in rows]), 'metrics': metric}


def is_better(current, best):
    return best is None or (current['value'] > best['value'] if current['maximize'] else current['value'] < best['value'])


def save_checkpoint(path, model, stats, identity, epoch):
    state = {key: value.detach().cpu().clone() for key, value in model.trainable_state_dict().items()}
    if not state or any(key.startswith('base_lm.') for key in state):
        raise ValueError('checkpoint must contain trainable state only')
    if any(value.is_floating_point() and value.dtype != torch.float32 for value in state.values()):
        raise ValueError('trainable checkpoint must be FP32')
    payload = {'schema': 'mt-checkpoint.v1', 'identity': identity, 'epoch': epoch,
               'trainable_state': state, 'numeric_stats': asdict(stats) if stats is not None else None}
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def fit_model(model, train, validation, collator, config, identity, output, task, candidate_ids,
              max_updates=None, progress=None):
    if not train or any(row['split'] != 'train' for row in train):
        raise ValueError('optimizer consumes TRAIN only')
    canary = max_updates is not None
    if canary and validation is not None:
        raise ValueError('canary must not receive validation')
    if not canary and validation is None:
        raise ValueError('formal training requires its frozen selection split')
    training = config['training']
    micro, accumulation, effective = batch_dimensions(training)
    stats = None if task == 'elm' else NumericStats.fit(train)
    if stats is not None:
        write_json(output / 'numeric_stats.json', asdict(stats))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters or any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError('trainable parameters must be nonempty FP32')
    optimizer = torch.optim.AdamW(parameters, lr=training.get('lr', .0003), weight_decay=training.get('weight_decay', .01))
    history, updates, examples, best = [], 0, 0, None
    first_update = None
    started = time.perf_counter()
    model.train()
    first_indices = None
    if canary and getattr(collator, 'tokenizer', None) is not None:
        tokenizer = collator.tokenizer
        lengths = [(len(tokenizer.encode(prompt_text(row, tokenizer), add_special_tokens=False))
                    + len(tokenizer.encode(answer_text(row, collator.catalog), add_special_tokens=False)) + 1, index)
                   for index, row in enumerate(train)]
        first_indices = [index for _, index in sorted(lengths, key=lambda pair: (-pair[0], pair[1]))[:effective]]
        write_json(output / 'train_length_qualification.json', {'source_split': 'train', 'all_train_rows_inspected': len(train),
                    'max_supervised_tokens': max(pair[0] for pair in lengths), 'lengths_sha256': digest_json(lengths),
                    'first_effective_batch_sample_ids': [train[index]['sample_id'] for index in first_indices],
                    'validation_or_test_opened': False})
    epochs = max(training['epochs'], math.ceil(max_updates * effective / len(train))) if canary else training['epochs']
    for epoch in range(1, epochs + 1):
        order = list(range(len(train)))
        random.Random(identity['seed'] + epoch).shuffle(order)
        if epoch == 1 and first_indices is not None:
            order = first_indices + [index for index in order if index not in first_indices]
        head_total, ce_total, ce_tokens, epoch_examples = 0., 0., 0, 0
        for start in range(0, len(order), effective):
            group = [train[i] for i in order[start:start + effective]]
            # Ensure 4 x 4 canary examples even for tiny injected TRAIN fixtures.
            if canary and len(group) < effective:
                group += [train[order[i % len(order)]] for i in range(effective - len(group))]
            parts = [group[i:i + micro] for i in range(0, len(group), micro)]
            is_lm = identity['method'] != 'ESM'
            group_tokens = 0
            if is_lm:
                for part in parts:
                    counted = collator(part, supervised=True)
                    group_tokens += int((counted['labels'][:, 1:] != -100).sum().item())
                if group_tokens == 0:
                    raise ValueError('empty answer-token group')
            optimizer.zero_grad(set_to_none=True)
            for part in parts:
                batch = collator(part, supervised=is_lm)
                result = forward(model, batch)
                if task == 'elm':
                    head_sum = ranking_loss_sum(result['ranking_logits'], part, candidate_ids)
                else:
                    predicted = result['numeric_z'].float().reshape(-1)
                    if predicted.shape != (len(part),):
                        raise ValueError('numeric readout batch shape mismatch')
                    head_sum = F.mse_loss(predicted, stats.targets(part, predicted.device), reduction='sum')
                if not torch.isfinite(head_sum):
                    raise FloatingPointError('nonfinite readout objective')
                loss = head_sum / len(group)
                head_total += float(head_sum.detach().item())
                if is_lm:
                    ce, count = supervised_ce(result['logits'], batch['labels'])
                    loss = loss + ce / group_tokens
                    ce_total += float(ce.detach().item())
                    ce_tokens += count
                loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in parameters):
                raise FloatingPointError('nonfinite gradient')
            norm = torch.nn.utils.clip_grad_norm_(parameters, training.get('max_grad_norm', 1.))
            optimizer.step()
            updates += 1
            examples += len(group)
            epoch_examples += len(group)
            elapsed = time.perf_counter() - started
            if first_update is None:
                first_update = elapsed
            event = {'event': 'optimizer_step', 'epoch': epoch, 'optimizer_step': updates, 'optimizer_steps': updates,
                     'examples_seen': examples, 'elapsed_seconds': elapsed, 'gradient_norm': float(norm)}
            if progress:
                progress(event)
            if canary and updates == max_updates:
                break
        record = {'epoch': epoch, 'train_examples': epoch_examples, 'train_head_mean': head_total / epoch_examples,
                  'train_answer_token_ce': ce_total / ce_tokens if ce_tokens else None, 'answer_tokens': ce_tokens,
                  'optimizer_steps': updates, 'elapsed_seconds': time.perf_counter() - started}
        if not canary:
            predictions = readout_predictions(model, validation, collator, task, stats, candidate_ids,
                                               training.get('readout_batch_size', 8))
            current = selection_value(task, validation, predictions, candidate_ids)
            record['selection'] = current
            if is_better(current, best):
                best = {**current, 'epoch': epoch}
                save_checkpoint(output / 'checkpoint.pt', model, stats, identity, epoch)
                write_json(output / 'selection_predictions.json', predictions)
            write_json(output / 'selection.json', {**best, 'earliest_exact_tie': True,
                       'test_labels_used_for_selection': False, 'calibration_labels_used_for_selection': False,
                       'written_before_test': True, 'checkpoint_sha256': file_sha(output / 'checkpoint.pt')})
        history.append(record)
        write_json(output / 'training_history.json', history)
        if progress:
            progress({'event': 'epoch_completed' if not canary else 'canary_progress', **record})
        if canary and updates == max_updates:
            save_checkpoint(output / 'checkpoint.pt', model, stats, identity, epoch)
            break
    if canary:
        if updates != 4 or examples != 16:
            raise ValueError('canary must finish four updates and sixteen examples')
        write_json(output / 'selection.json', {'selection': 'none_TRAIN_only_canary', 'written_before_test': True,
                   'test_labels_used_for_selection': False, 'checkpoint_sha256': file_sha(output / 'checkpoint.pt')})
    return {'epochs_completed': 0 if canary else training['epochs'], 'optimizer_steps': updates,
            'examples_seen': examples, 'elapsed_seconds': time.perf_counter() - started,
            'first_update_seconds': first_update}, stats


def language_predictions(model, rows, collator, tokenizer, task, batch_size=1, progress=None):
    predictions = []
    limit = 256 if task == 'elm' else 24
    with inference(model):
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            batch = collator(selected, supervised=False)
            sequences = [[] for _ in selected]
            done = torch.zeros(len(selected), dtype=torch.bool, device=batch['input_ids'].device)
            for _ in range(limit):
                output = forward(model, batch)
                last = output['logits'][:, -1].float()
                if not torch.isfinite(last).all():
                    raise FloatingPointError('nonfinite LM generation logits')
                tokens = last.argmax(dim=-1)
                active = ~done
                tokens = torch.where(active, tokens, torch.full_like(tokens, tokenizer.pad_token_id))
                for i, token in enumerate(tokens.tolist()):
                    if active[i].item() and token != tokenizer.eos_token_id:
                        sequences[i].append(token)
                batch['input_ids'] = torch.cat((batch['input_ids'], tokens[:, None]), dim=1)
                batch['attention_mask'] = torch.cat((batch['attention_mask'], active.long()[:, None]), dim=1)
                done = done | (tokens == tokenizer.eos_token_id)
                if done.all():
                    break
            predictions.extend({'sample_id': row['sample_id'], 'text': tokenizer.decode(tokens, skip_special_tokens=True),
                                'generated_tokens': len(tokens), 'max_new_tokens': limit}
                               for row, tokens in zip(selected, sequences))
            if progress:
                progress({'event': 'generation_progress', 'completed_rows': len(predictions), 'total_rows': len(rows)})
    return predictions


def prompt_anchor_check(model, rows, collator, task):
    """Perturb only future token values at fixed shape, retaining the strict tolerance.

    BF16 matrix multiplication can choose different arithmetic for T and T+2.
    That shape-sensitivity is retained as a diagnostic, rather than mistaken for
    future-token access. Both causal comparisons use the identical T+2 layout.
    """
    original = collator(rows, supervised=False)
    key = 'ranking_logits' if task == 'elm' else 'numeric_z'
    futures = []
    for token_id in (1, 2):
        suffix = torch.full((len(rows), 2), token_id, dtype=torch.long, device=original['input_ids'].device)
        futures.append({**original, 'input_ids': torch.cat((original['input_ids'], suffix), dim=1),
                        'attention_mask': torch.cat((original['attention_mask'], torch.ones_like(suffix)), dim=1)})
    def anchors(output):
        routes = output.get('routes')
        if not isinstance(routes, dict) or set(routes) != {'view', 'bank'}:
            raise ValueError('causality qualification requires both prompt-conditioned route anchors')
        values = {key: output[key], 'prompt_hidden': output['prompt_hidden'],
                  'routes.view': routes['view'], 'routes.bank': routes['bank']}
        if any(not torch.is_tensor(value) or not torch.isfinite(value).all() for value in values.values()):
            raise ValueError('nonfinite or missing prompt-anchor qualification tensor')
        return {name: value.detach().float() for name, value in values.items()}
    with inference(model):
        before = anchors(forward(model, original))
        left, right = [anchors(forward(model, batch)) for batch in futures]
    components = {}
    for name in left:
        if left[name].shape != right[name].shape or before[name].shape != left[name].shape:
            raise ValueError('prompt-anchor tensor shape changed')
        components[name] = {'passed': bool(torch.allclose(left[name], right[name], atol=1e-5, rtol=1e-5)),
                            'max_abs_difference': float((left[name] - right[name]).abs().max().item())}
    return {'passed': all(value['passed'] for value in components.values()),
            'max_abs_difference': max(value['max_abs_difference'] for value in components.values()),
            'comparison': 'same_shape_different_future_token_values', 'components': components,
            'atol': 1e-5, 'rtol': 1e-5, 'source_split': 'train', 'future_tokens': 2,
            'synthetic_future_token_ids': [1, 2],
            'append_shape_diagnostic': {'used_for_pass': False, 'max_abs_difference': {
                name: float((before[name] - left[name]).abs().max().item()) for name in before}}}


def load_candidate_vectors(cache_root, catalog):
    path = Path(cache_root) / 'description_vectors.npz'
    with np.load(path, allow_pickle=False) as values:
        ids, vectors = values['candidate_ids'].tolist(), values['vectors'].copy()
    if ids != [entry['candidate_id'] for entry in catalog] or vectors.shape != (31, 1536) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError('frozen description vectors/catalog differ')
    return torch.from_numpy(vectors), {'sha256': file_sha(path), 'candidate_ids_sha256': digest_json(ids)}


def verify_model_assets(settings):
    root = Path(settings['backbone_path']).resolve(strict=True)
    manifest_path = Path(settings['model_manifest_path'])
    records = json.loads(manifest_path.read_text(encoding='utf-8'))
    if isinstance(records, dict):
        records = records.get('files', [])
    selected = [row for row in records if row.get('repo') == settings['repo']]
    if not selected or not any(row['file'].endswith('.safetensors') for row in selected):
        raise ValueError('official model manifest has no pinned weights')
    seen = set()
    for row in selected:
        path = (root / row['file']).resolve(strict=True)
        if row['file'] in seen or not path.is_relative_to(root):
            raise ValueError('duplicate or escaping model asset')
        seen.add(row['file'])
        if row['revision'] != settings['revision'] or path.stat().st_size != row['bytes'] or file_sha(path) != row['sha256']:
            raise ValueError('official frozen model asset mismatch: ' + row['file'])
    return {'manifest_sha256': file_sha(manifest_path), 'repo': settings['repo'], 'revision': settings['revision'],
            'verified_files': len(selected)}


def load_components(config, task, method, backbone, device, candidate_vectors):
    from .mt_model import ESMTaskModel, SingleMemoryLM
    settings = config['model']
    if method == 'ESM':
        return ESMTaskModel(task, esm_dim=640, memory_dim=settings.get('memory_dim', 128),
                            candidate_vectors=candidate_vectors).to(device), None, None
    from transformers import AutoModelForCausalLM, AutoTokenizer
    selected = config['backbones'][backbone]
    if selected['revision'] != REVISIONS[backbone]:
        raise ValueError('backbone revision differs from fixed comparison')
    asset = verify_model_assets(selected)
    tokenizer = AutoTokenizer.from_pretrained(selected['backbone_path'], local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer._mt_backbone_id = backbone
    base = AutoModelForCausalLM.from_pretrained(selected['backbone_path'], local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.bfloat16 if torch.device(device).type == 'cuda' else torch.float32,
            attn_implementation='eager').to(device)
    base.requires_grad_(False).eval()
    if base.config.hidden_size != selected['hidden_size'] or base.config.num_hidden_layers != selected['num_hidden_layers']:
        raise ValueError('loaded backbone dimensions differ')
    model = SingleMemoryLM(base, task=task, variant=method, hidden_dim=selected['hidden_size'], esm_dim=640,
            memory_dim=settings.get('memory_dim', 128), num_heads=settings.get('num_heads', 4),
            hook_layer=selected['hook_layer'], candidate_vectors=candidate_vectors).to(device)
    return model, tokenizer, asset


def model_policy(model, device):
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(parameter.dtype != torch.float32 for _, parameter in trainable):
        raise ValueError('trainable tensors must be FP32')
    if hasattr(model, 'base_lm'):
        base = model.base_lm
        expected = torch.bfloat16 if torch.device(device).type == 'cuda' else torch.float32
        if base.training or any(parameter.requires_grad or parameter.dtype != expected for parameter in base.parameters()):
            raise ValueError('LM must be frozen eval BF16 on CUDA / FP32 on CPU')
    return {'trainable_parameters': sum(parameter.numel() for _, parameter in trainable),
            'active_tensor_names': [name for name, _ in trainable], 'trainable_dtype': 'torch.float32',
            'capacity_report': plain(model.capacity_report())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'task', 'method', 'backbone', 'seed', 'data-root', 'cache-root', 'output', 'device'):
        parser.add_argument('--' + name, required=True, type=int if name == 'seed' else str)
    parser.add_argument('--phase', choices=('train-canary', 'formal'), required=True)
    parser.add_argument('--max-updates', type=int)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started, manifest = time.perf_counter(), None
    try:
        from .mt_data import load_task, load_catalog, load_manifest
        from .mt_extract import FeatureStore
        from .mt_metrics import build_metrics, calibration_radius, score_intervals
        config = json.loads(Path(args.config).read_text(encoding='utf-8'))
        signature = check_protocol(config, args.task, args.method, args.backbone, args.seed, args.phase, args.max_updates)
        runtime = configure(args.seed)
        if torch.device(args.device).type == 'cuda':
            torch.cuda.set_device(torch.device(args.device))
            cap = config.get('numerical_policy', {}).get('gpu_allocator_gib', {}).get(args.backbone)
            if cap is not None:
                total = torch.cuda.get_device_properties(torch.device(args.device)).total_memory
                torch.cuda.set_per_process_memory_fraction(min(1., float(cap) * 1024**3 / total), torch.device(args.device))
            torch.cuda.reset_peak_memory_stats()
        code_pins = {path.name: file_sha(path) for path in sorted(Path(__file__).parent.glob('*.py'))}
        identity = {'scope_id': SCOPE_ID, 'task': args.task, 'method': args.method, 'backbone': args.backbone,
                    'seed': args.seed, 'config_sha256': file_sha(args.config), 'training_signature': signature,
                    'code_sha256': code_pins}
        manifest = {'schema': 'mt-run.v1', **identity, 'phase': args.phase, 'status': 'partial',
                    'training': {}, 'test_labels_used_for_fit': False, 'artifacts': {}, 'runtime': runtime,
                    'created_utc': datetime.now(timezone.utc).isoformat(), 'metrics': {},
                    'data_manifest_sha256': file_sha(Path(args.data_root) / 'manifest.json')}
        # Reading the manifest does not open any split labels. Feature verification below is split-local.
        load_manifest(args.data_root)
        store = FeatureStore(args.cache_root)
        cache_manifest = Path(args.cache_root) / 'manifest.json'
        if cache_manifest.exists():
            manifest['cache_manifest_sha256'] = file_sha(cache_manifest)
        elif args.phase == 'train-canary':
            manifest['cache_manifest_sha256'] = file_sha(Path(args.cache_root) / 'cache_spec.json')
            manifest['cache_manifest_kind'] = 'partial_cache_spec_TRAIN_only_canary'
        else:
            raise ValueError('complete feature cache manifest required for formal fits')
        train = load_task(args.data_root, args.task, 'train')
        validation = load_task(args.data_root, args.task, 'selection' if args.task == 'rt' else 'validation') if args.phase == 'formal' else None
        manifest['train_cache_coverage'] = plain(store.verify_all(train))
        if validation is not None:
            manifest['selection_cache_coverage'] = plain(store.verify_all(validation))
        catalog = load_catalog(args.data_root) if args.task == 'elm' else []
        candidate_ids = [entry['candidate_id'] for entry in catalog]
        vectors, vector_pin = load_candidate_vectors(args.cache_root, catalog) if args.task == 'elm' else (None, None)
        if vector_pin:
            manifest['candidate_vectors_sha256'] = vector_pin['sha256']
        model, tokenizer, assets = load_components(config, args.task, args.method, args.backbone, args.device, vectors)
        manifest['model_policy'] = model_policy(model, args.device)
        manifest['model_assets'] = assets
        collator = Collator(store, tokenizer, args.device, catalog)
        write_json(output / 'protocol.json', config)
        write_json(output / 'run_manifest.json', manifest)
        def progress(value):
            with (output / 'progress.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(canonical(value) + '\n')
            print(canonical(value), flush=True)
        training, stats = fit_model(model, train, validation, collator, config, identity, output,
                                    args.task, candidate_ids, max_updates=args.max_updates, progress=progress)
        manifest['training'] = training
        for key, filename in (('checkpoint', 'checkpoint.pt'), ('history', 'training_history.json'), ('selection', 'selection.json')):
            manifest['artifacts'][key] = artifact(output / filename, output)
        selected = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=True)
        if selected['identity'] != identity:
            raise ValueError('checkpoint identity mismatch')
        model.load_trainable_state_dict(selected['trainable_state'])
        selected_parameters = {key: value.detach().cpu().clone() for key, value in model.trainable_state_dict().items()}
        if args.phase == 'train-canary':
            if tokenizer is not None:
                anchor = prompt_anchor_check(model, train[:1], collator, args.task)
                write_json(output / 'train_anchor_qualification.json', anchor)
                if not anchor['passed']:
                    raise ValueError('prompt-anchor readout depends on future tokens')
            probe = train[:8]
            predictions = readout_predictions(model, probe, collator, args.task, stats, candidate_ids,
                                              config['training'].get('readout_batch_size', 8))
            write_jsonl(output / 'train_readout_predictions.jsonl', predictions)
            if tokenizer is not None:
                generation_started = time.perf_counter()
                generated = language_predictions(model, probe, collator, tokenizer, args.task,
                              config['training'].get('generation_batch_size', 8), progress)
                write_jsonl(output / 'train_LM_predictions.jsonl', generated)
                manifest['runtime']['canary_generation_seconds'] = time.perf_counter() - generation_started
                manifest['artifacts']['train_LM_predictions'] = artifact(output / 'train_LM_predictions.jsonl', output)
            manifest['canary_qualification'] = {'train_rows_scored': len(probe), 'validation_opened': False,
                    'calibration_opened': False, 'test_opened': False, 'development_only': True}
        else:
            freeze = {'schema': 'mt-test-blind-freeze.v1', 'identity': identity,
                      'checkpoint_sha256': file_sha(output / 'checkpoint.pt'),
                      'selection_sha256': file_sha(output / 'selection.json'),
                      'frozen_utc': datetime.now(timezone.utc).isoformat(), 'test_opened': False,
                      'test_labels_used_for_fit': False, 'optimizer_destroyed_before_test': True}
            write_json(output / 'model_frozen_before_test.json', freeze)
            manifest['artifacts']['model_frozen_before_test'] = artifact(output / 'model_frozen_before_test.json', output)
            radius = None
            if args.task == 'rt':
                calibration = load_task(args.data_root, args.task, 'calibration')
                store.verify_all(calibration)
                calibration_predictions = readout_predictions(model, calibration, collator, args.task, stats, candidate_ids,
                                                  config['training'].get('readout_batch_size', 8))
                write_jsonl(output / 'calibration_predictions.jsonl', calibration_predictions)
                radius = calibration_radius(calibration, calibration_predictions, alpha=.1)
                write_json(output / 'calibration.json', radius)
                manifest['artifacts']['calibration'] = artifact(output / 'calibration.json', output)
                manifest['artifacts']['calibration_predictions'] = artifact(output / 'calibration_predictions.jsonl', output)
            test = load_task(args.data_root, args.task, 'test')
            manifest['test_cache_coverage'] = plain(store.verify_all(test))
            readout = readout_predictions(model, test, collator, args.task, stats, candidate_ids,
                                         config['training'].get('readout_batch_size', 8))
            if radius is not None:
                for prediction in readout:
                    prediction['interval_lower'] = prediction['prediction'] - radius['radius'] if radius['radius'] is not None else None
                    prediction['interval_upper'] = prediction['prediction'] + radius['radius'] if radius['radius'] is not None else None
                    prediction['interval_unbounded'] = radius['unbounded']
            write_jsonl(output / 'readout_predictions.jsonl', readout)
            manifest['artifacts']['readout_predictions'] = artifact(output / 'readout_predictions.jsonl', output)
            language, c1 = None, None
            if tokenizer is not None:
                language = language_predictions(model, test, collator, tokenizer, args.task,
                                                 config['training'].get('generation_batch_size', 1), progress)
                write_jsonl(output / 'LM_predictions.jsonl', language)
                off = readout_predictions(model, test, collator, args.task, stats, candidate_ids,
                                           config['training'].get('readout_batch_size', 8), off=True)
                c1 = []
                for on_row, off_row in zip(readout, off):
                    if on_row['sample_id'] != off_row['sample_id']:
                        raise ValueError('C1 identity/order mismatch')
                    item = dict(on_row)
                    item.update({key + '_off': value for key, value in off_row.items() if key != 'sample_id'})
                    c1.append(item)
                write_jsonl(output / 'C1_predictions.jsonl', c1)
                manifest['artifacts']['LM_predictions'] = artifact(output / 'LM_predictions.jsonl', output)
                manifest['artifacts']['C1_predictions'] = artifact(output / 'C1_predictions.jsonl', output)
            metrics = build_metrics(args.task, test, readout, language, c1, catalog or None)
            if radius is not None:
                metrics['calibration'] = score_intervals(calibration, calibration_predictions, test, readout, alpha=.1)
            if any(not torch.equal(selected_parameters[key], value.detach().cpu()) for key, value in model.trainable_state_dict().items()):
                raise ValueError('trainable model changed after test-blind freeze')
            if file_sha(output / 'checkpoint.pt') != freeze['checkpoint_sha256']:
                raise ValueError('selected checkpoint changed during evaluation')
            write_json(output / 'metrics.json', metrics)
            manifest['metrics'] = metrics
            manifest['artifacts']['metrics'] = artifact(output / 'metrics.json', output)
        manifest['runtime'].update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.device(args.device).type == 'cuda' else None,
                                   peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.device(args.device).type == 'cuda' else None)
        manifest['status'] = 'complete'
        manifest['elapsed_seconds'] = time.perf_counter() - started
        write_json(output / 'run_manifest.json', manifest)
        return 0
    except BaseException as exc:
        failure = {'status': 'failed', 'exception_type': type(exc).__name__, 'message': str(exc),
                   'traceback': traceback.format_exc(), 'elapsed_seconds': time.perf_counter() - started}
        write_json(output / 'failed.json', failure)
        if manifest is not None:
            manifest['status'] = 'failed'
            manifest['failure'] = {key: value for key, value in failure.items() if key != 'traceback'}
            write_json(output / 'run_manifest.json', manifest)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
