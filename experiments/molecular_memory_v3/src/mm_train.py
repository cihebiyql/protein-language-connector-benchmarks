"""Single-run training with TRAIN-only normalization and separately released evaluation.

The controller owns admission, deadline/process-tree enforcement and acceptance.
This worker never alters the preregistered matrix or fits across scenarios.
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

# Support both ``python -m src.mm_train`` and direct script invocation.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'src'

# Set before torch import, including on a fresh subprocess launched by the controller.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
import numpy as np
import torch
import torch.nn.functional as F

SCOPE_ID = 'molecular_memory_v3_20261017'
SELECTION = 'min_full_validation_answer_token_nll_earliest_tie'
MODEL_REVISIONS = {'Qwen/Qwen2.5-1.5B-Instruct': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
                   'facebook/esm2_t30_150M_UR50D': 'a695f6045e2e32885fa60af20c13cb35398ce30c'}


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


def artifact(path, root):
    return {'path': Path(path).relative_to(root).as_posix(), 'sha256': file_sha(path)}


def read_json(path):
    from .mm_data import strict_loads
    return strict_loads(Path(path).read_text(encoding='utf-8'))


def check_protocol(config, phase, scenario, variant, seed):
    if config.get('schema_version') != 'mm-v3.protocol.v1':
        raise ValueError('unrecognized new-scope protocol schema')
    if scenario not in config['scenarios'] or variant not in config['variants'] or seed not in config['seeds']:
        raise ValueError('run is outside the fixed matrix')
    training = config['training']
    if config['model'].get('slots') != 8 or config['model'].get('condition_slots') != 10:
        raise ValueError('this model contract requires eight residue slots and ten condition slots')
    signature = digest_json(training)
    if phase != 'train-canary' and (config.get('status') != 'frozen' or config.get('budget_signature') != signature):
        raise ValueError('formal fit/evaluate requires frozen config and exact training budget signature')
    if training['numeric_standardization'] != 'train_zscore' or training['checkpoint_selection'] != SELECTION:
        raise ValueError('unexpected numeric normalization or checkpoint selection')
    if training['grad_accumulation'] != 1 or training['max_new_tokens'] != 24:
        raise ValueError('only registered accumulation=1 and greedy24 are supported')
    for name in ('epochs', 'batch_size', 'generation_batch_size'):
        if type(training[name]) is not int or training[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name in ('lr', 'weight_decay', 'max_grad_norm', 'numeric_loss_weight'):
        value = training[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite nonnegative')
    if training['lr'] <= 0 or training['max_grad_norm'] <= 0:
        raise ValueError('learning rate and clipping threshold must be positive')
    deadline = datetime.fromisoformat(config['deadline'])
    if deadline.tzinfo is None:
        raise ValueError('deadline must have an explicit timezone')
    if deadline != datetime.fromisoformat('2026-10-17T10:29:40+08:00'):
        raise ValueError('this scope cannot reset its authorized deadline')
    if datetime.now(timezone.utc) >= deadline:
        raise ValueError('new-scope deadline has passed')
    return signature


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
    torch.use_deterministic_algorithms(True, warn_only=False)
    return {'seed': seed, 'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
            'cpu_threads': torch.get_num_threads(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'autocast': False,
            'float32_matmul_precision': torch.get_float32_matmul_precision(), 'tf32': False,
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cuda_launch_blocking': os.environ.get('CUDA_LAUNCH_BLOCKING')}


@dataclass(frozen=True)
class NumericStats:
    mean: float
    std: float
    count: int
    source_split: str = 'train'
    std_definition: str = 'population_std_floor_1e-8'

    @classmethod
    def fit(cls, rows):
        if not rows or any(row.get('split') != 'train' for row in rows):
            raise ValueError('numeric statistics require TRAIN rows exclusively')
        values = [float(row['pkd']) for row in rows]
        if any(isinstance(row['pkd'], bool) for row in rows) or not all(map(math.isfinite, values)):
            raise ValueError('numeric TRAIN gold must be finite real')
        mean = math.fsum(values) / len(values)
        std = max(math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values)), 1e-8)
        return cls(mean, std, len(values))

    def targets(self, rows, device):
        return torch.tensor([(float(row['pkd']) - self.mean) / self.std for row in rows],
                            dtype=torch.float32, device=device)

    def inverse(self, value):
        return float(value) * self.std + self.mean


def supervised_ce(logits, labels):
    shifted = labels[:, 1:].contiguous()
    count = int((shifted != -100).sum().item())
    if count == 0:
        raise ValueError('supervised batch has no answer tokens')
    total = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                            shifted.reshape(-1), ignore_index=-100, reduction='sum')
    if not torch.isfinite(total):
        raise FloatingPointError('nonfinite answer CE')
    return total, count


def model_inputs(batch):
    return {key: value for key, value in batch.items() if key != 'labels'}


def assert_model_policy(model, device):
    active = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if not active or any(value.dtype != torch.float32 for _, value in active):
        raise ValueError('all active trainable parameters must be FP32')
    base = model.base_lm
    expected = torch.bfloat16 if torch.device(device).type == 'cuda' else torch.float32
    if any(value.requires_grad or value.dtype != expected for value in base.parameters()) or base.training:
        raise ValueError('frozen base must be eval and bf16 on CUDA / FP32 on CPU')
    return {'trainable_parameters': sum(p.numel() for _, p in active),
            'active_tensor_names': [name for name, _ in active], 'base_dtype': str(expected),
            'trainable_dtype': 'torch.float32', 'frozen_base_eval': True}


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().item() if value.numel() == 1 else value.detach().float().cpu().tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def task_gradient_diagnostics(model, ce, numeric):
    """Task gradients before weight decay/clipping; gate-zero downstream zeros are expected."""
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    result = {}
    for task, loss in (('answer_ce', ce), ('numeric_mse', numeric)):
        grads = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
        groups = {}
        for (name, _), grad in zip(named, grads):
            key = name.split('.')[0]
            group = groups.setdefault(key, {'squared_l2': 0.0, 'connected_tensors': 0, 'nonzero_tensors': 0})
            if grad is not None:
                if not torch.isfinite(grad).all():
                    raise FloatingPointError('nonfinite task gradient')
                group['squared_l2'] += float(grad.detach().double().square().sum().item())
                group['connected_tensors'] += 1
                group['nonzero_tensors'] += int(torch.count_nonzero(grad).item() != 0)
        for group in groups.values():
            group['l2'] = math.sqrt(group.pop('squared_l2'))
        result[task] = groups
    return result


@contextmanager
def inference_context(model):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


def validation_nll(model, rows, collator, batch_size):
    if not rows or any(row.get('split') != 'validation' for row in rows):
        raise ValueError('checkpoint selection requires full validation view')
    loss_sum, token_count = 0.0, 0
    with inference_context(model):
        for start in range(0, len(rows), batch_size):
            batch = collator(rows[start:start + batch_size], supervised=True)
            output = model(**model_inputs(batch))
            loss, count = supervised_ce(output['logits'], batch['labels'])
            loss_sum += float(loss.item())
            token_count += count
    return {'validation_answer_loss_sum': loss_sum, 'validation_answer_token_count': token_count,
            'validation_answer_token_nll': loss_sum / token_count, 'validation_rows': len(rows),
            'validation_ordered_ids_sha256': digest_json([row['sample_id'] for row in rows])}


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def save_checkpoint(path, model, optimizer, stats, identity, epoch, history, selection):
    state = cpu_state(model.trainable_state_dict())
    if any(name.startswith('base_lm.') for name in state):
        raise ValueError('trainable checkpoint unexpectedly contains frozen LM')
    for value in state.values():
        if torch.is_tensor(value) and value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError('floating trainable state must be saved in FP32')
    payload = {'schema_version': 1, 'scope_id': SCOPE_ID, 'identity': identity,
               'epoch': epoch, 'trainable_state': state, 'optimizer': cpu_state(optimizer.state_dict()),
               'numeric_stats': asdict(stats), 'history': history, 'selection': selection,
               'model_diagnostics': plain(model.diagnostics())}
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def fit_model(model, train, validation, collator, config, identity, output, device,
              max_updates=None, progress=None):
    """Core injected runner: callers provide only splits admitted for this phase."""
    if not train or any(row.get('split') != 'train' for row in train):
        raise ValueError('optimizer may consume TRAIN only')
    canary = max_updates is not None
    if canary and (type(max_updates) is not int or max_updates < 1):
        raise ValueError('max_updates must be a positive integer')
    if not canary and validation is None:
        raise ValueError('formal fit requires validation')
    training = config['training']
    stats = NumericStats.fit(train)
    write_json(output / 'numeric_stats.json', asdict(stats))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=training['lr'], weight_decay=training['weight_decay'])
    history, updates, examples_seen = [], 0, 0
    best = None
    first_update_seconds = None
    started = time.perf_counter()
    model.train()
    assert_model_policy(model, device)
    # Canary is allowed to wrap TRAIN to reach the requested update count; no validation is opened.
    epochs = max(training['epochs'], math.ceil(max_updates * training['batch_size'] / len(train))) if canary else training['epochs']
    for epoch in range(1, epochs + 1):
        order = list(range(len(train)))
        random.Random(identity['seed'] + epoch).shuffle(order)
        ce_sum, ce_tokens, numeric_sum, epoch_examples = 0.0, 0, 0.0, 0
        epoch_started = time.perf_counter()
        task_grads = []
        for start in range(0, len(order), training['batch_size']):
            rows = [train[index] for index in order[start:start + training['batch_size']]]
            batch = collator(rows, supervised=True)
            optimizer.zero_grad(set_to_none=True)
            result = model(**model_inputs(batch))
            total, count = supervised_ce(result['logits'], batch['labels'])
            ce = total / count
            numeric = F.mse_loss(result['numeric_z'].float().reshape(-1), stats.targets(rows, device))
            if not torch.isfinite(numeric):
                raise FloatingPointError('nonfinite numeric MSE')
            if start == 0 or updates in (1, 3):
                task_grads.append({'optimizer_step_before_update': updates,
                                   'model_before_update': plain(model.diagnostics()),
                                   'tasks': task_gradient_diagnostics(model, ce, numeric)})
            loss = ce + training['numeric_loss_weight'] * numeric
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, training['max_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            examples_seen += len(rows)
            epoch_examples += len(rows)
            ce_sum += float(total.item())
            ce_tokens += count
            numeric_sum += float(numeric.item()) * len(rows)
            if first_update_seconds is None:
                first_update_seconds = time.perf_counter() - started
            if progress:
                progress({'event': 'update', 'epoch': epoch, 'optimizer_steps': updates,
                          'examples_seen': examples_seen, 'loss': float(loss.item()),
                          'gradient_norm_before_clip': float(grad_norm), 'elapsed_seconds': time.perf_counter() - started})
            if canary and updates >= max_updates:
                break
        item = {'epoch': epoch, 'optimizer_steps': updates, 'examples_seen': examples_seen,
                'train_examples_this_epoch': epoch_examples, 'train_answer_loss_sum': ce_sum,
                'train_answer_token_count': ce_tokens, 'train_numeric_mse_z': numeric_sum / epoch_examples,
                'epoch_fully_completed': epoch_examples == len(train),
                'seconds': time.perf_counter() - epoch_started, 'task_gradient_diagnostics': task_grads,
                'model_diagnostics': plain(model.diagnostics())}
        if not canary:
            item.update(validation_nll(model, validation, collator, training['batch_size']))
            if best is None or item['validation_answer_token_nll'] < best['validation_answer_token_nll']:
                best = {'best_epoch': epoch, 'validation_answer_token_nll': item['validation_answer_token_nll'],
                        'test_labels_used_for_selection': False, 'written_before_test': True}
        history.append(item)
        write_json(output / 'history.json', history)
        checkpoint_path = output / f'epoch_{epoch}.pt'
        save_checkpoint(checkpoint_path, model, optimizer, stats, identity, epoch, history, best)
        if best is not None and best['best_epoch'] == epoch:
            # Copy the exact serialized epoch checkpoint; no second serialization/weight change.
            import shutil
            shutil.copyfile(checkpoint_path, output / 'selected.pt')
            selection = {**best, 'checkpoint_sha256': file_sha(output / 'selected.pt')}
            write_json(output / 'selection.json', selection)
        if progress:
            progress({'event': 'epoch_complete' if item['epoch_fully_completed'] else 'canary_segment_complete',
                      **{k: v for k, v in item.items() if k not in ('task_gradient_diagnostics', 'model_diagnostics')}})
        if canary and updates >= max_updates:
            break
    if canary:
        checkpoint_path.rename(output / 'canary.pt')
    return {'epochs_completed': sum(item['epoch_fully_completed'] for item in history),
            'history_segments': len(history), 'optimizer_steps': updates, 'examples_seen': examples_seen,
            'elapsed_seconds': time.perf_counter() - started, 'first_update_seconds': first_update_seconds,
            'test_labels_used_for_fit': False, 'validation_used': not canary,
            'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated() if torch.device(device).type == 'cuda' else None}


def greedy_batch(model, rows, collator, tokenizer, stats, max_new_tokens=24):
    """Left-padded prompt batch, fixed prompt anchor and first-EOS truncation per row.

    Finished rows remain masked padding while unfinished rows continue. The ledger
    accounts for their actual batched forward work, separately from generated tokens.
    """
    batch = collator(rows, supervised=False)
    if 'labels' in batch:
        raise ValueError('generation must receive prompt-only batch')
    if not torch.all(batch['attention_mask'][:, -1] == 1):
        raise ValueError('generation requires left padding and final prompt token active')
    generated = [[] for _ in rows]
    finished = [False] * len(rows)
    prompt_tokens = batch['attention_mask'].sum(dim=1).tolist()
    numeric_pkd, ledger = None, []
    with inference_context(model):
        for _ in range(max_new_tokens):
            ledger.append({'batch_rows': len(rows), 'query_slots': int(batch['input_ids'].numel()),
                           'valid_query_tokens': int(batch['attention_mask'].sum().item()),
                           'padded_query_tokens': int((batch['attention_mask'] == 0).sum().item()),
                           'active_generation_rows': sum(not value for value in finished),
                           'source_consumption_rows_including_finished': len(rows)})
            result = model(**batch)
            if numeric_pkd is None:
                numeric_pkd = [stats.inverse(v) for v in result['numeric_z'].reshape(-1).tolist()]
            tokens = result['logits'][:, -1].argmax(dim=-1).tolist()
            active_before = [not value for value in finished]
            for index, token in enumerate(tokens):
                if not finished[index]:
                    generated[index].append(token)
                    finished[index] = token == tokenizer.eos_token_id
            if all(finished):
                break
            # The just-produced EOS remains attended; subsequent writes for finished rows are masked padding.
            appended = [token if active else tokenizer.pad_token_id for token, active in zip(tokens, active_before)]
            batch['input_ids'] = torch.cat((batch['input_ids'], batch['input_ids'].new_tensor(appended)[:, None]), dim=1)
            batch['attention_mask'] = torch.cat((batch['attention_mask'], batch['attention_mask'].new_tensor(active_before)[:, None]), dim=1)
    predictions = [{'raw_text': tokenizer.decode(tokens, skip_special_tokens=True),
                    'raw_generated_token_ids': tokens, 'first_eos_length': len(tokens),
                    'prediction_pkd': value, 'prompt_tokens': int(prompt_count),
                    'stopped_on_eos': bool(tokens and tokens[-1] == tokenizer.eos_token_id)}
                   for tokens, value, prompt_count in zip(generated, numeric_pkd, prompt_tokens)]
    return predictions, ledger


def greedy_and_numeric(model, row, collator, tokenizer, stats, max_new_tokens=24):
    return greedy_batch(model, [row], collator, tokenizer, stats, max_new_tokens)[0][0]


def write_predictions(model, rows, collator, tokenizer, stats, output, split, common, batch_size, progress):
    """Write both output channels plus engineering consumption ledger, preserving row order."""
    from .mm_evaluate import parse_answer
    lm_path, numeric_path = output / f'{split}_lm_predictions.jsonl', output / f'{split}_numeric_predictions.jsonl'
    ledger_path = output / f'{split}_generation_ledger.jsonl'
    lm_rows, numeric_rows = [], []
    with lm_path.open('x', encoding='utf-8') as lm_stream, numeric_path.open('x', encoding='utf-8') as num_stream, ledger_path.open('x', encoding='utf-8') as ledger_stream:
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            predictions, ledger = greedy_batch(model, selected, collator, tokenizer, stats)
            ledger_stream.write(canonical({'ordered_sample_ids': [r['sample_id'] for r in selected],
                                           'generation_cap_per_row': 24, 'forward_steps': ledger,
                                           'row_prompt_tokens': [p['prompt_tokens'] for p in predictions],
                                           'row_first_eos_lengths': [p['first_eos_length'] for p in predictions],
                                           'stopped_on_eos': [p['stopped_on_eos'] for p in predictions]}) + '\n')
            for row, prediction in zip(selected, predictions):
                envelope = {**common, 'sample_id': row['sample_id'], 'split': split, 'gold_pkd': row['pkd']}
                lm = {**envelope, **{k: prediction[k] for k in ('raw_text', 'raw_generated_token_ids', 'first_eos_length')}}
                lm['parse'] = parse_answer(lm['raw_text'])
                value = prediction['prediction_pkd']
                numeric = {**envelope, 'prediction_pkd': value if math.isfinite(value) else None,
                           'error': None if math.isfinite(value) else 'nonfinite_prediction'}
                lm_stream.write(canonical(lm) + '\n'); lm_stream.flush()
                num_stream.write(canonical(numeric) + '\n'); num_stream.flush()
                lm_rows.append(lm); numeric_rows.append(numeric)
            if progress:
                progress({'event': 'prediction', 'split': split, 'rows_completed': start + len(selected)})
    return {'lm_predictions': artifact(lm_path, output), 'numeric_predictions': artifact(numeric_path, output),
            'generation_ledger': artifact(ledger_path, output)}, lm_rows, numeric_rows


def row_answer_nll(logits, labels):
    targets = labels[:, 1:]
    counts = (targets != -100).sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError('C1 row has no supervised answer tokens')
    losses = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                            targets.reshape(-1), ignore_index=-100, reduction='none').reshape(targets.shape)
    if not torch.isfinite(losses).all():
        raise FloatingPointError('nonfinite C1 answer NLL')
    return [{'loss_sum': float(value), 'token_count': int(count)}
            for value, count in zip(losses.sum(dim=1).tolist(), counts.tolist())]


def prompt_anchor_invariance(model, rows, collator):
    """TRAIN-only same-shape future-token perturbation, not a gold-conditioned head."""
    if any(row['split'] != 'train' for row in rows):
        raise ValueError('canary anchor qualification must use TRAIN only')
    batch = collator(rows, supervised=True)
    inputs = model_inputs(batch)
    with inference_context(model):
        original = model(**inputs)['numeric_z'].detach().clone()
        changed = {**inputs, 'input_ids': inputs['input_ids'].clone()}
        positions = torch.arange(inputs['input_ids'].shape[1], device=inputs['input_ids'].device)[None, :]
        future = (positions > inputs['prompt_end_index'][:, None]) & inputs['attention_mask'].bool()
        vocab_size = model.base_lm.get_input_embeddings().weight.shape[0] if hasattr(model.base_lm, 'get_input_embeddings') else None
        if vocab_size is None:
            # CPU fixture models expose no tokenizer embedding; LM logits supply its vocabulary.
            vocab_size = model(**inputs)['logits'].shape[-1]
        changed['input_ids'][future] = (changed['input_ids'][future] + 1) % vocab_size
        perturbed = model(**changed)['numeric_z'].detach().clone()
    passed = bool(torch.equal(original, perturbed))
    receipt = {'probe': 'same_shape_future_answer_token_perturbation', 'split': 'train',
               'ordered_sample_ids': [r['sample_id'] for r in rows], 'changed_future_tokens': int(future.sum().item()),
               'numeric_z_before': original.float().cpu().tolist(), 'numeric_z_after': perturbed.float().cpu().tolist(),
               'max_absolute_difference': float((original - perturbed).abs().max().item()), 'passed': passed,
               'prompt_end_index_fixed': True, 'attention_mask_fixed': True}
    return receipt


def write_c1(model, rows, collator, stats, output, checkpoint_sha, batch_size, filename='c1_predictions.jsonl'):
    """Same selected theta: remove the branch; prompt-only numeric head stays separate.

    The explicit evaluation-only collation capability never enters fit_model.
    TRAIN-mean/OOD-memory and biological donor probes are not computed here.
    """
    records = []
    started = time.perf_counter()
    with (output / filename).open('x', encoding='utf-8') as stream, inference_context(model):
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            batch = collator(selected, supervised=True, evaluation_only=True)
            on = row_answer_nll(model(**model_inputs(batch))['logits'], batch['labels'])
            off = row_answer_nll(model(**model_inputs(batch), off=True)['logits'], batch['labels'])
            prompt = collator(selected, supervised=False)
            numeric_on = model(**prompt)['numeric_z'].reshape(-1).tolist()
            numeric_off = model(**prompt, off=True)['numeric_z'].reshape(-1).tolist()
            for row, correct, removed, y_on, y_off in zip(selected, on, off, numeric_on, numeric_off):
                if correct['token_count'] != removed['token_count']:
                    raise ValueError('C1 on/off denominator differs')
                item = {'sample_id': row['sample_id'], 'answer_tokens': correct['token_count'],
                        'correct': correct, 'off': removed, 'checkpoint_sha256': checkpoint_sha,
                        'numeric_correct_pkd': stats.inverse(y_on), 'numeric_off_pkd': stats.inverse(y_off)}
                stream.write(canonical(item) + '\n'); stream.flush()
                records.append(item)
    return artifact(output / filename, output), records, time.perf_counter() - started


def verify_checkpoint_identity(payload, identity):
    if payload.get('scope_id') != SCOPE_ID or payload.get('identity') != identity:
        raise ValueError('checkpoint identity/config/scenario/variant/seed differs')
    if payload.get('selection') is None or payload['selection']['best_epoch'] != payload.get('epoch'):
        raise ValueError('evaluate requires a validation-selected checkpoint')


def source_pins():
    root = Path(__file__).resolve().parent
    return {p.name: file_sha(p) for p in sorted(root.glob('*.py'))}


def verify_model_assets(project_root, config):
    """Check local bytes before deserialization, including frozen ESM cache provenance."""
    manifest_path = project_root / 'sources/model_manifest.json'
    records = read_json(manifest_path)
    result = {'manifest_sha256': file_sha(manifest_path), 'files': {}}
    for repo, revision in MODEL_REVISIONS.items():
        rows = [row for row in records if row['repo'] == repo]
        if not rows or not any(row['file'].endswith('.safetensors') for row in rows):
            raise ValueError('pinned local backbone weights absent from model manifest')
        key = 'backbone_path' if repo.startswith('Qwen/') else 'esm_backbone_path'
        root = project_root / config['model'][key]
        for row in rows:
            path = root / row['file']
            if row['revision'] != revision or path.stat().st_size != row['bytes'] or file_sha(path) != row['sha256']:
                raise ValueError(f'pinned model asset differs: {repo}/{row["file"]}')
            result['files'][f'{repo}/{row["file"]}'] = {'sha256': row['sha256'], 'bytes': row['bytes'], 'revision': revision}
    return result


def load_components(project_root, config, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .mm_model import MolecularMemoryLM
    path = project_root / config['model']['backbone_path']
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16 if torch.device(device).type == 'cuda' else torch.float32,
            attn_implementation='eager').to(device)
    base.requires_grad_(False).eval()
    settings = config['model']
    model = MolecularMemoryLM(base, variant=config['_variant'], hidden_dim=base.config.hidden_size,
            esm_dim=640, memory_dim=settings['memory_dim'], num_heads=settings['num_heads'],
            hook_layer=settings['hook_layer']).to(device)
    return model, tokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'project-root', 'cache-root', 'output', 'scenario', 'variant', 'seed', 'device'):
        parser.add_argument('--' + name, required=True, type=int if name == 'seed' else str)
    parser.add_argument('--phase', choices=('train-canary', 'fit', 'evaluate'), required=True)
    parser.add_argument('--max-updates', type=int, default=None)
    parser.add_argument('--checkpoint', help='selected.pt from the corresponding fit; evaluate only')
    parser.add_argument('--view-root', help='prepared gold-separated views; default config data_views_root or scope/data_views')
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = None
    try:
        from .mm_data import read_input_view, collate, FROZEN_SHAS, DATA_RELATIVE, read_jsonl
        from .mm_extract import SequenceStore
        project = Path(args.project_root).resolve()
        config_path = Path(args.config).resolve()
        config = read_json(config_path)
        training_signature = check_protocol(config, args.phase, args.scenario, args.variant, args.seed)
        if (args.phase == 'train-canary') != (args.max_updates is not None):
            raise ValueError('only train-canary requires --max-updates')
        if (args.phase == 'evaluate') != (args.checkpoint is not None):
            raise ValueError('only evaluate requires --checkpoint')
        config_sha = file_sha(config_path)
        identity = {'scenario': args.scenario, 'variant': args.variant, 'seed': args.seed,
                    'protocol_sha256': config_sha, 'training_budget_sha256': training_signature}
        runtime = configure(args.seed)
        scope_root = Path(__file__).resolve().parents[1]
        views = Path(args.view_root or config.get('data_views_root', str(scope_root / 'data_views')))
        if not views.is_absolute():
            views = project / views
        view_manifest = read_json(views / 'manifest.json')
        manifest = {'schema_version': 1, 'scope_id': SCOPE_ID, 'run_id': output.name,
                    'phase': 'development' if args.phase == 'train-canary' else 'formal',
                    'execution_phase': args.phase, 'status': 'partial', 'scenario': args.scenario,
                    'arm': args.variant, 'seed': args.seed, 'protocol_sha256': config_sha,
                    'training_budget_sha256': training_signature,
                    'data_sha256': FROZEN_SHAS[args.scenario], 'test_labels_used_for_fit': False,
                    'artifacts': {key: None for key in ('history', 'selection', 'checkpoint', 'lm_predictions', 'numeric_predictions', 'c1')},
                    'runtime': runtime, 'identity': identity,
                    'provenance': {'code_sha256': source_pins(), 'view_manifest_sha256': file_sha(views / 'manifest.json'),
                                   'views': view_manifest['scenarios'][args.scenario], 'audit_artifacts': {}},
                    'deadline': config['deadline'], 'created_utc': datetime.now(timezone.utc).isoformat()}
        write_json(output / 'protocol.json', config)
        write_json(output / 'run_manifest.json', manifest)
        assets_started = time.perf_counter()
        manifest['provenance']['model_assets'] = verify_model_assets(project, config)
        store = SequenceStore(Path(args.cache_root))
        manifest['provenance']['cache_before_run'] = store.verify()
        manifest['runtime']['asset_verification_seconds'] = time.perf_counter() - assets_started
        config_for_model = {**config, '_variant': args.variant}
        loading_started = time.perf_counter()
        model, tokenizer = load_components(project, config_for_model, args.device)
        manifest['runtime']['model_loading_seconds'] = time.perf_counter() - loading_started
        manifest['model_policy'] = assert_model_policy(model, args.device)
        def collator(rows, supervised=True, evaluation_only=False):
            if supervised and any(row['split'] == 'test' for row in rows):
                if not evaluation_only or args.phase != 'evaluate':
                    raise ValueError('test supervision is only an explicit evaluate-only C1 intervention')
                from .mm_data_evaluation import collate_evaluation
                return collate_evaluation(rows, tokenizer, store, device=args.device, phase='evaluate')
            return collate(rows, tokenizer, store, device=args.device, supervised=supervised)
        def progress(value):
            with (output / 'progress.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(canonical(value) + '\n')
            print(canonical(value), flush=True)
        if args.phase in ('fit', 'train-canary'):
            train = read_input_view(views, args.scenario, 'train')
            val = read_input_view(views, args.scenario, 'validation') if args.phase == 'fit' else None
            training_result = fit_model(model, train, val, collator, config, identity, output, args.device,
                                        max_updates=args.max_updates, progress=progress)
            manifest['training'] = {key: training_result[key] for key in ('epochs_completed', 'optimizer_steps', 'examples_seen')}
            manifest['runtime']['training_performance'] = {key: value for key, value in training_result.items()
                                                          if key not in manifest['training']}
            for name, filename in (('history', 'history.json'), ('selection', 'selection.json'),
                                   ('checkpoint', 'selected.pt' if args.phase == 'fit' else 'canary.pt')):
                if (output / filename).exists():
                    manifest['artifacts'][name] = artifact(output / filename, output)
            manifest['provenance']['audit_artifacts']['numeric_stats'] = artifact(output / 'numeric_stats.json', output)
            if args.phase == 'train-canary':
                from .mm_evaluate import generation_metrics, numeric_metrics, c1_metrics
                stats = NumericStats(**read_json(output / 'numeric_stats.json'))
                probe_count = max(2, config['training']['generation_batch_size'])
                probe = [{**row, 'groups': {'target_sequence_sha256': hashlib.sha256(row['target_sequence'].encode()).hexdigest()}}
                         for row in train[:probe_count]]
                qualification_started = time.perf_counter()
                qualification = prompt_anchor_invariance(model, probe, collator)
                write_json(output / 'train_anchor_qualification.json', qualification)
                if not qualification['passed']:
                    raise ValueError('numeric prompt anchor changed when only future answer tokens changed')
                manifest['runtime']['canary_anchor_qualification_seconds'] = time.perf_counter() - qualification_started
                generation_started = time.perf_counter()
                refs, lm_rows, numeric_rows = write_predictions(model, probe, collator, tokenizer, stats,
                    output, 'train', {'run_id': output.name, 'protocol_sha256': config_sha,
                    'checkpoint_sha256': manifest['artifacts']['checkpoint']['sha256']},
                    config['training']['generation_batch_size'], progress)
                manifest['runtime']['canary_generation_seconds'] = time.perf_counter() - generation_started
                c1_ref, c1_rows, c1_seconds = write_c1(model, probe, collator, stats, output,
                    manifest['artifacts']['checkpoint']['sha256'], config['training']['batch_size'], 'train_c1_predictions.jsonl')
                write_json(output / 'train_canary_metrics.json', {'development_only': True, 'split': 'train',
                    'rows': len(probe), 'lm': generation_metrics(probe, lm_rows),
                    'numeric': numeric_metrics(probe, numeric_rows), 'c1': c1_metrics(probe, c1_rows), 'c1_seconds': c1_seconds})
                manifest['artifacts'].update({key: refs[key] for key in ('lm_predictions', 'numeric_predictions')})
                manifest['artifacts']['c1'] = c1_ref
                manifest['provenance']['audit_artifacts'].update(generation_ledger=refs['generation_ledger'],
                    train_canary_metrics=artifact(output / 'train_canary_metrics.json', output),
                    train_anchor_qualification=artifact(output / 'train_anchor_qualification.json', output))
        else:
            checkpoint = Path(args.checkpoint).resolve()
            fit_root = checkpoint.parent
            fit_manifest = read_json(fit_root / 'run_manifest.json')
            if fit_manifest.get('status') != 'complete' or fit_manifest.get('execution_phase') != 'fit':
                raise ValueError('evaluation source fit is not complete')
            if fit_manifest.get('identity') != identity:
                raise ValueError('evaluation cannot cross scenario/config/variant/seed')
            if fit_manifest['provenance']['code_sha256'] != manifest['provenance']['code_sha256']:
                raise ValueError('evaluation source code differs from the fit pin')
            if fit_manifest['provenance']['cache_before_run'] != manifest['provenance']['cache_before_run']:
                raise ValueError('evaluation cache closure differs from the fit pin')
            source_selection = read_json(fit_root / 'selection.json')
            for name in ('history', 'selection', 'checkpoint'):
                ref = fit_manifest['artifacts'][name]
                candidate = (fit_root / ref['path']).resolve()
                if not candidate.is_relative_to(fit_root) or file_sha(candidate) != ref['sha256']:
                    raise ValueError('source fit artifact escapes or differs from its SHA pin')
            if source_selection['checkpoint_sha256'] != file_sha(checkpoint):
                raise ValueError('selected checkpoint SHA differs')
            if source_selection.get('test_labels_used_for_selection') is not False or source_selection.get('written_before_test') is not True:
                raise ValueError('source selection lacks a test-blind commitment')
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            verify_checkpoint_identity(payload, identity)
            model.load_trainable_state_dict(payload['trainable_state'])
            stats = NumericStats(**payload['numeric_stats'])
            import shutil
            for name in ('history.json', 'selection.json'):
                shutil.copyfile(fit_root / name, output / name)
            shutil.copyfile(checkpoint, output / 'selected.pt')
            manifest['source_fit'] = {'path': str(fit_root), 'manifest_sha256': file_sha(fit_root / 'run_manifest.json'),
                                      'checkpoint_sha256': file_sha(checkpoint), 'identity': identity}
            manifest['training'] = fit_manifest['training']
            manifest['artifacts'].update(history=artifact(output / 'history.json', output),
                                         selection=artifact(output / 'selection.json', output))
            manifest['artifacts']['checkpoint'] = artifact(output / 'selected.pt', output)
            manifest['evaluation'] = {}
            from .mm_evaluate import generation_metrics, numeric_metrics, c1_metrics
            for split in ('validation', 'test'):
                rows = read_input_view(views, args.scenario, split)
                if split == 'test':
                    gold_path = project / DATA_RELATIVE / args.scenario / 'test_gold.jsonl'
                    if file_sha(gold_path) != FROZEN_SHAS[args.scenario]['test_gold.jsonl']:
                        raise ValueError('test gold SHA differs')
                    gold = read_jsonl(gold_path)
                    if [r['sample_id'] for r in gold] != [r['sample_id'] for r in rows]:
                        raise ValueError('test input/gold identity/order differs')
                    metric_rows = gold
                else:
                    metric_rows = [{**row, 'groups': {'target_sequence_sha256': hashlib.sha256(row['target_sequence'].encode()).hexdigest()}}
                                   for row in rows]
                labelled = [{**row, 'pkd': gold_row['pkd']} for row, gold_row in zip(rows, metric_rows)]
                refs, lm_rows, numeric_rows = write_predictions(model, labelled, collator, tokenizer, stats,
                    output, split, {'checkpoint_sha256': file_sha(checkpoint), 'protocol_sha256': config_sha,
                    'run_id': output.name}, config['training']['generation_batch_size'], progress)
                metrics = {'rows': len(rows), 'lm': generation_metrics(metric_rows, lm_rows),
                           'numeric': numeric_metrics(metric_rows, numeric_rows)}
                write_json(output / f'{split}_metrics.json', metrics)
                manifest['evaluation'][split] = {**metrics, **refs}
                manifest['provenance']['audit_artifacts'][split + '_generation_ledger'] = refs['generation_ledger']
                manifest['provenance']['audit_artifacts'][split + '_metrics'] = artifact(output / f'{split}_metrics.json', output)
                if split == 'test':
                    manifest['artifacts'].update({key: refs[key] for key in ('lm_predictions', 'numeric_predictions')})
                    c1_ref, c1_rows, c1_seconds = write_c1(model, labelled, collator, stats, output,
                        manifest['artifacts']['checkpoint']['sha256'], config['training']['batch_size'])
                    manifest['artifacts']['c1'] = c1_ref
                    write_json(output / 'c1_metrics.json', c1_metrics(metric_rows, c1_rows))
                    manifest['runtime']['c1_seconds'] = c1_seconds
                    manifest['provenance']['audit_artifacts']['c1_metrics'] = artifact(output / 'c1_metrics.json', output)
        manifest['provenance']['cache'] = store.provenance()
        manifest['status'] = 'complete'
        manifest['elapsed_seconds'] = time.perf_counter() - started
        manifest['worker_completion_is_controller_acceptance'] = False
        write_json(output / 'run_manifest.json', manifest)
        write_json(output / 'receipt.json', {'status': 'complete', 'phase': args.phase,
                    'run_manifest_sha256': file_sha(output / 'run_manifest.json'),
                    'elapsed_seconds': manifest['elapsed_seconds'], 'scientific_acceptance': 'controller_pending'})
        return 0
    except BaseException as exc:
        if manifest is not None:
            manifest['status'] = 'evaluation_failed' if args.phase == 'evaluate' else 'training_failed'
            manifest['elapsed_seconds'] = time.perf_counter() - started
            write_json(output / 'run_manifest.json', manifest)
        write_json(output / 'failed.json', {'status': 'failed', 'phase': args.phase,
                   'exception_type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc(),
                   'elapsed_seconds': time.perf_counter() - started, 'scientific_acceptance': False})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
