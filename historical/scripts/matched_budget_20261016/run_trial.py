"""One prospectively registered full matched-message A1 training trial.

This command never edits historical data/results or chooses its own matrix.
The parent supervisor admits trials and owns resource/contract supervision.
Only fixed-message lookup differs from the frozen foundation collation: text,
answer labels, validation criterion, strict parser and main scoring are reused.
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import sys
import time

NUMERICAL_ENVIRONMENT = {'CUDA_LAUNCH_BLOCKING': '1', 'TORCH_SHOW_CPP_STACKTRACES': '1',
                         'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
PRE_TORCH_ENVIRONMENT = {key: os.environ.get(key) for key in NUMERICAL_ENVIRONMENT}

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_v02.cache_panel_features import file_sha, verify_local_models
from bench_v02.panel_execution import validate_panel_data
from foundation_20261016 import run_connectors as legacy
from foundation_20261016.information_diagnostics import answer_nll as frozen_c1_answer_nll
from matched_budget_20261016.token_diagnostics import batch_token_categories, token_nll_breakdown
from matched_budget_20261016.prepare_cache import (
    ROOT, SCENARIOS, EXPECTED_COUNTS, SEEDS, LEGACY_SHA256, relative,
    tensor_sha, verify_legacy_source, verify_pins)

FROZEN_PROTOCOL_SHA256 = '9940d77d82e3ff23a085aa026219a4f4ee4e7ecd18bc1afe6952646a78e8b208'
TOKEN_BUCKETS = ('format', 'numeric', 'mixed', 'full')
EXPECTED_NUMERICAL_FLAGS = {'float32_matmul_precision': 'highest', 'cuda_matmul_allow_tf32': False,
                           'cudnn_allow_tf32': False, 'cudnn_benchmark': False, 'cudnn_deterministic': True,
                           'deterministic_algorithms': True, 'deterministic_warn_only': False,
                           'cuda_autocast_enabled': False, 'cpu_autocast_enabled': False}


def numerical_policy_receipt():
    return {'pre_torch_environment': dict(PRE_TORCH_ENVIRONMENT),
            'current_environment': {key: os.environ.get(key) for key in NUMERICAL_ENVIRONMENT},
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cuda_autocast_enabled': torch.is_autocast_enabled('cuda'),
            'cpu_autocast_enabled': torch.is_autocast_enabled('cpu')}


def configure_numerical_policy(require_cuda_environment):
    """Apply the fixed policy before any CUDA/model operation; no warn-only fallback."""
    if require_cuda_environment and (PRE_TORCH_ENVIRONMENT != NUMERICAL_ENVIRONMENT or
            {key: os.environ.get(key) for key in NUMERICAL_ENVIRONMENT} != NUMERICAL_ENVIRONMENT):
        raise ValueError('Strict CUDA environment must be set by supervisor before importing torch')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    value = numerical_policy_receipt()
    if any(value[key] != expected_value for key, expected_value in EXPECTED_NUMERICAL_FLAGS.items()):
        raise ValueError('Strict numerical flags/autocast differ; execution refused')
    return value


def verify_model_numerics(model):
    base_dtypes = sorted({str(p.dtype) for p in model.base_lm.parameters()})
    reader_dtypes = sorted({str(p.dtype) for p in model.reader.parameters()})
    if base_dtypes != ['torch.bfloat16'] or reader_dtypes != ['torch.float32']:
        raise ValueError('Frozen policy requires base bfloat16 and reader float32')
    receipt = numerical_policy_receipt()
    if any(receipt[key] != expected_value for key, expected_value in EXPECTED_NUMERICAL_FLAGS.items()):
        raise ValueError('Strict numerical flags/autocast changed during execution')
    if next(model.base_lm.parameters()).device.type == 'cuda' and (
            receipt['pre_torch_environment'] != NUMERICAL_ENVIRONMENT or
            receipt['current_environment'] != NUMERICAL_ENVIRONMENT):
        raise ValueError('Strict CUDA environment changed during execution')
    return {**receipt, 'base_parameter_dtypes': base_dtypes, 'reader_parameter_dtypes': reader_dtypes,
            'source_message_dtype': 'torch.float32', 'reader_autocast': False,
            'unsupported_deterministic_kernel_policy': 'raise and stop; never warn_only or fallback'}


def aggregate_token_breakdowns(items):
    """Aggregate loss sums/counts; category means themselves are never added."""
    result = {}
    for name in TOKEN_BUCKETS:
        result[name] = {'loss_sum': math.fsum(x[name]['loss_sum'] for x in items),
                        'count': sum(x[name]['count'] for x in items)}
    # Exact category composition for diagnostic accounting, independent of the
    # original fp32 reduction that alone defines checkpoint selection.
    result['full']['loss_sum'] = sum(result[name]['loss_sum'] for name in TOKEN_BUCKETS[:3])
    count = result['full']['count']
    for value in result.values():
        value['nll_nats_per_token'] = value['loss_sum'] / value['count'] if value['count'] else None
        value['contribution_nats_per_full_token'] = value['loss_sum'] / count if count else None
    reference = {'loss_sum': math.fsum(x['reference_supervised_nll']['loss_sum'] for x in items),
                 'count': sum(x['reference_supervised_nll']['count'] for x in items)}
    reference['nll_nats_per_token'] = reference['loss_sum'] / reference['count'] if reference['count'] else None
    result['reference_supervised_nll'] = reference
    result['accounting'] = {'diagnostic_sum_minus_reference_fp32_sum': result['full']['loss_sum'] - reference['loss_sum'],
                            'category_count_equals_full': sum(result[name]['count'] for name in TOKEN_BUCKETS[:3]) == count,
                            'checkpoint_selection': 'unchanged legacy.supervised_nll full-answer value only'}
    return result


class MessageStore:
    """Hash-checked immutable fixed messages; never reads raw residue tensors."""
    def __init__(self, manifest_path, expected_sha256, scenario, sender_mode, message_input):
        self.manifest_path = Path(manifest_path)
        if file_sha(self.manifest_path) != expected_sha256:
            raise ValueError('Fixed-message manifest SHA differs from admitted trial')
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        if self.manifest.get('purpose') != 'matched_budget_fixed_message_cache':
            raise ValueError('Wrong cache purpose')
        if self.manifest.get('legacy_semantics_sha256') != LEGACY_SHA256:
            raise ValueError('Cache does not bind frozen text/scoring semantics')
        if self.manifest.get('source_pins', {}).get('scripts/foundation_20261016/run_connectors.py') != LEGACY_SHA256:
            raise ValueError('Cache lacks the actual immutable legacy source pin')
        verify_pins(self.manifest['source_pins'])
        if message_input not in ('real', 'neutral') or sender_mode not in ('balanced', 'global16'):
            raise ValueError('Unregistered content or fixed pooling mode')
        self.scenario, self.sender_mode, self.message_input = scenario, sender_mode, message_input
        item = self.manifest['datasets'][scenario]['layouts'][sender_mode]
        self.payload_path = ROOT / item['path']
        if file_sha(self.payload_path) != item['sha256']:
            raise ValueError('Cached messages changed')
        payload = torch.load(self.payload_path, map_location='cpu', weights_only=True)
        ids, splits = payload['sample_ids'], payload['splits']
        self.messages, self.neutral = payload['messages'], payload['neutral']
        if (payload.get('scenario') != scenario or payload.get('sender_mode') != sender_mode
                or len(ids) != len(set(ids)) or len(splits) != len(ids)
                or self.messages.shape != (len(ids), 16, 128)
                or self.messages.dtype != torch.float32 or self.messages.requires_grad
                or self.neutral.shape != (16, 128) or self.neutral.dtype != torch.float32
                or not torch.isfinite(self.messages).all() or not torch.isfinite(self.neutral).all()):
            raise ValueError('Malformed fixed-message payload')
        self.indices = {key: index for index, key in enumerate(ids)}
        self.splits = dict(zip(ids, splits))
        train = [i for i, split in enumerate(splits) if split == 'train']
        if (not train or payload['neutral_fit_train_sample_ids'] != [ids[i] for i in train]
                or payload.get('neutral_fit_uses_labels') is not False
                or not torch.equal(self.neutral, self.messages[train].mean(0))
                or tensor_sha(self.neutral) != item['neutral_sha256']):
            raise ValueError('Neutral message is not the exact scenario TRAIN-only mean')
        self.samples_sha256 = payload['samples_sha256']

    def batch(self, rows):
        if any(row['sample_id'] not in self.indices or self.splits[row['sample_id']] != row['split'] for row in rows):
            raise ValueError('Row identity/split differs from fixed-message manifest')
        if self.message_input == 'neutral':
            return self.neutral.unsqueeze(0).expand(len(rows), -1, -1).clone()
        return self.messages[[self.indices[row['sample_id']] for row in rows]]


def collate(rows, tokenizer, store, device, supervised=True):
    """Exactly the frozen text/label collation, replacing only residue memory."""
    if not rows:
        raise ValueError('Cannot collate an empty batch')
    ids, targets = [], []
    for row in rows:
        prefix = tokenizer.encode(legacy.prompt_text(row, tokenizer), add_special_tokens=False)
        suffix = tokenizer.encode(legacy.answer_text(row), add_special_tokens=False) + [tokenizer.eos_token_id] if supervised else []
        if not prefix or (supervised and len(suffix) < 2):
            raise ValueError('Empty prompt or answer tokenization')
        ids.append(torch.tensor(prefix + suffix, dtype=torch.long))
        targets.append(torch.tensor([-100] * len(prefix) + suffix, dtype=torch.long))
    max_length = max(map(len, ids))
    input_ids = torch.full((len(rows), max_length), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    attention = torch.zeros_like(input_ids)
    for i, (tokens, answer) in enumerate(zip(ids, targets)):
        input_ids[i, -len(tokens):] = tokens
        labels[i, -len(tokens):] = answer
        attention[i, -len(tokens):] = 1
    memory = store.batch(rows)
    if memory.shape != (len(rows), 16, 128) or memory.dtype != torch.float32 or not torch.isfinite(memory).all():
        raise ValueError('Cache must supply finite float32 [B,16,128]')
    result = {'input_ids': input_ids, 'attention_mask': attention, 'memory': memory.detach(),
              'memory_mask': torch.ones((len(rows), 16), dtype=torch.bool)}
    if supervised:
        result['labels'] = labels
    return {key: value.to(device) for key, value in result.items()}


def validation_nll(model, rows, tokenizer, store, device, batch_size, check=lambda: None,
                   diagnostic_sink=None):
    model.eval()
    total, count = 0., 0
    breakdowns = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            check()
            selected = rows[start:start + batch_size]
            batch = collate(selected, tokenizer, store, device)
            output = model(**batch)
            summed, n = legacy.supervised_nll(output, batch['labels'])
            total += float(summed)
            count += n
            if diagnostic_sink is not None:
                categories = batch_token_categories(tokenizer, [legacy.answer_text(r) for r in selected],
                    batch['input_ids'], batch['labels'], batch['attention_mask'])
                value = token_nll_breakdown(output, batch['labels'], categories)
                if value['reference_supervised_nll']['loss_sum'] != float(summed) or value['reference_supervised_nll']['count'] != n:
                    raise ValueError('Diagnostic reference differs from unchanged validation criterion')
                breakdowns.append(value)
    if not count:
        raise ValueError('Full validation contains no answer tokens')
    if diagnostic_sink is not None:
        diagnostic_sink.update(aggregate_token_breakdowns(breakdowns))
    return total / count


def test_c1_diagnostics(model, rows, tokenizer, store, args, check=lambda: None):
    """Best-checkpoint teacher-forced C1; original row-weighted definition is frozen."""
    if not rows or any(row['split'] != 'test' for row in rows):
        raise ValueError('Formal C1 diagnosis requires the full locked test rows')
    model.eval()
    records, breakdowns = [], {'correct': [], 'off': []}
    with (args.output / 'test_C1_answer_nll.jsonl').open('x', encoding='utf-8') as handle:
        with torch.no_grad():
            for start in range(0, len(rows), 4):
                check()
                selected = rows[start:start + 4]
                batch = collate(selected, tokenizer, store, args.device)
                labels = batch.pop('labels')
                categories = batch_token_categories(tokenizer, [legacy.answer_text(r) for r in selected],
                    batch['input_ids'], labels, batch['attention_mask'])
                values = {}
                for mode in ('correct', 'off'):
                    output = model(**batch, message_mode=mode)
                    means, counts = frozen_c1_answer_nll(output, labels)
                    diagnostic = token_nll_breakdown(output, labels, categories)
                    values[mode] = {'nll': means.tolist(),
                                    'counts': counts.tolist(),
                                    'breakdown': diagnostic['per_sample']}
                    breakdowns[mode].append(diagnostic)
                    del output
                for index, row in enumerate(selected):
                    record = {'sample_id': row['sample_id'], 'answer_tokens': int(values['correct']['counts'][index]),
                              'correct_original_gold_mean_token_nll': float(values['correct']['nll'][index]),
                              'off_original_gold_mean_token_nll': float(values['off']['nll'][index]),
                              'token_breakdown': {mode: values[mode]['breakdown'][index] for mode in ('correct', 'off')}}
                    if values['correct']['counts'][index] != values['off']['counts'][index]:
                        raise ValueError('C1 correct/off label denominator differs')
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    means = {key: math.fsum(r[key] for r in records) / len(records)
             for key in ('correct_original_gold_mean_token_nll', 'off_original_gold_mean_token_nll')}
    gain = math.fsum(r['off_original_gold_mean_token_nll'] - r['correct_original_gold_mean_token_nll'] for r in records) / len(records)
    decomposition = {name: math.fsum(
        r['token_breakdown']['off'][name]['contribution_nats_per_full_token']
        - r['token_breakdown']['correct'][name]['contribution_nats_per_full_token'] for r in records) / len(records)
        for name in TOKEN_BUCKETS[:3]}
    decomposition.update({'sum_categories': sum(decomposition.values()), 'original_G': gain})
    decomposition['reduction_rounding_residual'] = gain - decomposition['sum_categories']
    return {'n_test': len(records), 'n_matched': 0, 'G_all_test': gain,
            'P_H_status': 'undefined_zero_independent_donors', 'algebra_matched_mean': None,
            'means_all_test': means, 'units': 'nats_per_answer_token',
            'score_definition': 'Original per-sample mean answer-token NLL, then mean across test rows; G=off NLL-correct NLL',
            'token_breakdown': {mode: aggregate_token_breakdowns(values) for mode, values in breakdowns.items()},
            'G_decomposition': decomposition,
            'decomposition_denominator': 'Each category loss contribution divides by that sample full-answer token count, then averages rows; category means themselves are not additive',
            'scope': 'Best-checkpoint teacher-forced computational sensitivity, separate from autonomous generation and biological causality',
            'selection_or_primary_score_changed': False}


def generated_predictions(model, rows, tokenizer, store, args, handle, check=lambda: None):
    model.eval()
    records = []
    for row in rows:
        check()
        batch = collate([row], tokenizer, store, args.device, supervised=False)
        started = time.perf_counter()
        with torch.no_grad():
            tokens = model.greedy_generate(**batch, max_new_tokens=24,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                message_mode='correct')
        ids = tokens[0].cpu().tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True)
        record = {'sample_id': row['sample_id'], 'split': row['split'], 'raw_text': text,
                  'parse': legacy.parse_answer(text), 'gold_pkd': float(row['pkd']),
                  'raw_generated_token_ids': ids, 'generated_token_count': len(ids),
                  'prompt_token_count': int(batch['attention_mask'].sum()),
                  'elapsed_seconds': time.perf_counter() - started,
                  'channel': 'generation', 'message_input': args.message_input}
        records.append(record)
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
    return records


def train_and_evaluate(model, tokenizer, splits, store, args, registration, log, check=lambda: None):
    """Injectable training core; CPU fixtures are declared outside real contracts."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError('Empty reader optimizer')
    legacy.assert_frozen(model.base_lm)
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=.01)
    best, best_epoch, steps, history = math.inf, None, 0, []
    checkpoint = args.output / 'best_reader.pt'
    started = time.perf_counter()
    phase_diagnostics = []
    for epoch in range(1, 4):
        model.train()
        model.diagnostics(reset=True)
        order = list(range(len(splits['train'])))
        random.Random(args.seed + epoch).shuffle(order)
        train_sum, train_tokens = 0., 0
        for start in range(0, len(order), 4):
            check()
            selected = [splits['train'][i] for i in order[start:start + 4]]
            batch = collate(selected, tokenizer, store, args.device)
            optimizer.zero_grad(set_to_none=True)
            summed, count = legacy.supervised_nll(model(**batch), batch['labels'])
            (summed / count).backward()
            gradients = [p.grad for p in parameters if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                raise ValueError('Missing or nonfinite reader gradients')
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            legacy.assert_frozen(model.base_lm)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in parameters):
                raise ValueError('Nonfinite reader state after update')
            steps += 1
            train_sum += float(summed.detach())
            train_tokens += count
            entry = {'event': 'optimizer_step', 'epoch': epoch, 'step': steps,
                     'answer_token_nll': float(summed.detach()) / count, 'gradient_norm_before_clip': float(norm),
                     'alpha': float(model.reader.alpha.detach()),
                     'gate_tanh_alpha': float(model.reader.alpha.detach().tanh())}
            if steps == 1 or steps % 50 == 0 or start + 4 >= len(order):
                entry['cumulative_train_reader_diagnostics'] = model.diagnostics()
            log(entry)
            if steps % 100 == 0:
                legacy.save_checkpoint(args.output / 'latest_partial_reader.pt',
                    {'reader_state': model.reader_state(), 'epoch': epoch, 'optimizer_steps': steps,
                     'registration': registration, 'selection_eligible': False})
        train_diagnostic = model.diagnostics(reset=True)
        token_diagnostic = {}
        score = validation_nll(model, splits['validation'], tokenizer, store, args.device, 4, check,
                               diagnostic_sink=token_diagnostic)
        validation_diagnostic = model.diagnostics(reset=True)
        if not math.isfinite(score):
            raise ValueError('Nonfinite full validation NLL')
        if score < best:
            best, best_epoch = score, epoch
            legacy.save_checkpoint(checkpoint, {'reader_state': model.reader_state(), 'best_epoch': epoch,
                'validation_answer_token_nll': best, 'registration': registration,
                'normalization': 'none; unchanged two-decimal text serialization; original pKD scoring'})
        item = {'epoch': epoch, 'train_answer_token_nll': train_sum / train_tokens,
                'validation_answer_token_nll': score, 'best_epoch': best_epoch, 'optimizer_steps': steps,
                'train_reader_diagnostics': train_diagnostic, 'validation_reader_diagnostics': validation_diagnostic,
                'validation_token_nll_breakdown': token_diagnostic}
        history.append(item)
        phase_diagnostics.append({'epoch': epoch, 'train': train_diagnostic, 'validation': validation_diagnostic})
        legacy.write_json(args.output / 'history.json', history)
        log({'event': 'epoch_complete', **item})
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    model.load_reader_state(saved['reader_state'])
    legacy.assert_frozen(model.base_lm)
    legacy.write_json(args.output / 'selection_before_test.json', {'best_epoch': best_epoch,
        'validation_answer_token_nll': best, 'checkpoint_sha256': file_sha(checkpoint),
        'all_training_epochs_completed': 3, 'optimizer_steps': steps,
        'selection': 'minimum full validation answer-token NLL; earliest tie',
        'test_labels_used_for_selection': False})
    model.diagnostics(reset=True)
    c1 = test_c1_diagnostics(model, splits['test'], tokenizer, store, args, check)
    c1_reader_diagnostics = model.diagnostics(reset=True)
    with (args.output / 'test_predictions_correct.jsonl').open('x', encoding='utf-8') as handle:
        predictions = generated_predictions(model, splits['test'], tokenizer, store, args, handle, check)
    metrics = legacy.generation_metrics(splits['test'], predictions)
    inference_diagnostics = model.diagnostics(reset=True)
    return {'status': 'complete', 'actual_epochs': 3, 'optimizer_steps': steps, 'best_epoch': best_epoch,
            'best_validation_answer_token_nll': best,
            'selection': 'minimum full validation answer-token NLL; earliest tie',
            'checkpoint_path': str(checkpoint), 'checkpoint_sha256': file_sha(checkpoint),
            'test_generation': {'correct': metrics}, 'test_evaluation_passes': 1,
            'test_evaluation_passes_meaning': 'Autonomous correct-generation pass only; C1 teacher-forcing is a separate two-forward diagnostic',
            'test_generation_passes': 1, 'test_teacher_forced_C1_passes': {'correct': 1, 'off': 1},
            'C1': c1, 'C1_reader_diagnostics': c1_reader_diagnostics,
            'train_validation_reader_diagnostics': phase_diagnostics, 'test_reader_diagnostics': inference_diagnostics,
            'actual_generated_token_count': sum(x['generated_token_count'] for x in predictions),
            'training_and_generation_seconds': time.perf_counter() - started,
            'backbone_gradients': 'none_verified', 'synthetic_results_counted': False,
            'formal_multitask_connector_result': False,
            'additional_token_format_numeric_NLL': 'Same-forward validation and separate frozen-definition test C1 diagnostics only; never used for checkpoint selection or primary metrics'}


def train_only_canary(model, tokenizer, train_rows, store, args, registration, log, check=lambda: None):
    """One TRAIN update and TRAIN generation/scoring; never opens test predictions."""
    if not train_rows or any(row['split'] != 'train' for row in train_rows):
        raise ValueError('Canary accepts TRAIN rows only')
    # Fixed before any outcome: cover the longest allowed prompt among TRAIN.
    selected = sorted(train_rows, key=lambda row: (
        -len(tokenizer.encode(legacy.prompt_text(row, tokenizer), add_special_tokens=False)), row['sample_id']))[:4]
    check()
    model.train()
    model.diagnostics(reset=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=.01)
    batch = collate(selected, tokenizer, store, args.device)
    optimizer.zero_grad(set_to_none=True)
    output = model(**batch)
    summed, count = legacy.supervised_nll(output, batch['labels'])
    categories = batch_token_categories(tokenizer, [legacy.answer_text(r) for r in selected],
        batch['input_ids'], batch['labels'], batch['attention_mask'])
    canary_tokens = token_nll_breakdown(output, batch['labels'], categories)
    if canary_tokens['reference_supervised_nll']['loss_sum'] != float(summed.detach()):
        raise ValueError('TRAIN canary token diagnostic changes reference loss')
    (summed / count).backward()
    gradients = [p.grad for p in trainable if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise ValueError('Canary missing/nonfinite gradients')
    alpha_gradient = float(model.reader.alpha.grad) if model.reader.alpha.grad is not None else None
    norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
    legacy.assert_frozen(model.base_lm)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in trainable):
        raise ValueError('Canary nonfinite parameters')
    train_diagnostics = model.diagnostics(reset=True)
    del output
    model.eval()
    with torch.no_grad():
        off_output = model(**batch, message_mode='off')
        off_sum, off_count = legacy.supervised_nll(off_output, batch['labels'])
        canary_off_tokens = token_nll_breakdown(off_output, batch['labels'], categories)
    if not math.isfinite(float(off_sum)) or off_count != count:
        raise ValueError('TRAIN canary frozen OFF path failed')
    del off_output
    legacy.save_checkpoint(args.output / 'canary_reader_not_for_training.pt',
        {'reader_state': model.reader_state(), 'registration': registration,
         'initialization_or_selection_eligible': False, 'purpose': 'TRAIN canary only'})
    with (args.output / 'canary_train_predictions.jsonl').open('x', encoding='utf-8') as handle:
        predictions = generated_predictions(model, selected, tokenizer, store, args, handle, check)
    metrics = legacy.generation_metrics(selected, predictions)
    result = {'status': 'passed', 'purpose': 'TRAIN-only real forward/backward and generation/scoring serialization',
              'scientific_trial_count': 0, 'optimizer_steps': 1, 'sample_ids': [r['sample_id'] for r in selected],
              'sample_split': 'train', 'selection': 'longest TRAIN prompts, sample ID tie-breaker, first four',
              'answer_token_nll': float(summed.detach()) / count, 'gradient_norm_before_clip': float(norm),
              'alpha_gradient_before_update': alpha_gradient, 'train_diagnostics': train_diagnostics,
              'train_token_nll_breakdown_before_update': canary_tokens,
              'train_off_token_nll_breakdown_after_update': canary_off_tokens,
              'canary_correct_off_comparison_claim': False,
              'train_generation_diagnostics': model.diagnostics(reset=True), 'train_generation_metrics': metrics,
              'test_examples_evaluated': 0, 'backbone_gradients': 'none_verified',
              'scope': 'canary_only_not_a_scientific_benchmark_result'}
    log({'event': 'canary_complete', 'answer_token_nll': result['answer_token_nll'],
         'n_train': len(selected), 'test_examples_evaluated': 0})
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', required=True, choices=SCENARIOS)
    p.add_argument('--sender-mode', choices=('balanced', 'global16'), default='balanced')
    p.add_argument('--message-input', choices=('real', 'neutral'), default='real')
    p.add_argument('--placement', choices=('early', 'mid'), required=True)
    p.add_argument('--gate-init', choices=('zero', 'near_unit'), default='zero')
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--cache-manifest', type=Path, required=True)
    p.add_argument('--cache-manifest-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--timeout-seconds', type=int, default=1800)
    p.add_argument('--protocol', type=Path, default=ROOT / 'configs/matched_budget_20261016.frozen.json')
    p.add_argument('--protocol-sha256', required=True)
    p.add_argument('--canary', action='store_true', help='TRAIN-only one-update transport/scoring canary; no test inference')
    return p


def validate_trial_configuration(args, protocol):
    """Admit only the prospectively enumerated 36-case matrix, never test-selected branches."""
    if args.scenario not in SCENARIOS or args.seed not in SEEDS:
        raise ValueError('Unregistered scenario or training seed')
    primary = args.sender_mode == 'balanced' and args.gate_init == 'zero'
    gate_extension = (args.placement == 'mid' and args.message_input == 'real'
                      and args.sender_mode == 'balanced' and args.gate_init == 'near_unit')
    pool_extension = (args.placement == 'mid' and args.message_input == 'real'
                      and args.sender_mode == 'global16' and args.gate_init == 'zero')
    if args.placement not in ('early', 'mid') or args.message_input not in ('real', 'neutral') or not (
            primary or gate_extension or pool_extension):
        raise ValueError('Trial is outside the prospective 36-case matrix')
    expected = {'epochs': 3, 'batch_size': 4, 'learning_rate': .0003,
                'weight_decay': .01, 'max_grad_norm': 1., 'optimizer_steps': 1548}
    if any(protocol.get('training', {}).get(k) != value for k, value in expected.items()):
        raise ValueError('Protocol changed the fixed training budget')
    if protocol.get('generation', {}).get('max_new_tokens') != 24:
        raise ValueError('Protocol changed the frozen decoding limit')
    if protocol.get('timing', {}).get('effective_deadline_at') != '2026-10-16T23:20:00+08:00':
        raise ValueError('New-scope absolute deadline differs')


def main():
    args = parser().parse_args()
    if args.threads not in (1, 2) or not 1 <= args.timeout_seconds <= 1800:
        raise ValueError('Parent-authorized trial permits at most two threads and 1800 seconds')
    verify_legacy_source()
    if file_sha(args.protocol) != args.protocol_sha256:
        raise ValueError('Prospective protocol SHA differs')
    if args.protocol_sha256 != FROZEN_PROTOCOL_SHA256:
        raise ValueError('Only the admitted immutable final protocol is executable')
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    if protocol.get('status') not in ('frozen', 'approved_for_execution', 'frozen_for_execution'):
        raise ValueError('Protocol is not frozen/approved for execution')
    validate_trial_configuration(args, protocol)
    numerical_receipt = configure_numerical_policy(require_cuda_environment=args.device == 'cuda')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    def check():
        if time.perf_counter() - started > args.timeout_seconds:
            raise TimeoutError('Trial wall-clock budget exhausted')
        if datetime.now().astimezone() >= datetime.fromisoformat('2026-10-16T23:20:00+08:00'):
            raise TimeoutError('Absolute matched-budget scope deadline exhausted')

    process = legacy.process_record(args)
    legacy.write_json(args.output / 'process.json', process)
    with (args.output / 'run.jsonl').open('x', encoding='utf-8', buffering=1) as log_file:
        def log(value):
            record = {'utc': legacy.utc_now(), **value}
            line = json.dumps(record, ensure_ascii=False, allow_nan=False)
            log_file.write(line + '\n')
            print(line, flush=True)
        try:
            store = MessageStore(args.cache_manifest, args.cache_manifest_sha256,
                                 args.scenario, args.sender_mode, args.message_input)
            data = store.manifest['datasets'][args.scenario]
            rows, gate = validate_panel_data(ROOT / data['samples_path'], ROOT / data['lock_path'], 'A1')
            if gate['split_counts'] != EXPECTED_COUNTS[args.scenario] or gate['samples_sha256'] != store.samples_sha256:
                raise ValueError('Full frozen scenario identity mismatch')
            splits, selection = legacy.select_splits(rows, {})
            if set(store.indices) != {r['sample_id'] for r in rows}:
                raise ValueError('Fixed messages do not cover exact full scenario')
            for split, selected in splits.items():
                with (args.output / f'{split}_input_snapshot.jsonl').open('x', encoding='utf-8') as handle:
                    for row in selected:
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            model_receipt = verify_local_models()
            check()
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from matched_budget_20261016.connectors import MatchedConnectorLM
            model_path = ROOT / 'models/Qwen--Qwen2.5-1.5B-Instruct'
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
            base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                dtype=torch.bfloat16,
                attn_implementation='sdpa').to(args.device)
            model = MatchedConnectorLM(base, placement=args.placement, gate_init=args.gate_init).to(args.device)
            initial = store.manifest['initializations'][f'{args.seed}:{args.gate_init}']
            initial_path = ROOT / initial['path']
            if file_sha(initial_path) != initial['sha256']:
                raise ValueError('Shared untrained reader initialization changed')
            initial_payload = torch.load(initial_path, weights_only=True, map_location='cpu')
            if (initial_payload['seed'] != args.seed or initial_payload['gate_init'] != args.gate_init
                    or initial_payload.get('untrained') is not True or initial_payload.get('source_labels_used') is not False):
                raise ValueError('Initialization provenance is not fresh untrained state')
            model.load_reader_state(initial_payload['reader_state'])
            numerical_receipt = verify_model_numerics(model)
            if model.capacity_report()['trainable_parameters'] != 464001:
                raise ValueError('Matched reader parameter count differs')
            legacy.assert_frozen(base)
            max_text = max(len(tokenizer.encode(legacy.prompt_text(r, tokenizer), add_special_tokens=False))
                           + len(tokenizer.encode(legacy.answer_text(r), add_special_tokens=False)) + 1
                           for split in ('train', 'validation') for r in splits[split])
            max_prompt = max(len(tokenizer.encode(legacy.prompt_text(r, tokenizer), add_special_tokens=False)) for r in rows)
            if max(max_text, max_prompt + 24) > base.config.max_position_embeddings:
                raise ValueError('Unchanged prompt exceeds model context; no truncation permitted')
            registration = {'schema_version': 1, 'benchmark_version': 'matched_budget_20261016',
                'panel': 'A1', 'role': 'matched_fixed_message_generation', 'scope': 'full_locked_A1_panel',
                'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'capacity': model.capacity_report(), 'communication_at_max_training_text': model.communication_report(4, max_text),
                'source_pins': store.manifest['source_pins'], 'cache_manifest_sha256': args.cache_manifest_sha256,
                'initialization': initial, 'model_receipt': model_receipt, 'data_gate': gate,
                'split_selection': selection, 'legacy_prompt_answer_parser_metric_source_sha256': LEGACY_SHA256,
                'training': {'epochs': 3, 'batch_size': 4, 'learning_rate': .0003, 'weight_decay': .01,
                             'max_grad_norm': 1., 'expected_optimizer_steps': 1548},
                'selection': 'minimum full validation answer-token NLL, earliest tie; no early stopping',
                'generation': {'max_new_tokens': 24, 'sampling': False, 'KV_cache': False, 'strict_parser_unchanged': True},
                'test_evaluation_plan': {'autonomous_correct_generation_passes': 1,
                                         'teacher_forced_C1_passes': {'correct': 1, 'off': 1},
                                         'test_used_for_checkpoint_selection': False},
                'source_message_shape': [16, 128], 'source_message_dtype': 'float32',
                'scope_started_at': '2026-10-16T13:20:00+08:00', 'scope_deadline_at': '2026-10-16T23:20:00+08:00',
                'historical_contracts_rerun': False, 'test_already_observed': True,
                'protocol_sha256': args.protocol_sha256, 'runner_sha256': file_sha(__file__),
                'numerical_policy': numerical_receipt,
                'formal_multitask_connector_result': False}
            legacy.write_json(args.output / 'run_preregistration.json', registration)
            log({'event': 'training_start', 'capacity': registration['capacity']})
            result = (train_only_canary(model, tokenizer, splits['train'], store, args, registration, log, check)
                      if args.canary else train_and_evaluate(model, tokenizer, splits, store, args, registration, log, check))
            if not args.canary and (result['optimizer_steps'] != 1548 or result['actual_epochs'] != 3):
                raise ValueError('Full training budget not completed')
            verify_pins(store.manifest['source_pins'])
            if file_sha(args.cache_manifest) != args.cache_manifest_sha256 or file_sha(args.protocol) != args.protocol_sha256:
                raise ValueError('Executing cache/protocol changed mid-run')
            if file_sha(store.payload_path) != data['layouts'][args.sender_mode]['sha256']:
                raise ValueError('Message payload changed mid-run')
            result.update({'registration': registration, 'process': process, 'finished_utc': legacy.utc_now(),
                           'numerical_policy': verify_model_numerics(model),
                           'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0,
                           'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved() if args.device == 'cuda' else 0})
            log({'event': 'complete', 'best_epoch': result.get('best_epoch'),
                 'n_test': 0 if args.canary else len(splits['test'])})
            log_file.flush()
            result['artifact_sha256'] = {p.name: file_sha(p) for p in args.output.iterdir() if p.is_file()}
            legacy.write_json(args.output / 'result.json', result)
        except BaseException as error:
            legacy.write_json(args.output / 'failure.json', {'status': 'failed', 'utc': legacy.utc_now(),
                'error_type': type(error).__name__, 'error': str(error), 'process': process})
            raise


if __name__ == '__main__':
    main()
