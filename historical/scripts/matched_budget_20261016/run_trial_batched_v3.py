"""V3 final batch2-only revision of the fixed matched-message A1 trial.

This command never edits historical data/results or chooses its own matrix.
The parent supervisor admits trials and owns resource/contract supervision.
Only fixed-message lookup differs from the frozen foundation collation: text,
answer labels, validation criterion, strict parser and main scoring are reused.
"""
import argparse
import ast
from datetime import datetime
import hashlib
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
V1_RUNNER_SHA256 = 'dca9c255e5403af0207390f8a247403aa9728fba50034b85dec108656d877c2c'
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
    from matched_budget_20261016.batched_generation_v3 import generated_predictions as batched
    return batched(model, rows, tokenizer, store, args, handle, check, batch_size=2)


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
    from matched_budget_20261016.batched_generation_v3 import train_only_canary as qualify
    return qualify(model, tokenizer, train_rows, store, args, registration, log, check)


def execution_revision():
    original = Path(__file__).with_name('run_trial.py')
    if file_sha(original) != V1_RUNNER_SHA256:
        raise ValueError('Immutable V1 runner changed')
    def training_bytes(path):
        text = path.read_text(encoding='utf-8')
        node = next(x for x in ast.parse(text).body if isinstance(x, ast.FunctionDef) and x.name == 'train_and_evaluate')
        return ''.join(text.splitlines(keepends=True)[node.lineno - 1:node.end_lineno]).encode('utf-8')
    old, new = training_bytes(original), training_bytes(Path(__file__))
    if old != new:
        raise ValueError('V3 training function is not byte-identical to V1')
    files = [Path(__file__), Path(__file__).with_name('batched_generation_v3.py'), original,
             ROOT / 'reports/benchmark_clarity_20261016/engineering_addendum_v3.json']
    return {'id': 'fixed_order_generation_batch2_v3', 'generation_batch_size': 2,
            'source_pins': {relative(path): file_sha(path) for path in files},
            'training_function_verbatim_V1_sha256': hashlib.sha256(new).hexdigest(),
            'scientific_protocol_sha256_unchanged': FROZEN_PROTOCOL_SHA256,
            'original_V1_evaluations_preserved': True, 'original_V2_evaluations_preserved': True,
            'last_throughput_candidate_no_fallback': True,
            'reason': 'Throughput qualification only; no source checkpoint or metric chosen from test scores'}


def validate_complete_v1(source, args, registration, store, require_project_root=True):
    """Admit complete, accepted V1 training by provenance, never by test score."""
    candidate = Path(source).resolve()
    job = candidate.parent if candidate.name == 'artifacts' else candidate
    if require_project_root and not job.is_relative_to((ROOT / 'results/matched_budget_20261016').resolve()):
        raise ValueError('Reuse source is outside the registered matched-budget results root')
    names = ['execution-receipt.json', 'result-record.json', 'result-validation.json', 'task-contract.json',
             'artifacts/result.json', 'artifacts/history.json', 'artifacts/selection_before_test.json',
             'artifacts/best_reader.pt', 'artifacts/run_preregistration.json']
    if any(not (job / name).is_file() for name in names):
        raise ValueError('V1 source lacks complete physical training/acceptance evidence')
    load = lambda name: json.loads((job / name).read_text(encoding='utf-8'))
    receipt, standard, accepted, contract = [load(name) for name in names[:4]]
    if (receipt.get('performed') is not True or receipt.get('exit_code') != 0
            or receipt.get('process_group_absent') is not True or receipt.get('termination_reason') is not None
            or accepted.get('valid_for_acceptance') is not True
            or standard.get('is_template') is not False or standard.get('lifecycle_status') != 'succeeded'
            or standard.get('review', {}).get('status') != 'accepted'
            or standard.get('reproduction_level') != 'training'
            or standard.get('execution', {}).get('performed') is not True
            or standard.get('execution', {}).get('exit_code') != 0
            or standard.get('task_id') != contract.get('task_id')
            or standard.get('run_id') != contract.get('run_id')
            or standard.get('contract_sha256') != file_sha(job / 'task-contract.json')
            or contract.get('matched_budget_registration', {}).get('case', {}).get('phase') != 'trial'):
        raise ValueError('Source is not a complete exited accepted V1 training trial')
    source_result = load('artifacts/result.json')
    original = source_result.get('registration', {})
    if (source_result.get('status') != 'complete' or source_result.get('optimizer_steps') != 1548
            or source_result.get('actual_epochs') != 3 or original.get('runner_sha256') != V1_RUNNER_SHA256
            or original.get('capacity', {}).get('trainable_parameters') != 464001
            or original.get('protocol_sha256') != FROZEN_PROTOCOL_SHA256
            or original != load('artifacts/run_preregistration.json')):
        raise ValueError('Source is not the exact immutable V1 training implementation/budget')
    for key in ('scenario', 'seed', 'placement', 'message_input', 'sender_mode', 'gate_init'):
        if original.get('configuration', {}).get(key) != getattr(args, key):
            raise ValueError(f'Source training configuration mismatch: {key}')
    if original.get('data_gate', {}).get('samples_sha256') != store.samples_sha256:
        raise ValueError('Source training used different frozen sample rows')
    for key in ('samples_sha256', 'benchmark_lock_sha256'):
        if original.get('data_gate', {}).get(key) != registration['data_gate'][key]:
            raise ValueError('Source training lock differs from this admitted evaluation')
    verify_pins(original['source_pins'])
    initial = original.get('initialization', {})
    if initial.get('tensor_sha256') != registration['initialization'].get('tensor_sha256'):
        raise ValueError('Source and V3 shared seed initialization differ')
    # The new cache may append source-code pins. Compare the actual fixed-message
    # payloads instead of demanding identical administrative manifest bytes.
    old_manifest_path = Path(original['configuration']['cache_manifest'])
    if file_sha(old_manifest_path) != original['cache_manifest_sha256']:
        raise ValueError('Original V1 cache manifest changed')
    old_manifest = json.loads(old_manifest_path.read_text(encoding='utf-8'))
    if old_manifest['projection']['tensor_sha256'] != store.manifest['projection']['tensor_sha256']:
        raise ValueError('V3 projection differs from source training')
    layout = old_manifest['datasets'][args.scenario]['layouts'][args.sender_mode]
    old_payload_path = ROOT / layout['path']
    if file_sha(old_payload_path) != layout['sha256']:
        raise ValueError('Original V1 fixed messages changed')
    payload = torch.load(old_payload_path, map_location='cpu', weights_only=True)
    if ({key: index for index, key in enumerate(payload['sample_ids'])} != store.indices
            or dict(zip(payload['sample_ids'], payload['splits'])) != store.splits
            or not torch.equal(payload['messages'], store.messages)
            or not torch.equal(payload['neutral'], store.neutral)):
        raise ValueError('V3 source content/neutral/order differs from original training')
    history = load('artifacts/history.json')
    if ([x.get('epoch') for x in history] != [1, 2, 3]
            or [x.get('optimizer_steps') for x in history] != [516, 1032, 1548]
            or any(not math.isfinite(x['validation_answer_token_nll']) for x in history)):
        raise ValueError('V1 full three-epoch training history is incomplete')
    selected = min(history, key=lambda x: x['validation_answer_token_nll'])
    selection = load('artifacts/selection_before_test.json')
    checkpoint = job / 'artifacts/best_reader.pt'
    if (source_result.get('best_epoch') != selected['epoch']
            or source_result.get('best_validation_answer_token_nll') != selected['validation_answer_token_nll']
            or selection.get('best_epoch') != selected['epoch']
            or selection.get('validation_answer_token_nll') != selected['validation_answer_token_nll']
            or selection.get('test_labels_used_for_selection') is not False
            or selection.get('checkpoint_sha256') != file_sha(checkpoint)
            or source_result.get('checkpoint_sha256') != file_sha(checkpoint)):
        raise ValueError('Source checkpoint is not the earliest validation-selected full-budget checkpoint')
    checkpoint_payload = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if (checkpoint_payload.get('best_epoch') != selected['epoch']
            or checkpoint_payload.get('validation_answer_token_nll') != selected['validation_answer_token_nll']
            or checkpoint_payload.get('registration') != original):
        raise ValueError('Source checkpoint metadata differs from validation selection')
    state = checkpoint_payload.get('reader_state', {})
    if (sum(value.numel() for value in state.values()) != 464001
            or any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values())):
        raise ValueError('Source reader checkpoint has the wrong parameter budget/dtype or nonfinite values')
    archived = standard.get('artifacts', [])
    required_archive_paths = {(job / name).resolve() for name in names
                              if name not in ('result-record.json', 'result-validation.json')}
    if not required_archive_paths.issubset({Path(item['path']).resolve() for item in archived}):
        raise ValueError('Source standard archive does not bind all required training evidence')
    for item in archived:
        path = Path(item['path']).resolve()
        if not path.is_relative_to(job) or not path.is_file() or file_sha(path) != item['sha256']:
            raise ValueError('A source standard archived artifact is missing/changed')
    evidence = {str(job / name): file_sha(job / name) for name in names}
    evidence[str(old_manifest_path)] = file_sha(old_manifest_path)
    evidence[str(old_payload_path)] = file_sha(old_payload_path)
    return {'job': job, 'result': source_result, 'history': history,
            'checkpoint': checkpoint, 'checkpoint_payload': checkpoint_payload,
            'evidence_sha256': evidence, 'test_score_used_for_admission': False}


def evaluate_reused_v1(model, tokenizer, splits, store, args, registration, log, check=lambda: None):
    source = validate_complete_v1(args.reuse_complete_v1, args, registration, store)
    model.load_reader_state(source['checkpoint_payload']['reader_state'])
    legacy.assert_frozen(model.base_lm)
    for filename in ('history.json', 'best_reader.pt'):
        data = (source['job'] / 'artifacts' / filename).read_bytes()
        with (args.output / filename).open('xb') as handle:
            handle.write(data)
    original = source['result']
    checkpoint = args.output / 'best_reader.pt'
    legacy.write_json(args.output / 'selection_before_test.json', {
        'best_epoch': original['best_epoch'], 'validation_answer_token_nll': original['best_validation_answer_token_nll'],
        'checkpoint_sha256': file_sha(checkpoint), 'all_training_epochs_completed': 3, 'optimizer_steps': 1548,
        'training_performed_in_this_run': False, 'test_labels_used_for_selection': False,
        'selection': 'Unchanged V1 earliest minimum validation NLL; no V3 selection or optimization',
        'training_source_evidence_sha256': source['evidence_sha256']})
    started = time.perf_counter()
    model.diagnostics(reset=True)
    c1 = test_c1_diagnostics(model, splits['test'], tokenizer, store, args, check)
    c1_diagnostics = model.diagnostics(reset=True)
    with (args.output / 'test_predictions_correct.jsonl').open('x', encoding='utf-8') as handle:
        predictions = generated_predictions(model, splits['test'], tokenizer, store, args, handle, check)
    metrics = legacy.generation_metrics(splits['test'], predictions)
    return {'status': 'complete', 'actual_epochs': 3, 'optimizer_steps': 1548,
            'new_optimizer_steps': 0, 'performed_new_optimizer_steps': 0, 'training_performed_in_this_run': False,
            'training_reused_from_v1': True, 'training_budget_original': {'epochs': 3, 'optimizer_steps': 1548},
            'training_source_evidence_sha256': source['evidence_sha256'], 'training_reused_source': str(source['job']),
            'source_admission_test_scores_used': False, 'best_epoch': original['best_epoch'],
            'best_validation_answer_token_nll': original['best_validation_answer_token_nll'],
            'selection': original['selection'], 'checkpoint_path': str(checkpoint), 'checkpoint_sha256': file_sha(checkpoint),
            'test_generation': {'correct': metrics}, 'test_evaluation_passes': 1, 'test_generation_passes': 1,
            'test_evaluation_passes_meaning': 'New V3 batch2 autonomous generation only; C1 is separate teacher forcing',
            'test_teacher_forced_C1_passes': {'correct': 1, 'off': 1}, 'C1': c1, 'C1_reader_diagnostics': c1_diagnostics,
            'test_reader_diagnostics': model.diagnostics(reset=True),
            'train_validation_reader_diagnostics': [
                {'epoch': x['epoch'], 'train': x['train_reader_diagnostics'], 'validation': x['validation_reader_diagnostics'],
                 'origin': 'V1 source training history; not recomputed in this inference-only run'} for x in source['history']],
            'actual_generated_token_count': sum(x['generated_token_count'] for x in predictions),
            'training_and_generation_seconds': time.perf_counter() - started,
            'elapsed_scope': 'This V3 inference-only run; original training time remains in V1 records',
            'backbone_gradients': 'none_verified', 'synthetic_results_counted': False,
            'formal_multitask_connector_result': False}


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
    p.add_argument('--reuse-complete-v1', type=Path, help='Only an exited, accepted full V1 training job; perform no new optimization')
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
    revision = execution_revision()
    if args.canary and args.reuse_complete_v1 is not None:
        raise ValueError('Canary qualification cannot reuse a trained scientific checkpoint')
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
                'execution_revision': revision,
                'generation_batch_size': 2,
                'training_reuse_requested': str(args.reuse_complete_v1) if args.reuse_complete_v1 else None,
                'formal_multitask_connector_result': False}
            legacy.write_json(args.output / 'run_preregistration.json', registration)
            log({'event': 'evaluation_start' if args.reuse_complete_v1 else 'training_start', 'capacity': registration['capacity']})
            result = (train_only_canary(model, tokenizer, splits['train'], store, args, registration, log, check)
                      if args.canary else evaluate_reused_v1(model, tokenizer, splits, store, args, registration, log, check)
                      if args.reuse_complete_v1 else train_and_evaluate(model, tokenizer, splits, store, args, registration, log, check))
            if not args.canary and (result['optimizer_steps'] != 1548 or result['actual_epochs'] != 3):
                raise ValueError('Full training budget not completed')
            verify_pins(store.manifest['source_pins'])
            verify_pins(revision['source_pins'])
            if file_sha(args.cache_manifest) != args.cache_manifest_sha256 or file_sha(args.protocol) != args.protocol_sha256:
                raise ValueError('Executing cache/protocol changed mid-run')
            if file_sha(store.payload_path) != data['layouts'][args.sender_mode]['sha256']:
                raise ValueError('Message payload changed mid-run')
            result.update({'registration': registration, 'process': process, 'finished_utc': legacy.utc_now(),
                           'execution_revision': revision, 'generation_batch_size': 2,
                           'new_optimizer_steps': result.get('new_optimizer_steps', result['optimizer_steps']),
                           'performed_new_optimizer_steps': result.get('performed_new_optimizer_steps', result['optimizer_steps']),
                           'training_performed_in_this_run': result.get('training_performed_in_this_run', True),
                           'training_reused_from_v1': result.get('training_reused_from_v1', False),
                           'training_budget_original': {'epochs': 3, 'optimizer_steps': 1548},
                           'actual_generated_token_count_semantics': 'Full raw generated tensor positions including post-EOS padding; canonical lengths are retained per row',
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
