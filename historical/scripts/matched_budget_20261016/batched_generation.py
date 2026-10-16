"""Fixed-order B4 greedy evaluation and TRAIN-only B1/B2/B4 qualification.

Raw generated tensors and original decode/parser semantics remain intact. EOS
canonicalization exists solely for qualification/accounting; it never rewrites
the generated answer used by the frozen scorer. No shared batch time is divided.
"""
import json
from pathlib import Path
import time

import torch

from foundation_20261016 import run_connectors as legacy
from matched_budget_20261016.run_trial import collate

EVAL_BATCH_SIZE = 4
MAX_NEW_TOKENS = 24


def canonical_generation(raw_ids, eos_token_id, pad_token_id):
    eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
    eos.discard(None)
    if not raw_ids or any(type(x) is not int or x < 0 for x in raw_ids):
        raise ValueError('Raw generated IDs must be nonempty nonnegative integer IDs')
    first = next((i for i, value in enumerate(raw_ids) if value in eos), None)
    length = len(raw_ids) if first is None else first + 1
    tail = raw_ids[length:]
    if tail and any(value != pad_token_id for value in tail):
        raise ValueError('A finished row has non-padding tokens after its first EOS')
    return {'canonical_generated_token_ids': raw_ids[:length],
            'canonical_generated_token_count': length,
            'first_eos_generated_length_including_eos': length if first is not None else None,
            'padding_tail_count': len(tail), 'raw_generated_token_count': len(raw_ids)}


def check_generation_ledger(before, after, batch_size, padded_prompt_length, forward_calls):
    expected = {'reader_forward_calls': forward_calls,
                'realized_padded_text_queries': batch_size * (
                    forward_calls * padded_prompt_length + forward_calls * (forward_calls - 1) // 2),
                'realized_logical_message_payload_bytes': batch_size * forward_calls * 8192}
    actual = {}
    for key, value in expected.items():
        if key not in before or key not in after:
            raise ValueError(f'Reader diagnostics lack required cumulative ledger: {key}')
        actual[key] = after[key] - before[key]
        if actual[key] != value:
            raise ValueError(f'Batched generation accounting mismatch for {key}: {actual[key]} != {value}')
    return {'expected': expected, 'actual': actual, 'verified': True,
            'query_semantics': 'All physical padded queries, including already-EOS rows; no dummy calls',
            'payload_semantics': 'Logical module-boundary message bytes; not physical network traffic'}


def generated_predictions(model, rows, tokenizer, store, args, handle, check=lambda: None,
                          batch_size=EVAL_BATCH_SIZE):
    """Only qualification may request B1/B2; scientific trials always use B4."""
    if batch_size not in (1, 2, 4):
        raise ValueError('Only fixed B1/B2 qualification or B4 evaluation is admitted')
    model.eval()
    records = []
    batch_path = Path(handle.name).with_suffix('.batches.jsonl')
    with batch_path.open('x', encoding='utf-8') as batch_handle:
        for start in range(0, len(rows), batch_size):
            check()
            selected = rows[start:start + batch_size]
            batch = collate(selected, tokenizer, store, args.device, supervised=False)
            before = model.diagnostics()
            started = time.perf_counter()
            with torch.no_grad():
                tokens = model.greedy_generate(**batch, max_new_tokens=MAX_NEW_TOKENS,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                    message_mode='correct')
            # Synchronize the returned tensor before recording elapsed batch time.
            raw = tokens.detach().cpu().tolist()
            elapsed = time.perf_counter() - started
            if (tokens.dtype != torch.long or tokens.ndim != 2 or tokens.shape[0] != len(selected)
                    or not 1 <= tokens.shape[1] <= MAX_NEW_TOKENS):
                raise ValueError('Greedy generator returned an invalid complete raw tensor')
            g, B, length = tokens.shape[1], len(selected), batch['input_ids'].shape[1]
            after = model.diagnostics()  # Never reset: retain end-to-end case ledgers.
            ledger_error = None
            try:
                accounting = check_generation_ledger(before, after, B, length, g)
            except ValueError as error:
                ledger_error = str(error)
                accounting = {'verified': False, 'error': ledger_error, 'before': before, 'after': after}
            batch_id = f'{Path(handle.name).stem}:batch{start // batch_size:06d}'
            members = [row['sample_id'] for row in selected]
            receipt = {'generation_batch_id': batch_id, 'member_sample_ids': members,
                       'B': B, 'Lmax': length, 'g': g, 'max_new_tokens': MAX_NEW_TOKENS,
                       'whole_batch_elapsed_seconds': elapsed, 'accounting': accounting,
                       'raw_generated_token_ids_by_row': raw,
                       'elapsed_semantics': 'One whole-batch wall time; not divided into row latency'}
            batch_handle.write(json.dumps(receipt, ensure_ascii=False, allow_nan=False) + '\n')
            batch_handle.flush()
            for row, raw_ids in zip(selected, raw):
                canonical = canonical_generation(raw_ids, tokenizer.eos_token_id, tokenizer.pad_token_id)
                text = tokenizer.decode(raw_ids, skip_special_tokens=True)
                record = {'sample_id': row['sample_id'], 'split': row['split'], 'raw_text': text,
                          'parse': legacy.parse_answer(text), 'gold_pkd': float(row['pkd']),
                          'raw_generated_token_ids': raw_ids, 'generated_token_count': len(raw_ids),
                          **canonical, 'prompt_token_count': int(batch['attention_mask'][members.index(row['sample_id'])].sum()),
                          'generation_batch_id': batch_id, 'generation_batch_member_ids': members,
                          'generation_batch_size': B, 'generation_prompt_Lmax': length,
                          'batch_generation_forward_calls': g, 'whole_batch_elapsed_seconds': elapsed,
                          'elapsed_seconds': None, 'row_latency_attribution': 'unavailable_shared_batch_execution',
                          'channel': 'generation', 'message_input': args.message_input}
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                handle.flush()
            if ledger_error is not None:
                raise ValueError(ledger_error)
    return records


def canary_rows(train_rows, tokenizer):
    if len(train_rows) < 4 or any(row['split'] != 'train' for row in train_rows):
        raise ValueError('Batched qualification needs at least four TRAIN rows')
    ordered = sorted(train_rows, key=lambda row: (
        len(tokenizer.encode(legacy.prompt_text(row, tokenizer), add_special_tokens=False)), row['sample_id']))
    n = len(ordered)
    indices = [0, (n - 1) // 2, (3 * (n - 1)) // 4, n - 1]
    if len(set(indices)) != 4:
        raise ValueError('TRAIN population is too small for four distinct length strata')
    return [ordered[i] for i in indices]


def qualify_batch_invariance(model, rows, tokenizer, store, args, check=lambda: None):
    """Same post-update reader state throughout; any difference fails qualification."""
    if len(rows) != 4 or any(row['split'] != 'train' for row in rows):
        raise ValueError('Qualification must use exactly four prespecified TRAIN rows')
    before_state = model.reader_state()
    runs = {}
    for name, size, selected in (('B1', 1, rows), ('B4', 4, rows), ('B2_first_two', 2, rows[:2])):
        with (args.output / f'canary_train_{name}.jsonl').open('x', encoding='utf-8') as handle:
            runs[name] = generated_predictions(model, selected, tokenizer, store, args, handle, check, batch_size=size)
    mismatches = []
    baseline = {x['sample_id']: x for x in runs['B1']}
    for name in ('B4', 'B2_first_two'):
        for actual in runs[name]:
            expected = baseline[actual['sample_id']]
            for key in ('canonical_generated_token_ids', 'raw_text', 'parse'):
                if actual[key] != expected[key]:
                    mismatches.append({'comparison': name, 'sample_id': actual['sample_id'],
                                       'field': key, 'B1': expected[key], 'actual': actual[key]})
    after_state = model.reader_state()
    same_state = set(before_state) == set(after_state) and all(
        torch.equal(value, after_state[key]) for key, value in before_state.items())
    result = {'accepted': not mismatches and same_state, 'sample_split': 'train',
              'sample_ids': [row['sample_id'] for row in rows], 'batches': ['B1', 'B4', 'B2_first_two'],
              'comparison_fields': ['canonical_generated_token_ids', 'raw_text', 'parse'],
              'reader_state_unchanged_during_qualification': same_state, 'mismatches': mismatches,
              'test_rows_used': 0, 'failure_policy': 'Stop qualification; retain every raw output and mismatch; no test-driven selection'}
    legacy.write_json(args.output / 'batch_invariance_qualification.json', result)
    if not result['accepted']:
        raise ValueError('TRAIN B1/B4/B2 generation differs or reader state changed; qualification failed')
    return result, runs['B4']


def train_only_canary(model, tokenizer, train_rows, store, args, registration, log, check=lambda: None):
    from matched_budget_20261016.token_diagnostics import batch_token_categories, token_nll_breakdown
    selected = canary_rows(train_rows, tokenizer)
    check()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=.01)
    batch = collate(selected, tokenizer, store, args.device)
    optimizer.zero_grad(set_to_none=True)
    output = model(**batch)
    summed, count = legacy.supervised_nll(output, batch['labels'])
    categories = batch_token_categories(tokenizer, [legacy.answer_text(row) for row in selected],
                                        batch['input_ids'], batch['labels'], batch['attention_mask'])
    diagnostic = token_nll_breakdown(output, batch['labels'], categories)
    if diagnostic['reference_supervised_nll']['loss_sum'] != float(summed.detach()):
        raise ValueError('Canary token accounting differs from frozen full loss')
    (summed / count).backward()
    gradients = [p.grad for p in trainable if p.grad is not None]
    if not gradients or any(not torch.isfinite(value).all() for value in gradients):
        raise ValueError('Canary reader gradients missing/nonfinite')
    alpha_gradient = float(model.reader.alpha.grad) if model.reader.alpha.grad is not None else None
    norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
    legacy.assert_frozen(model.base_lm)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in trainable):
        raise ValueError('Canary reader state is nonfinite')
    del output
    legacy.save_checkpoint(args.output / 'canary_reader_not_for_training.pt',
        {'reader_state': model.reader_state(), 'registration': registration,
         'initialization_or_selection_eligible': False, 'purpose': 'TRAIN transport qualification only'})
    qualification, predictions = qualify_batch_invariance(model, selected, tokenizer, store, args, check)
    result = {'status': 'passed', 'purpose': 'TRAIN-only one-update B1/B2/B4 batching qualification',
              'scientific_trial_count': 0, 'optimizer_steps': 1, 'test_examples_evaluated': 0,
              'sample_split': 'train', 'sample_ids': [row['sample_id'] for row in selected],
              'selection': 'TRAIN prompt length sorted ascending then sample ID; indices 0,floor((n-1)/2),floor(3(n-1)/4),n-1',
              'answer_token_nll': float(summed.detach()) / count, 'gradient_norm_before_clip': float(norm),
              'alpha_gradient_before_update': alpha_gradient, 'train_token_nll_breakdown_before_update': diagnostic,
              'batch_invariance_qualification': qualification,
              'train_generation_metrics': legacy.generation_metrics(selected, predictions),
              'end_to_end_reader_diagnostics': model.diagnostics(),
              'diagnostics_reset_policy': 'No reset during this end-to-end canary, including all B1/B2/B4 qualification forwards',
              'backbone_gradients': 'none_verified', 'scope': 'canary_only_not_a_scientific_benchmark_result'}
    log({'event': 'canary_complete', 'batch_invariance_accepted': qualification['accepted'], 'test_examples_evaluated': 0})
    return result
