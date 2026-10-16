"""BASE03: validation-selected frozen A1 predictor -> frozen Qwen JSON rendering.

This is zero-shot structured prediction rendering, with no language-model
training and no B2 knowledge-composition claim. Gold pKD values never enter a
prompt. Invalid or truncated answers remain failures without repair.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_v02.cache_panel_features import DEFAULT_CACHE, ESM_REVISION, QWEN_REVISION, file_sha, verify_local_models
from bench_v02.panel_execution import validate_panel_data
from bench_v02.train_panel_baselines import A1Predictor, ContextEncoder, FeatureStore
from foundation_20261016.run_connectors import context_view, parse_answer, process_record, utc_now, write_json

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'results/bench_v02/panel_baselines_20261015_v2'
SCENARIOS = ('new_peptide_known_target', 'new_peptide_new_target')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def select_predictor(candidates):
    """Only validation MAE participates; BASE01 is the fixed tie-breaker."""
    require(len(candidates) == 2 and {c['architecture'] for c in candidates} == {'BASE01', 'MEANPOOL'},
            'Exactly BASE01 and MEANPOOL candidates required')
    require(all(finite(c['best_validation_metric']) for c in candidates), 'Invalid validation MAE')
    return min(candidates, key=lambda c: (c['best_validation_metric'], c['architecture'] != 'BASE01'))


def structured_input(row, estimate):
    require(finite(estimate), 'Predictor estimate must be finite')
    # Match the predictor's allowed chemistry and context, including assay and
    # buffer strings. Direct gold, row IDs, names, sequences, and raw metadata
    # are excluded from this view.
    return {'prediction': {'endpoint': 'pKD = -log10(KD in molar)',
                           'pKD_estimate': float(estimate),
                           'uncertainty': None,
                           'uncertainty_status': 'source_predictor_has_no_native_uncertainty'},
            'allowed_context': context_view(row),
            'scope': 'reference_target_prediction; experimental_assay_construct_may_be_unknown'}


def prompt_text(payload, tokenizer):
    user = ('Render the supplied frozen prediction as strict JSON. Return exactly '
            '{"pKD":number}, preserving the supplied pKD_estimate as the numeric value. '
            'Do not add fields, explanations, markdown, or inferred uncertainty. '
            'The estimate is a model prediction, not an experimental measurement.\n'
            + json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False))
    return tokenizer.apply_chat_template([
        {'role': 'system', 'content': 'You render structured numerical model predictions.'},
        {'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def greedy_tokens(model, input_ids, attention_mask, max_new_tokens=16, eos_token_id=None, pad_token_id=0):
    """Same finite greedy/no-cache convention as the connector runner."""
    require(isinstance(max_new_tokens, int) and not isinstance(max_new_tokens, bool)
            and 1 <= max_new_tokens <= 256, 'max_new_tokens must be in [1,256]')
    require(input_ids.ndim == 2 and input_ids.shape == attention_mask.shape
            and bool(((attention_mask == 0) | (attention_mask == 1)).all())
            and bool(attention_mask.bool().any(-1).all()), 'Invalid text batch/mask')
    eos = [] if eos_token_id is None else (list(eos_token_id) if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
    model.eval()
    finished = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)
    tokens = []
    for _ in range(max_new_tokens):
        output = model(input_ids=input_ids, attention_mask=attention_mask,
                       position_ids=(attention_mask.cumsum(-1) - 1).clamp_min(0),
                       use_cache=False, return_dict=True)
        require(bool(torch.isfinite(output.logits).all()), 'Nonfinite LM logits')
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        last = positions.masked_fill(~attention_mask.bool(), -1).max(-1).values
        next_token = output.logits[torch.arange(len(input_ids), device=input_ids.device), last].argmax(-1)
        next_token = torch.where(finished, torch.full_like(next_token, pad_token_id), next_token)
        tokens.append(next_token)
        active = ~finished
        for value in eos:
            finished |= active & (next_token == value)
        if bool(finished.all()):
            break
        input_ids = torch.cat((input_ids, next_token[:, None]), 1)
        attention_mask = torch.cat((attention_mask, active[:, None].to(attention_mask.dtype)), 1)
    return torch.stack(tokens, 1)


@torch.inference_mode()
def predictor_values(model, store, context, rows, device, label_center, label_scale, batch_size=8):
    require(finite(label_center) and finite(label_scale) and label_scale > 0, 'Invalid frozen label normalization')
    values = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        peptide, pm = store.batch([r['peptide_sequence'] for r in selected], device)
        target, tm = store.batch([r['target_sequence'] for r in selected], device)
        encoded = torch.stack([context(r) for r in selected]).to(device)
        predicted = model(peptide, pm, target, tm, encoded).float() * label_scale + label_center
        require(bool(torch.isfinite(predicted).all()), 'Nonfinite predictor output')
        values.extend(predicted.cpu().tolist())
        # Bound CPU residue caching without changing its hash-checked provenance.
        if hasattr(store, 'memory') and sum(v.numel() * v.element_size() for v in store.memory.values()) > 384 * 1024**2:
            store.memory.clear()
    return values


def evaluate_records(rows, records):
    require(bool(rows) and len(rows) == len(records), 'Empty or missing prediction records')
    require([r['sample_id'] for r in rows] == [r['sample_id'] for r in records], 'Sample order/identity mismatch')
    require(all(finite(r['pkd']) for r in rows), 'Nonfinite evaluation gold')
    valid = [i for i, record in enumerate(records) if record['parse']['valid']]
    source = [abs(row['pkd'] - record['source_predictor_pKD']) for row, record in zip(rows, records)]
    rendered = [abs(rows[i]['pkd'] - records[i]['parse']['value']) for i in valid]
    preservation = [abs(records[i]['source_predictor_pKD'] - records[i]['parse']['value']) for i in valid]
    return {'n_total': len(rows), 'n_valid': len(valid), 'n_failed': len(rows) - len(valid),
            'parse_valid_rate': len(valid) / len(rows),
            'failure_counts': dict(Counter(record['parse']['reason'] for record in records if not record['parse']['valid'])),
            'source_predictor_mae_same_all_examples': float(np.mean(source)),
            'source_predictor_mae_same_valid_examples': float(np.mean([source[i] for i in valid])) if valid else None,
            'generated_pKD_mae_valid_only': float(np.mean(rendered)) if valid else None,
            'parsed_estimate_preservation_mae_valid_only': float(np.mean(preservation)) if valid else None,
            'parsed_estimate_preservation_max_error_valid_only': max(preservation) if valid else None,
            'full_sample_generated_mae': float(np.mean(rendered)) if len(valid) == len(rows) else None,
            'failure_policy': 'no imputation, clipping, repair, or gold substitution',
            'native_uncertainty_available': False}


def generate_records(rows, values, tokenizer, lm, device, max_new_tokens=16, handle=None):
    require(len(rows) == len(values), 'Missing source predictor values')
    records = []
    for row, value in zip(rows, values):
        started = time.perf_counter()
        payload = structured_input(row, value)
        prompt = prompt_text(payload, tokenizer)
        input_values = tokenizer.encode(prompt, add_special_tokens=False)
        require(bool(input_values), 'Empty tokenized prompt')
        ids = torch.tensor([input_values], dtype=torch.long, device=device)
        generated = greedy_tokens(lm, ids, torch.ones_like(ids), max_new_tokens,
                                  tokenizer.eos_token_id, tokenizer.pad_token_id)
        token_values = generated[0].cpu().tolist()
        text = tokenizer.decode(token_values, skip_special_tokens=True)
        record = {'sample_id': row['sample_id'], 'split': row['split'], 'source_predictor_pKD': value,
                  'structured_input': payload, 'prompt': prompt,
                  'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                  'input_token_ids': input_values, 'generated_token_ids': token_values,
                  'raw_text': text, 'parse': parse_answer(text),
                  'prompt_tokens': len(input_values), 'generated_tokens': len(token_values),
                  'elapsed_seconds': time.perf_counter() - started}
        records.append(record)
        if handle is not None:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()
    return records


def validate_candidate(directory, architecture, seed, scenario, gate, store, audited, pins):
    result_path = directory / 'result.json'
    require(file_sha(result_path) == audited['result_sha256'], 'Prior audited result changed')
    result = read_json(result_path)
    require(result['architecture'] == architecture and result['seed'] == seed and result['scenario'] == scenario,
            'Prior predictor identity mismatch')
    require(result['feature_cache_manifest_sha256'] == file_sha(store.manifest_path), 'Prior cache changed')
    for key in ('benchmark_lock_sha256', 'samples_sha256'):
        require(result['data_gate'][key] == gate[key], 'Prior frozen data changed')
    for filename, expected in result['artifact_sha256'].items():
        path = directory / filename
        require(file_sha(path) == expected, f'Prior artifact changed: {path}')
        pins[str(path)] = expected
    pins[str(result_path)] = audited['result_sha256']
    for filename, key in (('train_panel_baselines.py', 'script_sha256'), ('cache_panel_features.py', 'cache_script_sha256')):
        path = ROOT / 'scripts/bench_v02' / filename
        require(file_sha(path) == result[key], 'Prior predictor/cache implementation changed')
        pins[str(path)] = result[key]
    for filename, key in (('preregistered_protocol.json', 'preregistered_protocol_sha256'),
                          ('architecture_training_preregistration.json', 'architecture_registration_sha256')):
        path = directory.parent / filename
        require(file_sha(path) == result[key], 'Prior registration changed')
        pins[str(path)] = result[key]
    for kind in ('proteins', 'texts'):
        for key, expected in result['feature_receipts'][kind].items():
            require(store.manifest[kind][key]['sha256'] == expected, 'Prior feature receipt changed')
    history = read_json(directory / 'history.json')
    require(len(history) == 30 and result['actual_epochs'] == 30, 'Prior training budget incomplete')
    best = min(history, key=lambda item: item['validation']['mae'])
    require(best['epoch'] == result['best_epoch'] and best['validation']['mae'] == result['best_validation_metric'],
            'Prior validation-only earliest checkpoint selection differs')
    require(file_sha(directory / 'best.pt') == audited['checkpoint_sha256'], 'Prior audited checkpoint changed')
    return {**result, 'directory': str(directory), 'result_sha256': audited['result_sha256']}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', choices=SCENARIOS, required=True)
    p.add_argument('--seed', type=int, choices=(17, 42, 2026), default=17)
    p.add_argument('--split', choices=('validation', 'test', 'both'), default='both')
    p.add_argument('--test-limit', type=int)
    p.add_argument('--validation-limit', type=int)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    p.add_argument('--data-root', type=Path, default=ROOT / 'data/processed/bench_v02/a1_locked_v2')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--predictor-device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--max-new-tokens', type=int, default=16)
    p.add_argument('--threads', type=int, default=2)
    return p


def main():
    args = parser().parse_args()
    require(1 <= args.max_new_tokens <= 256 and args.threads > 0, 'Invalid resource/decoding limits')
    require(all(limit is None or limit > 0 for limit in (args.test_limit, args.validation_limit)), 'Subset limits must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    write_json(args.output / 'process.json', process_record(args))
    try:
        source = args.data_root / args.scenario
        rows, gate = validate_panel_data(source / 'samples.jsonl', source / 'benchmark_lock.json', 'A1')
        require(read_json(source / 'benchmark_lock.json')['scenario'] == args.scenario, 'Physical scenario differs')
        store = FeatureStore(args.cache)
        review_path = OLD / 'independent_review.json'
        review = read_json(review_path)
        require(review['status'] == 'passed', 'Prior independent audit required')
        pins = {str(review_path): file_sha(review_path), str(store.manifest_path): file_sha(store.manifest_path),
                str(source / 'samples.jsonl'): gate['samples_sha256'],
                str(source / 'benchmark_lock.json'): gate['benchmark_lock_sha256']}
        candidates = []
        for architecture in ('BASE01', 'MEANPOOL'):
            directory = OLD / f'A1_{args.scenario}_{architecture}_seed{args.seed}'
            audited = next(r for r in review['runs'] if r['run_id'] == directory.name)
            candidates.append(validate_candidate(directory, architecture, args.seed, args.scenario, gate, store, audited, pins))
        chosen = select_predictor(candidates)
        splits = ('validation', 'test') if args.split == 'both' else (args.split,)
        selections = {}
        for split in splits:
            full = [r for r in rows if r['split'] == split]
            limit = getattr(args, split + '_limit')
            selected = full if limit is None else full[:limit]
            require(bool(selected), f'Empty {split}')
            selections[split] = selected
        model_pins = verify_local_models()
        scripts = (Path(__file__), Path(__file__).with_name('run_connectors.py'))
        pins.update({str(path): file_sha(path) for path in scripts})
        registration = {'method': 'BASE03', 'role': 'frozen_predictor_to_frozen_lm',
                        'scope': 'A1 structured prediction rendering; zero-shot LM; no B2 composition claim',
                        'new_training': False, 'predictor_selection': 'lowest stored full-validation MAE; BASE01 fixed tie-breaker',
                        'selection_candidates': [{k: c[k] for k in ('architecture', 'best_validation_metric', 'best_epoch', 'result_sha256')}
                                                 for c in candidates],
                        'selected_architecture': chosen['architecture'], 'seed': args.seed, 'scenario': args.scenario,
                        'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        'data_gate': gate, 'input_pins': pins, 'model_pins': model_pins,
                        'foundation_revisions': {'ESM2': ESM_REVISION, 'Qwen': QWEN_REVISION},
                        'selection_manifest': {s: {'available_n': sum(r['split'] == s for r in rows),
                                                   'selected_n': len(values), 'sample_ids': [r['sample_id'] for r in values]}
                                               for s, values in selections.items()},
                        'subset_scope': any(len(v) < sum(r['split'] == s for r in rows) for s, v in selections.items()),
                        'decode': {'algorithm': 'greedy_argmax', 'use_cache': False, 'max_new_tokens': args.max_new_tokens},
                        'gold_in_prompt': False, 'native_uncertainty': None, 'started_utc': utc_now()}
        write_json(args.output / 'run_preregistration.json', registration)
        saved = torch.load(Path(chosen['directory']) / 'best.pt', weights_only=True, map_location='cpu')
        require(saved['registration'] == read_json(Path(chosen['directory']) / 'run_preregistration.json'), 'Checkpoint registration mismatch')
        require(saved['best_epoch'] == chosen['best_epoch'], 'Checkpoint epoch mismatch')
        training = [r for r in rows if r['split'] == 'train']
        context = ContextEncoder(training)
        require(context.statistics == saved['context_statistics'], 'Frozen train-only context normalization differs')
        require(saved['label_center'] == float(np.mean([r['pkd'] for r in training]))
                and saved['label_scale'] == max(float(np.std([r['pkd'] for r in training])), 1e-5), 'Frozen label normalization differs')
        predictor = A1Predictor(chosen['architecture'] == 'BASE01').to(args.predictor_device).eval().requires_grad_(False)
        predictor.load_state_dict(saved['model'], strict=True)
        source_estimates = {split: predictor_values(predictor, store, context, selected, args.predictor_device,
                                                  saved['label_center'], saved['label_scale'])
                            for split, selected in selections.items()}
        del predictor, saved
        store.memory.clear()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_path = ROOT / 'models/Qwen--Qwen2.5-1.5B-Instruct'
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        lm = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                dtype=torch.bfloat16 if args.device == 'cuda' else torch.float32,
                attn_implementation='sdpa').to(args.device).eval().requires_grad_(False)
        summaries = {}
        for split, selected in selections.items():
            with (args.output / f'{split}_predictions.jsonl').open('x', encoding='utf-8') as handle:
                records = generate_records(selected, source_estimates[split], tokenizer, lm, args.device,
                                           args.max_new_tokens, handle)
            summaries[split] = evaluate_records(selected, records)
            print(json.dumps({'event': 'split_complete', 'split': split, 'metrics': summaries[split]}), flush=True)
        require(all(file_sha(path) == expected for path, expected in pins.items()), 'Pinned input changed during run')
        result = {'status': 'complete', 'method': 'BASE03', 'scope': registration['scope'],
                  'subset_scope': registration['subset_scope'], 'seed': args.seed, 'scenario': args.scenario,
                  'selected_architecture': chosen['architecture'], 'metrics': summaries,
                  'cost': {'elapsed_seconds': time.perf_counter() - started, 'optimizer_steps': 0,
                           'lm_trainable_parameters': 0, 'predictor_trainable_parameters': 0},
                  'feature_receipts': store.checked_proteins,
                  'registration_sha256': file_sha(args.output / 'run_preregistration.json'),
                  'prediction_sha256': {split: file_sha(args.output / f'{split}_predictions.jsonl') for split in selections},
                  'input_pins_unchanged': True, 'formal_multitask_connector_result': False,
                  'B2_composition_claim': False, 'synthetic_results_counted': False}
        write_json(args.output / 'result.json', result)
    except Exception as exc:
        write_json(args.output / 'failure.json', {'status': 'failed', 'type': type(exc).__name__, 'error': str(exc),
                                                'utc': utc_now(), 'elapsed_seconds': time.perf_counter() - started})
        raise


if __name__ == '__main__':
    main()
