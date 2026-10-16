"""CPU-only V2 batching/reuse fixtures; no real test scores or GPU are read."""
import ast
import copy
import inspect
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from matched_budget_20261016 import run_trial as v1
from matched_budget_20261016 import run_trial_batched as v2
from matched_budget_20261016 import batched_generation as generation
from matched_budget_20261016.connectors import MatchedReader, MatchedConnectorLM
from test_matched_budget_runner import Tokenizer, Messages, FixtureModel, row


class LabelMessages:
    def batch(self, rows):
        tensor = torch.zeros(len(rows), 16, 128)
        for i, item in enumerate(rows):
            tensor[i, 0, 0] = int(item['sample_id'].split('-')[-1])
        return tensor


class ScriptedGenerator(FixtureModel):
    def __init__(self, mismatch=False, wrong_ledger=False):
        super().__init__()
        self.calls = self.queries = self.bytes = 0
        self.mismatch, self.wrong_ledger = mismatch, wrong_ledger

    def diagnostics(self, reset=False):
        value = {'reader_forward_calls': self.calls, 'realized_padded_text_queries': self.queries,
                 'realized_logical_message_payload_bytes': self.bytes}
        if reset:
            self.calls = self.queries = self.bytes = 0
        return value

    def greedy_generate(self, input_ids, memory, max_new_tokens, eos_token_id, pad_token_id, **kwargs):
        outputs = []
        for code in memory[:, 0, 0].long().tolist():
            text = ('{"pKD":5}' if code % 4 == 0 else '{"pKD":-1.25}' if code % 4 == 1
                    else '{"pKD":12.000}' if code % 4 == 2 else 'x' * 24)
            ids = Tokenizer().encode(text)
            if code % 4 != 3:
                ids.append(eos_token_id)
            outputs.append(ids[:max_new_tokens])
        g = max(len(x) for x in outputs)
        tokens = torch.full((len(outputs), g), pad_token_id, dtype=torch.long)
        for i, ids in enumerate(outputs):
            tokens[i, :len(ids)] = torch.tensor(ids)
        if self.mismatch and len(outputs) > 1:
            tokens[0, 0] = ord('!') + 3
        self.calls += g
        B, length = input_ids.shape
        self.queries += B * (g * length + g * (g - 1) // 2) - int(self.wrong_ledger)
        self.bytes += B * g * 8192
        return tokens


class BatchedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_training_function_is_byte_identical_and_original_sha_unchanged(self):
        self.assertEqual(inspect.getsource(v1.train_and_evaluate), inspect.getsource(v2.train_and_evaluate))
        self.assertEqual(v1.file_sha(Path(v1.__file__)), v2.V1_RUNNER_SHA256)
        self.assertEqual(v1.FROZEN_PROTOCOL_SHA256, v2.FROZEN_PROTOCOL_SHA256)
        revision = v2.execution_revision()
        self.assertEqual(revision['generation_batch_size'], 4)

    def test_first_eos_canonicalization_keeps_full_raw_ids_for_both_padding_policies(self):
        for pad in (0, 1):
            raw = [9, 8, 1, pad, pad]
            value = generation.canonical_generation(raw, 1, pad)
            self.assertEqual(raw, [9, 8, 1, pad, pad])
            self.assertEqual(value['canonical_generated_token_ids'], [9, 8, 1])
            self.assertEqual(value['raw_generated_token_count'], 5)
            self.assertEqual(value['canonical_generated_token_count'], 3)
            self.assertEqual(value['padding_tail_count'], 2)
            self.assertEqual(value['first_eos_generated_length_including_eos'], 3)
        with self.assertRaises(ValueError):
            generation.canonical_generation([9, 1, 8], 1, 0)
        value = generation.canonical_generation([9] * 24, 1, 0)
        self.assertIsNone(value['first_eos_generated_length_including_eos'])
        self.assertEqual(value['canonical_generated_token_count'], 24)
        self.assertEqual(value['padding_tail_count'], 0)

    def test_batch4_fixed_order_tail_batch_and_exact_physical_ledger_with_mixed_eos(self):
        for pad in (0, 1):
            tokenizer = Tokenizer()
            tokenizer.pad_token_id = pad
            records = [row(f'train-{i}') for i in range(5)]
            records[1]['experimental_context']['assay'] = 'a much longer allowed context'
            with tempfile.TemporaryDirectory() as temporary:
                args = SimpleNamespace(output=Path(temporary), device='cpu', message_input='real')
                model = ScriptedGenerator()
                with (args.output / 'predictions.jsonl').open('x', encoding='utf8') as handle:
                    predictions = generation.generated_predictions(model, records, tokenizer, LabelMessages(), args, handle)
                self.assertEqual([x['sample_id'] for x in predictions], [x['sample_id'] for x in records])
                ledgers = [json.loads(x) for x in (args.output / 'predictions.batches.jsonl').read_text().splitlines()]
                self.assertEqual([x['B'] for x in ledgers], [4, 1])
                self.assertEqual([x['g'] for x in ledgers], [24, 10])
                self.assertEqual(predictions[3]['canonical_generated_token_count'], 24)
                self.assertIsNone(predictions[3]['first_eos_generated_length_including_eos'])
                self.assertGreater(predictions[0]['padding_tail_count'], 0)
                self.assertEqual(len(predictions[0]['raw_generated_token_ids']), 24)
                self.assertEqual(predictions[0]['canonical_generated_token_count'], 10)
                self.assertIsNone(predictions[0]['elapsed_seconds'])
                self.assertEqual(predictions[0]['whole_batch_elapsed_seconds'], ledgers[0]['whole_batch_elapsed_seconds'])
                for item in ledgers:
                    self.assertTrue(item['accounting']['verified'])
                    self.assertEqual(item['accounting']['actual']['realized_padded_text_queries'],
                        item['B'] * (item['g'] * item['Lmax'] + item['g'] * (item['g'] - 1) // 2))
                self.assertEqual(model.calls, sum(x['g'] for x in ledgers))
                self.assertEqual(model.bytes, sum(x['B'] * x['g'] * 8192 for x in ledgers))

    def test_train_only_b1_b4_first_two_b2_qualification_and_failure_retention(self):
        records = [row(f'train-{i}') for i in range(4)]
        tokenizer = Tokenizer()
        tokenizer.pad_token_id = tokenizer.eos_token_id
        for mismatch in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                args = SimpleNamespace(output=Path(temporary), device='cpu', message_input='neutral')
                model = ScriptedGenerator(mismatch=mismatch)
                if mismatch:
                    with self.assertRaisesRegex(ValueError, 'qualification failed'):
                        generation.qualify_batch_invariance(model, records, tokenizer, LabelMessages(), args)
                else:
                    result, predictions = generation.qualify_batch_invariance(model, records, tokenizer, LabelMessages(), args)
                    self.assertTrue(result['accepted'])
                    self.assertEqual(len(predictions), 4)
                result = json.loads((args.output / 'batch_invariance_qualification.json').read_text())
                self.assertEqual(result['accepted'], not mismatch)
                self.assertEqual(result['test_rows_used'], 0)
                self.assertEqual(len(list(args.output.glob('canary_train_*.jsonl'))), 6)
                self.assertTrue((args.output / 'canary_train_B4.jsonl').read_text())
        with self.assertRaises(ValueError):
            generation.canary_rows([row('test-0', 'test')] * 4, tokenizer)

    def test_actual_tiny_qwen_train_update_and_batched_qualification(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        class ShortTokenizer(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return 'prompt:' + str(len(messages[-1]['content']))
        v2.configure_numerical_policy(require_cuda_environment=False)
        torch.manual_seed(123)
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=256, hidden_size=24, intermediate_size=40,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            attention_dropout=0., eos_token_id=1, pad_token_id=0))
        model = MatchedConnectorLM(base, placement='mid', insertion_layer=1)
        records = [row(f'train-{i}') for i in range(4)]
        for i, record in enumerate(records):
            record['experimental_context']['assay'] = 'x' * (10**i)
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(output=Path(temporary), device='cpu', message_input='real')
            result = generation.train_only_canary(model, ShortTokenizer(), records, LabelMessages(), args,
                                                   {'scope': 'cpu_tiny_fixture_only'}, lambda _: None)
            self.assertEqual(result['optimizer_steps'], 1)
            self.assertTrue(result['batch_invariance_qualification']['accepted'])
            self.assertEqual(result['test_examples_evaluated'], 0)
            self.assertGreater(result['end_to_end_reader_diagnostics']['reader_forward_calls'], 1)
            self.assertTrue(all(p.grad is None for p in model.base_lm.parameters()))

    def test_ledger_failure_preserves_raw_batch_and_row_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(output=Path(temporary), device='cpu', message_input='real')
            with (args.output / 'bad.jsonl').open('x', encoding='utf8') as handle:
                with self.assertRaisesRegex(ValueError, 'accounting mismatch'):
                    generation.generated_predictions(ScriptedGenerator(wrong_ledger=True), [row('train-0')],
                        Tokenizer(), LabelMessages(), args, handle)
            self.assertTrue((args.output / 'bad.jsonl').read_text())
            batch = json.loads((args.output / 'bad.batches.jsonl').read_text())
            self.assertFalse(batch['accounting']['verified'])
            self.assertTrue(batch['raw_generated_token_ids_by_row'])

    def test_canary_prompt_length_strata_are_fixed_without_labels(self):
        records = [row(f'train-{i}') for i in range(8)]
        for i, record in enumerate(records):
            record['experimental_context']['assay'] = 'x' * (i + 1)
        chosen = generation.canary_rows(records, Tokenizer())
        self.assertEqual([x['sample_id'] for x in chosen], ['train-0', 'train-3', 'train-5', 'train-7'])
        changed = copy.deepcopy(records)
        for record in changed:
            record['pkd'] = -999.
        self.assertEqual([x['sample_id'] for x in generation.canary_rows(changed, Tokenizer())],
                         [x['sample_id'] for x in chosen])

    def test_reuse_cli_is_explicit_without_optimizer_or_eval_batch_tuning(self):
        actions = {x.dest for x in v2.parser()._actions}
        self.assertIn('reuse_complete_v1', actions)
        self.assertNotIn('generation_batch_size', actions)
        self.assertNotIn('epochs', actions)

    def reuse_fixture(self, folder):
        folder = Path(folder)
        job = folder / 'source_V1'
        artifacts = job / 'artifacts'
        artifacts.mkdir(parents=True)
        args = SimpleNamespace(scenario=v2.SCENARIOS[0], seed=17, placement='early',
                               sender_mode='balanced', message_input='real', gate_init='zero')
        messages = torch.randn(3, 16, 128, generator=torch.Generator().manual_seed(18))
        payload = {'sample_ids': ['a', 'b', 'c'], 'splits': ['train', 'validation', 'test'],
                   'messages': messages, 'neutral': messages[:1].mean(0)}
        cache = folder / 'cache.pt'
        torch.save(payload, cache)
        cache_manifest = folder / 'cache.json'
        old_manifest = {'projection': {'tensor_sha256': 'fixed_projection_fixture'},
                        'datasets': {args.scenario: {'layouts': {'balanced': {
                            'path': v2.relative(cache), 'sha256': v2.file_sha(cache)}}}}}
        cache_manifest.write_text(json.dumps(old_manifest), encoding='utf8')
        gate = {'samples_sha256': 'frozen_rows_fixture', 'benchmark_lock_sha256': 'frozen_lock_fixture'}
        original = {'runner_sha256': v2.V1_RUNNER_SHA256, 'protocol_sha256': v2.FROZEN_PROTOCOL_SHA256,
                    'capacity': {'trainable_parameters': 464001}, 'configuration': {
                        **vars(args), 'cache_manifest': str(cache_manifest)},
                    'cache_manifest_sha256': v2.file_sha(cache_manifest), 'data_gate': gate,
                    'initialization': {'tensor_sha256': {'fixture': 'same_untrained_initial_state'}},
                    'source_pins': {'scripts/matched_budget_20261016/run_trial.py': v2.V1_RUNNER_SHA256}}
        history = [{'epoch': i, 'optimizer_steps': 516 * i, 'validation_answer_token_nll': nll,
                    'train_reader_diagnostics': {}, 'validation_reader_diagnostics': {}}
                   for i, nll in enumerate([.5, .3, .3], 1)]
        checkpoint = artifacts / 'best_reader.pt'
        torch.save({'reader_state': MatchedReader().state_dict(), 'best_epoch': 2,
                    'validation_answer_token_nll': .3, 'registration': original}, checkpoint)
        selection = {'best_epoch': 2, 'validation_answer_token_nll': .3,
                     'test_labels_used_for_selection': False, 'checkpoint_sha256': v2.file_sha(checkpoint)}
        actual = {'status': 'complete', 'optimizer_steps': 1548, 'actual_epochs': 3,
                  'best_epoch': 2, 'best_validation_answer_token_nll': .3, 'checkpoint_sha256': v2.file_sha(checkpoint),
                  'selection': 'minimum full validation answer-token NLL; earliest tie',
                  'registration': original, 'test_generation': {'not_a_score': 'Source admission must never inspect this field'}}
        contract = {'task_id': 'fixture_only', 'run_id': 'source_V1',
                    'matched_budget_registration': {'case': {'phase': 'trial'}}}
        receipt = {'performed': True, 'exit_code': 0, 'process_group_absent': True, 'termination_reason': None}
        for path, value in [(artifacts / 'history.json', history), (artifacts / 'selection_before_test.json', selection),
                            (artifacts / 'result.json', actual), (artifacts / 'run_preregistration.json', original),
                            (job / 'task-contract.json', contract), (job / 'execution-receipt.json', receipt)]:
            path.write_text(json.dumps(value), encoding='utf8')
        names = ['execution-receipt.json', 'task-contract.json', 'artifacts/result.json', 'artifacts/history.json',
                 'artifacts/selection_before_test.json', 'artifacts/best_reader.pt', 'artifacts/run_preregistration.json']
        standard = {'is_template': False, 'lifecycle_status': 'succeeded', 'review': {'status': 'accepted'},
                    'reproduction_level': 'training', 'execution': {'performed': True, 'exit_code': 0},
                    'task_id': contract['task_id'], 'run_id': contract['run_id'],
                    'contract_sha256': v2.file_sha(job / 'task-contract.json'),
                    'artifacts': [{'path': str(job / name), 'sha256': v2.file_sha(job / name)} for name in names]}
        (job / 'result-record.json').write_text(json.dumps(standard), encoding='utf8')
        (job / 'result-validation.json').write_text(json.dumps({'valid_for_acceptance': True}), encoding='utf8')
        registration = {'data_gate': gate, 'initialization': original['initialization']}
        store = SimpleNamespace(samples_sha256=gate['samples_sha256'], manifest=old_manifest,
            indices={'a': 0, 'b': 1, 'c': 2}, splits={'a': 'train', 'b': 'validation', 'c': 'test'},
            messages=messages.clone(), neutral=payload['neutral'].clone())
        return job, args, registration, store

    def test_complete_v1_reuse_requires_full_accepted_budget_and_earliest_validation(self):
        with tempfile.TemporaryDirectory(dir=v2.ROOT) as temporary:
            job, args, registration, store = self.reuse_fixture(temporary)
            source = v2.validate_complete_v1(job, args, registration, store, require_project_root=False)
            self.assertFalse(source['test_score_used_for_admission'])
            self.assertEqual(source['result']['best_epoch'], 2)
            self.assertEqual(source['result']['optimizer_steps'], 1548)
            self.assertTrue(source['evidence_sha256'])
            self.assertEqual(v2.validate_complete_v1(job / 'artifacts', args, registration, store,
                require_project_root=False)['checkpoint'], source['checkpoint'])
            with self.assertRaisesRegex(ValueError, 'registered matched-budget'):
                v2.validate_complete_v1(job, args, registration, store)

    def test_v1_reuse_refuses_wrong_config_partial_budget_reselection_and_changed_content(self):
        for problem in ('config', 'partial', 'validation_selection', 'cache_content', 'acceptance', 'artifact_tamper'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory(dir=v2.ROOT) as temporary:
                job, args, registration, store = self.reuse_fixture(temporary)
                if problem == 'config':
                    args.placement = 'mid'
                elif problem == 'partial':
                    p = job / 'artifacts/result.json'
                    value = json.loads(p.read_text());value['optimizer_steps'] = 1547;p.write_text(json.dumps(value))
                elif problem == 'validation_selection':
                    p = job / 'artifacts/history.json'
                    value = json.loads(p.read_text());value[2]['validation_answer_token_nll'] = .1;p.write_text(json.dumps(value))
                elif problem == 'cache_content':
                    store.messages[2, 0, 0] += 1
                elif problem == 'acceptance':
                    (job / 'result-validation.json').write_text(json.dumps({'valid_for_acceptance': False}))
                else:
                    with (job / 'artifacts/run_preregistration.json').open('a') as handle:
                        handle.write(' ')
                with self.assertRaises(ValueError):
                    v2.validate_complete_v1(job, args, registration, store, require_project_root=False)

    def test_eval_reused_performs_zero_optimizer_updates_and_keeps_original_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=v2.ROOT) as temporary:
            job, args, registration, store = self.reuse_fixture(temporary)
            source = v2.validate_complete_v1(job, args, registration, store, require_project_root=False)
            args.reuse_complete_v1 = job
            args.output = Path(temporary) / 'new_evaluation'
            args.output.mkdir()
            args.device = 'cpu'
            args.message_input = 'real'
            class InferenceOnly:
                base_lm = torch.nn.Linear(1, 1).requires_grad_(False)
                reader = MatchedReader()
                def load_reader_state(self, state):
                    self.reader.load_state_dict(state)
                def diagnostics(self, reset=False):
                    return {'fixture_only': True}
            def predictions(model, rows, tokenizer, store, args, handle, check):
                result = [{'sample_id': r['sample_id'], 'parse': v2.legacy.parse_answer('{"pKD":5.2}'),
                           'generated_token_count': 10} for r in rows]
                handle.write(json.dumps(result))
                return result
            with patch.object(v2, 'validate_complete_v1', return_value=source), \
                    patch.object(v2, 'test_c1_diagnostics', return_value={'fixture_only': True}), \
                    patch.object(v2, 'generated_predictions', side_effect=predictions), \
                    patch.object(torch.optim, 'AdamW', side_effect=AssertionError('No optimizer allowed')):
                result = v2.evaluate_reused_v1(InferenceOnly(), Tokenizer(), {'test': [row('c', 'test')]},
                    store, args, registration, lambda _: None)
            self.assertFalse(result['training_performed_in_this_run'])
            self.assertTrue(result['training_reused_from_v1'])
            self.assertEqual(result['new_optimizer_steps'], 0)
            self.assertEqual(result['performed_new_optimizer_steps'], 0)
            self.assertEqual(result['actual_epochs'], 3)
            self.assertEqual(result['optimizer_steps'], 1548)
            self.assertEqual(result['checkpoint_sha256'], v2.file_sha(source['checkpoint']))
            self.assertEqual(v2.file_sha(args.output / 'history.json'), v2.file_sha(job / 'artifacts/history.json'))


if __name__ == '__main__':
    unittest.main()
