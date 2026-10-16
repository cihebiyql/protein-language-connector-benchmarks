"""CPU-only runner fixtures; never counted as scientific training or inference."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
    from torch import nn
    import transformers
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
if torch is not None:
    from foundation_20261016 import run_connectors as legacy
    from matched_budget_20261016 import run_trial as trial
    from matched_budget_20261016 import prepare_cache as prep
    from matched_budget_20261016.fixed_message import FixedMessageSender


def row(identifier, split='train', label=5.2, target='EEEEEE'):
    return {'sample_id': identifier, 'split': split, 'group_id': identifier,
            'peptide_sequence': 'ACD', 'target_sequence': target, 'pkd': label,
            'chemical_state': {'topology': 'Linear', 'c_terminal': 'Amide'},
            'experimental_context': {'assay': 'binding assay', 'buffer': None, 'ph': None},
            'target_sequence_role': 'database_reference_sequence_actual_assay_construct_unverified',
            'groups': {'target_sequence_sha256': 'target-' + target},
            'source_doi': 'never_prompt_this', 'raw_metadata': {'gold': 999.}}


class Tokenizer:
    pad_token_id, eos_token_id = 0, 1
    is_fast = True
    all_special_ids = [0, 1]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return json.dumps(messages, sort_keys=True) + '\nassistant:'

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 3 for c in text]

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        return ''.join(chr(x - 3) for x in ids if x >= 3)

    def __call__(self, text, **kwargs):
        return {'input_ids': self.encode(text), 'offset_mapping': [(i, i + 1) for i in range(len(text))],
                'special_tokens_mask': [0] * len(text)}


@unittest.skipIf(torch is None, 'Use existing WSL Torch/Transformers environment')
class MatchedRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_text_input_ids_mask_and_labels_exactly_equal_frozen_collation(self):
        records = [row('a', target='E' * 1030), row('b', label=-1.06307)]
        records[1]['experimental_context']['assay'] = 'a different and longer assay name'
        for supervised in (True, False):
            old = legacy.collate(records, Tokenizer(), Residues(), 'cpu', supervised)
            new = trial.collate(records, Tokenizer(), Messages(), 'cpu', supervised)
            for key in ('input_ids', 'attention_mask') + (('labels',) if supervised else ()):
                self.assertTrue(torch.equal(old[key], new[key]), key)
            self.assertNotIn('memory_type_ids', new)
            self.assertEqual(tuple(new['memory'].shape), (2, 16, 128))
            self.assertTrue(new['memory_mask'].all())
            if not supervised:
                self.assertNotIn('labels', new)

    def test_cache_sender_does_not_read_gold_and_neutral_fits_only_train_inputs(self):
        records = [row('train'), row('validation', 'validation'), row('test', 'test')]
        sender = FixedMessageSender()
        payload, _ = prep.build_messages(records, Residues(), sender, 'balanced')
        changed = copy.deepcopy(records)
        for r in changed:
            r['pkd'] = 999999.
            r['source_doi'] = 'other'
            r['raw_metadata'] = {'anything': 'different'}
        other, _ = prep.build_messages(changed, Residues(), sender, 'balanced')
        self.assertTrue(torch.equal(payload['messages'], other['messages']))
        self.assertTrue(torch.equal(payload['neutral'], payload['messages'][0]))
        self.assertEqual(payload['neutral_fit_train_sample_ids'], ['train'])
        # Held-out sequence change changes held-out messages but never the neutral fit.
        changed[2]['target_sequence'] = 'Y' * 9
        other, _ = prep.build_messages(changed, Residues(), sender, 'balanced')
        self.assertTrue(torch.equal(payload['neutral'], other['neutral']))
        self.assertFalse(torch.equal(payload['messages'][2], other['messages'][2]))

    def test_three_seed_initial_states_share_every_non_gate_tensor_across_gate_modes(self):
        with tempfile.TemporaryDirectory(dir=prep.ROOT) as temporary:
            records = prep.save_initializations(Path(temporary))
            self.assertEqual(len(records), 6)
            for seed in prep.SEEDS:
                states = [torch.load(prep.ROOT / records[f'{seed}:{mode}']['path'], weights_only=True)['reader_state']
                          for mode in ('zero', 'near_unit')]
                self.assertEqual(float(states[0]['alpha']), 0.)
                self.assertEqual(float(states[1]['alpha']), 3.)
                for key in states[0]:
                    if key != 'alpha':
                        self.assertTrue(torch.equal(states[0][key], states[1][key]))
                self.assertEqual(sum(x.numel() for x in states[0].values()), 464001)
            a = torch.load(prep.ROOT / records['17:zero']['path'], weights_only=True)['reader_state']
            b = torch.load(prep.ROOT / records['42:zero']['path'], weights_only=True)['reader_state']
            self.assertTrue(any(not torch.equal(a[k], b[k]) for k in a))

    def test_full_training_selection_precedes_one_test_pass_and_preserves_failure_denominator(self):
        model = FixtureModel()
        splits = {s: [row(s, s)] for s in ('train', 'validation', 'test')}
        logs = []
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(output=Path(temporary), seed=17, device='cpu', message_input='real')
            with patch.object(trial, 'validation_nll', side_effect=[.3, .2, .4]) as validation:
                result = trial.train_and_evaluate(model, Tokenizer(), splits, Messages(), args,
                                                  {'scope': 'cpu_fixture_only'}, logs.append)
            self.assertEqual(validation.call_count, 3)
            self.assertEqual(result['actual_epochs'], 3)
            self.assertEqual(result['optimizer_steps'], 3)
            self.assertEqual(result['best_epoch'], 2)
            self.assertEqual(model.generation_calls, 1)
            self.assertEqual(result['test_evaluation_passes'], 1)
            self.assertEqual(result['test_generation_passes'], 1)
            self.assertEqual(result['test_teacher_forced_C1_passes'], {'correct': 1, 'off': 1})
            self.assertTrue((args.output / 'selection_before_test.json').is_file())
            self.assertEqual(result['test_generation']['correct']['n_total'], 1)
            self.assertEqual(result['test_generation']['correct']['n_failed'], 1)
            self.assertIsNone(result['test_generation']['correct']['full_sample_mae'])
            self.assertTrue(any('alpha' in x for x in logs))
            saved = torch.load(args.output / 'best_reader.pt', weights_only=True)
            self.assertEqual(saved['best_epoch'], 2)
            self.assertFalse(any('base_lm' in x for x in saved['reader_state']))

    def test_legacy_source_pin_and_changed_manifest_fail_closed(self):
        prep.verify_legacy_source()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'manifest.json'
            path.write_text('{}', encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'manifest SHA'):
                trial.MessageStore(path, '0' * 64, prep.SCENARIOS[0], 'balanced', 'real')
        with patch.object(prep, 'file_sha', return_value='0' * 64):
            with self.assertRaisesRegex(ValueError, 'implementation changed'):
                prep.verify_legacy_source()

    def test_canary_is_train_only_and_does_not_create_test_output(self):
        model = FixtureModel()
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(output=Path(temporary), seed=17, device='cpu', message_input='neutral')
            result = trial.train_only_canary(model, Tokenizer(), [row('train')], Messages(), args,
                                             {'scope': 'cpu_fixture_only'}, lambda _: None)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['optimizer_steps'], 1)
            self.assertEqual(result['test_examples_evaluated'], 0)
            self.assertEqual(result['scientific_trial_count'], 0)
            self.assertFalse(list(Path(temporary).glob('*test*')))
            with self.assertRaises(ValueError):
                trial.train_only_canary(model, Tokenizer(), [row('test', 'test')], Messages(), args,
                                        {}, lambda _: None)

    def test_message_store_verifies_payload_identity_train_mean_and_tamper(self):
        with tempfile.TemporaryDirectory(dir=prep.ROOT) as temporary:
            folder = Path(temporary)
            data = folder / 'messages.pt'
            messages = torch.randn(3, 16, 128, generator=torch.Generator().manual_seed(82))
            payload = {'scenario': prep.SCENARIOS[0], 'sender_mode': 'balanced',
                       'sample_ids': ['a', 'b', 'c'], 'splits': ['train', 'validation', 'test'],
                       'messages': messages, 'neutral': messages[:1].mean(0),
                       'neutral_fit_train_sample_ids': ['a'], 'neutral_fit_uses_labels': False,
                       'samples_sha256': 'fixture_only'}
            torch.save(payload, data)
            manifest = {'purpose': 'matched_budget_fixed_message_cache',
                        'legacy_semantics_sha256': prep.LEGACY_SHA256,
                        'source_pins': {'scripts/foundation_20261016/run_connectors.py': prep.LEGACY_SHA256},
                        'datasets': {prep.SCENARIOS[0]: {'layouts': {'balanced': {
                            'path': prep.relative(data), 'sha256': prep.file_sha(data),
                            'neutral_sha256': prep.tensor_sha(payload['neutral'])}}}}}
            path = folder / 'manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf8')
            store = trial.MessageStore(path, prep.file_sha(path), prep.SCENARIOS[0], 'balanced', 'real')
            self.assertTrue(torch.equal(store.batch([row('c', 'test')]), messages[2:3]))
            neutral = trial.MessageStore(path, prep.file_sha(path), prep.SCENARIOS[0], 'balanced', 'neutral')
            self.assertTrue(torch.equal(neutral.batch([row('c', 'test')])[0], messages[0]))
            with self.assertRaises(ValueError):
                store.batch([row('c', 'train')])
            payload['neutral'] = messages.mean(0)  # Forbidden held-out influence.
            torch.save(payload, data)
            manifest['datasets'][prep.SCENARIOS[0]]['layouts']['balanced']['sha256'] = prep.file_sha(data)
            path.write_text(json.dumps(manifest), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'TRAIN-only mean'):
                trial.MessageStore(path, prep.file_sha(path), prep.SCENARIOS[0], 'balanced', 'neutral')

    def test_fixed_matrix_cannot_admit_unregistered_gate_or_pool_combinations(self):
        protocol = {'training': {'epochs': 3, 'batch_size': 4, 'learning_rate': .0003,
                    'weight_decay': .01, 'max_grad_norm': 1., 'optimizer_steps': 1548},
                    'generation': {'max_new_tokens': 24},
                    'timing': {'effective_deadline_at': '2026-10-16T23:20:00+08:00'}}
        admitted = 0
        for scenario in prep.SCENARIOS:
            for seed in prep.SEEDS:
                for placement in ('early', 'mid'):
                    for content in ('real', 'neutral'):
                        for mode in ('balanced', 'global16'):
                            for gate in ('zero', 'near_unit'):
                                args = SimpleNamespace(scenario=scenario, seed=seed, placement=placement,
                                                       message_input=content, sender_mode=mode, gate_init=gate)
                                try:
                                    trial.validate_trial_configuration(args, protocol)
                                    admitted += 1
                                except ValueError:
                                    pass
        self.assertEqual(admitted, 36)
        protocol['training']['epochs'] = 4
        with self.assertRaises(ValueError):
            trial.validate_trial_configuration(SimpleNamespace(scenario=prep.SCENARIOS[0], seed=17,
                placement='early', message_input='real', sender_mode='balanced', gate_init='zero'), protocol)

    def test_validation_diagnostics_reuse_forward_and_keep_criterion_exact(self):
        records = [row(str(i), 'validation', label=-1.23 + i) for i in range(5)]
        model = FixtureModel()
        expected = trial.validation_nll(model, records, Tokenizer(), Messages(), 'cpu', 4)
        before = model.forward_calls
        diagnostic = {}
        actual = trial.validation_nll(model, records, Tokenizer(), Messages(), 'cpu', 4,
                                       diagnostic_sink=diagnostic)
        self.assertEqual(actual, expected)
        self.assertEqual(model.forward_calls - before, 2)  # No diagnostic forward.
        self.assertEqual(actual, diagnostic['reference_supervised_nll']['nll_nats_per_token'])
        self.assertEqual(diagnostic['full']['count'], sum(diagnostic[k]['count'] for k in ('format', 'numeric', 'mixed')))
        self.assertEqual(diagnostic['full']['loss_sum'], sum(diagnostic[k]['loss_sum'] for k in ('format', 'numeric', 'mixed')))
        self.assertAlmostEqual(diagnostic['full']['loss_sum'], diagnostic['reference_supervised_nll']['loss_sum'], delta=.001)

    def test_c1_uses_original_per_sample_mean_and_separate_token_diagnostics(self):
        records = [row('a', 'test', -1.23), row('b', 'test', 12.8)]
        model, tokenizer = FixtureModel(), Tokenizer()
        batch = trial.collate(records, tokenizer, Messages(), 'cpu')
        labels = batch.pop('labels')
        old_correct, counts = trial.frozen_c1_answer_nll(model(**batch, message_mode='correct'), labels)
        old_off, _ = trial.frozen_c1_answer_nll(model(**batch, message_mode='off'), labels)
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(output=Path(temporary), device='cpu', message_input='real')
            before = model.forward_calls
            result = trial.test_c1_diagnostics(model, records, tokenizer, Messages(), args)
            self.assertEqual(model.forward_calls - before, 2)
            expected = sum(float(b) - float(a) for a, b in zip(old_correct, old_off)) / len(records)
            self.assertEqual(result['G_all_test'], expected)
            self.assertEqual(result['n_test'], 2)
            self.assertEqual(result['n_matched'], 0)
            self.assertIsNone(result['algebra_matched_mean'])
            self.assertAlmostEqual(result['G_decomposition']['sum_categories'], expected, delta=1e-5)
            self.assertFalse(result['selection_or_primary_score_changed'])
            for mode in ('correct', 'off'):
                value = result['token_breakdown'][mode]
                self.assertEqual(value['full']['count'], int(counts.sum()))
                self.assertAlmostEqual(value['full']['loss_sum'], sum(value[k]['loss_sum'] for k in ('format', 'numeric', 'mixed')), places=12)

    def test_strict_numerical_policy_requires_pre_torch_environment_and_correct_dtypes(self):
        with patch.object(trial, 'PRE_TORCH_ENVIRONMENT', {k: None for k in trial.NUMERICAL_ENVIRONMENT}):
            with self.assertRaisesRegex(ValueError, 'before importing torch'):
                trial.configure_numerical_policy(require_cuda_environment=True)
        receipt = trial.configure_numerical_policy(require_cuda_environment=False)
        self.assertTrue(receipt['deterministic_algorithms'])
        self.assertFalse(receipt['deterministic_warn_only'])
        self.assertEqual(receipt['float32_matmul_precision'], 'highest')
        self.assertFalse(receipt['cuda_matmul_allow_tf32'])
        self.assertFalse(receipt['cudnn_allow_tf32'])
        model = FixtureModel()
        with self.assertRaisesRegex(ValueError, 'base bfloat16'):
            trial.verify_model_numerics(model)
        model.base_lm.bfloat16()
        checked = trial.verify_model_numerics(model)
        self.assertEqual(checked['base_parameter_dtypes'], ['torch.bfloat16'])
        self.assertEqual(checked['reader_parameter_dtypes'], ['torch.float32'])

    def test_no_subset_or_primary_optimizer_tuning_arguments(self):
        parser = trial.parser()
        destinations = {action.dest for action in parser._actions}
        for forbidden in ('train_limit', 'validation_limit', 'test_limit', 'epochs', 'batch_size',
                          'learning_rate', 'max_grad_norm', 'max_new_tokens'):
            self.assertNotIn(forbidden, destinations)

    def test_malformed_message_shapes_and_missing_rows_fail(self):
        class Broken:
            def batch(self, rows):
                return torch.zeros(len(rows), 15, 128)
        with self.assertRaisesRegex(ValueError, 'B,16,128'):
            trial.collate([row('a')], Tokenizer(), Broken(), 'cpu')
        with self.assertRaises(ValueError):
            trial.collate([], Tokenizer(), Messages(), 'cpu')


if torch is not None:
    class Residues:
        def residues(self, sequence):
            # Sequence-dependent deterministic finite feature fixture.
            values = torch.arange(len(sequence) * 640, dtype=torch.float32).reshape(len(sequence), 640)
            code = sum(ord(x) for x in sequence) / max(len(sequence), 1)
            return torch.sin(values * (.001 + code / 10000.))

    class Messages:
        def batch(self, rows):
            return torch.ones(len(rows), 16, 128)

    class FixtureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_lm = nn.Linear(1, 1).requires_grad_(False)
            self.reader = nn.Module()
            self.reader.alpha = nn.Parameter(torch.tensor(0.))
            self.reader.weight = nn.Parameter(torch.tensor(.1))
            self.generation_calls = 0
            self.forward_calls = 0

        def forward(self, input_ids, labels=None, message_mode='correct', **kwargs):
            self.forward_calls += 1
            strength = self.reader.weight + self.reader.alpha if message_mode == 'correct' else torch.tensor(0.)
            logits = strength * torch.arange(256, dtype=torch.float32)
            return SimpleNamespace(logits=logits.expand(*input_ids.shape, -1))

        def diagnostics(self, reset=False):
            return {'alpha': float(self.reader.alpha.detach()), 'fixture_only': True}

        def reader_state(self):
            return {k: v.detach().clone() for k, v in self.reader.state_dict().items()}

        def load_reader_state(self, state):
            self.reader.load_state_dict(state)

        def greedy_generate(self, **kwargs):
            self.generation_calls += 1
            # Deliberately invalid output tests the frozen failure denominator.
            return torch.tensor([[ord('x') + 3, 1]])


if __name__ == '__main__':
    unittest.main()
