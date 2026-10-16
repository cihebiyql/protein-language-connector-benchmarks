"""CPU-only tiny Qwen and statistical-contract fixtures for C1/C2 diagnostics."""
import copy
from datetime import datetime, timedelta, timezone
import json
import os
import subprocess
import signal
import time
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from foundation_20261016.connectors import FrozenConnectorLM
from foundation_20261016.information_diagnostics import (
    ALPHAS, answer_nll, communication_algebra, conditioned_baseline, donor_key, fixed_projection,
    freeze_donors, grouped_bootstrap, last_active, plain_forward, probe_panel,
    prompt_states, ridge_fit)
from foundation_20261016.run_connectors import collate, prompt_text
from foundation_20261016.execute_information import checkpoint_receipt, parent_death_guard, sha, supervise


def model_fixture(method='internal_cross_attention', source_dim=8, insertion_layer=2):
    torch.manual_seed(71)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=48, hidden_size=24,
        intermediate_size=40, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, eos_token_id=47, pad_token_id=0)).eval()
    return FrozenConnectorLM(base, method, source_dim=source_dim, memory_dim=12,
                             num_memory_tokens=3, insertion_layer=insertion_layer).requires_grad_(False).eval()


def batch_fixture():
    torch.manual_seed(11)
    return {'input_ids': torch.tensor([[0, 0, 3, 4, 5], [2, 3, 4, 0, 0]]),
        'attention_mask': torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 0, 0]]),
        'memory': torch.randn(2, 7, 8),
        'memory_mask': torch.tensor([[1, 1, 0, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0]]),
        'memory_type_ids': torch.tensor([[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0]])}


def row_fixture(i, split='test', target='ACDE', peptide='AKLM'):
    return {'sample_id': str(i), 'group_id': 'locked:' + str(i), 'split': split, 'pkd': float(i % 7),
        'target_sequence': target, 'peptide_sequence': peptide,
        'chemical_state': {'n_terminal': 'Free', 'c_terminal': 'Amide', 'topology': 'Linear'},
        'experimental_context': {'temperature_kelvin': None, 'ph': None, 'buffer': None,
                                 'assay': 'test', 'actual_construct_sequence': None,
                                 'actual_construct_verified': False},
        'target_sequence_role': 'reference', 'groups': {'target_sequence_sha256': target}}


class InformationDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_last_real_token_handles_left_right_holes_and_empty(self):
        self.assertEqual(last_active(batch_fixture()['attention_mask']).tolist(), [4, 2])
        self.assertEqual(last_active(torch.tensor([[1, 0, 1]])).tolist(), [2])
        with self.assertRaises(ValueError):
            last_active(torch.zeros(1, 3))

    def test_tiny_qwen_capture_all_methods_no_grad_and_padding_invariance(self):
        for method in FrozenConnectorLM.METHODS:
            with self.subTest(method=method):
                model, batch = model_fixture(method), batch_fixture()
                initial = {name: value.clone() for name, value in model.state_dict().items()}
                values = prompt_states(model, batch)
                self.assertEqual(values['receiver'].shape, (2, 24))
                self.assertEqual(values['message'].shape, (2, 24))
                self.assertFalse(any(p.grad is not None or p.requires_grad for p in model.parameters()))
                changed = {key: value.clone() for key, value in batch.items()}
                changed['memory'][~changed['memory_mask'].bool()] = 10000
                other = prompt_states(model, changed)
                for key in values:
                    np.testing.assert_allclose(values[key], other[key], atol=1e-6)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, initial[key], atol=0, rtol=0)
                self.assertFalse(model.base_lm._forward_pre_hooks)
                self.assertTrue(all(not block._forward_hooks for block in model.base_lm.model.layers))

    def test_receiver_capture_includes_injection_in_final_decoder_layer(self):
        model, batch = model_fixture(), batch_fixture()
        captured = prompt_states(model, batch)
        originals = []
        handle = model.base_lm.model.layers[-1].register_forward_hook(
            lambda _module, _args, output: originals.append((output[0] if isinstance(output, tuple) else output).detach()))
        try:
            model(**batch)
        finally:
            handle.remove()
        positions = last_active(batch['attention_mask'])
        original = originals[0][torch.arange(2), positions].float().numpy()
        np.testing.assert_allclose(captured['receiver'], original + captured['message'], atol=1e-6)

    def test_capture_rejects_labels_and_cleans_up_on_failure(self):
        model, batch = model_fixture(), batch_fixture()
        with self.assertRaises(ValueError):
            prompt_states(model, {**batch, 'labels': batch['input_ids']})
        with patch.object(model.base_lm.model.layers[-1], 'forward', side_effect=RuntimeError('fixture')):
            with self.assertRaises(RuntimeError):
                prompt_states(model, batch)
        self.assertFalse(model.base_lm._forward_pre_hooks)
        self.assertFalse(model.output_projection._forward_hooks)
        self.assertTrue(all(not block._forward_hooks for block in model.base_lm.model.layers))

    def test_off_fully_frozen_matches_plain_and_answer_nll_is_token_mean(self):
        model, batch = model_fixture(), batch_fixture()
        output = plain_forward(model, batch)
        labels = batch['input_ids'].clone()
        labels[:, :3] = -100
        labels[1, 3] = 10
        nll, counts = answer_nll(output, labels)
        expected = torch.nn.functional.cross_entropy(output.logits[:, :-1].float().transpose(1, 2),
            labels[:, 1:], ignore_index=-100, reduction='none')
        np.testing.assert_allclose(nll, (expected.sum(1) / (labels[:, 1:] != -100).sum(1)).detach().numpy())
        self.assertEqual(counts.tolist(), [2, 2])
        changed = {**batch, 'memory': batch['memory'] + 100}
        torch.testing.assert_close(output.logits, plain_forward(model, changed).logits, rtol=0, atol=0)

    def test_inductive_projection_test_values_cannot_change_fit(self):
        train = np.arange(45, dtype=float).reshape(9, 5)
        all_values = np.concatenate((train, np.ones((2, 5))))
        first, state = fixed_projection(train, all_values, 128)
        changed = all_values.copy()
        changed[-2:] += 1e9
        second, other_state = fixed_projection(train, changed, 128)
        for key in state:
            np.testing.assert_array_equal(state[key], other_state[key])
        np.testing.assert_array_equal(first[:9], second[:9])
        self.assertEqual(first.shape, (11, 128))
        with self.assertRaises(ValueError):
            fixed_projection(train, np.zeros((1, 6)))

    def test_ridge_singular_and_group_bootstrap(self):
        x, y = np.zeros((5, 9)), np.arange(5.)
        for alpha in (0, .1):
            weights = ridge_fit(x, y, alpha)
            np.testing.assert_allclose(np.column_stack((x, np.ones(5))) @ weights, y.mean())
        self.assertEqual(grouped_bootstrap(y, y + 1, y, ['a'] * 5)['status'], 'undefined_fewer_than_two_target_groups')
        interval = grouped_bootstrap(y, y + 1, y, ['a', 'a', 'b', 'b', 'b'], repeats=20)
        self.assertEqual(interval['delta_mse_95_percentile_interval'], [1., 1.])

    def test_donor_map_exact_strata_unknowns_and_label_independence(self):
        rows = [row_fixture(0), row_fixture(1, peptide='GQLM'), row_fixture(2, peptide='G')]
        rows += [row_fixture(3, target='ACDF', peptide='AAAA')]
        first = freeze_donors(rows)
        self.assertEqual([r['donor_index'] for r in first], [1, 0, None, None])
        changed = copy.deepcopy(rows)
        for row in changed:
            row['pkd'] = 10000
            row['source_doi'] = 'DOI must not enter donor selection'
        self.assertEqual(first, freeze_donors(changed))
        changed[1]['experimental_context']['ph'] = 7
        self.assertIsNone(freeze_donors(changed)[0]['donor_index'])
        self.assertNotEqual(donor_key(changed[0]), donor_key(changed[1]))

    def test_conditioned_baseline_context_statistics_train_only(self):
        rows = [row_fixture(i, 'train' if i < 3 else 'test') for i in range(5)]
        for i, row in enumerate(rows):
            row['experimental_context']['temperature_kelvin'] = 280. + i
            row['experimental_context']['ph'] = 6. + i / 10
        first, first_receipt = conditioned_baseline(rows, [0, 1, 2])
        changed = copy.deepcopy(rows)
        for row in changed[3:]:
            row['pkd'] = -10000
            row['experimental_context']['ph'] = 10000
            row['experimental_context']['temperature_kelvin'] = 1e9
            row['experimental_context']['assay'] = 'heldout independent context'
        second, second_receipt = conditioned_baseline(changed, [0, 1, 2])
        self.assertEqual(first_receipt, second_receipt)
        self.assertEqual(first.shape, (5, 200))
        self.assertEqual(first_receipt['allowed_context_dimensions'], 148)
        self.assertEqual(first_receipt['baseline_dimensions'], 200)
        self.assertEqual(first_receipt['context_numeric_statistics_TRAIN_only']['ph']['n_observed_training_rows'], 3)
        np.testing.assert_array_equal(first[:3], second[:3])
        self.assertFalse(np.array_equal(first[3:], second[3:]))

    def test_donors_exclude_distinct_sequences_in_same_locked_group(self):
        rows = [row_fixture(0), row_fixture(1, peptide='GQLM')]
        rows[1]['group_id'] = rows[0]['group_id']
        self.assertTrue(all(r['donor_index'] is None for r in freeze_donors(rows)))
        rows[1].pop('group_id')
        with self.assertRaises(ValueError):
            freeze_donors(rows)

    def test_GPH_algebra(self):
        score = communication_algebra(2.3, 3.1, 4.7)
        self.assertAlmostEqual(score['G'], .8)
        self.assertAlmostEqual(score['P'], 2.4)
        self.assertAlmostEqual(score['H'], 1.6)
        self.assertLess(abs(score['identity_residual_P_minus_G_minus_H']), 1e-12)

    def test_probe_shapes_same_capacity_and_no_test_label_selection(self):
        rng = np.random.default_rng(17)
        rows = [row_fixture(i, 'train' if i < 12 else 'validation' if i < 18 else 'test',
                            target='ACDE' if i % 2 else 'ACDF', peptide='AAAA' if i % 2 else 'GGGG') for i in range(24)]
        arrays = {name: rng.normal(size=(24, size)) for name, size in [('raw', 8), ('message', 12), ('receiver', 16)]}
        changed = copy.deepcopy(rows)
        for row in changed[18:]:
            row['pkd'] += 100
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = probe_panel(rows, arrays, first)
            for name in arrays:
                self.assertEqual(a['models'][name]['grouped_bootstrap']['n_groups'], 6)
            b = probe_panel(changed, arrays, second)
            self.assertEqual({value['input_dimension'] for value in a['models'].values()}, {328})
            self.assertEqual(a['conditioning_baseline']['baseline_dimensions'], 200)
            for key in a['models']:
                self.assertEqual(a['models'][key]['selected_alpha'], b['models'][key]['selected_alpha'])
                self.assertEqual(len(a['models'][key]['validation_trials']), len(ALPHAS))
            with np.load(Path(first) / 'probe_states.npz') as sa, np.load(Path(second) / 'probe_states.npz') as sb:
                for key in sa:
                    np.testing.assert_array_equal(sa[key], sb[key])

    def test_prompt_collation_does_not_tokenize_gold_or_ids(self):
        class Tokenizer:
            pad_token_id, eos_token_id = 0, 47
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return str(messages)
            def encode(self, text, add_special_tokens=False):
                return [2 + ord(c) % 40 for c in text]
        class Store:
            def residues(self, sequence):
                return torch.zeros(len(sequence), 640)
        row = row_fixture(0)
        row['sample_id'] = 'SECRET_ID_1928'
        row['pkd'] = 777.1234
        tok = Tokenizer()
        text = prompt_text(row, tok)
        self.assertNotIn('777.1234', text)
        self.assertNotIn('SECRET_ID_1928', text)
        with patch('foundation_20261016.run_connectors.answer_text', side_effect=AssertionError('answer leak')):
            batch = collate([row], tok, Store(), 'cpu', supervised=False)
        self.assertNotIn('labels', batch)
        self.assertEqual(batch['input_ids'].shape[1], len(text))


class SourceArchiveAdmissionTests(unittest.TestCase):
    def fixture(self, directory):
        job = Path(directory)
        artifacts = job / 'artifacts'
        artifacts.mkdir()
        checkpoint = artifacts / 'best_connector.pt'
        checkpoint.write_bytes(b'fixture connector bytes, never a scientific model')
        def write(path, value):
            path.write_text(json.dumps(value), encoding='utf-8')
        (job / 'stdout.log').write_text('fixture stdout')
        (job / 'stderr.log').write_text('')
        contract = {'is_template': False, 'contract_status': 'frozen', 'task_id': 'fixture', 'run_id': 'fixture',
                    'execution': {'command': ['fixture']}}
        write(job / 'task-contract.json', contract)
        write(artifacts / 'input_manifest.json', {'scope': 'fixture'})
        write(artifacts / 'run_preregistration.json', {'scope': 'full_locked_A1_panel',
            'configuration': {'seed': 17, 'scenario': 'new_peptide_known_target'},
            'input_manifest_sha256': sha(artifacts / 'input_manifest.json')})
        write(artifacts / 'result.json', {'status': 'complete', 'actual_epochs': 3,
            'checkpoint_path': str(checkpoint), 'checkpoint_sha256': sha(checkpoint)})
        receipt = {'exit_code': 0, 'process_group_absent': True, 'termination_reason': None,
                   'pid': 123, 'pgid': 123, 'stdout_sha256': sha(job / 'stdout.log'), 'stderr_sha256': sha(job / 'stderr.log')}
        write(job / 'execution-receipt.json', receipt)
        write(job / 'result-validation.json', {'valid_for_acceptance': True})
        record = {'is_template': False, 'task_id': 'fixture', 'run_id': 'fixture',
            'lifecycle_status': 'succeeded', 'reproduction_level': 'training', 'review': {'status': 'accepted'},
            'contract_path': str(job / 'task-contract.json'), 'contract_sha256': sha(job / 'task-contract.json'),
            'execution': {'performed': True, 'exit_code': 0, 'pid': 123, 'pgid': 123, 'command': ['fixture'],
                'receipt_path': str(job / 'execution-receipt.json'), 'exit_verification_path': str(job / 'execution-receipt.json')},
            'artifacts': [{'path': str(path), 'location': 'local', 'available': True,
                'sha256': sha(path), 'size_bytes': path.stat().st_size} for path in
                (artifacts / 'result.json', job / 'execution-receipt.json', job / 'stdout.log', job / 'stderr.log')]}
        write(job / 'result-record.json', record)
        return SimpleNamespace(checkpoint=checkpoint, scenario='new_peptide_known_target'), job

    def mutate(self, path, **updates):
        value = json.loads(path.read_text(encoding='utf-8'))
        value.update(updates)
        path.write_text(json.dumps(value), encoding='utf-8')

    def test_source_archive_positive_bound_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(directory)
            result, registration, _ = checkpoint_receipt(args)
            self.assertEqual(result['actual_epochs'], 3)
            self.assertEqual(registration['configuration']['seed'], 17)

    def test_source_missing_terminal_archive_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            args, job = self.fixture(directory)
            (job / 'result-record.json').unlink()
            with self.assertRaisesRegex(ValueError, 'Source archive missing'):
                checkpoint_receipt(args)

    def test_source_nonzero_exit_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            args, job = self.fixture(directory)
            self.mutate(job / 'execution-receipt.json', exit_code=1)
            with self.assertRaisesRegex(ValueError, 'clean terminal exit'):
                checkpoint_receipt(args)

    def test_source_group_still_present_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            args, job = self.fixture(directory)
            self.mutate(job / 'execution-receipt.json', process_group_absent=False)
            with self.assertRaisesRegex(ValueError, 'absent process group'):
                checkpoint_receipt(args)

    def test_source_unaccepted_validation_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            args, job = self.fixture(directory)
            self.mutate(job / 'result-validation.json', valid_for_acceptance=False)
            with self.assertRaisesRegex(ValueError, 'accepted standard validation'):
                checkpoint_receipt(args)

    def test_source_contract_hash_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            args, job = self.fixture(directory)
            self.mutate(job / 'result-record.json', contract_sha256='0' * 64)
            with self.assertRaisesRegex(ValueError, 'contract binding/hash mismatch'):
                checkpoint_receipt(args)

    def test_source_scientific_artifact_hash_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            args, job = self.fixture(directory)
            self.mutate(job / 'artifacts/result.json', status='tampered')
            with self.assertRaisesRegex(ValueError, 'artifact hash/size/path mismatch'):
                checkpoint_receipt(args)

    def test_source_checkpoint_hash_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(directory)
            args.checkpoint.write_bytes(b'tampered connector')
            with self.assertRaisesRegex(ValueError, 'checkpoint path/hash mismatch'):
                checkpoint_receipt(args)

    def test_source_record_receipt_binding_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            args, job = self.fixture(directory)
            record = json.loads((job / 'result-record.json').read_text())
            record['execution']['pid'] = 999
            (job / 'result-record.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'receipt binding mismatch'):
                checkpoint_receipt(args)


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux process controls')
class InformationWrapperTests(unittest.TestCase):
    def test_stdlib_wrapper_import_and_bad_parent_refuse_before_torch(self):
        script = (Path(__file__).resolve().parents[1] / 'scripts/foundation_20261016/execute_information.py')
        command = [sys.executable, '-c', "import importlib.util,sys; spec=importlib.util.spec_from_file_location('guard',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); assert 'torch' not in sys.modules; m.parent_death_guard(0)", str(script)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Worker parent changed before guard installation', result.stderr)

    def test_successful_child_wait_and_absence_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            receipt = supervise([sys.executable, '-c', 'print(123)'], job, 5,
                                datetime.now(timezone.utc) + timedelta(seconds=10))
            self.assertEqual(receipt['exit_code'], 0)
            self.assertTrue(receipt['process_group_absent'])
            self.assertEqual(receipt['remaining_processes'], [])
            self.assertEqual((job / 'stdout.log').read_text().strip(), '123')
            self.assertTrue((job / 'execution-receipt.json').exists())

    def test_timeout_kills_only_owned_group_and_waits(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = supervise([sys.executable, '-c', 'import time; time.sleep(20)'], Path(directory), .1,
                                datetime.now(timezone.utc) + timedelta(seconds=10))
            self.assertEqual(receipt['termination_reason'], 'timeout')
            self.assertNotEqual(receipt['exit_code'], 0)
            self.assertTrue(receipt['process_group_absent'])

    def test_controller_sigterm_cleanup_and_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            driver = job / 'driver.py'
            scripts = str(Path(__file__).resolve().parents[1] / 'scripts')
            driver.write_text("import sys; sys.path.insert(0, " + repr(scripts) + "); "
                "from foundation_20261016.execute_information import supervise; "
                "from pathlib import Path; from datetime import datetime,timedelta,timezone; "
                "supervise([sys.executable,'-c','import time;time.sleep(20)'],Path(sys.argv[1]),10,datetime.now(timezone.utc)+timedelta(seconds=20))")
            process = subprocess.Popen([sys.executable, str(driver), str(job)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 10
                while not (job / 'running-process.json').exists() and time.monotonic() < deadline and process.poll() is None:
                    time.sleep(.02)
                if not (job / 'running-process.json').exists():
                    _, stderr = process.communicate(timeout=5)
                    self.fail('Fixture controller failed to start: ' + stderr.decode())
                os.kill(process.pid, signal.SIGTERM)
                _, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, stderr)
                receipt = json.loads((job / 'execution-receipt.json').read_text())
                self.assertEqual(receipt['termination_reason'], 'controller_signal_15')
                self.assertTrue(receipt['process_group_absent'])
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                process.stderr.close()

    def test_memory_negative_control_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = supervise([sys.executable, '-c', 'import time; x=bytearray(10000000); time.sleep(20)'], Path(directory), 5,
                                datetime.now(timezone.utc) + timedelta(seconds=10), memory_limit=1)
            self.assertEqual(receipt['termination_reason'], 'whole_group_host_rss_limit')
            self.assertTrue(receipt['process_group_absent'])


if __name__ == '__main__':
    unittest.main()
