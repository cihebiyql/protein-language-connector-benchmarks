"""CPU fixtures only; never load pretrained weights or run the real probe."""
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src import mm_probe as p
from src import mm_data as data
from test_data import Store, row
from test_model import fixture


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fixed_protocol_dimensions_and_no_tuning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'protocol.json'
            p.write_new(path, p.PROTOCOL)
            self.assertEqual(p.load_protocol(path), p.PROTOCOL)
            for key, value in (('seed', 42), ('ridge_lambda', 1.), ('variant', 'M2')):
                altered = {**p.PROTOCOL, key: value}
                path.write_text(json.dumps(altered))
                with self.assertRaises(ValueError):
                    p.load_protocol(path)
        self.assertEqual(64 + 16 + 10 + 2, p.PROTOCOL['base_dim'])
        self.assertEqual(p.PROTOCOL['base_dim'] + 64, p.PROTOCOL['joint_dim'])

    def test_base_features_are_input_only_and_fixed_92_dimensions(self):
        matrices = p.projections()
        self.assertEqual(matrices['B'].shape, (2560, 64))
        self.assertEqual(matrices['R'].shape, (1536, 64))
        np.testing.assert_array_equal(matrices['B'], p.projections()['B'])
        original = row()
        altered = copy.deepcopy(original)
        altered.update(pkd=-200, sample_id='other', source_doi='other', raw_metadata={'pkd': 999})
        np.testing.assert_array_equal(p.base_features(original, Store(), matrices),
                                      p.base_features(altered, Store(), matrices))
        x = p.base_features(original, Store(), matrices)
        self.assertEqual(x.shape, (92,))
        self.assertEqual(x.dtype, np.float32)
        np.testing.assert_array_equal(x[-2:], np.log1p([3, 4]).astype(np.float32))
        observed = data.condition_encoding(original)[1]
        np.testing.assert_array_equal(x[80:90], observed)

    def test_capture_pre_head_only_and_hook_cleanup(self):
        model, inputs = fixture()
        inputs['input_ids'] = inputs['input_ids'][:, :3]
        inputs['attention_mask'] = inputs['attention_mask'][:, :3]
        model.alpha.data.fill_(.5)
        model.train()
        old = len(model.numeric_head._forward_pre_hooks)
        captured = p.capture_prompt(model, inputs, width=12)
        self.assertEqual(captured.shape, (2, 12))
        self.assertEqual(captured.dtype, torch.float32)
        self.assertFalse(captured.requires_grad)
        self.assertTrue(model.training)
        self.assertFalse(model.base_lm.training)
        self.assertEqual(len(model.numeric_head._forward_pre_hooks), old)
        model.numeric_head.weight.data.fill_(999)
        np.testing.assert_array_equal(captured.numpy(), p.capture_prompt(model, inputs, 12).numpy())
        model.base_lm.fail = True
        with self.assertRaises(RuntimeError):
            p.capture_prompt(model, inputs, 12)
        self.assertEqual(len(model.numeric_head._forward_pre_hooks), old)
        self.assertTrue(model.training)

    def test_capture_rejects_labels_and_teacher_suffix(self):
        model, batch = fixture()
        with self.assertRaises(ValueError):
            p.capture_prompt(model, {**batch, 'labels': batch['input_ids']}, 12)
        with self.assertRaises(ValueError):
            p.capture_prompt(model, batch, 12)

    def test_ridge_is_same_width_train_only_and_reports_zero_columns(self):
        rng = np.random.default_rng(9)
        b, r = rng.normal(size=(24, 92)), rng.normal(size=(24, 64))
        y = 10 * r[:, 0] + 7
        state = p.fit_pair(b, r, y)
        self.assertEqual(state['baseline']['nominal_width'], 156)
        self.assertEqual(state['conditioned']['nominal_width'], 156)
        np.testing.assert_array_equal(state['baseline']['coefficients'][92:156], np.zeros(64))
        self.assertEqual(state['baseline']['receiver_coefficient_max_abs'], 0.)
        self.assertLessEqual(state['baseline']['centered_train_rank'], 23)
        self.assertGreater(state['conditioned']['effective_ridge_degrees_of_freedom_with_intercept'], 1)
        np.testing.assert_allclose(state['conditioned']['mean'][:92], b.mean(0))
        before = copy.deepcopy(state)
        p.predict(state['conditioned'], b[:2] * 100, r[:2] - 1000, 'conditioned')
        np.testing.assert_array_equal(before['conditioned']['mean'], state['conditioned']['mean'])
        with self.assertRaises(ValueError):
            p.fit_pair(b[:, :91], r, y)

    def test_metrics_delta_units_and_undefined_spearman(self):
        rows = [row(pkd=2), row(pkd=4)]
        score = p.metrics(rows, [3., 3.])
        self.assertEqual(score['MSE'], 1.)
        self.assertEqual(score['micro_MAE'], 1.)
        self.assertIsNone(score['global_Spearman'])
        self.assertEqual(score['Spearman_status'], 'undefined_constant_or_singleton')

    @contextlib.contextmanager
    def assets(self, root):
        """Tiny CPU fit identities; no checkpoint tensors are deserialized."""
        root = Path(root)
        scenario = data.SCENARIOS[0]
        protocol = root / 'probe.json'; p.write_new(protocol, p.PROTOCOL)
        config = root / 'config.json'
        training = {'checkpoint_selection': 'min_full_validation_answer_token_nll_earliest_tie'}
        p.write_new(config, {'schema_version': 'mm-v3.protocol.v1', 'status': 'frozen',
            'training': training, 'budget_signature': p.digest(training), 'scenarios': list(data.SCENARIOS),
            'variants': ['M3'], 'seeds': [17], 'deadline': '2026-10-17T10:29:40+08:00'})
        fit_root = root / 'fit'; fit_root.mkdir()
        checkpoint = fit_root / 'selected.pt'; checkpoint.write_bytes(b'not deserialized in readiness')
        p.write_new(fit_root / 'selection.json', {'checkpoint_sha256': data.file_sha(checkpoint)})
        views = root / 'views'; views.mkdir()
        splits = {'train': [], 'validation': [], 'test': []}
        for split, n in [('train', 8), ('validation', 3), ('test', 2)]:
            for i in range(n):
                sample = row(split=split, pkd=2 + i)
                sample['sample_id'] = f'{split}_{i}'
                splits[split].append(data.input_view(sample, labelled=split != 'test'))
        gold_root = root / data.DATA_RELATIVE / scenario; gold_root.mkdir(parents=True)
        gold = [{'sample_id': row['sample_id'], 'pkd': 9. + i} for i, row in enumerate(splits['test'])]
        gold_path = gold_root / 'test_gold.jsonl'
        gold_path.write_text(''.join(data.canonical(row) + '\n' for row in gold), encoding='utf8')
        pins = {**data.FROZEN_SHAS[scenario], 'test_gold.jsonl': data.file_sha(gold_path)}
        receipt = {'source_shas': pins, 'views': {}}
        for split, rows in splits.items():
            path = views / f'{split}.jsonl'
            path.write_text(''.join(data.canonical(row) + '\n' for row in rows), encoding='utf8')
            receipt['views'][split] = {'path': path.name, 'rows': len(rows), 'sha256': data.file_sha(path)}
        p.write_new(views / 'manifest.json', {'schema': 'mm_a1_views_v1',
            'condition_slots': list(data.CONDITION_SLOTS), 'scenarios': {scenario: receipt}})
        cache = root / 'cache'; cache.mkdir(); p.write_new(cache / 'manifest.json', {})
        identity = {'scenario': scenario, 'variant': 'M3', 'seed': 17,
            'protocol_sha256': data.file_sha(config), 'training_budget_sha256': p.digest(training)}
        p.write_new(fit_root / 'run_manifest.json', {'status': 'complete', 'execution_phase': 'fit',
            'identity': identity, 'artifacts': {'checkpoint': {'sha256': data.file_sha(checkpoint)}},
            'provenance': {'view_manifest_sha256': data.file_sha(views / 'manifest.json'),
                'code_sha256': {name: data.file_sha(Path(p.__file__).parent / name)
                    for name in ('mm_model.py', 'mm_data.py', 'mm_extract.py', 'mm_train.py')},
                'cache_before_run': {'manifest_sha256': data.file_sha(cache / 'manifest.json')}}})
        args = SimpleNamespace(command='extract', probe_protocol=str(protocol), config=str(config),
            checkpoint=str(checkpoint), scenario=scenario, view_root=str(views), cache_root=str(cache),
            project_root=str(root), output=str(root / 'out'), features=None,
            contract=None, validator=None, validator_sha256=None, device='cpu')
        with patch.dict(data.FROZEN_SHAS, {scenario: pins}), patch.dict(data.SPLIT_COUNTS,
                {scenario: {split: len(rows) for split, rows in splits.items()}}):
            yield args, splits, identity

    def test_readiness_no_model_load_and_exact_selected_identity(self):
        with tempfile.TemporaryDirectory() as root, self.assets(root) as (args, _, identity):
            with patch('torch.load', side_effect=AssertionError('no load')):
                config, actual, binding = p._context(args)
            self.assertEqual(actual, identity)
            self.assertEqual(binding['stage'], 'extract')
            self.assertEqual(p.execution_binding(args)['kind'], 'conditional_probe')
            with self.assertRaises(ValueError):
                p.authorize(args, binding)
            path = Path(args.checkpoint).parent / 'run_manifest.json'
            manifest = p.read_json(path); manifest['identity']['seed'] = 42
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                p._context(args)

    def test_readiness_incomplete_assets_is_explicit_not_ready(self):
        with tempfile.TemporaryDirectory() as root:
            protocol = Path(root) / 'probe.json'; p.write_new(protocol, p.PROTOCOL)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream), patch('torch.load', side_effect=AssertionError('no load')):
                code = p.main(['readiness', '--probe-protocol', str(protocol)])
            self.assertEqual(code, 3)
            self.assertEqual(json.loads(stream.getvalue())['status'], 'waiting_selected_fit_assets')

    def test_contract_binding_pins_device_and_rejected_validator_stops(self):
        with tempfile.TemporaryDirectory() as root, self.assets(root) as (args, _, _):
            _, _, binding = p._context(args)
            validator = Path(root) / 'validator.py'
            validator.write_text("def validate(*args, **kwargs):\n    return {'valid_for_execution': False}\n")
            args.validator = str(validator); args.validator_sha256 = data.file_sha(validator)
            contract = Path(root) / 'contract.json'; args.contract = str(contract)
            value = {'training_run_binding': {'kind': 'conditional_probe', 'probe': binding},
                'timing': {'effective_deadline_at': (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()},
                'execution': {'timeout_seconds': 30}}
            p.write_new(contract, value)
            with self.assertRaisesRegex(ValueError, 'standard validator rejected'):
                p.authorize(args, binding)
            altered = {**binding, 'device': 'cuda:0'}
            with self.assertRaisesRegex(ValueError, 'exact probe stage'):
                p.authorize(args, altered)
            validator.write_text("def validate(*args, **kwargs):\n    return {'valid_for_execution': True}\n")
            args.validator_sha256 = data.file_sha(validator)
            check, receipt = p.authorize(args, binding)
            check()
            self.assertEqual(receipt['contract_sha256'], data.file_sha(contract))

    def test_source_code_must_match_fitted_checkpoint_provenance(self):
        with tempfile.TemporaryDirectory() as root, self.assets(root) as (args, _, _):
            path = Path(args.checkpoint).parent / 'run_manifest.json'
            manifest = p.read_json(path)
            manifest['provenance']['code_sha256']['mm_model.py'] = '0' * 64
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'code differs'):
                p._context(args)

    def test_cpu_fixture_fit_freezes_predictions_before_test_gold(self):
        with tempfile.TemporaryDirectory() as root, self.assets(root) as (args, rows, identity):
            features = Path(root) / 'features'; features.mkdir()
            _, _, extract_binding = p._context(args)
            matrices = p.projections(); np.savez(features / 'projections.npz', **matrices)
            manifest = {'schema': 'mm_conditional_probe_features_v1', 'status': 'complete',
                'identity': identity, 'binding': extract_binding, 'protocol': p.PROTOCOL,
                'projection_sha256': data.file_sha(features / 'projections.npz'), 'splits': {}}
            rng = np.random.default_rng(4)
            for split, values in rows.items():
                path = features / f'{split}.npz'
                np.savez(path, B=rng.normal(size=(len(values), 92)).astype(np.float32),
                         R=rng.normal(size=(len(values), 1536)).astype(np.float32))
                manifest['splits'][split] = {'path': path.name, 'sha256': data.file_sha(path),
                    'ordered_ids': [row['sample_id'] for row in values]}
            p.write_new(features / 'manifest.json', manifest)
            args.command = 'fit'; args.features = str(features)
            config, actual, binding = p._context(args)
            original = data.read_jsonl
            def gold_guard(path):
                if Path(path).name == 'test_gold.jsonl':
                    self.assertTrue((Path(args.output) / 'probe_states.json').is_file())
                    self.assertTrue(all((Path(args.output) / f'{s}_predictions.jsonl').is_file()
                                        for s in rows))
                return original(path)
            with patch.object(data, 'read_jsonl', side_effect=gold_guard):
                result = p.fit(args, config, actual, binding, lambda: None, {'fixture_only': True})
            self.assertEqual(result['status'], 'complete')
            saved = p.read_json(Path(args.output) / 'result.json')
            self.assertTrue(saved['single_seed_diagnostic'])
            self.assertTrue(saved['not_mutual_information'])
            self.assertEqual(saved['splits']['test']['baseline']['n_total'], 2)
            states = p.read_json(Path(args.output) / 'probe_states.json')
            self.assertEqual(states['baseline']['receiver_coefficient_max_abs'], 0.)
            np.testing.assert_array_equal(states['baseline']['mean'][92:], np.zeros(64))


if __name__ == '__main__':
    unittest.main()
