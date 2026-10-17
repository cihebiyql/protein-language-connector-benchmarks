"""Stdlib CPU fixtures; never read or fit the real A1 dataset."""
import copy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import mm_baselines as b
from src import mm_baselines_cli as cli
from src import mm_data as d
from src import mm_evaluate as evaluation

S = d.SCENARIOS[0]


def row(i, split='train', pkd=1., sequence='ACD', target='EFGH'):
    value = {'sample_id': str(i), 'split': split, 'peptide_sequence': sequence,
        'target_sequence': target, 'target_sequence_role': 'fixture',
        'chemical_state': {'n_terminal': 'free', 'c_terminal': 'amide'},
        'experimental_context': {'temperature_kelvin': None, 'ph': None, 'buffer': None,
                                 'actual_construct_sequence_missing': True}}
    if split != 'test':
        value['pkd'] = pkd
    return value


class BaselineTests(unittest.TestCase):
    def test_train_statistics_exact_and_scenario_isolation(self):
        rows = [row('a', pkd=1), row('b', pkd=1), row('c', pkd=10)]
        stats = b.train_statistics(rows, S)
        self.assertEqual((stats['train_sum'], stats['train_count'], stats['train_mean'], stats['train_median']), (12, 3, 4, 1))
        self.assertEqual(stats['train_ids_sha256'], b._digest(['a', 'b', 'c']))
        model = b.fit_reference(rows, S, 'train_mean')
        self.assertEqual(b.predict_reference(model, [row('t', 'test')], S), [4])
        with self.assertRaises(ValueError):
            b.predict_reference(model, [row('t', 'test')], d.SCENARIOS[1])

    def test_invalid_train_labels_splits_and_duplicates_rejected(self):
        for rows in ([row('a', 'validation')], [row('a', pkd=True)], [row('a', pkd=math.nan)],
                     [row('a'), row('a')], [{**row('a'), 'scenario': d.SCENARIOS[1]}], []):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                b.train_statistics(rows, S)

    def test_features_allowlist_and_dimension(self):
        original = row('a')
        altered = copy.deepcopy(original)
        altered.update(pkd=-99, sample_id='other', split='test', source_doi='secret', raw_metadata={'gold': 99})
        altered['experimental_context']['assay'] = 'gold 99'
        self.assertEqual(len(b.old52_features(original)), 52)
        self.assertEqual(b.old52_features(original), b.old52_features(altered))
        self.assertEqual(b.old52_features(original)[-4:], [1., 1., 1., 1.])
        with self.assertRaises(ValueError):
            b.old52_features(row('a', sequence='AX'))

    def test_ridge_constant_intercept_is_not_penalized(self):
        state = b.fit_reference([row('a', pkd=3), row('b', pkd=3)], S, 'ridge52', 1000.)
        self.assertEqual(len(state['coefficients']), 53)
        self.assertAlmostEqual(b.predict_reference(state, [row('x', 'test')], S)[0], 3.)

    def test_knn_tied_distance_uses_train_order_and_requested_k(self):
        rows = [row('a', pkd=2), row('b', pkd=6)]
        one = b.fit_reference(rows, S, 'knn52', 1)
        both = b.fit_reference(rows, S, 'knn52', 25)
        self.assertEqual(b.predict_reference(one, [row('x', 'test')], S), [2.])
        self.assertEqual(b.predict_reference(both, [row('x', 'test')], S), [4.])

    def test_numeric_metrics_full_denominators_and_tied_spearman(self):
        rows = [row('a', pkd=1), row('b', pkd=1), row('c', pkd=3, target='AC')]
        result = b.numeric_metrics(rows, [2, 2, 2])
        self.assertEqual(result['micro_MAE'], 1.)
        self.assertIsNone(result['global_Spearman'])
        self.assertEqual((result['n_total'], result['n_finite'], result['n_exact_targets']), (3, 3, 2))
        self.assertAlmostEqual(b.numeric_metrics(rows, [0, 0, 9])['global_Spearman'], 1.)
        self.assertAlmostEqual(b.numeric_metrics(rows, [9, 9, 0])['global_Spearman'], -1.)
        for predictions in ([2, 2], [2, 2, math.inf]):
            with self.assertRaises(ValueError):
                b.numeric_metrics(rows, predictions)

    def test_metrics_match_registered_numeric_evaluator(self):
        import hashlib
        rows = [row('a', pkd=-2), row('b', pkd=1), row('c', pkd=9, target='AC')]
        for r in rows:
            r['groups'] = {'target_sequence_sha256': hashlib.sha256(r['target_sequence'].encode('ascii')).hexdigest()}
        pred = [3., -1., 8.]
        observed, expected = b.numeric_metrics(rows, pred), evaluation.regression(rows, pred)
        for ours, official in [('micro_MAE', 'mae'), ('RMSE', 'rmse'), ('global_Spearman', 'spearman'),
                               ('exact_target_macro_MAE', 'target_macro_mae')]:
            self.assertEqual(observed[ours], expected[official])

    def test_request_rejects_paths_hashes_and_unknown_methods(self):
        request = b.make_request(S, 'views/manifest.json', 'a' * 64, 'out')
        for key, value in [('view_manifest_sha256', None), ('view_manifest_relative', '../escape'),
                           ('output_relative', '/absolute'), ('methods', ['train_mean', 'train_mean']),
                           ('methods', ['pooled_H15']), ('knn_ks', [True]), ('ridge_alphas', [0])]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                b._check_request({**request, key: value})

    def fixture(self, root, scenario=S):
        """Create tiny pinned sources/views; patch production identities only in tests."""
        source = root / d.DATA_RELATIVE / scenario
        source.mkdir(parents=True)
        gold = [{'sample_id': 't1', 'pkd': -8}, {'sample_id': 't2', 'pkd': 20}]
        for name in ('samples.jsonl', 'benchmark_lock.json', 'manifest.json'):
            (source / name).write_text('{}\n', encoding='utf-8')
        (source / 'test_gold.jsonl').write_text(''.join(d.canonical(r) + '\n' for r in gold), encoding='utf-8')
        pins = {p.name: d.file_sha(p) for p in source.iterdir()}
        views = root / 'views'
        views.mkdir()
        rows = {'train': [row('a', pkd=1), row('b', pkd=1), row('c', pkd=10)],
                'validation': [row('v1', 'validation', 7), row('v2', 'validation', -1)],
                'test': [row('t1', 'test'), row('t2', 'test')]}
        entries = {}
        for split, records in rows.items():
            path = views / (split + '.jsonl')
            path.write_text(''.join(d.canonical(r) + '\n' for r in records), encoding='utf-8')
            entries[split] = {'path': path.name, 'rows': len(records), 'sha256': d.file_sha(path)}
        manifest = {'schema': 'mm_a1_views_v1', 'condition_slots': list(d.CONDITION_SLOTS),
                    'scenarios': {scenario: {'source_shas': pins, 'views': entries}}}
        path = views / 'manifest.json'
        path.write_text(d.canonical(manifest), encoding='utf-8')
        request = b.make_request(scenario, 'views/manifest.json', d.file_sha(path), 'out')
        return request, pins, {k: len(v) for k, v in rows.items()}

    def contract(self, root, request, accepted=True):
        validator = root / 'validator.py'
        validator.write_text('def validate(*args, **kwargs):\n    return {"valid_for_execution": ' + repr(accepted) + '}\n', encoding='utf-8')
        contract = root / 'contract.json'
        contract.write_text(d.canonical({'baseline_run_binding': b.execution_binding(request, root),
            'timing': {'effective_deadline_at': (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()},
            'execution': {'timeout_seconds': 60}}), encoding='utf-8')
        return {'contract_path': contract, 'validator_path': validator, 'validator_sha256': d.file_sha(validator)}

    def test_plan_does_not_fit_parse_gold_or_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request, pins, counts = self.fixture(root)
            with patch.dict(d.FROZEN_SHAS, {S: pins}), patch.dict(d.SPLIT_COUNTS, {S: counts}), \
                 patch.object(b, 'fit_reference', side_effect=AssertionError('fit forbidden')):
                real_read = d.read_jsonl
                def restricted(path):
                    self.assertFalse(str(path).endswith('test_gold.jsonl'))
                    self.assertFalse(str(path).endswith('samples.jsonl'))
                    return real_read(path)
                with patch.object(d, 'read_jsonl', side_effect=restricted):
                    result = b.run_baselines(request, root)
                self.assertFalse(result['fit_performed'])
                self.assertFalse((root / 'out').exists())

    def test_authorized_pipeline_freezes_before_test_and_never_parses_mixed_source(self):
        for scenario in d.SCENARIOS:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                request, pins, counts = self.fixture(root, scenario)
                authorization = self.contract(root, request)
                with patch.dict(d.FROZEN_SHAS, {scenario: pins}), patch.dict(d.SPLIT_COUNTS, {scenario: counts}):
                    real_read = d.read_jsonl
                    def checked(path):
                        self.assertNotEqual(Path(path).name, 'samples.jsonl')
                        if Path(path).name == 'test_gold.jsonl':
                            self.assertTrue((root / 'out/selection_before_test.json').is_file())
                            self.assertTrue((root / 'out/train_median.state.json').is_file())
                        return real_read(path)
                    with patch.object(d, 'read_jsonl', side_effect=checked):
                        result = b.run_baselines(request, root, execute=True, **authorization)
                    self.assertEqual(result['status'], 'complete')
                    self.assertEqual(result['models']['train_mean']['test']['micro_MAE'], 14.)
                    state = d.strict_loads((root / 'out/train_mean.state.json').read_text())
                    self.assertEqual(state['train_statistics']['train_sum'], 12.)
                    self.assertEqual(state['scenario'], scenario)
                    self.assertEqual(len(d.read_jsonl(root / 'out/train_mean.test.predictions.jsonl')), 2)
                    with self.assertRaises(FileExistsError):
                        b.run_baselines(request, root, execute=True, **authorization)

    def test_missing_rejected_or_changed_binding_contract_never_fits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request, pins, counts = self.fixture(root)
            with patch.object(b, 'fit_reference', side_effect=AssertionError('fit forbidden')):
                with self.assertRaises(ValueError):
                    b.run_baselines(request, root, execute=True)
                auth = self.contract(root, request, accepted=False)
                with self.assertRaisesRegex(ValueError, 'rejected'):
                    b.run_baselines(request, root, execute=True, **auth)
                with self.assertRaisesRegex(ValueError, 'bind'):
                    b.run_baselines({**request, 'output_relative': 'other'}, root, execute=True, **auth)

    def test_tampered_view_or_source_is_rejected(self):
        for target in ('views/train.jsonl', str(d.DATA_RELATIVE / S / 'test_gold.jsonl')):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                request, pins, counts = self.fixture(root)
                with (root / target).open('a') as stream:
                    stream.write(' ')
                with patch.dict(d.FROZEN_SHAS, {S: pins}), patch.dict(d.SPLIT_COUNTS, {S: counts}), self.assertRaises(ValueError):
                    b.run_baselines(request, root)

    def test_gold_field_in_test_view_rejected_even_if_rehashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request, pins, counts = self.fixture(root)
            test = root / 'views/test.jsonl'
            test.write_text(d.canonical({**row('t1', 'test'), 'pkd': 99}) + '\n' + d.canonical(row('t2', 'test')) + '\n')
            manifest = d.strict_loads((root / 'views/manifest.json').read_text())
            manifest['scenarios'][S]['views']['test']['sha256'] = d.file_sha(test)
            (root / 'views/manifest.json').write_text(d.canonical(manifest))
            request['view_manifest_sha256'] = d.file_sha(root / 'views/manifest.json')
            with patch.dict(d.FROZEN_SHAS, {S: pins}), patch.dict(d.SPLIT_COUNTS, {S: counts}), self.assertRaisesRegex(ValueError, 'gold'):
                b.run_baselines(request, root)

    def test_fixed_suite_resolves_actual_manifest_sha_without_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request, _, _ = self.fixture(root)
            recipe = {'schema': 'mm_numeric_baseline_suite.v1', 'view_manifest_relative': 'views/manifest.json',
                      'scenario_requests': {s: {'methods': ['train_mean', 'train_median'], 'output_relative': 'results/' + s} for s in d.SCENARIOS}}
            path = root / 'recipe.json'
            path.write_text(d.canonical(recipe))
            resolved = cli.resolve_request(path, root, S)
            self.assertEqual(resolved['view_manifest_sha256'], request['view_manifest_sha256'])
            self.assertEqual(resolved['methods'], ['train_mean', 'train_median'])
            recipe['scenario_requests'][S]['methods'].append('ridge52')
            path.write_text(d.canonical(recipe))
            with self.assertRaises(ValueError):
                cli.resolve_request(path, root, S)


if __name__ == '__main__':
    unittest.main()
