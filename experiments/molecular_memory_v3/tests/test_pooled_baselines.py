"""CPU-only tiny arrays/cache fixtures; never fit or read the real A1 corpus."""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src import mm_pooled_baselines as b
from src import mm_data as d
from src import mm_extract as e

CONFIG = Path(__file__).resolve().parents[1] / 'configs/pooled_baselines.json'
HAS_TORCH = importlib.util.find_spec('torch') is not None


def row(identity, split='train', value=1., peptide='AC', target='EFGH'):
    result = {'sample_id': identity, 'split': split, 'peptide_sequence': peptide, 'target_sequence': target,
        'target_sequence_role': 'fixture_reference', 'chemical_state': {'n_terminal': 'free', 'c_terminal': 'amide'},
        'experimental_context': {'temperature_kelvin': None, 'ph': None, 'buffer': None, 'assay': 'binding',
                                 'actual_construct_sequence_missing': True, 'actual_construct_verified': False}}
    if split != 'test': result['pkd'] = value
    return result


class Store:
    def pooled(self, sequence):
        offset = 0 if sequence == 'AC' else 2
        return {'H15': np.full((8, 640), 1. + offset), 'H30': np.full((8, 640), 2. + offset)}


class PooledTests(unittest.TestCase):
    def test_exact_2700_layout_and_unweighted_slot_mean(self):
        r = row('a')
        features = b.feature_matrix([r], Store())[0]
        self.assertEqual(features.shape, (2700,))
        for i, value in enumerate((1., 2., 3., 4.)):
            np.testing.assert_array_equal(features[i * 640:(i + 1) * 640], value)
        np.testing.assert_array_equal(features[2560:2562], np.log1p([2, 4]))
        np.testing.assert_array_equal(features[2562:2572], np.asarray(d.condition_encoding(r)[1], dtype=np.float64))
        self.assertEqual(features[2572:].shape, (128,))

    def test_condition_hash_gold_id_and_unlisted_metadata_invariant(self):
        original = row('a')
        changed = copy.deepcopy(original)
        changed.update(sample_id='other', pkd=999, source_doi='secret', raw_metadata={'gold': -999})
        changed['experimental_context']['unlisted_gold'] = 999
        np.testing.assert_array_equal(b.condition_features(original), b.condition_features(changed))
        changed['experimental_context']['ph'] = 0.
        self.assertFalse(np.array_equal(b.condition_features(original), b.condition_features(changed)))
        self.assertEqual(b.condition_features(original).shape, (138,))

    def test_standardization_is_train_only_with_population_sd_and_floor(self):
        train = np.array([[1., 8.], [3., 8.]])
        center, scale = b.standardize_fit(train)
        np.testing.assert_array_equal(center, [2., 8.])
        np.testing.assert_array_equal(scale, [1., 1e-6])
        np.testing.assert_array_equal(b.transform(train, center, scale), [[-1., 0.], [1., 0.]])
        held = b.transform([[100., 8.]], center, scale)
        np.testing.assert_array_equal(held, [[98., 0.]])
        np.testing.assert_array_equal(center, [2., 8.])

    def test_primal_dual_ridge_equivalence_and_unpenalized_intercept(self):
        x = np.array([[1., 2., 7.], [3., -1., 7.], [8., 5., 7.], [2., 4., 7.]])
        y = np.array([1., 6., -3., 9.])
        primal = b.ridge_fit(x, y, formulation='primal')
        dual = b.ridge_fit(x, y, formulation='dual')
        np.testing.assert_allclose(b.ridge_predict(primal, x), b.ridge_predict(dual, x), rtol=0, atol=1e-12)
        constant = b.ridge_fit(x, np.full(4, 123.))
        np.testing.assert_array_equal(constant['coefficients'], np.zeros(3))
        np.testing.assert_array_equal(b.ridge_predict(constant, x), np.full(4, 123.))
        self.assertEqual(b.ridge_fit(x[:2], y[:2])['formulation'], 'dual')
        self.assertEqual(b.ridge_fit(x[:, :1], y)['formulation'], 'primal')

    def test_knn_fixed_distance_weights_stable_order_and_min_k(self):
        train = np.array([[0.], [0.], [2.]])
        y = np.array([2., 6., 10.])
        np.testing.assert_array_equal(b.knn_predict(train, y, [[0.]], k=1), [2.])
        weights = np.array([1e6, 1e6, 1 / (4 + 1e-6)])
        expected = (weights * y).sum() / weights.sum()
        self.assertEqual(b.knn_predict(train, y, [[0.]], k=25)[0], expected)
        np.testing.assert_array_equal(b.knn_predict(train[::-1], y[::-1], [[0.]], k=1), [6.])

    def test_nonfinite_arrays_invalid_parameters_or_payloads_rejected(self):
        for fn in (lambda: b.standardize_fit([[np.nan]]), lambda: b.ridge_fit([[0.]], [np.inf]),
                   lambda: b.ridge_fit([[0.]], [1.], alpha=True),
                   lambda: b.knn_predict([[0.]], [1.], [[0.]], k=True)):
            with self.assertRaises(ValueError): fn()
        class BadStore:
            def pooled(self, _): return {'H15': np.zeros((8, 640)), 'H30': np.full((8, 640), np.nan)}
        with self.assertRaises(ValueError): b.feature_matrix([row('a')], BadStore())

    def fixture(self, root):
        import torch
        torch.set_num_threads(2)
        datasets, views, cache = root / 'explicit_datasets', root / 'views', root / 'cache'
        datasets.mkdir(); views.mkdir(); cache.mkdir()
        config_path = root / 'config.json'
        config_path.write_bytes(CONFIG.read_bytes())
        manifest = {'schema': 'mm_a1_views_v1', 'condition_slots': list(d.CONDITION_SLOTS), 'scenarios': {}}
        pins, counts = {}, {}
        for number, scenario in enumerate(d.SCENARIOS):
            source = datasets / scenario
            destination = views / scenario
            source.mkdir(); destination.mkdir()
            offset = number * 100
            records = {'train': [row(scenario + 'a', value=1 + offset, peptide='AC'), row(scenario + 'b', value=2 + offset, peptide='AD'), row(scenario + 'c', value=9 + offset, peptide='AA')],
                'validation': [row(scenario + 'v', 'validation', value=2000., peptide='AD')],
                'test': [row(scenario + 't1', 'test'), row(scenario + 't2', 'test', peptide='AA')]}
            gold = [{'sample_id': records['test'][0]['sample_id'], 'pkd': -9.}, {'sample_id': records['test'][1]['sample_id'], 'pkd': 20.}]
            for name in ('samples.jsonl', 'manifest.json', 'benchmark_lock.json'):
                (source / name).write_text('{}\n', encoding='utf-8')
            (source / 'test_gold.jsonl').write_text(''.join(d.canonical(r) + '\n' for r in gold), encoding='utf-8')
            pins[scenario] = {p.name: d.file_sha(p) for p in source.iterdir()}
            counts[scenario] = {key: len(value) for key, value in records.items()}
            receipt = {'source_shas': pins[scenario], 'views': {}}
            for split, rows in records.items():
                path = destination / (split + '.jsonl')
                path.write_text(''.join(d.canonical(r) + '\n' for r in rows), encoding='utf-8')
                receipt['views'][split] = {'path': f'{scenario}/{path.name}', 'rows': len(rows), 'sha256': d.file_sha(path)}
            manifest['scenarios'][scenario] = receipt
        (views / 'manifest.json').write_text(d.canonical(manifest), encoding='utf-8')
        entries = {}
        for i, sequence in enumerate(('AC', 'AD', 'AA', 'EFGH', 'WV')):  # WV is unreferenced: still must be hashed.
            path = cache / (e.sequence_sha(sequence) + '.pt')
            torch.save({'H15': torch.full((8, 640), float(i + 1)), 'H30': torch.full((8, 640), float(i + 10))}, path)
            entries[e.sequence_sha(sequence)] = {'path': path.name, 'sha256': d.file_sha(path), 'length': len(sequence),
                'raw_residues_covered': len(sequence), 'raw_full_coverage': True, 'chunk_spans': [{'start': 0, 'stop': len(sequence)}]}
        cache_manifest = {'schema': 'mm_paired_esm_pooled_v1', 'model_revision': e.ESM_REVISION,
            'cache_status': 'complete', 'phase': 'full', 'expected_sequences': len(entries), 'sequences': entries,
            'semantics': e.SEMANTICS, 'model_pins': e.MODEL_PINS, 'source_semantics': {'verified': True,
                'source_sha256': e.REMOTE_TF4481_SOURCE_SHA, 'transformers_version': '4.48.1'}}
        (cache / 'manifest.json').write_text(d.canonical(cache_manifest), encoding='utf-8')
        request = b.make_request(config_path, root, datasets, views, cache, d.SCENARIOS[0])
        return request, pins, counts

    def authorize(self, root, prepared):
        standard = root / 'standard'
        standard.mkdir(exist_ok=True)
        validator = standard / 'validate_reproduction.py'
        validator.write_text('def validate(*args, **kwargs):\n    return {"valid_for_execution": True, "files_skipped": []}\n', encoding='utf-8')
        contract = root / ('contract_' + prepared['request']['scenario'] + '.json')
        contract.write_text(d.canonical({'training_run_binding': prepared['training_run_binding'],
            'timing': {'effective_deadline_at': (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()},
            'execution': {'timeout_seconds': 60}}), encoding='utf-8')
        return {'contract_path': contract, 'standard_root': standard, 'validator_sha256': d.file_sha(validator)}

    @unittest.skipUnless(HAS_TORCH, 'cache tensor deserialization fixture requires existing task Torch environment')
    def test_prepare_hashes_all_payloads_and_never_fits_or_parses_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            request, pins, counts = self.fixture(Path(directory))
            with patch.dict(d.FROZEN_SHAS, pins), patch.dict(d.SPLIT_COUNTS, counts), \
                 patch.object(b, 'ridge_fit', side_effect=AssertionError('fit forbidden')), \
                 patch.object(b, 'feature_matrix', side_effect=AssertionError('features forbidden')):
                original = d.read_jsonl
                def checked(path):
                    self.assertNotIn(Path(path).name, ('samples.jsonl', 'test_gold.jsonl'))
                    return original(path)
                with patch.object(d, 'read_jsonl', side_effect=checked): prepared = b.run(request)
                self.assertFalse(prepared['fit_performed'])
                self.assertFalse(prepared['test_gold_parsed'])
                self.assertEqual(prepared['training_run_binding']['cache_payloads_count'], 5)
                self.assertEqual(len(prepared['view_closure']['views']), 6)
                self.assertEqual(len(prepared['view_closure']['sources']), 8)
                self.assertFalse(Path(request['output_directory']).exists())

    @unittest.skipUnless(HAS_TORCH, 'cache tensor deserialization fixture requires existing task Torch environment')
    def test_complete_two_scenario_pipeline_and_saved_state_before_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, pins, counts = self.fixture(root)
            with patch.dict(d.FROZEN_SHAS, pins), patch.dict(d.SPLIT_COUNTS, counts):
                for number, scenario in enumerate(d.SCENARIOS):
                    request = b.make_request(first['config_file'], root, first['dataset_root'], first['view_root'], first['cache_root'], scenario)
                    auth = self.authorize(root, b.prepare(request))
                    original = d.read_jsonl
                    def checked(path):
                        self.assertNotEqual(Path(path).name, 'samples.jsonl')
                        if Path(path).name == 'test_gold.jsonl':
                            self.assertTrue((Path(request['output_directory']) / 'selected_states_before_test.json').is_file())
                        return original(path)
                    with patch.object(d, 'read_jsonl', side_effect=checked): result = b.run(request, execute=True, **auth)
                    self.assertEqual(result['status'], 'complete')
                    self.assertEqual(result['models'][b.METHODS[0]]['test']['n_finite'], 2)
                    self.assertEqual(result['train_state_metadata']['TRAIN_statistics']['train_sum'], 12. + 300 * number)
                    self.assertEqual(result['train_state_metadata']['TRAIN_statistics']['train_ids_sha256'], b.digest([scenario + 'a', scenario + 'b', scenario + 'c']))
                    self.assertFalse(result['generation_performed'])
                    with np.load(Path(request['output_directory']) / 'standardizer.npz', allow_pickle=False) as state:
                        self.assertEqual(state['center'].shape, (2700,))
                    with self.assertRaises(FileExistsError): b.run(request, execute=True, **auth)

    @unittest.skipUnless(HAS_TORCH, 'cache tensor deserialization fixture requires existing task Torch environment')
    def test_gate_cache_config_view_and_request_tamper_rejected(self):
        for mode in ('missing_contract', 'binding', 'unreferenced_payload', 'view', 'config'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                request, pins, counts = self.fixture(root)
                with patch.dict(d.FROZEN_SHAS, pins), patch.dict(d.SPLIT_COUNTS, counts):
                    prepared = b.prepare(request)
                    auth = self.authorize(root, prepared)
                    if mode == 'missing_contract': auth = {}
                    elif mode == 'binding':
                        contract = d.strict_loads(Path(auth['contract_path']).read_text())
                        contract['training_run_binding']['cache_manifest_sha256'] = '0' * 64
                        Path(auth['contract_path']).write_text(d.canonical(contract))
                    elif mode == 'unreferenced_payload':
                        path = Path(request['cache_root']) / (e.sequence_sha('WV') + '.pt')
                        path.write_bytes(b'tamper')
                    elif mode == 'view':
                        path = Path(request['view_root']) / d.SCENARIOS[1] / 'validation.jsonl'
                        path.write_text('{}\n')
                    else: Path(request['config_file']).write_text('{}')
                    with patch.object(b, 'ridge_fit', side_effect=AssertionError('fit forbidden')), self.assertRaises(ValueError):
                        b.run(request, execute=True, **auth)


if __name__ == '__main__':
    unittest.main()
