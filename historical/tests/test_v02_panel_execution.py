import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from bench_v02.contracts import ContractError, sha256
from bench_v02.panel_execution import disjointness, validate_panel_data
from bench_v02.run_simple_baselines import features, knn_predict, ridge_fit
from bench_v02.summarize_panel_baselines import paired_group_gain
import numpy as np


class PanelExecutionTests(unittest.TestCase):
    def evidence(self, directory, lock):
        paths = []
        for name in ('alignment_protocol.json', 'groups.jsonl', 'alignments.tsv'):
            path = Path(directory) / name
            path.write_text('unit fixture only', encoding='utf-8')
            paths.append(str(path))
            lock['pins'][str(path)] = sha256(path)
        lock['homology_evidence'] = {'protocol_path': paths[0], 'group_audit_path': paths[1], 'alignment_paths': paths[2:]}
        return lock

    def fixture(self):
        return [{'sample_id': str(i), 'panel_id': 'A1', 'peptide_sequence': ('ACDEFG', 'HIKLMN', 'PQRSTV')[i],
                 'target_sequence': 'FGHIKLMN', 'measurement_type': 'KD', 'relation': '=',
                 'label_exact': True, 'kd_molar': 1e-6, 'pkd': 6.0,
                 'chemical_state': {'topology': 'Linear', 'n_terminal': 'Free', 'c_terminal': 'Amide',
                                    'explicitly_representable': True},
                 'experimental_context': {}, 'eligibility': {'broad_reference_target': True},
                 'group_id': f'component{i}',
                 'groups': {'peptide_family': f'family{i}', 'source_doi': f'doi{i}', 'target_family': 'target'},
                 'split': split} for i, split in enumerate(('train', 'validation', 'test'))]

    def test_partial_external_scope_does_not_claim_full_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            samples, lock_path = Path(directory)/'samples.jsonl', Path(directory)/'lock.json'
            samples.write_text(''.join(json.dumps(r)+'\n' for r in self.fixture()), encoding='utf-8')
            lock = {'scope': 'panel_specific_predictor', 'panel_id': 'A1',
                    'scenario': 'new_peptide_known_target', 'status': 'frozen_restricted_scope',
                    'pins': {str(samples): sha256(samples)}, 'checks': {'external_isolation_verified': False}}
            self.evidence(directory, lock)
            lock_path.write_text(json.dumps(lock), encoding='utf-8')
            _, receipt = validate_panel_data(samples, lock_path, 'A1')
            self.assertFalse(receipt['external_isolation_verified'])
            self.assertFalse(receipt['unified_multitask_connector_claim'])
            lock['scenario'] = 'unregistered_target_setting'
            lock_path.write_text(json.dumps(lock), encoding='utf-8')
            with self.assertRaises(ContractError):
                validate_panel_data(samples, lock_path, 'A1')
            lock['scenario'] = 'new_peptide_known_target'
            lock['status'] = 'frozen'; lock_path.write_text(json.dumps(lock), encoding='utf-8')
            with self.assertRaises(ContractError):
                validate_panel_data(samples, lock_path, 'A1')

    def test_each_publication_is_checked_in_multivalue_lists(self):
        rows = [{'sample_id': '1', 'split': 'train', 'groups': {'publication': ['a', 'b']}},
                {'sample_id': '2', 'split': 'test', 'groups': {'publication': ['b', 'c']}}]
        self.assertFalse(disjointness(rows, 'groups.publication')['verified'])
        rows[1]['groups']['publication'] = ['c']
        self.assertTrue(disjointness(rows, 'groups.publication')['verified'])

    def test_tampered_data_and_real_family_overlap_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            samples, lock_path = Path(directory)/'samples.jsonl', Path(directory)/'lock.json'
            rows = self.fixture()
            rows[2]['groups']['peptide_family'] = rows[0]['groups']['peptide_family']
            samples.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
            lock = {'scope': 'panel_specific_predictor', 'panel_id': 'A1', 'status': 'frozen_restricted_scope',
                    'scenario': 'new_peptide_known_target', 'pins': {str(samples): sha256(samples)}}
            self.evidence(directory, lock)
            lock_path.write_text(json.dumps(lock), encoding='utf-8')
            with self.assertRaises(ContractError):
                validate_panel_data(samples, lock_path, 'A1')
            rows[2]['groups']['peptide_family'] = 'test_only'
            samples.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
            with self.assertRaises(ContractError):
                validate_panel_data(samples, lock_path, 'A1')

    def test_chemistry_case_and_numeric_baseline_predictions(self):
        row = self.fixture()[0]
        canonical = features(row)
        changed = copy.deepcopy(row)
        changed['chemical_state'].update(n_terminal='free', c_terminal='amide')
        self.assertEqual(canonical, features(changed))
        changed['chemical_state']['n_terminal'] = 'Acetyl'
        self.assertNotEqual(canonical, features(changed))
        x, y = np.array([[0.], [1.], [2.]]), np.array([0., 2., 4.])
        np.testing.assert_allclose(np.column_stack([x, np.ones(3)]) @ ridge_fit(x, y, .001), y, atol=.01)
        np.testing.assert_allclose(knn_predict(x, y, x, 1), y)

    def test_paired_bootstrap_preserves_cluster_sizes_and_direction(self):
        result=paired_group_gain([5.,5.,1.],[0.,0.,0.],['experiment_a','experiment_a','experiment_b'])
        self.assertAlmostEqual(result['gain_in_pKD_MAE'],11/3)
        self.assertEqual(result['n_independent_groups'],2)
        self.assertEqual(result['ci95'],[1.,5.])
        self.assertEqual(paired_group_gain([2.,4.],[2.,4.],['same','same'])['ci95'],[0.,0.])


if __name__ == '__main__':
    unittest.main()
