"""Scientific-report regressions: do not hide failure denominators or schema."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

MODULES = Path(__file__).resolve().parents[1] / 'scripts/foundation_20261016'
sys.path.insert(0, str(MODULES))
from make_report import normalized_generation, group_error_interval
import foundation

class FoundationReportingTests(unittest.TestCase):
    def test_information_index_requires_successful_validated_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'results'
            job = out / 'information_diagnostics/v3_all8/method_known_both'
            (job / 'artifacts').mkdir(parents=True)
            (job / 'artifacts/result.json').write_text(json.dumps({
                'status': 'complete', 'method': 'constant', 'scenario': 'known',
                'C1': {}, 'C2': {}}), encoding='utf8')
            with patch.object(foundation, 'ROOT', root), patch.object(foundation, 'OUT', out):
                self.assertEqual(foundation.build_index()['information_results'], [])
                (job / 'result-validation.json').write_text(json.dumps({
                    'valid_for_acceptance': True}), encoding='utf8')
                (job / 'execution-receipt.json').write_text(json.dumps({
                    'exit_code': 1, 'process_group_absent': True}), encoding='utf8')
                self.assertEqual(foundation.build_index()['information_results'], [])
                (job / 'execution-receipt.json').write_text(json.dumps({
                    'exit_code': 0, 'process_group_absent': True}), encoding='utf8')
                index = foundation.build_index()
                self.assertEqual(len(index['information_results']), 1)
                self.assertEqual(index['panels']['C2']['accepted_execution_jobs'], 1)

    def test_group_bootstrap_preserves_exact_paired_difference(self):
        value = group_error_interval([0., 0., 0.], [1., 1., 1.], [3., 3., 3.],
                                     ['family_a', 'family_a', 'family_b'], draws=100)
        self.assertEqual(value['baseline_minus_candidate_MAE'], 2.)
        self.assertEqual(value['paired_group_95_interval'], [2., 2.])
        self.assertEqual(value['groups'], 2)
        self.assertFalse(value['training_seed_uncertainty_included'])

    def test_single_independent_group_does_not_produce_interval(self):
        value = group_error_interval([0., 0.], [1., 1.], [2., 2.], ['a', 'a'])
        self.assertEqual(value['status'], 'undefined_fewer_than_two_independent_groups')
        self.assertNotIn('paired_group_95_interval', value)

    def test_bootstrap_rejects_nonpaired_nonfinite_and_invalid_draws(self):
        for gold, pred, control, groups, draws in [
            ([0.], [1., 2.], [2.], ['a'], 10),
            ([0.], [float('nan')], [2.], ['a'], 10),
            ([0.], [1.], [2.], [''], 10),
            ([0.], [1.], [2.], ['a'], 0)]:
            with self.assertRaises(ValueError):
                group_error_interval(gold, pred, control, groups, draws=draws)

    def test_actual_pipeline_schema_preserves_valid_full_score(self):
        value = normalized_generation({'n_total': 152, 'n_valid': 152, 'n_failed': 0,
            'generated_pKD_mae_valid_only': 1.5, 'full_sample_generated_mae': 1.5}, pipeline=True)
        self.assertEqual(value['full_sample_mae'], 1.5)

    def test_partial_off_stays_undefined_for_full_population(self):
        value = normalized_generation({'n_total': 152, 'n_valid': 27, 'n_failed': 125,
            'regression_valid_only': {'mae': 3.149}, 'full_sample_mae': None})
        self.assertIsNone(value['full_sample_mae'])
        self.assertEqual(value['n_failed'], 125)

    def test_false_full_score_on_partial_population_is_rejected(self):
        with self.assertRaises(ValueError):
            normalized_generation({'n_total': 152, 'n_valid': 27, 'n_failed': 125,
                'regression_valid_only': {'mae': 3.149}, 'full_sample_mae': 3.149})

    def test_invalid_count_or_nonfinite_error_is_rejected(self):
        for invalid in [dict(n_total=2, n_valid=1, n_failed=0),
                        dict(n_total=1, n_valid=True, n_failed=0),
                        dict(n_total=1, n_valid=1, n_failed=0,
                             regression_valid_only={'mae': float('nan')})]:
            with self.assertRaises(ValueError):
                normalized_generation(invalid)

if __name__ == '__main__':
    unittest.main()
