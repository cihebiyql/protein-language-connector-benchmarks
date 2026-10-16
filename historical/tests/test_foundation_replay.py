import copy
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
_spec = importlib.util.spec_from_file_location('replay_predictors',
    Path(__file__).resolve().parents[1] / 'scripts/foundation_20261016/replay_predictors.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
calibrate, interval_evaluation = _module.calibrate, _module.interval_evaluation
numerically_equal = _module.numerically_equal
compare_control_selection = _module.compare_control_selection


class ValidationCalibrationTests(unittest.TestCase):
    def rows(self, split, values):
        return [{'sample_id': f'{split}{i}', 'split': split, 'pkd': value}
                for i, value in enumerate(values)]

    def test_order_statistic_and_test_metrics(self):
        calibration = calibrate(self.rows('validation', [0, 1, 2, 3]), [0, 0, 0, 0], .75)
        self.assertEqual(calibration['threshold'], 2.)
        self.assertEqual(calibration['order_statistic_1based'], 3)
        result = interval_evaluation(self.rows('test', [2, 4]), [0, 0], calibration)
        self.assertEqual(result['test_empirical_coverage'], .5)
        self.assertEqual(result['interval_width_pKD'], 4.)
        self.assertEqual(result['test_mae_pKD'], 3.)
        self.assertFalse(result['guarantee_under_family_shift'])

    def test_test_labels_cannot_fit_threshold(self):
        rows = self.rows('validation', [1, 2, 3])
        fitted = calibrate(rows, [0, 0, 0], .9)
        original = copy.deepcopy(fitted)
        for test_labels in ([1, 2], [1000, -1000]):
            interval_evaluation(self.rows('test', test_labels), [0, 0], fitted)
            self.assertEqual(fitted, original)
        with self.assertRaises(ValueError):
            calibrate(self.rows('test', [1]), [0], .9)
        rows[0]['split'] = 'train'
        with self.assertRaises(ValueError):
            calibrate(rows, [0, 0, 0], .9)

    def test_invalid_input_rejected(self):
        for values, predictions, coverage in (([], [], .9), ([1], [float('nan')], .9),
                                               ([1], [], .9), ([1], [0], 1.),
                                               ([1], [0], 0.), ([1], [0], True)):
            with self.subTest(values=values, coverage=coverage), self.assertRaises(ValueError):
                calibrate(self.rows('validation', values), predictions, coverage)
        rows = self.rows('validation', [1, 2])
        rows[1]['sample_id'] = rows[0]['sample_id']
        with self.assertRaises(ValueError):
            calibrate(rows, [0, 0], .9)

    def test_overlap_and_invalid_evaluation_rejected(self):
        fitted = calibrate(self.rows('validation', [1]), [0], .9)
        overlapping = [{'sample_id': 'validation0', 'split': 'test', 'pkd': 1}]
        with self.assertRaises(ValueError):
            interval_evaluation(overlapping, [0], fitted)
        for split in ('train', 'validation'):
            with self.assertRaises(ValueError):
                interval_evaluation(self.rows(split, [1]), [0], fitted)
        for threshold in (-1, float('inf')):
            invalid = {**fitted, 'threshold': threshold}
            with self.assertRaises(ValueError):
                interval_evaluation(self.rows('test', [1]), [0], invalid)

    def test_portable_float_comparison_preserves_structure_and_tight_tolerance(self):
        self.assertTrue(numerically_equal({'x': [1., None, 'id']}, {'x': [1. + 1e-14, None, 'id']}))
        self.assertFalse(numerically_equal({'x': [1.]}, {'x': [1.01]}))
        self.assertFalse(numerically_equal({'x': [1.]}, {'x': [1., 2.]}))
        self.assertFalse(numerically_equal({'x': [1.]}, {'y': [1.]}))

    def test_changed_trial_is_disclosed_and_changed_selection_rejected(self):
        previous = {'selected_parameter': 25, 'test': {'mae': 1.},
                    'validation_trials': [{'parameter': 3, 'validation_mae': 2.},
                                          {'parameter': 25, 'validation_mae': 1.}]}
        current = copy.deepcopy(previous)
        current['validation_trials'][0]['validation_mae'] += .001
        self.assertEqual(len(compare_control_selection(current, previous)), 1)
        current['selected_parameter'] = 3
        with self.assertRaises(ValueError):
            compare_control_selection(current, previous)
        current = copy.deepcopy(previous)
        current['validation_trials'][1]['validation_mae'] += .001
        with self.assertRaises(ValueError):
            compare_control_selection(current, previous)


if __name__ == '__main__':
    unittest.main()
