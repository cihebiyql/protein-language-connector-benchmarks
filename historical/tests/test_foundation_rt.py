import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'scripts/foundation_20261016/rt_baselines.py'
spec = importlib.util.spec_from_file_location('rt_foundation', PATH)
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)


def row(seq='ACM', mods=None, y=10):
    return {'sample_id': seq + str(mods), 'peptide_sequence': seq, 'modifications': mods or [], 'rt_value': y, 'rt_unit': 'min', 'condition_id': rt.CONDITION, 'eligibility': {'measurement_valid': True}}


class FoundationRT(unittest.TestCase):
    def test_split_all_modforms_same_sequence(self):
        rows = [row('A' * n + 'CM') for n in range(1, 101)]
        rows += [row('ACM', [{'name': 'Oxidation', 'position': 3}])]
        a, roles = rt.assignments(rows)
        b, roles2 = rt.assignments(list(reversed(rows)))
        self.assertEqual(a, b)
        self.assertEqual(roles, roles2)
        self.assertEqual(len(a), 100)
        self.assertEqual(set(a.values()), {'train', 'validation', 'test'})
        self.assertTrue(all(a[s] == 'validation' for s in roles))

    def test_unknown_modification_not_silently_stripped(self):
        r = row(mods=[{'name': 'Unknown', 'position': 1}])
        with self.assertRaises(ValueError):
            rt.features([r])

    def test_modification_positions_and_unit_rejected(self):
        r = row(mods=[{'name': 'Oxidation', 'position': 1}])
        self.assertIn('modification_residue_mismatch', rt.chemistry_errors(r))
        r = row(); r['rt_unit'] = 'iRT'
        self.assertIn('wrong_condition_or_unit', rt.chemistry_errors(r))

    def test_duplicate_modifications_rejected(self):
        m = {'name': 'Oxidation', 'position': 3}
        self.assertIn('duplicate_modification', rt.chemistry_errors(row(mods=[m, m])))

    def test_known_ptm_changes_features(self):
        a = rt.features([row(), row(mods=[{'name': 'Oxidation', 'position': 3}])])
        self.assertEqual(a[1, -1] - a[0, -1], 1)

    def test_error_metric_math_and_constant_rank(self):
        m = rt.metrics([1, 3], [2, 2])
        self.assertEqual(m['MAE_min'], 1)
        self.assertEqual(m['RMSE_min'], 1)
        self.assertIsNone(m['Spearman_same_condition'])
        with self.assertRaises(ValueError):
            rt.metrics([1], [float('nan')])

    def test_scaler_uses_only_fit_array(self):
        import numpy as np
        x, mean, std = rt.normalize_fit(np.array([[0., 4.], [2., 4.]]))
        self.assertEqual(mean.tolist(), [1., 4.])
        self.assertEqual(std.tolist(), [1., 1.])
        self.assertEqual(x.tolist(), [[-1., 0.], [1., 0.]])

    def test_group_conformal_order_statistic(self):
        r = [row('A' * i, y=i) for i in range(1, 11)]
        r.append(row('A', y=30))  # same group: maximum residual, not extra vote
        q, n, rank = rt.conformal_radius(r, [0] * len(r))
        self.assertEqual((q, n, rank), (30, 10, 10))
        with self.assertRaises(ValueError):
            rt.conformal_radius([row()], [0])

    def test_ridge_knn_known_answers(self):
        import numpy as np
        x = np.array([[-1.], [0.], [1.]])
        y = np.array([1., 2., 3.])
        coef, intercept = rt.ridge_fit(x, y, 1)
        self.assertAlmostEqual(coef[0], 2/3)
        self.assertEqual(intercept, 2)
        self.assertEqual(rt.knn_predict(x, y, np.array([[0.]]), 3).tolist(), [2.])


if __name__ == '__main__':
    unittest.main()
