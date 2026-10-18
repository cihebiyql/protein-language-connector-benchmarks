"""CPU checks for target isolation, grouped cross-fitting and safe inputs."""
import hashlib
import importlib.util
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location('qa_features', Path(__file__).resolve().parents[1]/'src/qa_features.py')
features = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(features)


class FeatureTests(unittest.TestCase):
    def test_safe_allowlist_ignores_measured_values(self):
        rows = [{'sequence':'ADEFGHIK', 'experiment':str(i)} for i in range(3)]
        expected = features.physicochemical_features(rows)
        poisoned = [{**row, 'measured Mass':1e20, 'Intensity':-1e20, 'Score':999,
                     'rt':99999, 'ccs':-99999, 'id':'different', 'group_id':'other'} for row in rows]
        np.testing.assert_array_equal(expected, features.physicochemical_features(poisoned))
        self.assertEqual(expected.shape, (3,25))
        self.assertAlmostEqual(expected[0,:20].sum(), 1.)

    def test_group_hash_folds_preserve_groups(self):
        groups = ['shared']*4 + [str(i) for i in range(40)]
        folds = features.grouped_fold_ids(groups)
        self.assertEqual(len(set(folds[:4])), 1)
        self.assertEqual(set(folds), set(range(5)))
        for fold in range(5):
            self.assertFalse(set(np.asarray(groups)[folds==fold]) & set(np.asarray(groups)[folds!=fold]))

    def test_train_only_nested_oof_and_prediction(self):
        rng = np.random.default_rng(99)
        x = rng.normal(size=(100,8))
        y = np.column_stack((x[:,0]*2 + rng.normal(size=100)*.1, x[:,2]*3+rng.normal(size=100)*.1))
        groups = np.asarray([hashlib.sha256(str(i//2).encode()).hexdigest() for i in range(100)])
        query = rng.normal(size=(12,8))
        first = features.fit_predict_specialists(x,y,groups,query,physicochemical_dim=3)
        oof, prediction, folds, receipt, models = first
        self.assertEqual(oof.shape, (100,2))
        self.assertEqual(prediction.shape, (12,2))
        self.assertTrue(np.isfinite(oof).all())
        self.assertFalse(receipt['external_labels_used'])
        self.assertTrue(all(r['outer_group_overlap']==0 and r['inner_group_overlap']==0 for r in receipt['outer_folds']))
        # Changing heldout fold labels cannot alter that fold's OOF predictions.
        changed = y.copy()
        changed[folds==0] += 10000
        second = features.fit_predict_specialists(x,changed,groups,query,physicochemical_dim=3)
        np.testing.assert_allclose(first[0][folds==0],second[0][folds==0],rtol=0,atol=0)
        third = features.fit_predict_specialists(x,y,groups,query,physicochemical_dim=3)
        np.testing.assert_allclose(prediction, third[1],rtol=0,atol=0)
        packed = features._model_arrays(models)
        self.assertIn('final_coefficient', packed)

    def test_panel_never_reads_heldout_labels(self):
        rng = np.random.default_rng(108)
        rows = [{'split':'train' if i<100 else 'validation' if i<106 else 'test',
                 'group_id':str(i), 'rt':float(i), 'ccs':float(i*2)} for i in range(112)]
        x = rng.normal(size=(112,8))
        # Reduced synthetic covariates exercise the real panel-to-fit boundary.
        original_fit = features.fit_predict_specialists
        def small_fit(a,b,c,d):
            return original_fit(a,b,c,d,physicochemical_dim=3)
        for i,row in enumerate(rows):
            row['experiment'] = str(i%3)
        with patch.object(features, 'build_specialist_covariates', return_value=x), patch.object(features, 'fit_predict_specialists', side_effect=small_fit):
            first = features.fit_panel_specialists(rows,None,None)
            changed = [{k:v for k,v in row.items() if k not in ('rt','ccs')} if row['split']!='train' else dict(row) for row in rows]
            # Missing external targets would raise immediately if accessed.
            second = features.fit_panel_specialists(changed,None,None)
        np.testing.assert_array_equal(first[0],second[0])
        self.assertTrue(np.all(first[1][100:] == -1))

    def test_cache_order_and_nonfinite_are_rejected(self):
        rows = [{'id':'a'*64}, {'id':'b'*64}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'states.npz'
            states = np.zeros((2,8,640), dtype=np.float32)
            features.atomic_npz(path, ids=np.asarray([r['id'] for r in rows],dtype='U64'), p15=states,p30=states)
            features.validate_features(path,rows)
            with self.assertRaises(ValueError):
                features.validate_features(path,rows[::-1])
            states[0,0,0] = np.nan
            features.atomic_npz(path, ids=np.asarray([r['id'] for r in rows],dtype='U64'), p15=states,p30=states)
            with self.assertRaises(ValueError):
                features.validate_features(path,rows)


if __name__ == '__main__':
    unittest.main()
