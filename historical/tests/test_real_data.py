"""Regression tests for comparison data and sample-weighted evaluation."""
import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from real_data import epoch_order, load_t1, paired_group_bootstrap, ranking_metrics


class RealDataTests(unittest.TestCase):
    def test_fixed_split_and_file_order(self):
        rows, target_ids = load_t1()
        self.assertEqual(len(target_ids), 34)
        self.assertEqual({split: sum(r['split'] == split for r in rows)
                          for split in ('train', 'validation', 'test')},
                         {'train': 1914, 'validation': 393, 'test': 387})
        self.assertEqual(rows[0]['sample_id'], 'benz_531c5bc6785f')

    def test_known_positive_metrics_and_stable_ties(self):
        rows = [{'sample_id': 'a', 'targets': ['T1', 'T4']},
                {'sample_id': 'b', 'targets': ['T3']}]
        measured, predictions = ranking_metrics(np.ones((2, 4)), rows, ['T1', 'T2', 'T3', 'T4'])
        self.assertEqual(measured['known_positive_recall_at_3'], .75)
        self.assertAlmostEqual(measured['mrr'], (1 + 1 / 3) / 2)
        self.assertEqual(predictions[0]['ranked_target_ids'], ['T1', 'T2', 'T3', 'T4'])
        with self.assertRaises(ValueError):
            ranking_metrics(np.full((2, 4), np.nan), rows, ['T1', 'T2', 'T3', 'T4'])

    def test_cpu_epoch_order_matches_declared_generator(self):
        indices = [9, 1, 3, 6]
        actual = epoch_order(indices, seed=17, epoch=1)
        expected = torch.tensor(indices)[torch.randperm(4, generator=torch.Generator().manual_seed(18))]
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.device.type, 'cpu')

    def test_bootstrap_uses_group_and_keeps_sample_weighting(self):
        rows = [{'group_id': 'g1'}, {'group_id': 'g1'}, {'group_id': 'g2'}]
        actual = paired_group_bootstrap([1., 1., 0.], [0., 0., 0.], rows)
        self.assertAlmostEqual(actual['mean_difference'], 2 / 3)
        self.assertEqual(actual['n_groups'], 2)
        self.assertEqual(actual['ci95'], [0., 1.])
        self.assertEqual(actual['bootstrap_seed'], 2026)


if __name__ == '__main__':
    unittest.main()
