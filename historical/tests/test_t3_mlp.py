import sys
from pathlib import Path
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_t3_mlp import PairedMLP, flip_training_pair, grouped_metric_cis, order_fingerprints, pair_metrics, read_rows


class T3MLPTests(unittest.TestCase):
    def test_balanced_accuracy_is_not_majority_accuracy(self):
        result = pair_metrics([1, 1, 1, 1], [0, 1, 1, 1])
        self.assertEqual(result['accuracy'], .75)
        self.assertEqual(result['balanced_accuracy'], .5)

    def test_initialization_preserves_t1_front_and_head(self):
        state = {'0.weight': torch.randn(256, 640), '0.bias': torch.randn(256),
                 '3.weight': torch.randn(34, 256), '3.bias': torch.randn(34)}
        model = PairedMLP({'head': state}, 7).eval()
        x = torch.randn(3, 640)
        expected = torch.nn.functional.linear(torch.nn.functional.gelu(
            torch.nn.functional.linear(x, state['0.weight'], state['0.bias'])),
            state['3.weight'], state['3.bias'])
        self.assertTrue(torch.equal(model.d1_logits(x), expected))
        self.assertEqual(model(x, x, torch.zeros(3, 7)).shape, (3, 2))

    def test_flip_is_repeatable_and_relabels_both_classes(self):
        flips = [flip_training_pair(17, e, 'pair1') for e in range(1, 31)]
        self.assertEqual(flips, [flip_training_pair(17, e, 'pair1') for e in range(1, 31)])
        self.assertEqual(set(flips), {False, True})
        self.assertEqual([1 - y for y in (0, 1)], [1, 0])

    def test_group_ci_keeps_series_pockets_together(self):
        rows = [{'group_id': 'parent1'}, {'group_id': 'parent1'}, {'group_id': 'parent2'}]
        result = grouped_metric_cis(np.array([0, 0, 1]), np.array([0, 0, 1]), rows, n_boot=100)
        self.assertEqual(result['n_groups'], 2)
        self.assertEqual(result['accuracy_ci95'], [1., 1.])
        self.assertEqual(result['balanced_accuracy_ci95'], [1., 1.])
        self.assertLess(result['balanced_accuracy_valid_draws'], 100)

    def test_order_fingerprints_include_truncated_replay(self):
        d1 = [{'split': 'train'} for _ in range(8)]
        d2 = [{'split': 'train'} for _ in range(3)] + [{'split': 'test'}]
        result = order_fingerprints(d1, d2, seeds=[17])
        self.assertEqual(len(result['fingerprints']), 90)
        self.assertEqual({r['n'] for r in result['fingerprints'] if r['stream'] == 't3_d1_replay'}, {3})
        self.assertEqual(result['fingerprints'], order_fingerprints(d1, d2, seeds=[17])['fingerprints'])

    def test_final_qc_real_data_and_sensitivity_denominators(self):
        d2, g1 = read_rows('t3_d2_pairs.jsonl'), read_rows('t3_g1_direction_pairs.jsonl')
        self.assertEqual({s: sum(r['split'] == s for r in d2) for s in ('train', 'val', 'test')},
                         {'train': 34, 'val': 16, 'test': 9})
        self.assertTrue(all(r['source_evidence']['ev1_wt_input_library_coverage'] > 0
                            and r['source_evidence']['ev1_mut_input_library_coverage'] > 0 for r in d2))
        self.assertTrue(all(r['source_evidence']['single_pair_significance_verified'] is False for r in d2))
        self.assertEqual(len(g1), 17)
        self.assertEqual(sum(r['sequence_source_consistent'] for r in g1), 14)
        self.assertEqual({r['pair_id'] for r in read_rows('t3_g1_source_consistent_direction_pairs.jsonl')},
                         {r['pair_id'] for r in g1 if r['sequence_source_consistent']})


if __name__ == '__main__':
    unittest.main()
