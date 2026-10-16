"""Independent checks of fixed inputs, intervention matching, and paired inference."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from real_data import make_donors, paired_group_bootstrap


class FrozenInputTests(unittest.TestCase):
    def test_registered_input_bytes_and_budget(self):
        expected = {
            'benz_samples.jsonl': '57038156925fb5d772824716eef68cf6ce55d288175913e5d86a095007c9e8da',
            'benz_split_manifest.json': '530cd6db379132b353cb1eff6776552232efc1c0e77013eae072cbe716ce492a',
            'benz_targets.json': 'ce93a9376f8d5c7f4bff4138b6d67b83e86e28014be6338aae361e6ba8e8689f',
        }
        for filename, fingerprint in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / 'data/processed' / filename).read_bytes()).hexdigest(),
                             fingerprint, filename)
        manifest = json.loads((ROOT / 'data/processed/benz_split_manifest.json').read_text(encoding='utf-8'))
        n_train = manifest['counts']['train']
        self.assertEqual((n_train, (n_train + 63) // 64, n_train % 64), (1914, 30, 58))
        self.assertEqual(30 * ((n_train + 63) // 64), 900)
        self.assertEqual(manifest['status'], 'engineering_pilot_split')


class DonorContractTests(unittest.TestCase):
    @staticmethod
    def example_rows():
        # Unique target labels permit a donor pool; pairs of adjacent rows share
        # a group to ensure that same-group exclusion is actually exercised.
        return [{'sample_id': f's{i}', 'group_id': f'g{i // 2}', 'split': 'test',
                 'sequence': 'A' * 16, 'targets': [f'T{i}']} for i in range(14)]

    def test_ten_unique_donors_are_test_only_disjoint_and_from_other_groups(self):
        rows = self.example_rows()
        assignments = make_donors(rows, seed=2026, n=10)
        for receiver, indices in zip(rows, assignments):
            self.assertEqual(len(indices), 10)
            self.assertEqual(len(set(indices)), 10)
            for index in indices:
                donor = rows[index]
                self.assertEqual(donor['split'], 'test')
                self.assertNotEqual(donor['sample_id'], receiver['sample_id'])
                self.assertNotEqual(donor['group_id'], receiver['group_id'])
                self.assertLessEqual(abs(len(donor['sequence']) - len(receiver['sequence'])), 2)
                self.assertFalse(set(donor['targets']) & set(receiver['targets']))

    def test_matching_excludes_shared_targets_and_distant_lengths(self):
        rows = self.example_rows()
        rows[2]['targets'] = rows[0]['targets'][:]
        rows[3]['sequence'] = 'A' * 19
        rows[4]['sequence'] = 'A' * 17
        # n=1 keeps every row eligible while testing the recipient's full pool
        # through many fixed seeds rather than depending on one random draw.
        for seed in range(20):
            donors = make_donors(rows, seed=seed, n=1)
            self.assertNotIn(donors[0][0], (0, 1, 2, 3))

    def test_reordering_rows_preserves_donor_id_assignment(self):
        rows = self.example_rows()

        def ids(data):
            return {r['sample_id']: [data[i]['sample_id'] for i in selected]
                    for r, selected in zip(data, make_donors(data, seed=2026, n=10))}

        self.assertEqual(ids(rows), ids(list(reversed(rows))))

    def test_insufficient_candidates_are_explicit_not_repeated(self):
        with self.assertRaisesRegex(ValueError, 'eligible donors; need 10'):
            make_donors(self.example_rows()[:4], seed=2026, n=10)

    def test_train_rows_and_duplicate_ids_are_rejected(self):
        rows = self.example_rows()
        rows[0]['split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'test-only'):
            make_donors(rows, seed=2026, n=2)
        rows[0]['split'] = 'test'
        rows[1]['sample_id'] = rows[0]['sample_id']
        with self.assertRaisesRegex(ValueError, 'unique'):
            make_donors(rows, seed=2026, n=2)

    def test_real_test_data_have_ten_eligible_donors_per_sample(self):
        rows = [json.loads(line) for line in
                (ROOT / 'data/processed/benz_samples.jsonl').read_text(encoding='utf-8').splitlines()]
        rows = [r for r in rows if r['split'] == 'test']
        self.assertEqual(len(rows), 387)
        assignments = make_donors(rows, seed=2026, n=10)
        self.assertEqual(sum(len(d) for d in assignments), 3870)
        for row, indices in zip(rows, assignments):
            self.assertEqual(len(set(indices)), 10)
            for index in indices:
                self.assertNotEqual(rows[index]['group_id'], row['group_id'])
                self.assertFalse(set(rows[index]['targets']) & set(row['targets']))


class PairedInferenceTests(unittest.TestCase):
    def test_constant_paired_shift_has_point_interval_despite_heterogeneous_scores(self):
        rows = [{'group_id': 'a'}, {'group_id': 'a'}, {'group_id': 'b'}, {'group_id': 'c'}]
        reference = np.array([0.0, .125, .5, .75])
        result = paired_group_bootstrap(reference + .125, reference, rows, seed=2026, n_boot=2000)
        self.assertEqual(result['mean_difference'], .125)
        self.assertEqual(result['ci95'], [.125, .125])
        self.assertEqual((result['n_samples'], result['n_groups']), (4, 3))

    def test_estimand_is_sample_weighted_despite_unequal_group_sizes(self):
        rows = [{'group_id': 'a'}] * 3 + [{'group_id': 'b'}]
        result = paired_group_bootstrap([0, 0, 0, 1], [0, 0, 0, 0], rows, seed=2026, n_boot=2000)
        self.assertEqual(result['mean_difference'], .25)
        self.assertEqual(result['n_groups'], 2)
        self.assertEqual(result['ci95'], [0, 1])

    def test_donor_draw_dimension_must_be_aggregated_before_bootstrap(self):
        rows = [{'group_id': 'a'}, {'group_id': 'b'}]
        with self.assertRaisesRegex(ValueError, 'one finite paired difference'):
            paired_group_bootstrap(np.zeros((2, 10)), np.zeros((2, 10)), rows, seed=2026)


@unittest.skipUnless((ROOT / 'results/real_run_20261015/mlp_summary.json').exists(),
                     'Matched MLP run has not completed')
class SavedBaselineAuditTests(unittest.TestCase):
    directory = ROOT / 'results/real_run_20261015'

    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in
                    (ROOT / 'data/processed/benz_samples.jsonl').read_text(encoding='utf-8').splitlines()]
        cls.rows = [r for r in cls.rows if r['split'] == 'test']

    def test_all_seeds_complete_equal_budget_and_earliest_validation_checkpoint(self):
        summary = json.loads((self.directory / 'mlp_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['seeds'], [17, 42, 2026])
        for seed in summary['seeds']:
            run = json.loads((self.directory / f'mlp_seed{seed}.json').read_text(encoding='utf-8'))
            self.assertEqual((run['epochs'], run['optimizer_steps'], run['effective_batch_size']), (30, 900, 64))
            self.assertEqual([h['epoch'] for h in run['history']], list(range(1, 31)))
            best = max(run['history'], key=lambda h: h['validation']['mrr'])
            self.assertEqual(run['best_epoch'], best['epoch'])

    def test_saved_donors_match_prefixed_seed_and_all_seed_score_files(self):
        mapping = json.loads((self.directory / 'mlp_donors.json').read_text(encoding='utf-8'))
        self.assertEqual(mapping['seed'], 42)
        self.assertEqual(mapping['test_sample_ids'], [r['sample_id'] for r in self.rows])
        self.assertEqual(mapping['donor_indices'], make_donors(self.rows, seed=42, n=10))
        for seed in (17, 42, 2026):
            with np.load(self.directory / f'mlp_seed{seed}.scores.npz', allow_pickle=False) as score:
                np.testing.assert_array_equal(score['donor_indices'], mapping['donor_indices'])
                self.assertEqual(score['mismatched'].shape, (387, 10, 34))

    def test_saved_correct_metrics_recompute_from_scores_without_metric_helper(self):
        for seed in (17, 42, 2026):
            run = json.loads((self.directory / f'mlp_seed{seed}.json').read_text(encoding='utf-8'))
            with np.load(self.directory / f'mlp_seed{seed}.scores.npz', allow_pickle=False) as score:
                self.assertEqual(score['sample_ids'].tolist(), [r['sample_id'] for r in self.rows])
                target_ids = score['target_ids'].tolist()
                recalls, reciprocal_ranks = [], []
                for row, values in zip(self.rows, score['correct']):
                    order = sorted(range(len(target_ids)), key=lambda i: (-float(values[i]), i))
                    positives = set(row['targets'])
                    hit_ranks = [rank for rank, i in enumerate(order, 1) if target_ids[i] in positives]
                    recalls.append(sum(rank <= 3 for rank in hit_ranks) / len(positives))
                    reciprocal_ranks.append(1 / min(hit_ranks))
                self.assertAlmostEqual(np.mean(recalls), run['scores']['correct']['known_positive_recall_at_3'])
                self.assertAlmostEqual(np.mean(reciprocal_ranks), run['scores']['correct']['mrr'])


if __name__ == '__main__':
    unittest.main()
