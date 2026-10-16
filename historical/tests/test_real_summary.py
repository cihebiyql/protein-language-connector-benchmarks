"""Integrity and statistical-unit tests for the final real-data summary."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import summarize_real_results as summary


class RealSummaryTests(unittest.TestCase):
    def test_group_bootstrap_keeps_linked_pockets(self):
        rows = [{'group_id': 'parent1'}, {'group_id': 'parent1'}, {'group_id': 'parent2'}]
        result = summary.interval([1., 1., 0.], rows)
        self.assertAlmostEqual(result['estimate'], 2 / 3)
        self.assertEqual(result['ci95'], [0., 1.])
        self.assertEqual(result['n_groups'], 2)
        self.assertEqual(result['bootstrap_draws'], 2000)

    def test_balanced_ci_omits_missing_class_draws(self):
        rows = [{'group_id': 'a'}, {'group_id': 'b'}]
        result = summary.interval([1., 0.], rows, np.array(['A', 'B']))
        self.assertEqual(result['estimate'], .5)
        self.assertEqual(result['ci95'], [.5, .5])
        self.assertGreater(result['valid_draws'], 0)
        self.assertLess(result['valid_draws'], 2000)

    def test_ranking_ties_stable_and_positives_sample_weighted(self):
        rows = [{'targets': ['t1', 't4']}, {'targets': ['t3']}]
        values = summary.ranking_arrays(np.ones((2, 4)), rows, ['t1', 't2', 't3', 't4'])
        np.testing.assert_allclose(values['recall_at_3'], [.5, 1])
        np.testing.assert_allclose(values['mrr'], [1, 1 / 3])
        with self.assertRaises(summary.IntegrityError):
            summary.ranking_arrays(np.full((2, 4), np.nan), rows, ['t1', 't2', 't3', 't4'])

    def test_seed_sd_and_average_before_group_ci(self):
        rows = [{'group_id': 'a'}, {'group_id': 'b'}]
        runs = {}
        for method in summary.METHODS:
            runs[method] = {}
            for seed, value in zip(summary.SEEDS, (0., .5, 1.)):
                scalar = np.repeat(value, 2)
                runs[method][seed] = {mode: {metric: scalar for metric in ('recall_at_3', 'mrr')}
                                      for mode in ('correct', 'off', 'mismatched')}
        result = summary.summarize_t1(runs, rows)
        measured = result['methods']['L15-J14']['correct']['mrr']
        self.assertEqual(measured['mean'], .5)
        self.assertEqual(measured['sd'], .5)
        self.assertEqual(measured['group_ci']['ci95'], [.5, .5])
        paired = result['paired_comparisons']['L15-J14 - L30-J0']['mrr']
        self.assertEqual(paired['ci95'], [0., 0.])
        self.assertEqual(paired['n_samples'], 2)

    @staticmethod
    def result(seed=17):
        return {'seed': seed, 'epochs': 30, 'optimizer_steps': 900,
                'effective_batch_size': 64, 'best_epoch': 1, 'loss': 'test CE',
                'history': [{'epoch': i, 'train_loss': 1.,
                             'validation': {'mrr': .5, 'balanced_accuracy': .5}} for i in range(1, 31)],
                'input_sha256': summary.ORIGINAL_HASHES['benz_samples.jsonl']}

    def test_wrong_budget_seed_hash_or_partial_history_rejected(self):
        for key, value in [('optimizer_steps', 899), ('seed', 42), ('input_sha256', 'old'),
                           ('history', [{'epoch': 1, 'train_loss': 1.}])]:
            result = self.result()
            result[key] = value
            with self.subTest(key=key), self.assertRaises(summary.IntegrityError):
                summary.check_run(result, 17, 900, 'T1', {})

    def test_stale_t3_data_hash_rejected(self):
        result = self.result()
        result.update(optimizer_steps=60, inputs_sha256={'x': 'old'}, target_catalog_sha256='cat',
                      d2_batch_size=32, d1_d2_samples_per_epoch=[34, 34])
        with self.assertRaisesRegex(summary.IntegrityError, 'stale'):
            summary.check_run(result, 17, 60, 'T3', {'t3_hashes': {'x': 'new'}, 'target_catalog_sha256': 'cat'})

    def test_checkpoint_selection_uses_first_validation_maximum(self):
        result = self.result()
        result['history'][3]['validation']['mrr'] = .8
        result['history'][9]['validation']['mrr'] = .8
        result['best_epoch'] = 10
        with self.assertRaisesRegex(summary.IntegrityError, 'earliest validation'):
            summary.check_run(result, 17, 900, 'T1', {})
        result['best_epoch'] = 4
        summary.check_run(result, 17, 900, 'T1', {})

    def test_t3_materialized_derived_panel_hash_is_optional_but_verified(self):
        result = self.result()
        base_hashes = {name: name + '-frozen' for name in summary.T3_FILES if name != summary.T3_DERIVED_FILE}
        all_hashes = {**base_hashes, summary.T3_DERIVED_FILE: 'derived-frozen'}
        result.update(optimizer_steps=60, inputs_sha256=dict(base_hashes), target_catalog_sha256='cat',
                      d2_batch_size=32, d1_d2_samples_per_epoch=[34, 34])
        ctx = {'t3_hashes': all_hashes, 'target_catalog_sha256': 'cat'}
        summary.check_run(result, 17, 60, 'T3', ctx)
        result['inputs_sha256'] = dict(all_hashes)
        summary.check_run(result, 17, 60, 'T3', ctx)
        result['inputs_sha256'][summary.T3_DERIVED_FILE] = 'stale-derived'
        with self.assertRaisesRegex(summary.IntegrityError, 'stale data hash: ' + summary.T3_DERIVED_FILE):
            summary.check_run(result, 17, 60, 'T3', ctx)
        result['inputs_sha256'] = {summary.T3_DERIVED_FILE: 'derived-frozen'}
        with self.assertRaisesRegex(summary.IntegrityError, 'incomplete data hashes'):
            summary.check_run(result, 17, 60, 'T3', ctx)

    def test_npz_recomputation_and_donor_count(self):
        rows = [{'sample_id': str(i), 'group_id': str(i), 'targets': [f't{i}'],
                 'sequence': 'A' * 16} for i in range(30)]
        targets = [f't{i}' for i in range(30)]
        logits = np.eye(30)
        ctx = {'t1': rows, 'targets': targets}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / 'mlp_seed17.json').write_text(json.dumps(self.result()), encoding='utf-8')
            scores = {'correct': logits, 'closed_raw_zero': np.ones_like(logits),
                      'mismatched': np.repeat(logits[:, None], 10, axis=1),
                      'sample_ids': np.array([r['sample_id'] for r in rows]),
                      'target_ids': np.array(targets), 'donor_indices': summary.expected_donors(rows)}
            path = base / 'mlp_seed17.scores.npz'
            np.savez(path, **scores)
            meta, arrays = summary.load_t1('MLP', 17, base, ctx)
            self.assertEqual(meta['status'], 'complete')
            self.assertEqual(arrays['correct']['mrr'].mean(), 1.)
            self.assertAlmostEqual(arrays['off']['recall_at_3'].mean(), .1)
            self.assertEqual(arrays['mismatched']['mrr'].shape, (30,))
            scores['mismatched'] = scores['mismatched'][:, :9]
            np.savez(path, **scores)
            with self.assertRaisesRegex(summary.IntegrityError, 'ten donor'):
                summary.load_t1('MLP', 17, base, ctx)
            scores['mismatched'] = np.repeat(logits[:, None], 10, axis=1)
            scores['sample_ids'] = scores['sample_ids'][::-1]
            np.savez(path, **scores)
            with self.assertRaisesRegex(summary.IntegrityError, 'sample order'):
                summary.load_t1('MLP', 17, base, ctx)

    def test_missing_run_pending_and_no_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = {'t3_hashes': {}, 'target_catalog_sha256': 'test'}
            with patch.object(summary, 'inputs', return_value=ctx):
                result = summary.build_summary(root)
            self.assertEqual(result['status'], 'pending')
            for payload in result['tasks'].values():
                self.assertEqual(payload['status'], 'pending')
                self.assertNotIn('comparison', payload)
            report = summary.render_report(result)
            self.assertIn('pending', report)
            self.assertNotIn('0.0000 ±', report)

    def test_t3_reversal_and_source_consistent_subset_from_full_logits(self):
        rows = [{'pair_id': f'p{i}', 'group_id': f'g{i}', 'label': label, 'target_id': 't'}
                for i, label in enumerate(('A', 'B', 'A'))]
        ctx = {'t3_hashes': {'data': 'frozen'}, 'target_catalog_sha256': 'cat',
               'subsets': {'d2_test': rows, 'd2_test_clean_after_t1_pretraining': rows[:1],
                           'g1_direction': rows, 'g1_source_consistent_direction': rows[:2]}}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            result = self.result()
            result.update(optimizer_steps=60, inputs_sha256=ctx['t3_hashes'], target_catalog_sha256='cat',
                          d2_batch_size=32, d1_d2_samples_per_epoch=[34, 34])
            (base / 't3_mlp_seed17.json').write_text(json.dumps(result), encoding='utf-8')
            arrays = {}
            for name, selected in ctx['subsets'].items():
                if name == 'g1_source_consistent_direction':
                    continue
                logits = np.array([[2, 1] if r['label'] == 'A' else [1, 2] for r in selected])
                arrays.update({f'{name}__pair_ids': np.array([r['pair_id'] for r in selected]),
                               f'{name}__correct': logits, f'{name}__closed_raw_zero': logits,
                               f'{name}__swapped_messages': logits[:, ::-1]})
            np.savez(base / 't3_mlp_seed17.scores.npz', **arrays)
            meta, measured = summary.load_t3('MLP', 17, base, base, ctx)
            self.assertEqual(meta['status'], 'complete')
            panel = measured['g1_source_consistent_direction']
            np.testing.assert_array_equal(panel['correct'], [1., 1.])
            np.testing.assert_array_equal(panel['swapped_messages'], [0., 0.])
            np.testing.assert_array_equal(panel['reversed_input_and_label'], [1., 1.])
            np.testing.assert_array_equal(panel['prediction_reversal_rate'], [1., 1.])


if __name__ == '__main__':
    unittest.main()
