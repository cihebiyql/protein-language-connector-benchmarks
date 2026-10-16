"""Hand-calculated fixtures test metric semantics; these are not benchmark data."""
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from bench_v02.metrics import (atomic_facts, causal_fact_intervention,
                               comprehensiveness, grouped_regression,
                               intervention_effects, known_positive_retrieval,
                               pair_direction, regression, sufficiency)


class RegressionTests(unittest.TestCase):
    def test_ties_and_group_macro(self):
        result = regression([1, 1, 2], [1, 2, 3], ['a', 'a', 'b'])
        self.assertAlmostEqual(result['mae'], 2 / 3)
        self.assertAlmostEqual(result['spearman'], math.sqrt(3) / 2)
        self.assertEqual(result['group_macro_mae'], .75)
        self.assertIsNone(result['groups'][0]['spearman'])

    def test_constant_single_empty_and_strict_json(self):
        for actual, predicted in [([1, 1], [2, 3]), ([1], [4]), ([], [])]:
            result = regression(actual, predicted)
            self.assertIsNone(result['spearman'])
            json.dumps(result, allow_nan=False)

    def test_translation_and_permutation_invariance(self):
        original = regression([1, 2, 4], [2, 3, 1], ['a', 'b', 'a'])
        changed = regression([14, 11, 12], [11, 12, 13], ['a', 'a', 'b'])
        for key in ('mae', 'spearman', 'group_macro_mae'):
            self.assertAlmostEqual(original[key], changed[key])

    def test_strata_do_not_pool_rt_minutes(self):
        result = grouped_regression([1, 2, 100, 200], [2, 3, 90, 190],
                                    ['short', 'short', 'long', 'long'], ['t'] * 4)
        self.assertFalse(result['pooled'])
        self.assertNotIn('mae', result)
        self.assertEqual([row['mae'] for row in result['strata']], [1, 10])

    def test_malformed_rejected(self):
        for values, predictions in [([1], []), ([math.nan], [1]), ([1], [math.inf])]:
            with self.assertRaises(ValueError):
                regression(values, predictions)


class RetrievalTests(unittest.TestCase):
    def test_true_class_macro_uses_own_rank(self):
        result = known_positive_retrieval([['a', 'b', 'c'], ['b', 'a', 'c']],
                                          [{'a', 'c'}, {'a'}])
        self.assertEqual(result['mrr'], .75)
        self.assertEqual(result['recall_at_1'], .25)
        self.assertEqual(result['recall_at_3'], 1)
        self.assertAlmostEqual(result['class_macro_mrr'], (.75 + 1 / 3) / 2)

    def test_unknown_candidates_do_not_become_negatives(self):
        result = known_positive_retrieval([['unknown', 'a']], [{'a', 'missing'}])
        self.assertEqual(result['mrr'], .5)
        self.assertEqual(result['recall_at_3'], .5)
        self.assertEqual(result['class_macro_mrr'], .25)
        self.assertNotIn('auroc', result)

    def test_class_macro_recalls_differ_from_imbalanced_sample_recalls(self):
        result = known_positive_retrieval([['a', 'b'], ['a', 'b'], ['a', 'b']],
                                          [{'a'}, {'a'}, {'b'}])
        self.assertAlmostEqual(result['recall_at_1'], 2 / 3)
        self.assertEqual(result['class_macro_recall_at_1'], .5)
        self.assertEqual(result['class_macro_recall_at_3'], 1)

    def test_multilabel_macro_recalls_use_each_true_class(self):
        result = known_positive_retrieval([['a', 'x', 'b', 'c'], ['a', 'b', 'x', 'c']],
                                          [{'a', 'c'}, {'a', 'b'}])
        # Class a: hit both; class b: hit rank 2; class c: missed rank 4.
        self.assertEqual(result['recall_at_1'], .5)
        self.assertAlmostEqual(result['class_macro_recall_at_1'], 1 / 3)
        self.assertEqual(result['recall_at_3'], .75)
        self.assertAlmostEqual(result['class_macro_recall_at_3'], 2 / 3)

    def test_duplicate_ranking_and_empty_positive_rejected(self):
        for ranking, positives in [(['a', 'a'], {'a'}), (['a'], set())]:
            with self.assertRaises(ValueError):
                known_positive_retrieval([ranking], [positives])


class InterventionTests(unittest.TestCase):
    def test_donors_averaged_per_recipient_and_coverage(self):
        result = intervention_effects([1, 1, 1], [0, .5, 0], [[0, 1], [0], []])
        self.assertAlmostEqual(result['u'], 5 / 6)
        self.assertEqual(result['d'], .75)
        self.assertEqual(result['n_mismatch_covered'], 2)
        self.assertEqual(result['u_common_coverage'], .75)
        # Duplicating all donors for one recipient never reweights recipients.
        repeated = intervention_effects([1, 1, 1], [0, .5, 0], [[0, 1] * 10, [0], []])
        self.assertEqual(repeated['d'], result['d'])

    def test_mae_direction_and_empty_donors(self):
        result = intervention_effects([1, 2], [3, 4], [[2, 4], None], larger_is_better=False)
        self.assertEqual(result['u'], 2)
        self.assertEqual(result['d'], 2)
        self.assertIsNone(intervention_effects([1], [2], [[]])['d'])

    def test_selective_intervention_denominators(self):
        result = causal_fact_intervention([True, True, False], [True, False, True],
                                          {'x': 'yes', 'y': 'no'}, {'x': 'yes', 'y': 'yes'})
        self.assertEqual(result['target_intervention_success'], .5)
        self.assertEqual(result['n_target_eligible'], 2)
        self.assertEqual(result['irrelevant_fact_retention'], .5)
        empty = causal_fact_intervention([False], [True], [], [])
        self.assertIsNone(empty['target_intervention_success'])
        self.assertIsNone(empty['irrelevant_fact_retention'])

    def test_fact_ids_and_booleans_checked(self):
        with self.assertRaises(ValueError):
            causal_fact_intervention([1], [True], [], [])
        with self.assertRaises(ValueError):
            causal_fact_intervention([True], [True], {'a': 1}, {'b': 1})

    def test_rationale_probability_gaps_remain_signed(self):
        result = sufficiency([.8, .4], [.6, .7])
        self.assertAlmostEqual(result['mean'], -.05)
        self.assertAlmostEqual(comprehensiveness([.8], [.3])['mean'], .5)
        with self.assertRaises(ValueError):
            sufficiency([2], [.5])


class PairDirectionTests(unittest.TestCase):
    def test_class_balance_and_group_weighting(self):
        result = pair_direction(['A', 'A', 'A', 'B'], ['A', 'A', 'A', 'A'], ['x', 'x', 'x', 'y'])
        self.assertEqual(result['n'], 4)
        self.assertEqual(result['accuracy'], .75)
        self.assertEqual(result['balanced_accuracy'], .5)
        self.assertEqual(result['group_macro_accuracy'], .5)
        self.assertEqual(result['per_class'][1]['recall'], 0)

    def test_swap_labels_and_predictions_is_invariant(self):
        original = pair_direction(['A', 'B', 'A'], ['A', 'A', 'B'], ['x', 'y', 'x'])
        swapped = pair_direction(['B', 'A', 'B'], ['B', 'B', 'A'], ['x', 'y', 'x'])
        for key in ('accuracy', 'balanced_accuracy', 'group_macro_accuracy'):
            self.assertEqual(original[key], swapped[key])

    def test_empty_and_single_direction_undefined_balance(self):
        empty = pair_direction([], [])
        self.assertIsNone(empty['accuracy'])
        self.assertIsNone(empty['balanced_accuracy'])
        self.assertIsNone(empty['group_macro_accuracy'])
        single = pair_direction(['A'], ['A'])
        self.assertEqual(single['accuracy'], 1)
        self.assertIsNone(single['balanced_accuracy'])
        json.dumps(empty, allow_nan=False)

    def test_ambiguous_and_invalid_directions_are_rejected(self):
        for labels, predictions in [(['n.b.'], ['A']), (['A'], ['unknown']), (['A'], []), (['a'], ['A'])]:
            with self.assertRaises(ValueError):
                pair_direction(labels, predictions)


class AtomicFactTests(unittest.TestCase):
    def test_four_statuses_coverage_and_duplicate_fact_ids(self):
        claims = [{'status': 'supported', 'fact_id': 'a'},
                  {'status': 'supported', 'fact_id': 'a'},
                  {'status': 'contradicted', 'fact_id': 'b'},
                  {'status': 'unsupported'}, {'status': 'unknown'}]
        result = atomic_facts(claims, ['a', 'b'])
        self.assertEqual(result['supported_precision'], .4)
        self.assertEqual(result['required_coverage'], .5)
        self.assertEqual(result['contradiction_rate'], .2)
        self.assertEqual(result['unsupported_rate'], .2)
        self.assertEqual(result['unknown_rate'], .2)

    def test_empty_answer_is_not_perfect(self):
        result = atomic_facts([], ['required'])
        self.assertIsNone(result['supported_precision'])
        self.assertEqual(result['required_coverage'], 0)
        json.dumps(result, allow_nan=False)

    def test_unknown_status_rejected(self):
        with self.assertRaises(ValueError):
            atomic_facts([{'status': 'correct'}], [])


if __name__ == '__main__':
    unittest.main()
