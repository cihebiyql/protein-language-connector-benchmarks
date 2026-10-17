import math
import unittest

from src.mt_metrics import build_metrics, calibration_radius, parse_task_answer, score_language, score_ranking, score_regression


def regression_rows():
    return [{"sample_id": str(i), "target": float(i), "group_id": "a" if i < 2 else "b"}
            for i in range(4)]


class MetricsTests(unittest.TestCase):
    def test_id_alignment_not_positional(self):
        rows = regression_rows()
        predictions = [{"sample_id": r["sample_id"], "prediction": r["target"]} for r in reversed(rows)]
        result = score_regression(rows, predictions)
        self.assertEqual(result["mae"], 0)
        self.assertEqual(result["spearman"], 1)
        self.assertEqual(result["group_macro_spearman"], 1)

    def test_missing_nonfinite_and_boolean_do_not_drop_rows(self):
        rows = regression_rows()
        for value in (float("nan"), float("inf"), True, None):
            result = score_regression(rows, [{"sample_id": "0", "prediction": value}])
            self.assertIsNone(result["mae"])
            self.assertEqual(result["coverage"], {"n_total": 4, "n_valid": 0, "n_failed": 4, "fraction": 0})

    def test_duplicate_extra_reference_ids_rejected(self):
        rows = regression_rows()
        for predictions in ([{"sample_id": "extra", "prediction": 0}],
                            [{"sample_id": "0", "prediction": 0}] * 2):
            with self.assertRaises(ValueError):
                score_regression(rows, predictions)
        with self.assertRaises(ValueError):
            score_regression(rows + rows[:1], [])

    def test_tied_spearman_and_constant(self):
        rows = regression_rows()
        result = score_regression(rows, [{"sample_id": str(i), "prediction": [0, 0, 1, 1][i]} for i in range(4)])
        self.assertAlmostEqual(result["spearman"], math.sqrt(0.8))
        self.assertIsNone(result["group_macro_spearman"])
        result = score_regression(rows, [{"sample_id": str(i), "prediction": 1} for i in range(4)])
        self.assertIsNone(result["spearman"])

    def test_class_macro_differs_from_sample_macro(self):
        rows = [{"sample_id": "a", "positive_ids": ["A"]},
                {"sample_id": "b", "positive_ids": ["A"]},
                {"sample_id": "c", "positive_ids": ["B", "C"]}]
        predictions = [{"sample_id": r["sample_id"], "ranking": ["A", "B", "C"]} for r in rows]
        score = score_ranking(rows, predictions, ["A", "B", "C"])
        self.assertAlmostEqual(score["class_macro_mrr"], (1 + 0.5 + 1 / 3) / 3)
        self.assertAlmostEqual(score["mrr"], (2 + (0.5 + 1 / 3) / 2) / 3)
        self.assertEqual(score["recall_at_3"], 1)

    def test_ranking_requires_full_unique_catalog(self):
        rows = [{"sample_id": "a", "positive_ids": ["A"]}]
        for ranking in (["A"], ["A", "A"], ["A", "X"]):
            result = score_ranking(rows, [{"sample_id": "a", "ranking": ranking}], ["A", "B"])
            self.assertIsNone(result["class_macro_mrr"])
            self.assertEqual(result["coverage"]["n_failed"], 1)
        with self.assertRaises(ValueError):
            score_ranking([{"sample_id": "a", "positive_ids": ["X"]}], [], ["A", "B"])

    def test_strict_json_no_repairs_or_coercion(self):
        self.assertEqual(parse_task_answer("rt", ' {"retention_time_min": 2.5} '), 2.5)
        for text in ('```json\n{"retention_time_min":2}\n```', '{"retention_time_min":"2"}',
                     '{"retention_time_min":true}', '{"retention_time_min":NaN}',
                     '{"retention_time_min":1,"retention_time_min":2}',
                     '{"retention_time_min":1,"other":2}', '{"retention_time_min":1e999}',
                     '[{"retention_time_min":2}]', '{"retention_time_min":2} trailing'):
            self.assertIsNone(parse_task_answer("rt", text), text)

    def test_language_failure_nulls_regression_and_keeps_denominator(self):
        rows = regression_rows()
        predictions = [{"sample_id": str(i), "text": '{"melting_temperature_c":' + str(i) + '}'} for i in range(3)]
        result = score_language("meltome", rows, predictions)
        self.assertIsNone(result["mae"])
        self.assertEqual(result["coverage"]["fraction"], 0.75)

    def test_language_exact_description_and_full_denominator(self):
        rows = [{"sample_id": str(i), "positive_ids": ["A"]} for i in range(3)]
        result = score_language("elm", rows,
                                [{"sample_id": "0", "text": '{"function":"Alpha"}'},
                                 {"sample_id": "1", "text": '{"function":"alpha"}'}],
                                [{"candidate_id": "A", "description": "Alpha"}])
        self.assertEqual(result["exact_description_known_positive_accuracy"], 1 / 3)
        self.assertEqual(result["coverage"]["n_valid"], 1)
        self.assertEqual(result["json_coverage"]["n_valid"], 2)

    def test_calibration_uses_group_max_not_row_quantile(self):
        rows = [{"sample_id": str(i), "target": 0, "group_id": str(i // 2)} for i in range(20)]
        predictions = [{"sample_id": str(i), "prediction": i if i % 2 else 0} for i in range(20)]
        result = calibration_radius(rows, predictions)
        self.assertEqual(result["calibration_groups"], 10)
        self.assertEqual(result["quantile_order"], 10)
        self.assertEqual(result["radius"], 19)
        small = calibration_radius(rows[:2], predictions[:2])
        self.assertTrue(small["unbounded"])
        self.assertIsNone(small["radius"])
        self.assertFalse(calibration_radius(rows, predictions[:-1])["complete"])

    def test_c1_requires_identical_on_predictions_not_just_equal_metric(self):
        rows = regression_rows()
        readout = [{"sample_id": str(i), "prediction": float(i)} for i in range(4)]
        c1 = [{**p, "prediction_off": 0} for p in readout]
        result = build_metrics("rt", rows, readout, c1=c1)
        self.assertEqual(result["C1_off_minus_on"], 1.5)
        self.assertIsNone(result["language"])
        c1[0]["prediction"] = 1
        with self.assertRaises(ValueError):
            build_metrics("rt", rows, readout, c1=c1)


if __name__ == "__main__":
    unittest.main()
