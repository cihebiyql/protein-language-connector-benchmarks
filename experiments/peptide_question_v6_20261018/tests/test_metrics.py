"""Manually calculable checks of the prospective generated-answer endpoint."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.qa_metrics import label, parse_answer, score


THRESHOLDS = {"rt": 10.0, "ccs": 400.0}


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"id": "a", "rt": 12., "ccs": 450., "group_id": "family1"},
            {"id": "b", "rt": 11., "ccs": 350., "group_id": "family1"},
            {"id": "c", "rt": 13., "ccs": 420., "group_id": "family2"},
            {"id": "d", "rt": 9., "ccs": 399., "group_id": "family3"},
        ]

    def perfect(self):
        return [{"id": row["id"], "question": q, "raw": label(row, q, THRESHOLDS)}
                for row in self.rows for q in ("rt", "ccs", "both")]

    def test_parse_is_strict_and_strips_only_whitespace(self):
        self.assertEqual(parse_answer(" \nyes\t"), "yes")
        self.assertEqual(parse_answer("no"), "no")
        for raw in ("Yes", "YES", "yes.", "no!", '{"answer":"yes"}', "yes because", "", None, 1):
            self.assertIsNone(parse_answer(raw), repr(raw))

    def test_strict_threshold_and_conjunction(self):
        row = {"rt": 10., "ccs": 401.}
        self.assertEqual(label(row, "rt", THRESHOLDS), "no")
        self.assertEqual(label(row, "ccs", THRESHOLDS), "yes")
        self.assertEqual(label(row, "both", THRESHOLDS), "no")

    def test_perfect_and_same_peptide_joint_success(self):
        result = score(self.rows, self.perfect(), THRESHOLDS)
        self.assertEqual(result["primary"], 1.)
        self.assertEqual(result["joint_all_correct"], 1.)
        self.assertEqual(result["joint_all_correct_count"], 4)
        self.assertEqual(result["per_peptide"][1]["group_id"], "family1")
        self.assertEqual(result["per_peptide"][1]["truth"], {"rt": "yes", "ccs": "no", "both": "no"})

    def test_imbalanced_labels_do_not_turn_accuracy_into_balanced_accuracy(self):
        predictions = [{"id": row["id"], "question": q, "raw": "yes"}
                       for row in self.rows for q in ("rt", "ccs", "both")]
        result = score(self.rows, predictions, THRESHOLDS)
        rt = result["questions"]["rt"]
        self.assertEqual(rt["accuracy"], .75)
        self.assertEqual(rt["balanced_accuracy"], .5)
        self.assertEqual(rt["majority_baseline"], {"label": "yes", "accuracy": .75, "balanced_accuracy": .5})
        self.assertEqual(result["primary"], .5)
        self.assertEqual(result["joint_all_correct"], .5)

    def test_invalid_and_missing_keep_full_class_denominators(self):
        predictions = self.perfect()
        predictions[0]["raw"] = "YES"  # a/rt: invalid positive.
        predictions = [p for p in predictions if (p["id"], p["question"]) != ("d", "rt")]
        result = score(self.rows, predictions, THRESHOLDS)
        rt = result["questions"]["rt"]
        self.assertEqual(rt["class_counts"], {"no": 1, "yes": 3})
        self.assertAlmostEqual(rt["balanced_accuracy"], 1 / 3)
        self.assertEqual(rt["accuracy"], .5)
        self.assertEqual(rt["valid_rate"], .5)
        self.assertEqual(rt["confusion"]["yes"]["invalid"], 1)
        self.assertEqual(rt["confusion"]["no"]["invalid"], 1)
        self.assertEqual(result["missing_prediction_count"], 1)
        self.assertAlmostEqual(result["primary"], 2 / 3)
        self.assertEqual(result["joint_all_correct"], .5)

    def test_both_does_not_enter_primary(self):
        predictions = self.perfect()
        for prediction in predictions:
            if prediction["question"] == "both":
                prediction["raw"] = "invalid"
        result = score(self.rows, predictions, THRESHOLDS)
        self.assertEqual(result["primary"], 1.)
        self.assertEqual(result["questions"]["both"]["balanced_accuracy"], 0.)
        self.assertEqual(result["joint_all_correct"], 0.)

    def test_zero_predictions_is_zero_performance_not_empty_subset(self):
        result = score(self.rows, [], THRESHOLDS)
        self.assertEqual(result["primary"], 0.)
        self.assertEqual(result["missing_prediction_count"], 12)
        self.assertEqual(result["valid_rate"], 0.)

    def test_duplicate_and_unexpected_predictions_are_alignment_failures(self):
        valid = self.perfect()
        bads = [valid + [valid[0]], [{"id": "unknown", "question": "rt", "raw": "yes"}],
                [{"id": "a", "question": "unknown", "raw": "yes"}], [None]]
        for predictions in bads:
            with self.assertRaises(ValueError):
                score(self.rows, predictions, THRESHOLDS)

    def test_invalid_source_labels_and_thresholds_raise(self):
        for value in (float("nan"), float("inf"), True, "12"):
            with self.assertRaises(ValueError):
                label({"rt": value, "ccs": 450.}, "rt", THRESHOLDS)
            with self.assertRaises(ValueError):
                label(self.rows[0], "rt", {"rt": value, "ccs": 400.})
        with self.assertRaises(ValueError):
            label(self.rows[0], "unknown", THRESHOLDS)
        with self.assertRaises(ValueError):
            score(self.rows + [self.rows[0]], [], THRESHOLDS)
        with self.assertRaises(ValueError):
            score([], [], THRESHOLDS)

    def test_single_class_balanced_accuracy_is_null_not_invented(self):
        result = score(self.rows[:1], [], THRESHOLDS)
        self.assertIsNone(result["primary"])
        self.assertIsNone(result["questions"]["rt"]["balanced_accuracy"])
        self.assertEqual(result["questions"]["rt"]["accuracy"], 0.)
        json.dumps(result, allow_nan=False)

    def test_input_prediction_order_is_not_used_for_alignment(self):
        original = score(self.rows, self.perfect(), THRESHOLDS)
        shuffled = score(self.rows, list(reversed(self.perfect())), THRESHOLDS)
        self.assertEqual(original, shuffled)


if __name__ == "__main__":
    unittest.main()
