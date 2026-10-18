"""Synthetic-only tests of frozen stop gates and grouped paired inference."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.qa_analyze import analyze_scores, analyze_scope, compare_scores, paired_group_bootstrap
from src.qa_metrics import label, score


THRESHOLDS = {"rt": 10., "ccs": 400.}


def config():
    return {
        "seeds": [17, 42, 2026], "arms": ["TEXT", "M0", "M3"], "thresholds": THRESHOLDS,
        "decision": {
            "candidate_memory_arms": ["M0", "M3"], "minimum_mean_primary_gain": .02,
            "other_endpoint_mean_gain_floor": -.02, "paired_group_bootstrap_replicates": 50,
            "bootstrap_seed": 156, "one_sided_lower_quantile": .025,
            "two_candidate_arms_familywise_one_sided_error": .05,
            "molecular_controls": ["mean", "shuffled"], "minimum_control_mean_primary_drop": .02,
        },
    }


def rows():
    # Every connected group contains all four combinations, so every bootstrap
    # draw has both physical classes. QA rows and seeds are not sampling units.
    return [{"id": f"g{group}-p{pattern}", "group_id": f"g{group}",
             "rt": 11. if pattern & 1 else 9., "ccs": 401. if pattern & 2 else 399.}
            for group in range(4) for pattern in range(4)]


def predictions(panel, mode):
    return [{"id": row["id"], "question": question,
             "raw": label(row, question, THRESHOLDS) if mode == "perfect" else "no"}
            for row in panel for question in ("rt", "ccs", "both")]


def run_scores(memory="perfect"):
    panel = rows()
    result = {}
    for arm in ("TEXT", "M0", "M3"):
        for seed in (17, 42, 2026):
            modes = ("normal", "shuffled", "removed") if arm == "TEXT" else ("normal", "mean", "shuffled", "off")
            result[f"{arm}__seed{seed}"] = {
                mode: score(panel, predictions(panel, memory if arm != "TEXT" and mode == "normal" else "constant"), THRESHOLDS)
                for mode in modes}
    return result


class AnalysisTests(unittest.TestCase):
    def test_obvious_stable_gain_passes_all_gates(self):
        summary = analyze_scores(run_scores(), config())
        self.assertEqual(summary["status"], "complete_stable_gain")
        self.assertEqual(summary["selected_arm"], "M0")  # Fixed candidate order for exact tie.
        for arm in ("M0", "M3"):
            self.assertTrue(summary["contrasts"][arm]["passes_all_frozen_gates"])
            self.assertEqual(summary["contrasts"][arm]["primary_gain"]["values"], [.5, .5, .5])
            self.assertEqual(summary["bootstrap"]["contrasts"][arm]["lower_primary_gain"], .5)

    def test_zero_gain_pauses_after_complete_verified_matrix(self):
        summary = analyze_scores(run_scores("constant"), config())
        self.assertEqual(summary["status"], "complete_direction_paused")
        self.assertIsNone(summary["selected_arm"])
        self.assertFalse(summary["contrasts"]["M3"]["passes_all_frozen_gates"])

    def test_missing_seed_cannot_be_averaged_or_scientifically_paused(self):
        scores = run_scores()
        del scores["M3__seed17"]
        with self.assertRaisesRegex(ValueError, "every frozen case"):
            analyze_scores(scores, config())

    def test_one_bad_seed_and_ignored_control_block_false_gain(self):
        scores = run_scores()
        scores["M0__seed17"]["normal"] = copy.deepcopy(scores["TEXT__seed17"]["normal"])
        for seed in (17, 42, 2026):
            scores[f"M3__seed{seed}"]["mean"] = copy.deepcopy(scores[f"M3__seed{seed}"]["normal"])
        summary = analyze_scores(scores, config())
        self.assertEqual(summary["status"], "complete_direction_paused")
        self.assertFalse(summary["contrasts"]["M0"]["gates"]["all_three_seed_primary_gains_positive"])
        self.assertFalse(summary["contrasts"]["M3"]["gates"]["mean_minimum_mean_drop"])

    def test_off_is_reported_but_not_a_substitute_for_required_controls(self):
        scores = run_scores()
        for arm in ("M0", "M3"):
            for seed in (17, 42, 2026):
                scores[f"{arm}__seed{seed}"]["off"] = copy.deepcopy(scores[f"{arm}__seed{seed}"]["normal"])
        summary = analyze_scores(scores, config())
        self.assertEqual(summary["status"], "complete_stable_gain")
        self.assertEqual(summary["contrasts"]["M0"]["control_primary_drops"]["off"]["mean"], 0.)

    def test_good_properties_without_joint_answers_do_not_pass(self):
        scores = run_scores()
        panel = rows()
        raw = predictions(panel, "perfect")
        for record in raw:
            if record["question"] == "both":
                record["raw"] = "invalid"
        for arm in ("M0", "M3"):
            for seed in (17, 42, 2026):
                scores[f"{arm}__seed{seed}"]["normal"] = score(panel, raw, THRESHOLDS)
        summary = analyze_scores(scores, config())
        self.assertEqual(summary["status"], "complete_direction_paused")
        for arm in ("M0", "M3"):
            self.assertFalse(summary["contrasts"][arm]["gates"]["joint_all_three_answers_mean_gain_positive"])

    def test_bootstrap_recomputes_class_denominators_inside_each_draw(self):
        panel = [{"id": f"g{group}-p{index}", "group_id": f"g{group}",
                  "rt": 11. if index < (3 if group == 0 else 1) else 9.,
                  "ccs": 401. if index % 2 else 399.}
                 for group in (0, 1) for index in range(4)]
        baseline = predictions(panel, "constant")
        memory = copy.deepcopy(baseline)
        for prediction in memory:
            if prediction["question"] == "rt":
                # group0: all yes, group1: opposite of the true RT answer.
                row = next(row for row in panel if row["id"] == prediction["id"])
                prediction["raw"] = "yes" if row["group_id"] == "g0" or row["rt"] < 10. else "no"
        scores = {}
        for arm in ("TEXT", "M0", "M3"):
            for seed in (17, 42, 2026):
                scores[f"{arm}__seed{seed}"] = {"normal": score(panel, baseline if arm == "TEXT" else memory, THRESHOLDS)}
        settings = config()
        receipt = paired_group_bootstrap(scores, settings["seeds"], settings["decision"])
        rng = np.random.default_rng(156)
        manual = []
        for _ in range(settings["decision"]["paired_group_bootstrap_replicates"]):
            drawn = rng.integers(0, 2, size=2, dtype=np.int64)
            g0, g1 = np.bincount(drawn, minlength=2)
            rt_ba = .5 * (3 * g0 / (3 * g0 + g1))
            manual.append((rt_ba + .5) / 2 - .5)
        self.assertAlmostEqual(receipt["contrasts"]["M0"]["lower_primary_gain"], float(np.quantile(manual, .025)))
        self.assertAlmostEqual(receipt["contrasts"]["M0"]["bootstrap_mean_primary_gain"], float(np.mean(manual)))

    def test_bootstrap_samples_connected_groups_and_keeps_seeds_questions(self):
        scores = run_scores()
        receipt = paired_group_bootstrap(scores, config()["seeds"], config()["decision"])
        self.assertEqual(receipt["group_count"], 4)
        self.assertEqual(receipt["peptide_count"], 16)
        self.assertEqual(receipt["sampling_unit"], "connected_group")
        self.assertEqual(receipt["seeds_kept_together"], [17, 42, 2026])
        self.assertEqual(receipt["questions_kept_together"], ["rt", "ccs", "both"])
        self.assertTrue(receipt["same_resamples_for_both_candidate_arms"])
        repeated = paired_group_bootstrap(scores, config()["seeds"], config()["decision"])
        self.assertEqual(receipt, repeated)
        self.assertEqual(receipt["contrasts"]["M0"], receipt["contrasts"]["M3"])

    def test_bootstrap_detects_unpaired_groups_or_truths(self):
        scores = run_scores()
        scores["M3__seed17"]["normal"]["per_peptide"][0]["group_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "group or truth changed"):
            paired_group_bootstrap(scores, config()["seeds"], config()["decision"])

    def test_saved_metric_mismatch_detection_and_runtime_extras(self):
        recomputed = score(rows(), predictions(rows(), "perfect"), THRESHOLDS)
        saved = copy.deepcopy(recomputed)
        saved["wall_seconds"] = 42.
        compare_scores(recomputed, saved)
        saved["primary"] -= .001
        with self.assertRaisesRegex(ValueError, "numeric mismatch"):
            compare_scores(recomputed, saved)
        saved = copy.deepcopy(recomputed)
        saved["questions"]["rt"]["confusion"]["yes"]["invalid"] = 1
        with self.assertRaisesRegex(ValueError, "numeric mismatch"):
            compare_scores(recomputed, saved)

    def test_incomplete_scope_does_not_even_open_test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = Path(directory)
            (scope / "config.json").write_text(json.dumps(config()), encoding="utf-8")
            def missing(*_):
                raise FileNotFoundError("synthetic missing case")
            verifier = types.SimpleNamespace(verify_freeze=lambda _: None, verify_result=missing)
            result = analyze_scope(scope, verifier)
            self.assertEqual(result["status"], "technical_inconclusive")
            self.assertEqual(result["accepted_cases"], 0)
            self.assertTrue(result["no_scientific_stop_decision"])
            self.assertEqual(len(result["problems"]), 9)

    def test_scope_recomputes_raw_answers_and_rejects_saved_metric_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = Path(directory)
            (scope / "config.json").write_text(json.dumps(config()), encoding="utf-8")
            (scope / "data").mkdir()
            panel = rows()
            (scope / "data" / "test.jsonl").write_text("".join(json.dumps(row) + "\n" for row in panel), encoding="utf-8")
            for arm in ("TEXT", "M0", "M3"):
                for seed in (17, 42, 2026):
                    output = scope / "runs" / f"{arm}__seed{seed}" / "formal"
                    output.mkdir(parents=True)
                    conditions = ("normal", "shuffled", "removed") if arm == "TEXT" else ("normal", "mean", "shuffled", "off")
                    summaries = {}
                    for condition in conditions:
                        raw = predictions(panel, "perfect" if arm != "TEXT" and condition == "normal" else "constant")
                        (output / f"test_{condition}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw), encoding="utf-8")
                        summaries[condition] = score(panel, raw, THRESHOLDS)
                    (output / "test_summary.json").write_text(json.dumps(summaries), encoding="utf-8")
            verifier = types.SimpleNamespace(verify_freeze=lambda _: None, verify_result=lambda *_: {"synthetic_verified": True})
            good = analyze_scope(scope, verifier)
            self.assertEqual(good["status"], "complete_stable_gain")
            self.assertTrue(good["independent_rescore"]["all_raw_answer_metrics_match_saved"])
            changed = scope / "runs" / "M0__seed17" / "formal" / "test_summary.json"
            saved = json.loads(changed.read_text(encoding="utf-8"))
            saved["normal"]["primary"] = .75
            changed.write_text(json.dumps(saved), encoding="utf-8")
            bad = analyze_scope(scope, verifier)
            self.assertEqual(bad["status"], "technical_inconclusive")
            self.assertTrue(bad["no_scientific_stop_decision"])


if __name__ == "__main__":
    unittest.main()
