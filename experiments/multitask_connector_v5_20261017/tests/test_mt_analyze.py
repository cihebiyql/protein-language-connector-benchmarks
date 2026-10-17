import json
import tempfile
import unittest
from pathlib import Path

from src.mt_analyze import collect, file_sha, fixed_cases, paired_comparisons, read_json, write_report
from src.mt_metrics import build_metrics, score_intervals


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_lines(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v) + "\n" for v in values), encoding="utf-8")


class FakeData:
    @staticmethod
    def load_manifest(root):
        return {"source_pins": {"official": "fixed"}}

    @staticmethod
    def load_task(root, task, split):
        if split == "calibration":
            return [{"sample_id": "cal" + str(i), "target": 0.0, "group_id": str(i), "split": "calibration"}
                    for i in range(10)]
        return [{"sample_id": "a", "target": 0.0, "group_id": "g", "positive_ids": ["A"]},
                {"sample_id": "b", "target": 1.0, "group_id": "g", "positive_ids": ["B"]}]

    @staticmethod
    def load_catalog(root):
        return [{"candidate_id": "A", "description": "Alpha"}, {"candidate_id": "B", "description": "Beta"}]


class AnalyzerTests(unittest.TestCase):
    def fixture(self, root, method="ESM", task="meltome"):
        data = root / "data"
        write(data / "manifest.json", {})
        cases = [{"case_id": str(i), "task": task, "method": m, "backbone": backbone, "seed": seed}
                 for i, (task, m, backbone, seed) in enumerate(fixed_cases())]
        plan = {"scope_id": "test", "paths": {"data_root": str(data)}, "cases": cases}
        plan_path = root / "plan.json"
        write(plan_path, plan)
        queue = root / "queue"
        case = next(c for c in cases if c["task"] == task and c["method"] == method)
        artifacts = queue / "runs" / case["case_id"] / "attempt_001" / "artifacts"
        predictions = [{"sample_id": "b", "prediction": 1.0}, {"sample_id": "a", "prediction": 0.0}]
        write_lines(artifacts / "readout_predictions.jsonl", predictions)
        language, c1 = None, None
        if method != "ESM":
            language = [{"sample_id": "a", "text": "invalid"}, {"sample_id": "b", "text": "invalid"}]
            c1 = [{**p, "prediction_off": 0.5} for p in predictions]
            write_lines(artifacts / "LM_predictions.jsonl", language)
            write_lines(artifacts / "C1_predictions.jsonl", c1)
        metrics = build_metrics(task, FakeData.load_task(None, None, None), predictions, language, c1)
        if task == "rt":
            calibration = FakeData.load_task(None, task, "calibration")
            calibration_predictions = [{"sample_id": r["sample_id"], "prediction": 1} for r in calibration]
            write_lines(artifacts / "calibration_predictions.jsonl", calibration_predictions)
            metrics["calibration"] = score_intervals(calibration, calibration_predictions,
                                                     FakeData.load_task(None, task, "test"), predictions)
        write(artifacts / "metrics.json", metrics)
        write(artifacts / "run_manifest.json", {"scope_id": "test", "phase": "formal", "status": "complete", "model_policy": {"parameters": 4}})
        binding = {"example": "fixed-sha", "plan_sha256": file_sha(plan_path)}
        write(queue / "queue_binding.json", {"plan_sha256": file_sha(plan_path)})
        write_lines(queue / "queue_events.jsonl", [{"case_id": case["case_id"], "status": "accepted",
                    "attempt": str(artifacts.parent), "binding": binding}])
        return plan_path, queue, data, artifacts, binding

    def test_all_45_statuses_and_partial_pending_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, queue, data, _, binding = self.fixture(Path(directory))
            result = collect(plan, queue, data, validator=lambda *args: binding, data_module=FakeData)
            self.assertEqual(len(result["cases"]), 45)
            self.assertEqual(result["analysis_status_counts"], {"accepted": 1, "not_accepted": 44})
            self.assertEqual(result["queue_status_counts"]["pending"], 44)
            self.assertTrue(all(c["mean"] is None for c in result["paired_comparisons"]))
            output = Path(directory) / "analysis"
            write_report(result, output)
            self.assertTrue((output / "REPORT.md").exists())
            with self.assertRaises(FileExistsError):
                write_report(result, output)

    def test_saved_metric_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, queue, data, artifacts, binding = self.fixture(Path(directory))
            saved = read_json(artifacts / "metrics.json")
            saved["readout"]["mae"] = 99
            write(artifacts / "metrics.json", saved)
            result = collect(plan, queue, data, validator=lambda *args: binding, data_module=FakeData)
            self.assertEqual(result["analysis_status_counts"]["revalidation_rejected"], 1)

    def test_invalid_lm_is_reported_not_hidden_or_case_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, queue, data, _, binding = self.fixture(Path(directory), method="M0")
            result = collect(plan, queue, data, validator=lambda *args: binding, data_module=FakeData)
            accepted = next(c for c in result["cases"] if c["analysis_status"] == "accepted")
            self.assertEqual(accepted["recomputed_metrics"]["language"]["coverage"]["fraction"], 0)
            self.assertIsNone(accepted["recomputed_metrics"]["language"]["mae"])

    def test_wrong_queue_binding_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, queue, data, _, _ = self.fixture(Path(directory))
            result = collect(plan, queue, data, validator=lambda *args: {"wrong": "sha"}, data_module=FakeData)
            self.assertEqual(result["analysis_status_counts"]["revalidation_rejected"], 1)

    def test_rt_intervals_independently_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, queue, data, artifacts, binding = self.fixture(Path(directory), task="rt")
            result = collect(plan, queue, data, validator=lambda *args: binding, data_module=FakeData)
            self.assertEqual(result["analysis_status_counts"]["accepted"], 1)
            saved = read_json(artifacts / "metrics.json")
            saved["calibration"]["interval_width"] = 99
            write(artifacts / "metrics.json", saved)
            result = collect(plan, queue, data, validator=lambda *args: binding, data_module=FakeData)
            self.assertEqual(result["analysis_status_counts"]["revalidation_rejected"], 1)

    def test_pairing_seed_direction_and_complete_mean_sd(self):
        rows = []
        for method, values in (("M0", [3, 4, 5]), ("M3", [2, 3, 4])):
            for seed, value in zip((17, 42, 2026), values):
                rows.append({"task": "rt", "method": method, "backbone": "qwen25_1p5b", "seed": seed,
                             "analysis_status": "accepted", "primary_value": value})
        result = next(c for c in paired_comparisons(rows) if c["task"] == "rt" and c["left"] == "M0" and c["backbone"] == "qwen25_1p5b")
        self.assertEqual(result["mean"], 1)
        self.assertEqual(result["sample_sd"], 0)
        self.assertTrue(result["right_better_all_three"])
        partial = next(c for c in paired_comparisons(rows[:-1]) if c["task"] == "rt" and c["left"] == "M0" and c["backbone"] == "qwen25_1p5b")
        self.assertIsNone(partial["mean"])
        self.assertEqual(partial["n_complete_pairs"], 2)

    def test_calibration_leakage_rejected_and_empirical_groups(self):
        rows = [{"sample_id": str(i), "target": 0, "group_id": str(i), "split": "calibration"} for i in range(10)]
        predictions = [{"sample_id": str(i), "prediction": 1} for i in range(10)]
        test = [{"sample_id": "test-a", "target": 0, "group_id": "a", "split": "test"},
                {"sample_id": "test-b", "target": 0, "group_id": "a", "split": "test"}]
        result = score_intervals(rows, predictions, test, [{"sample_id": "test-a", "prediction": 0}, {"sample_id": "test-b", "prediction": 2}])
        self.assertEqual(result["interval_width"], 2)
        self.assertEqual(result["empirical_sample_coverage"], 0.5)
        self.assertEqual(result["empirical_group_coverage"], 0)
        with self.assertRaises(ValueError):
            score_intervals([{**rows[0], "split": "test"}], predictions[:1], test, [])
        with self.assertRaises(ValueError):
            score_intervals(rows, predictions, [{**test[0], "sample_id": "0"}], [])


if __name__ == "__main__":
    unittest.main()
