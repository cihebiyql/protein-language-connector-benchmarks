"""CPU-only interpretation tests; no training/data/metric changes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

FILE = Path(__file__).resolve().parents[1] / "resources/recovery_summary_v1.py"
SPEC = importlib.util.spec_from_file_location("recovery_summary_v1", FILE)
summary_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary_module)


def analysis_fixture():
    comparisons = []
    for task in summary_module.TASKS:
        for backbone in summary_module.BACKBONES:
            for left in ("M0", "ESM"):
                # Higher is better for meltome/elm; lower is better for RT.
                before, after = (2.0, 1.0) if task == "rt" else (1.0, 2.0)
                comparisons.append({"task": task, "backbone": backbone, "left": left, "right": "M3",
                    "metric": summary_module.METRICS[task], "required_seeds": 3, "n_complete_pairs": 3,
                    "right_better_all_three": True,
                    "pairs": [{"seed": seed, "left": before, "right": after,
                               "left_minus_right": before - after} for seed in summary_module.SEEDS]})
    return {"schema": "multitask-independent-analysis.v1", "expected_cases": 45,
            "plan_sha256": "f" * 64, "data_manifest_sha256": "d" * 64,
            "analysis_status_counts": {"accepted": 45}, "queue_status_counts": {"accepted": 45},
            "paired_comparisons": comparisons,
            "arm_summaries": [{"task": "rt", "method": "M3", "backbone": "qwen3_4b_2507",
                              "metrics": {"mae": {"n_valid_seeds": 3, "mean": 1.0, "sample_sd": 0.2},
                                          "group_macro_mae": {"n_valid_seeds": 2, "mean": None, "sample_sd": None}}}]}


def write_input(tmp_path, value):
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def check_gain_signs_for_both_metric_directions_and_esm(tmp_path):
    result = summary_module.write_summary(write_input(tmp_path, analysis_fixture()), tmp_path / "report")
    for name in ("M3_vs_M0", "M3_vs_ESM"):
        for row in result["summary"][name]:
            assert row["mean_gain"] == 1
            assert row["sample_sd_gain"] == 0
            assert row["wins"] == 3
            assert row["three_of_three_favors_M3"] is True
    assert result["summary"]["overall_M3_vs_M0"]["supports_universal_stable_advantage"] is True


def check_missing_seed_retains_null_aggregates_and_overall(tmp_path):
    data = analysis_fixture()
    data["paired_comparisons"][0]["pairs"][2].update(left=None, right=None, left_minus_right=None)
    data["paired_comparisons"][0].update(n_complete_pairs=2, right_better_all_three=None)
    result = summary_module.write_summary(write_input(tmp_path, data), tmp_path / "report")["summary"]
    row = result["M3_vs_M0"][0]
    assert row["mean_gain"] is None
    assert row["sample_sd_gain"] is None
    assert row["three_of_three_favors_M3"] is None
    assert row["pairs"][2]["gain"] is None
    assert result["overall_M3_vs_M0"]["status"] == "incomplete"
    assert result["overall_M3_vs_M0"]["supports_universal_stable_advantage"] is None


def check_seed_reversal_prevents_universal_stability_even_with_positive_mean(tmp_path):
    data = analysis_fixture()
    row = data["paired_comparisons"][0]
    row["pairs"][2].update(left=1.0, right=0.5, left_minus_right=0.5)
    row["right_better_all_three"] = False
    result = summary_module.write_summary(write_input(tmp_path, data), tmp_path / "report")["summary"]
    assert result["M3_vs_M0"][0]["mean_gain"] > 0
    assert result["M3_vs_M0"][0]["three_of_three_favors_M3"] is False
    assert result["M3_vs_M0"][0]["losses"] == 1
    assert result["overall_M3_vs_M0"]["all_strata_mean_gain_positive"] is True
    assert result["overall_M3_vs_M0"]["supports_universal_stable_advantage"] is False
    assert "不支持" in result["overall_M3_vs_M0"]["conclusion"]


def check_tie_counts_and_source_flag_cannot_overrule_seed_values(tmp_path):
    data = analysis_fixture()
    data["paired_comparisons"][0]["pairs"][0].update(left=1.0, right=1.0, left_minus_right=0)
    result = summary_module.write_summary(write_input(tmp_path, data), tmp_path / "report")["summary"]
    row = result["M3_vs_M0"][0]
    assert row["ties"] == 1
    assert row["three_of_three_favors_M3"] is False
    assert row["source_direction_consistent"] is False


def check_missing_stratum_and_secondary_values_are_not_imputed(tmp_path):
    data = analysis_fixture()
    del data["paired_comparisons"][0]
    result = summary_module.write_summary(write_input(tmp_path, data), tmp_path / "report")["summary"]
    assert result["M3_vs_M0"][0]["source_present"] is False
    assert result["M3_vs_M0"][0]["mean_gain"] is None
    assert result["overall_M3_vs_M0"]["complete_strata"] == 5
    assert result["frozen_arm_metrics"][1]["mean"] is None
    assert result["frozen_arm_metrics"][1]["sample_sd"] is None


def check_output_provenance_bytes_and_existing_report_preserved(tmp_path):
    source = write_input(tmp_path, analysis_fixture())
    original = source.read_bytes()
    result = summary_module.write_summary(source, tmp_path / "report")
    assert source.read_bytes() == original
    assert result["summary"]["analysis_sha256"] == hashlib.sha256(original).hexdigest()
    assert json.loads(Path(result["summary_path"]).read_text(encoding="utf-8")) == result["summary"]
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    for phrase in ("3/3同方向", "统计显著", "31候选", "OOM", "生物学独立重复", "C1", "SHA256"):
        assert phrase in report
    with unittest.TestCase().assertRaises(FileExistsError):
        summary_module.write_summary(source, tmp_path / "report")


def check_corrupt_comparison_metadata_is_reported_not_reinterpreted(tmp_path, field, value):
    data = analysis_fixture()
    data["paired_comparisons"][0][field] = value
    with unittest.TestCase().assertRaises(ValueError):
        summary_module.write_summary(write_input(tmp_path, data), tmp_path / "report")


class RecoverySummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_gain_signs(self):
        check_gain_signs_for_both_metric_directions_and_esm(self.root)

    def test_missing_seed(self):
        check_missing_seed_retains_null_aggregates_and_overall(self.root)

    def test_seed_reversal(self):
        check_seed_reversal_prevents_universal_stability_even_with_positive_mean(self.root)

    def test_tie_and_source_flag(self):
        check_tie_counts_and_source_flag_cannot_overrule_seed_values(self.root)

    def test_missing_stratum_and_secondary(self):
        check_missing_stratum_and_secondary_values_are_not_imputed(self.root)

    def test_output_provenance_and_preservation(self):
        check_output_provenance_bytes_and_existing_report_preserved(self.root)

    def test_wrong_metric(self):
        check_corrupt_comparison_metadata_is_reported_not_reinterpreted(self.root, "metric", "accuracy")

    def test_duplicate_seed(self):
        check_corrupt_comparison_metadata_is_reported_not_reinterpreted(self.root, "pairs", [
            {"seed": 17, "left": 1, "right": 2}, {"seed": 17, "left": 1, "right": 2}])


if __name__ == "__main__":
    unittest.main()
