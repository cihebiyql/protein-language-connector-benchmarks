"""Predeclared pilot gates, equal-parent pairing, completion and privacy guards."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "src/pilot_report.py"
spec = importlib.util.spec_from_file_location("pilot_report_under_test", MODULE)
r = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r
spec.loader.exec_module(r)


def fixture():
    private = [{"sample_id": f"private_sample_{i}", "parent_accession": "private_parent_" + str(i % 8),
                "anchor_class_id": "MOD_CK2_1", "motifae_short_anchor_auc": .8,
                "raw_probe_short_anchor_auc": .7, "sequence": "PRIVATESEQUENCE"} for i in range(67)]
    feature = {"test_rows": 67, "test_parents": 8, "private_per_row": private,
               "qualified_classes": {"motifae": ["MOD_CK2_1"], "raw_probe": ["MOD_CK2_1"]},
               "metrics": {"motifae_short_anchor_auc": {"macro_parent": .8, "defined_rows": 67,
                                                         "description": "PRIVATE DESCRIPTION"},
                           "raw_probe_short_anchor_auc": {"macro_parent": .7, "defined_rows": 67},
                           "top32_full_short_jaccard": {"macro_parent": .2},
                           "private_sample_1": {"macro_parent": 1}},
               "cache_file_receipts": {"short/privatehash.npz": {"sha256": "a" * 64}}}
    class_counts = {"sequence": 10, "regex": 20, "raw_probe": 24,
                    "motifae": 35, "motifae_shuffled": 12, "template": 36}
    arms = {}
    for arm in r.ARMS:
        arms[arm] = {}
        for question in r.QUESTIONS:
            arms[arm][question] = {"expected": 67, "observed": 67, "valid": 67, "invalid": 0,
                                    "valid_rate": 1.0, "raw_answer": "PRIVATE ANSWER"}
        arms[arm]["class"].update(known_positive_recovered_count=class_counts[arm],
                                   known_positive_recovered_rate=class_counts[arm] / 67)
        arms[arm]["position"].update(known_positive_overlap_count=25, known_positive_overlap_rate=25 / 67)
        arms[arm]["activity"].update(activity_overclaim_count=0, activity_overclaim_rate=0.0)
        arms[arm]["evidence"].update(strongest_record_or_correct_abstention_count=67,
                                      strongest_record_or_correct_abstention_rate=1.0)
    qa = {"status": "complete", "test_peptides": 67, "expected_records": 1608, "observed_records": 1608,
          "qwen_generations": 1340, "template_answers": 268, "arms": arms,
          "private_card": {"sample_id": "PRIVATE CARD"}}
    manifest = {"summary": {"splits": {split: {"rows": n, "parents": n - 5, "components": n - 8,
                                                "classes": 18, "sample_ids": ["PRIVATE ID"]}
                                         for split, n in (("train", 201), ("qualification", 39), ("test", 67))}},
                "source_path": "D:/private/secret"}
    config = {"pilot_signals": dict(r.DEFAULT_SIGNALS),
              "dictionary": {"qualification_auc": .7, "min_train_parents": 2, "min_qualification_parents": 2},
              "qwen_path": "/data/private/model"}
    return feature, qa, manifest, config


class PilotReportTests(unittest.TestCase):
    def test_all_four_predeclared_gates_required_for_promising_status(self):
        feature, qa, manifest, config = fixture()
        summary = r.build_summary(feature, qa, manifest, config)
        self.assertEqual(summary["status"], "complete_pilot_promising")
        self.assertTrue(all(gate["passed"] for gate in summary["gates"].values()))
        self.assertFalse(summary["stable_multi_seed_gain_established"])
        self.assertIn("v6", summary["decision_scope"])

    def test_failed_paired_auc_gate_is_completed_no_incremental_gain(self):
        feature, qa, manifest, config = fixture()
        for row in feature["private_per_row"]:
            row["motifae_short_anchor_auc"] = .71
        summary = r.build_summary(feature, qa, manifest, config)
        self.assertEqual(summary["status"], "complete_pilot_no_incremental_gain")
        self.assertFalse(summary["gates"]["paired_short_auc_gain"]["passed"])
        self.assertTrue(summary["completion"]["qa_pipeline_complete"])

    def test_class_gain_uses_best_of_three_qwen_controls_not_template(self):
        feature, qa, manifest, config = fixture()
        qa["arms"]["template"]["class"].update(known_positive_recovered_count=67,
                                                  known_positive_recovered_rate=1.0)
        summary = r.build_summary(feature, qa, manifest, config)
        self.assertEqual(summary["status"], "complete_pilot_promising")
        self.assertAlmostEqual(summary["gates"]["qwen_class_gain_over_best_control"]["observed"], 11 / 67)

    def test_wrong_card_drop_and_overclaim_can_each_veto(self):
        feature, qa, manifest, config = fixture()
        qa["arms"]["motifae_shuffled"]["class"].update(known_positive_recovered_count=35,
                                                         known_positive_recovered_rate=35 / 67)
        self.assertEqual(r.build_summary(feature, qa, manifest, config)["status"],
                         "complete_pilot_no_incremental_gain")
        feature, qa, manifest, config = fixture()
        qa["arms"]["motifae"]["activity"].update(activity_overclaim_count=1, activity_overclaim_rate=1 / 67)
        summary = r.build_summary(feature, qa, manifest, config)
        self.assertEqual(summary["status"], "complete_pilot_no_incremental_gain")
        self.assertFalse(summary["gates"]["activity_overclaim_increase"]["passed"])

    def test_pairing_precedes_parent_average_and_does_not_subtract_unmatched_coverage(self):
        rows = [{"parent_accession": "A", "motifae_short_anchor_auc": .9, "raw_probe_short_anchor_auc": .7}] * 64
        rows += [{"parent_accession": "B", "motifae_short_anchor_auc": .5, "raw_probe_short_anchor_auc": .7}] * 3
        rows += [{"parent_accession": "C", "motifae_short_anchor_auc": 1.0, "raw_probe_short_anchor_auc": None}]
        paired = r.paired_short_auc(rows)
        self.assertAlmostEqual(paired["mean_gain"], 0.0)
        self.assertAlmostEqual(paired["sample_sd"], .2 * 2 ** .5)
        self.assertEqual(paired["paired_parents"], 2)
        self.assertEqual(paired["paired_rows"], 67)
        self.assertEqual(paired["excluded_unpaired_rows"], 1)

    def test_missing_partial_or_duplicate_outputs_are_technical_inconclusive(self):
        feature, qa, manifest, config = fixture()
        self.assertEqual(r.build_summary(None, qa, manifest, config)["status"], "technical_inconclusive")
        self.assertEqual(r.build_summary(feature, None, manifest, config)["status"], "technical_inconclusive")
        qa["observed_records"] = 1607
        self.assertEqual(r.build_summary(feature, qa, manifest, config)["status"], "technical_inconclusive")
        feature, qa, manifest, config = fixture()
        feature["private_per_row"][-1]["sample_id"] = feature["private_per_row"][0]["sample_id"]
        self.assertEqual(r.build_summary(feature, qa, manifest, config)["status"], "technical_inconclusive")

    def test_failure_stage_does_not_publish_private_exception_path(self):
        feature, qa, manifest, config = fixture()
        summary = r.build_summary(feature, qa, manifest, config, failure_stage="D:/private/file")
        self.assertEqual(summary["status"], "technical_inconclusive")
        self.assertEqual(summary["completion"]["failure_stage"], "unspecified")
        self.assertNotIn("D:/private", r.dumps(summary))

    def test_public_summary_and_report_contain_no_per_query_fields_or_payloads(self):
        feature, qa, manifest, config = fixture()
        summary = r.build_summary(feature, qa, manifest, config)
        encoded = r.dumps(summary) + r.render_report(summary)
        for forbidden in ("private_sample_", "private_parent_", "PRIVATESEQUENCE", "PRIVATE DESCRIPTION",
                          "PRIVATE ANSWER", "PRIVATE CARD", "PRIVATE ID", "D:/private", "/data/private",
                          "privatehash.npz", "raw_answer", "private_per_row", "cache_file_receipts"):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(summary["data_counts"]["test"]["rows"], 67)
        self.assertEqual(summary["feature"]["qualified_class_counts"]["motifae"], 1)
        json.loads(r.dumps(summary))

    def test_invalid_signal_config_does_not_silently_relax_gate(self):
        feature, qa, manifest, config = fixture()
        config["pilot_signals"]["posthoc_extra_gate"] = .01
        self.assertEqual(r.build_summary(feature, qa, manifest, config)["status"], "technical_inconclusive")

    def test_missing_files_cli_generates_safe_technical_report(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = r.run(Path(folder))
            self.assertEqual(summary["status"], "technical_inconclusive")
            self.assertTrue((Path(folder) / "reports/SUMMARY.json").is_file())
            self.assertTrue((Path(folder) / "reports/REPORT.md").is_file())
            self.assertNotIn(folder, (Path(folder) / "reports/SUMMARY.json").read_text(encoding="utf-8"))

    def test_changed_derived_report_preserves_prior_report_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "SUMMARY.json"
            r.preserve_and_write(path, "first")
            r.preserve_and_write(path, "second")
            self.assertEqual(path.read_text(), "second")
            archives = list((Path(folder) / "history").glob("*.json"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].read_text(), "first")


if __name__ == "__main__":
    unittest.main()
