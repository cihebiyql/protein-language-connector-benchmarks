"""CPU regression checks: python -m unittest discover -s tests -p test_meltome.py -v"""
import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("meltome", Path(__file__).parents[1] / "src" / "meltome.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def row(seq, target=1.0, split="train"):
    return {"sequence": seq, "target": target, "split": split}


class MeltomeTests(unittest.TestCase):
    def test_blocks_exact_sequence_leakage_without_filtering(self):
        with self.assertRaisesRegex(ValueError, "cross-split leakage"):
            m.audit_rows([row("AA"), row("AA", split="test"), row("CC", split="validation")])

    def test_keeps_same_split_duplicates_and_unknown_residues(self):
        report = m.audit_rows([row("AU"), row("AU", 2), row("CC", split="validation"), row("DD", split="test")])
        self.assertEqual(report["total_rows"], 4)
        self.assertEqual(report["within_split_extra_duplicate_rows"]["train"], 1)
        self.assertEqual(report["noncanonical_residue_counts"], {"U": 2})
        self.assertAlmostEqual(m.features([row("AU")])[0, -1], 0.5)
        self.assertAlmostEqual(m.features([row("AU")])[0, 1:].sum(), 1.0)

    def test_training_and_selection_reject_test_rows(self):
        with self.assertRaisesRegex(ValueError, "TRAIN only"):
            m.fit_and_select([row("A", split="test")], [row("C", split="validation")], [1.0])
        with self.assertRaisesRegex(ValueError, "TRAIN only"):
            m.fit_and_select([row("A")], [row("C", split="test")], [1.0])

    def test_standardization_and_target_mean_use_train_only(self):
        train = [row("A", 1.0), row("CC", 3.0), row("DDD", 5.0)]
        val = [row("E" * 100, 999.0, "validation"), row("F" * 200, -999.0, "validation")]
        model = m.fit_and_select(train, val, [0.1, 1.0])
        self.assertEqual(model["standardization_mean"][0], 2.0)
        self.assertEqual(model["train_mean"], 3.0)
        self.assertFalse(model["refit_on_train_plus_validation"])
        altered = m.fit_and_select(train, [row("G", 0.0, "validation"), row("HH", 1.0, "validation")], [0.1, 1.0])
        self.assertEqual(model["standardization_mean"], altered["standardization_mean"])
        self.assertEqual(model["standardization_scale"], altered["standardization_scale"])

    def test_exact_selection_ties_choose_first_registered_alpha(self):
        train = [row("A", 0), row("CC", 2)]
        val = [row("D", 1.0, "validation"), row("EE", 1.0, "validation")]
        model = m.fit_and_select(train, val, [100, 1, 0.1])
        self.assertEqual(model["alpha"], 100)

    def test_spearman_ties_and_constant_predictions(self):
        self.assertAlmostEqual(m.spearman([1, 1, 2, 3], [1, 1, 2, 3]), 1.0)
        self.assertAlmostEqual(m.spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertIsNone(m.spearman([1, 2, 3], [2, 2, 2]))

    def test_scoring_full_denominator_and_nonfinite(self):
        for truth, pred, n in [([1, 2], [1], 2), ([1], [1], 2), ([1, 2], [1, math.nan], 2), ([1, math.inf], [1, 2], 2)]:
            with self.assertRaisesRegex(ValueError, "complete finite"):
                m.score(truth, pred, n)
        result = m.score([1, 3], [2, 1], 2)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["mae_degrees_C"], 1.5)
        self.assertAlmostEqual(result["rmse_degrees_C"], math.sqrt(2.5))

    def test_group_metrics_equal_group_weighting(self):
        rows = [row("A", 1), row("A", 3), row("C", 10)]
        metrics = m.group_score(rows, np.array([0.0, 0.0, 10.0]))
        self.assertEqual(metrics["n_exact_sequence_groups"], 2)
        self.assertEqual(metrics["group_macro_mae_degrees_C"], 1.0)
        self.assertAlmostEqual(metrics["group_macro_rmse_degrees_C"], math.sqrt(2.5))

    def test_invalid_labels_and_splits_block(self):
        cases = [("A", "nan", "train", "False"), ("A", "1", "test", "True"),
                 ("A", "1", "random", "False"), ("", "1", "train", "False"),
                 ("A", "1", "train", ""), ("AX!", "1", "train", "False")]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            for case in cases:
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["sequence", "target", "set", "validation"])
                    writer.writerow(case)
                with self.assertRaises(ValueError):
                    m.read_rows(path)

    def test_frozen_data_changes_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "input.csv").write_text("v1", encoding="utf-8")
            config = {"files_sha256": {"input.csv": m.sha(root / "input.csv")}}
            m.verify_inputs(config, root)
            (root / "input.csv").write_text("v2", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                m.verify_inputs(config, root)

    def test_existing_results_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "configs").mkdir()
            config = root / "configs" / "fixture.json"
            config.write_text(json.dumps({}), encoding="utf-8")
            existing = root / "results" / "meltome_human_v1" / "fixture"
            existing.mkdir(parents=True)
            sentinel = existing / "sentinel"
            sentinel.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                m.run(config, "fixture")
            self.assertEqual(sentinel.read_bytes(), b"preserve")

    def test_nonfinite_fit_targets_block(self):
        with self.assertRaisesRegex(ValueError, "must be finite"):
            m.fit_and_select([row("A", math.nan)], [row("C", 2.0, "validation")], [1.0])

    def test_existing_download_bytes_are_never_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data" / "meltome"
            data.mkdir(parents=True)
            asset = data / "splits.zip"
            asset.write_bytes(b"old bytes")
            config = {"files_sha256": {"data/meltome/splits.zip": "0" * 64}}
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                m.download_inputs(config, root)
            self.assertEqual(asset.read_bytes(), b"old bytes")


if __name__ == "__main__":
    unittest.main()
