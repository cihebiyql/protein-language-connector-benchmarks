"""A2 record-state and label-budget regression checks; no model dependencies."""
import importlib.util
import json
import csv
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_rt", ROOT / "scripts/bench_v02/prepare_rt.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)
reconcile_spec = importlib.util.spec_from_file_location("reconcile_rt_source", ROOT / "scripts/bench_v02/reconcile_rt_source.py")
reconcile_rt = importlib.util.module_from_spec(reconcile_spec)
reconcile_spec.loader.exec_module(reconcile_rt)


class TestRetentionTimeProtocol(unittest.TestCase):
    def row(self, sequence="ACDMK", modifications="2|Carbamidomethyl|4|Oxidation"):
        return {"Sequence": sequence, "Modifications": modifications, "RT": "32.4", "RT_std": "0.2", "RT_count": "3"}

    def test_ptm_preservation_and_positions(self):
        obj = rt.normalize_march(self.row(), 1, True)
        self.assertEqual(obj["peptidoform"], "ACDMK|2:Carbamidomethyl|4:Oxidation")
        self.assertTrue(obj["eligibility"]["measurement_valid"])
        self.assertEqual(obj["rt_unit"], "min")

    def test_unknown_chemistry_not_coerced(self):
        obj = rt.normalize_march(self.row("aC[foo]M", "0|Cyclization"), 2, True)
        self.assertEqual(obj["peptide_sequence"], "aC[foo]M")
        self.assertEqual(obj["modifications"][0]["name"], "Cyclization")
        self.assertFalse(obj["eligibility"]["measurement_valid"])

    def test_legacy_zero_based_oxidation_not_silently_accepted(self):
        obj = rt.normalize_march(self.row("MACDK", "0|Oxidation"), 3, True)
        self.assertIn("oxidation_residue_mismatch", obj["readiness_reasons"])

    def test_source_semantics_gate(self):
        rows = [rt.normalize_march(self.row("A" * i + "C", ""), i, True) for i in range(1, 130)]
        budgets = rt.adaptation_budgets(rows, {"LLLL"})
        for panel in budgets.values():
            for budget in panel["budgets"].values():
                self.assertEqual(budget["status"], "unsupported")

    def test_insufficient_budget_never_resampled(self):
        rows = [rt.normalize_march(self.row("A" * i + "C", ""), i, True) for i in range(1, 130)]
        budgets = rt.adaptation_budgets(rows, {"LLLL"}, True)
        for panel in budgets.values():
            b = panel["budgets"]["2048"]
            self.assertEqual(b["status"], "unsupported")
            self.assertLess(b["actual_target_labels"], 2048)
            self.assertEqual(len(b["label_ids"]), len(set(b["label_ids"])))

    def test_chemical_variants_share_holdout_group(self):
        a = rt.normalize_march(self.row(), 1, True)
        b = rt.normalize_march(self.row(modifications=""), 2, True)
        self.assertEqual(a["groups"]["sequence"], b["groups"]["sequence"])

    def test_real_budget_accounting_and_fixed_holdout(self):
        path = ROOT / "data/processed/bench_v02/a2_target_budgets.json"
        if not path.exists():
            self.skipTest("real processed artifact not present")
        panels = json.loads(path.read_text())
        for panel in panels.values():
            test_ids = set(panel["test_ids"])
            previous = set()
            for number in (32, 128, 512, 2048):
                b = panel["budgets"][str(number)]
                ids = set(b["label_ids"])
                union = set(b["fit_ids"]) | set(b["validation_ids"]) | set(b["calibration_allowed_ids"])
                self.assertEqual(ids, union)
                self.assertLessEqual(len(ids), number)
                self.assertFalse(ids & test_ids)
                self.assertTrue(previous <= ids)
                self.assertFalse(set(b["fit_ids"]) & set(b["validation_ids"]))
                previous = ids

    def test_real_source_candidates_remain_blocked(self):
        path = ROOT / "data/processed/bench_v02/a2_source_candidates.jsonl"
        if not path.exists():
            self.skipTest("source candidate artifact not present")
        with path.open() as f:
            first = json.loads(next(f))
        self.assertIsNone(first["rt_unit"])
        self.assertFalse(first["eligibility"]["measurement_valid"])
        self.assertLess(first["rt_value"], 0)  # retained coordinate, not deleted


class TestOriginalIrtReconciliation(unittest.TestCase):
    def make_fixture(self, directory, label_mismatch=False):
        original = np.zeros((2, 41, 22), dtype="<f8")
        for i, sequence in enumerate(("BACJ", "DECK")):
            for position, symbol in enumerate(sequence):
                original[i, position, reconcile_rt.VOCABULARY.index(symbol)] = 1
        def text(value):
            data = value.encode()
            return b"X" + struct.pack("<I", len(data)) + data
        def byte_array(value):
            return b"B" + struct.pack("<I", len(value)) + value
        # An executable GLOBAL/REDUCE is deliberately present: the static reader
        # must consume metadata without ever resolving or calling it.
        raw = (b"\x80\x03" + text("training_X") + byte_array(original.tobytes())
               + text("training_y") + byte_array(np.array([-16.3, 42.0], dtype="<f8").tobytes())
               + b"cbuiltins\nprint\n)R.")
        compressor = zlib.compressobj(wbits=-15)
        compressed = compressor.compress(raw) + compressor.flush()
        data = directory / "source.deflate"
        data.write_bytes(compressed)
        legacy = directory / "legacy.csv"
        with legacy.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["seq", "modifications", "tr"])
            writer.writeheader()
            writer.writerows([{"seq": "ACM", "modifications": "2|Oxidation", "tr": -16.3},
                             {"seq": "DECK", "modifications": "", "tr": 43 if label_mismatch else 42}])
        evidence = directory / "evidence.json"
        evidence.write_text(json.dumps({"uncompressed_bytes": len(raw), "crc32": f"{zlib.crc32(raw):08x}"}))
        return data, legacy, evidence

    def test_static_reader_restores_ptms_and_preserves_irt(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            compressed, legacy, evidence = self.make_fixture(directory)
            with mock.patch("builtins.print", side_effect=AssertionError("GLOBAL executed")):
                summary = reconcile_rt.reconcile(compressed, legacy, directory, evidence)
            rows = [json.loads(x) for x in (directory / "a2_source_irt_reconciled.jsonl").read_text().splitlines()]
            self.assertEqual(summary["missing_N_acetyl_restored_rows"], 1)
            self.assertEqual(rows[0]["rt_unit"], "iRT")
            self.assertEqual(rows[0]["rt_value"], -16.3)
            self.assertEqual([(x["position"], x["name"]) for x in rows[0]["modifications"]],
                             [(0, "Acetyl"), (2, "Carbamidomethyl"), (3, "Oxidation")])
            self.assertTrue(rows[0]["eligibility"]["chemical_identity_resolved"])
            self.assertFalse(rows[1]["eligibility"]["chemical_identity_resolved"])
            self.assertFalse(rows[0]["eligibility"]["training_ready"])
            self.assertFalse(summary["min_shift_performed"])

    def test_mismatched_labels_cannot_be_reconciled(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            compressed, legacy, evidence = self.make_fixture(directory, True)
            with self.assertRaisesRegex(ValueError, "do not exactly match"):
                reconcile_rt.reconcile(compressed, legacy, directory, evidence)

    def test_wrong_archive_crc_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            compressed, legacy, evidence = self.make_fixture(directory)
            obj = json.loads(evidence.read_text())
            obj["crc32"] = "00000000"
            evidence.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError, "CRC"):
                reconcile_rt.reconcile(compressed, legacy, directory, evidence)


if __name__ == "__main__":
    unittest.main()
