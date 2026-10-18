"""Scientific split and coordinate guards for the separate MotifAE pilot."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "src" / "pilot_data.py"
spec = importlib.util.spec_from_file_location("pilot_data_under_test", MODULE)
data = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = data
spec.loader.exec_module(data)


def make_row(name, parent=None, homology=None, context=None, pmids=None, split="train"):
    return {"sample_id": name, "split": split, "parent_accession": parent or name,
            "groups": {"parent_homology": homology or name, "peptide_family": context or name,
                       "source_publication": pmids or ["pmid_" + name]},
            "start": 20, "end": 23, "segment_sequence": "ACDE", "class_id": "motif_a",
            "source_refs": ["https://example.org/evidence"], "elm_instance_id": name}


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.sequence = "A" * 19 + "ACDE" + "C" * 70
        self.catalog = {"motif_a": {"regex": "ACDE", "description": "Source-backed candidate"}}

    def build(self, rows, parents=None):
        return data.build_panel(rows, parents or {r["parent_accession"]: self.sequence for r in rows}, self.catalog)

    def test_transitive_parent_context_pmid_groups_never_cross_splits(self):
        rows = [make_row("a", homology="h1", pmids=["p1"]),
                make_row("b", homology="h1", pmids=["p2"]),
                make_row("c", context="c3", pmids=["p2", "p3"]),
                make_row("d", context="c3", pmids=["p4"])]
        rows += [make_row("independent_" + str(i)) for i in range(60)]
        panel, _ = self.build(rows)
        connected = [r for r in panel if r["sample_id"] in {"a", "b", "c", "d"}]
        self.assertEqual(len({r["component_id"] for r in connected}), 1)
        self.assertEqual(connected[0]["component_member_ids"], ["a", "b", "c", "d"])
        for field in ["parent_accession", "parent_homology_group", "peptide_family_group"]:
            observed = {}
            for row in panel:
                self.assertEqual(observed.setdefault(row[field], row["split"]), row["split"])
        pmids = {}
        for row in panel:
            for pmid in row["source_pmids"]:
                self.assertEqual(pmids.setdefault(pmid, row["split"]), row["split"])

    def test_coordinates_and_variable_flanks(self):
        panel, _ = self.build([make_row("id_" + str(i)) for i in range(50)])
        self.assertGreater(len({r["short_motif_start"] for r in panel}), 1)
        self.assertGreater(len({(r["requested_left_flank"], r["requested_right_flank"]) for r in panel}), 10)
        for row in panel:
            self.assertEqual(row["parent_sequence"][row["parent_motif_start"] - 1:row["parent_motif_end"]], "ACDE")
            self.assertEqual(row["short_sequence"][row["short_motif_start"] - 1:row["short_motif_end"]], "ACDE")
            self.assertEqual(row["split"], data.component_identity(row["component_member_ids"])[1])
            for side in ("left", "right"):
                self.assertIn(row["requested_" + side + "_flank"], range(4, 13))

    def test_hash_stable_and_order_independent(self):
        self.assertEqual(data.component_identity(["b", "a"]), data.component_identity(["a", "b"]))
        rows = [make_row("item_" + str(i)) for i in range(30)]
        self.assertEqual(self.build(rows), self.build(list(reversed(rows))))
        self.assertEqual(data.flank_size("example", "left"), data.flank_size("example", "left"))

    def test_forbidden_old_test_and_related_train_component(self):
        rows = [make_row("old_test", parent="protected", split="test", pmids=["shared"]),
                make_row("old_val", split="validation"),
                make_row("blocked", pmids=["shared"]),
                make_row("transitive", context="blocked"), make_row("safe")]
        panel, summary = self.build(rows)
        self.assertEqual([r["sample_id"] for r in panel], ["safe"])
        self.assertEqual(summary["excluded_counts"], {"component_intersects_old_validation_or_test": 2})
        self.assertTrue(all(r["original_split"] == "train" for r in panel))

    def test_length_alphabet_declared_filter_and_transitivity_before_filter(self):
        rows = [make_row("long", pmids=["shared"]), make_row("peer", pmids=["shared"]), make_row("bad")]
        panel, summary = self.build(rows, {"long": self.sequence + "A" * 1100,
                                          "peer": self.sequence, "bad": self.sequence + "X"})
        self.assertEqual(len(panel), 1)
        self.assertEqual(panel[0]["component_member_ids"], ["long", "peer"])
        self.assertEqual(summary["excluded_counts"], {"parent_length_exceeds_1022": 1,
                                                     "parent_not_standard_20_amino_acids": 1})
        self.assertEqual(len(panel[0]["parent_sequence"]), len(self.sequence))

    def test_coordinate_mismatch_fails_instead_of_filtering(self):
        row = make_row("bad_coord")
        row["start"] = 21
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.build([row])

    def test_boundary_clip_preserves_absolute_and_local_coordinates(self):
        row = make_row("terminus")
        row.update(start=1, end=4)
        panel, _ = self.build([row], {"terminus": "ACDE" + "A" * 20})
        self.assertEqual(panel[0]["actual_left_flank"], 0)
        self.assertEqual(panel[0]["short_motif_start"], 1)

    def test_other_documented_spans_are_evaluation_only_and_clipped(self):
        anchor = make_row("anchor", parent="same_parent")
        nearby = make_row("nearby", parent="same_parent")
        nearby.update(start=22, end=25, segment_sequence="DECC")
        partly = make_row("partial", parent="same_parent")
        partly.update(start=30, end=50, segment_sequence="C" * 21)
        panel, _ = self.build([anchor, nearby, partly])
        row = next(r for r in panel if r["sample_id"] == "anchor")
        spans = {r["sample_id"]: r for r in row["known_positive_spans"]}
        self.assertIn("nearby", spans)
        self.assertTrue(spans["nearby"]["fully_contained"])
        if "partial" in spans:
            self.assertTrue(spans["partial"]["clipped_right"])
            self.assertFalse(spans["partial"]["fully_contained"])
        self.assertEqual(row["annotation_use"], "evaluation_only_except_new_train_dictionary_fitting")

    def test_real_panel_if_prepared(self):
        scope = MODULE.parents[1]
        manifest_file = scope / "data/manifest.json"
        if not manifest_file.exists():
            self.skipTest("Real data not prepared")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        rows = data._jsonl(scope / "data/panel.jsonl")
        self.assertEqual(manifest["summary"]["original_train_rows"], 378)
        self.assertTrue(all(r["original_split"] == "train" for r in rows))
        self.assertTrue(all(0 < len(r["parent_sequence"]) <= 1022 for r in rows))
        self.assertTrue(all(set(r["parent_sequence"]) <= data.STANDARD_AA for r in rows))
        self.assertEqual(data._sha((scope / "data/panel.jsonl").read_bytes()),
                         manifest["outputs"]["data/panel.jsonl"]["sha256"])
        project_root = scope.parents[1]
        old_rows = data._jsonl(project_root / data.LOCKED_DIR / "b1_samples.jsonl")
        original_train = {r["sample_id"] for r in old_rows if r["split"] == "train"}
        protected_parents = {r["parent_accession"] for r in old_rows if r["split"] != "train"}
        self.assertTrue({r["sample_id"] for r in rows} <= original_train)
        self.assertFalse({r["parent_accession"] for r in rows} & protected_parents)
        for rel, pin in manifest["source_pins"].items():
            self.assertNotIn("\\", rel)
            self.assertEqual(data._sha((project_root / rel).read_bytes()), pin["sha256"])
        for field in ["parent_accession", "parent_homology_group", "peptide_family_group", "component_id"]:
            observed = {}
            for row in rows:
                self.assertEqual(observed.setdefault(row[field], row["split"]), row["split"])
        pmids = {}
        for row in rows:
            self.assertEqual(row["parent_sequence"][row["parent_motif_start"] - 1:row["parent_motif_end"]], row["motif_sequence"])
            self.assertEqual(row["short_sequence"][row["short_motif_start"] - 1:row["short_motif_end"]], row["motif_sequence"])
            for pmid in row["source_pmids"]:
                self.assertEqual(pmids.setdefault(pmid, row["split"]), row["split"])


if __name__ == "__main__":
    unittest.main()
