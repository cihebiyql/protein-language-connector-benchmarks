"""Regression checks for scientific metric semantics and split isolation."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benz", ROOT / "scripts/prepare_benz_baseline.py")
benz = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benz)


class MetricTests(unittest.TestCase):
    def test_multilabel_recall_uses_all_known_positives(self):
        result = benz.metrics(["C", "B", "D", "A"], ["A", "B"])
        self.assertEqual(result["known_positive_recall_at_3"], .5)
        self.assertEqual(result["mrr"], .5)

    def test_mrr_is_first_positive(self):
        self.assertEqual(benz.metrics(["C", "B", "A"], ["B", "A"])["mrr"], .5)
        self.assertEqual(benz.metrics(["C", "D"], ["B"])["mrr"], 0)

    def test_reject_invalid_labels_and_duplicate_rank(self):
        with self.assertRaises(ValueError):
            benz.metrics(["A"], [])
        with self.assertRaises(ValueError):
            benz.metrics(["A", "A"], ["A"])

    def test_sequence_weight_does_not_grow_with_label_count(self):
        order, weights = benz.rank_frequency([{"targets": ["A", "B"]}, {"targets": ["C"]}], ["A", "B", "C"])
        self.assertEqual(sum(weights.values()), 2)
        self.assertEqual(order, ["C", "A", "B"])


class GroupTests(unittest.TestCase):
    def test_similarity_dual_coverage(self):
        self.assertTrue(benz.near_duplicate("ACDEFGHIKLMNPQRS", "ACDEFGHIKLMNAAAA"))
        self.assertFalse(benz.near_duplicate("ACDEFGHIKLMNPQRS", "ACDEFGHI"))
        self.assertFalse(benz.near_duplicate("AAAAAAAAAAAAAAAA", "CCCCCCCCCCCCCCCC"))

    def test_gap_aware_threshold(self):
        # One internal deletion and one terminal extension; 15 matches in 17 columns.
        self.assertTrue(benz.near_duplicate("ACDEFGHIKLMNPQRS", "ACDEFGHKLMNPQRST"))
        self.assertTrue(benz.near_duplicate("ACDEFGHIKLMNPQRS", "CDEFGHIKLMNPQRST"))

    def test_connected_group_and_conservative_holdout(self):
        samples = [dict(sample_id="s1", sequence="ACDEFGHIKLMNPQRS", parents=[dict(accession="P1", gene="G1", start=1, end=16)]),
                   dict(sample_id="s2", sequence="YYYYYYYYYYYYYYYY", parents=[dict(accession="P1", gene="G1", start=200, end=215)]),
                   dict(sample_id="s3", sequence="ACDEFGHIKLMNPQRT", parents=[dict(accession="P2", gene="G2", start=1, end=16)])]
        holdout = [dict(sequences=["WWWWWWWWWWWWWWWW"], gene="G2", row=2)]
        groups, _, _ = benz.group_samples(samples, holdout)
        benz.assign_splits(samples, groups)
        self.assertEqual(len(groups), 1)
        self.assertTrue(all(s["split"] == "excluded_g1" for s in samples))

    def test_ev8_notation(self):
        self.assertEqual(benz.expand_ev8_sequence("H3COC-AA(P/R)TT-NH2"), ["AAPTT", "AARTT"])
        self.assertEqual(benz.expand_ev8_sequence("H3COC-AA(TVV)(ILR)GG-NH2"), ["AATVVGG", "AAILRGG"])


@unittest.skipUnless((ROOT / "data/processed/benz_samples.jsonl").exists(), "Run preparation first")
class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = [json.loads(line) for line in (ROOT / "data/processed/benz_samples.jsonl").read_text(encoding="utf-8").splitlines()]
        cls.manifest = json.loads((ROOT / "data/processed/benz_split_manifest.json").read_text(encoding="utf-8"))

    def test_source_count_and_unique_samples(self):
        self.assertEqual(self.manifest["raw_records"], 3049)
        self.assertEqual(len({s["sequence"] for s in self.samples}), len(self.samples))
        self.assertEqual(sum(len(s["sources"]) for s in self.samples), 3049)

    def test_group_parent_and_exact_sequence_cannot_cross(self):
        seen = {}
        for s in self.samples:
            keys = [("group", s["group_id"]), ("sequence", s["sequence"])]
            for p in s["parents"]:
                keys += [("accession", p["accession"].split("-")[0])]
                if p["gene"]:
                    keys.append(("gene", p["gene"]))
            for key in keys:
                self.assertEqual(seen.setdefault(key, s["split"]), s["split"])

    def test_fixed_manifest_split_reproduces(self):
        samples = [dict(s) for s in self.samples]
        groups = [dict(g) for g in self.manifest["groups"]]
        benz.assign_splits(samples, groups)
        self.assertEqual([s["split"] for s in samples], [s["split"] for s in self.samples])

    def test_no_threshold_similarity_across_splits(self):
        # Direct all-cross-split check; it does not trust the saved components.
        np = benz.np
        counts = np.array([[s["sequence"].count(a) for a in benz.AA] for s in self.samples])
        lengths = np.array([len(s["sequence"]) for s in self.samples])
        split = np.array([s["split"] for s in self.samples])
        for i, sample in enumerate(self.samples):
            upper = np.minimum(counts[i], counts[i + 1:]).sum(axis=1)
            bound = .8 * np.maximum(np.ceil(.9 * lengths[i]), np.ceil(.9 * lengths[i + 1:]))
            candidates = np.nonzero((upper + 1e-9 >= bound) & (split[i + 1:] != split[i]))[0] + i + 1
            for j in candidates:
                self.assertFalse(benz.near_duplicate(sample["sequence"], self.samples[int(j)]["sequence"]),
                                 f"Related sequences cross splits: {sample['sample_id']}, {self.samples[int(j)]['sample_id']}")

    def test_g1_exclusion_and_prediction_sources(self):
        for s in self.samples:
            self.assertEqual(bool(s["excluded_ev8_rows"]), s["split"] == "excluded_g1")
        path = ROOT / "results/b0_predictions.jsonl"
        if not path.exists():
            self.skipTest("B0 still running")
        train_ids = {s["sample_id"] for s in self.samples if s["split"] == "train"}
        predictions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        expected = 2 * sum(s["split"] in {"validation", "test"} for s in self.samples)
        self.assertEqual(len(predictions), expected)
        for p in predictions:
            self.assertEqual(len(p["ranking"]), self.manifest["target_count"])
            self.assertEqual(benz.metrics(p["ranking"], p["targets"])["mrr"], p["mrr"])
            if p["method"] == "sequence_1nn":
                self.assertIn(p["nearest_train_id"], train_ids)


if __name__ == "__main__":
    unittest.main()
