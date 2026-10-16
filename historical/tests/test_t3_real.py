import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_t3_real import affinity_direction, parse_kd, published_sequences, published_mutation, supported_pair
from prepare_benz_baseline import near_duplicate


def read_rows(name):
    return [json.loads(line) for line in (ROOT / "data/processed" / name).read_text(encoding="utf-8").splitlines()]


class T3EvidenceTests(unittest.TestCase):
    def test_censoring_not_imputed(self):
        self.assertEqual(affinity_direction(parse_kd("<0.00008"), parse_kd("860 ± 20")), ("A", "direction_resolvable"))
        self.assertEqual(affinity_direction(parse_kd("n.b."), parse_kd("39 ± 4")), (None, "no_numeric_bound"))
        self.assertEqual(affinity_direction(parse_kd("3.1 ± 0.2"), parse_kd("4.7 ± 0.3")), (None, "small_change"))
        self.assertEqual(affinity_direction(parse_kd("<30"), parse_kd("39 ± 4")), (None, "ambiguous_bounds"))

    def test_published_multimutant_preserves_all_substitutions(self):
        self.assertEqual(published_sequences("292- EDEVYE(T/I)(V/L)(V/R)DTSEEDS -307"),
                         ("EDEVYETVVDTSEEDS", "EDEVYEILRDTSEEDS"))

    def test_mutation_suffix_never_matches_inside_gene(self):
        examples = [("BCL11A C48F", "BCL11A", "C48F"),
                    ("KMT2A L2854Q", "KMT2A", "L2854Q"),
                    ("KMT2D P2301S", "KMT2D", "P2301S"),
                    ("BUB1S492F", "BUB1", "S492F"),
                    ("SCNN1B P616R", "SCNN1B", "P616R"),
                    ("SCNN1B P618R", "SCNN1B", "P618R"),
                    ("CHK2 Q51K", "CHEK2", "Q51K"),
                    ("BRCA2 T298I/V299L/V300R", "BRCA2", "T298I/V299L/V300R")]
        for text, gene, expected in examples:
            with self.subTest(text=text):
                self.assertEqual(published_mutation(text, gene), expected)
        for invalid in ("BCL11A", "BCL11A C48F trailing", "other C48F"):
            with self.assertRaises(ValueError):
                published_mutation(invalid, "BCL11A")

    def test_individual_counts_override_group_label(self):
        group = {"bait": "target", "mutation": "P1.A1G", "mutation_effect_on_PPI": "enhanced"}
        wt = {"protein_accession": "P1", "library_peptide": "AAAAAAAAAAAAAAAA"}
        mut = {"protein_accession": "P1", "library_peptide": "GAAAAAAAAAAAAAAA"}
        a = {"source_row": 2, "p_value": .001, "mean_counts_%": 20, "wt_or_mut": "wt", "mutation": None}
        b = {"source_row": 3, "p_value": .001, "mean_counts_%": 1, "wt_or_mut": "mut", "mutation": "P1.A1G"}
        hits = {("target", "P1", wt["library_peptide"]): a, ("target", "P1", mut["library_peptide"]): b}
        self.assertEqual(supported_pair(group, wt, mut, hits)[0][0], "A")
        b["p_value"] = .02
        self.assertIsNone(supported_pair(group, wt, mut, hits)[0])

    def test_missing_endpoint_is_not_automatically_zero(self):
        group = {"bait": "target", "mutation": "P1.A1G", "mutants_observed": 1, "mutants_proportion": "0,0.001"}
        wt = {"protein_accession": "P1", "library_peptide": "AAAAAAAAAAAAAAAA"}
        mut = {"protein_accession": "P1", "library_peptide": "GAAAAAAAAAAAAAAA"}
        hits = {("target", "P1", wt["library_peptide"]): {"source_row": 2, "p_value": .001, "mean_counts_%": 20, "wt_or_mut": "wt", "mutation": None}}
        self.assertIsNone(supported_pair(group, wt, mut, hits)[0])
        group.update(mutants_observed=0, mutants_proportion="0,0")
        self.assertEqual(supported_pair(group, wt, mut, hits)[0][0], "A")


class T3ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d2 = read_rows("t3_d2_pairs.jsonl")
        cls.g1 = read_rows("t3_g1_records.jsonl")

    def test_all_candidates_accounted_for(self):
        rows = read_rows("t3_d2_candidate_audit.jsonl")
        self.assertEqual(len(rows), 201)
        self.assertEqual(sum(r["status"] == "included" for r in rows), len(self.d2))
        self.assertTrue(all(r["relationship_checks"] and all(x["reason"] for x in r["relationship_checks"]) for r in rows))

    def test_g1_full_table_and_identity_conflict_retained(self):
        self.assertEqual(len(self.g1), 29)
        self.assertEqual(sum(r["label"] is not None for r in self.g1), 17)
        krt = next(r for r in self.g1 if r["gene"] == "KRT1")
        self.assertEqual(krt["direction_status"], "identity_conflict")
        self.assertIsNone(krt["label"])
        cdc45 = next(r for r in self.g1 if r["gene"] == "CDC45" and "KPNA7" in r["target_info"])
        self.assertEqual(cdc45["kd_a"]["value"], 6.6)
        self.assertEqual(cdc45["label"], "A")
        self.assertTrue(all(r["clean_external"] for r in self.g1 if r["label"]))
        self.assertEqual(len({r["series_id"] for r in self.g1 if r["label"]}), 13)
        for row in self.g1:
            self.assertEqual(row["series_id"], row["gene"] + ":" + published_mutation(row["gene_mutation_as_published"], row["gene"]))

    def test_parent_and_near_sequences_do_not_cross_d2_split(self):
        g1_genes = {r["gene"] for r in self.g1}
        for i, row in enumerate(self.d2):
            self.assertNotIn(row["gene"], g1_genes)
            self.assertEqual(len(row["sequence_a"]), 16)
            self.assertEqual(len(row["sequence_b"]), 16)
            self.assertFalse(row["source_evidence"]["single_pair_significance_verified"])
            self.assertNotIn("pair_p_value", row["source_evidence"])
            self.assertGreater(row["source_evidence"]["ev1_wt_input_library_coverage"], 0)
            self.assertGreater(row["source_evidence"]["ev1_mut_input_library_coverage"], 0)
            for other in self.d2[:i]:
                if row["split"] != other["split"]:
                    self.assertNotEqual(row["parent_accession"], other["parent_accession"])
                    for a in (row["sequence_a"], row["sequence_b"]):
                        for b in (other["sequence_a"], other["sequence_b"]):
                            self.assertFalse(near_duplicate(a, b))
        for split in ("train", "val", "test"):
            self.assertEqual({r["label"] for r in self.d2 if r["split"] == split}, {"A", "B"})

    def test_benz_fixed_split_preserved(self):
        manifest = json.loads((ROOT / "data/processed/t3_split_manifest.json").read_text(encoding="utf-8"))
        path = ROOT / "data/processed/benz_split_manifest.json"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest["benz_split_manifest_sha256"])
        self.assertEqual(manifest["d2_group_splits"], {"train": 26, "val": 6, "test": 6})

    def test_qc_removes_zero_coverage_without_changing_split(self):
        old = read_rows("t3_d2_pairs_precoverage65.jsonl")
        old_by_id = {r["pair_id"]: r for r in old}
        excluded = read_rows("t3_d2_input_coverage_exclusions.jsonl")
        self.assertEqual(len(old), 65)
        self.assertEqual(len(self.d2), 59)
        self.assertEqual(len(excluded), 6)
        for row in self.d2 + excluded:
            self.assertEqual(row["split"], old_by_id[row["pair_id"]]["split"])
            self.assertEqual(row["group_id"], old_by_id[row["pair_id"]]["group_id"])
        self.assertEqual({r["pair_id"] for r in old}, {r["pair_id"] for r in self.d2 + excluded})

    def test_g1_source_sensitivity_and_target_identity(self):
        rows = read_rows("t3_g1_source_consistent_direction_pairs.jsonl")
        self.assertEqual(len(rows), 14)
        self.assertTrue(all(r["sequence_source_consistent"] for r in rows))
        self.assertFalse(any(r["gene"] in ("ABRAXAS1", "BCL11A") for r in rows))
        kpna4 = [r for r in self.g1 if "KPNA4" in r["target_id"]]
        self.assertEqual({r["measurement_pocket"] for r in kpna4}, {"major", "minor"})
        self.assertEqual(len({r["target_id"] for r in kpna4}), 2)
        keap = next(r for r in self.g1 if r["target_info"] == "KEAP1 KELCH")
        self.assertEqual(keap["target_id"], "Keap1_KELCH__Q14145_321_609")

    def test_series_metadata_fix_preserves_model_inputs_and_labels(self):
        old = {r["pair_id"]: r for r in read_rows("t3_g1_records_preseriesfix_v2.jsonl")}
        for row in self.g1:
            actual = {k: v for k, v in row.items() if k != "series_id"}
            expected = {k: v for k, v in old[row["pair_id"]].items() if k != "series_id"}
            self.assertEqual(actual, expected)
        manifest = json.loads((ROOT / "data/processed/t3_split_manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["generated_data_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / "data/processed" / name).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
