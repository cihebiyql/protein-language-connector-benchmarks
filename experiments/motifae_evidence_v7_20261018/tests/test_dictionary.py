"""Independent localization math, frozen naming, coordinates and leakage tests."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score


MODULE = Path(__file__).resolve().parents[1] / "src/motif_dictionary.py"
spec = importlib.util.spec_from_file_location("dictionary_under_test", MODULE)
d = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = d
spec.loader.exec_module(d)


def sample(name, split, class_id="class_a", start=2, end=3):
    return {"sample_id": name, "split": split, "class_id": class_id,
            "parent_accession": name, "parent_sequence": name + "ACDEFG",
            "short_sequence": "ACDEFG", "parent_motif_start": start,
            "parent_motif_end": end, "short_motif_start": start,
            "short_motif_end": end, "short_parent_start": 1, "short_parent_end": 6,
            "known_positive_spans": [], "known_positive_class_ids": [class_id]}


class FakeStore:
    """Small raw dimensions keep unit tests cheap; production validates1280."""

    def __init__(self, rows, feature=7):
        self.read_receipts = {}
        self.data = {}
        for row in rows:
            for kind, sequence in (("full", row["parent_sequence"]), ("short", row["short_sequence"])):
                values = np.zeros((len(sequence), 40960), dtype=np.float32)
                values[1:3, feature] = 10
                values[:, 40959] = np.arange(len(sequence)) / 100
                h33 = np.zeros((len(sequence), 4), dtype=np.float32)
                h33[1:3, 0] = 10
                h33[:, 1] = np.arange(len(sequence))
                self.data[kind, sequence] = (h33, sparse.csr_matrix(values))

    def load(self, kind, sequence):
        key = kind + "/" + d.sha256_bytes(sequence.encode())
        self.read_receipts[key] = {"sha256": "synthetic", "bytes": 0}
        return self.data[kind, sequence]


def fit_fixture():
    rows = [sample("a", "train"), sample("b", "train"),
            sample("c", "qualification"), sample("d", "qualification"), sample("e", "test")]
    store = FakeStore(rows)
    catalog = [{"class_id": "class_a", "description": "Source association", "regex": "CD"},
               {"class_id": "absent", "description": "Unsupported candidate", "regex": "Z"}]
    dictionary = d.fit_dictionary(rows, catalog, store)
    return rows, catalog, store, dictionary


class SparseAucTests(unittest.TestCase):
    def test_every_sparse_column_equals_independent_sklearn_with_zero_and_value_ties(self):
        rng = np.random.default_rng(124)
        values = rng.integers(-2, 4, size=(31, 49)).astype(np.float32)
        values[rng.random(values.shape) < 0.82] = 0
        values[:, 0] = 0
        values[:, 1] = 2
        values[:, 2] = -1
        labels = rng.random(31) > 0.7
        actual = d.sparse_column_auc(sparse.csr_matrix(values), labels)
        expected = [roc_auc_score(labels, values[:, i]) for i in range(values.shape[1])]
        np.testing.assert_allclose(actual, expected, atol=1e-15)

    def test_all_zero_features_are_half_and_undefined_mask_is_nan(self):
        matrix = sparse.csr_matrix((4, 40960), dtype=np.float32)
        np.testing.assert_array_equal(d.sparse_column_auc(matrix, [0, 1, 0, 1]), 0.5)
        self.assertTrue(np.isnan(d.sparse_column_auc(matrix, [1, 1, 1, 1])).all())

    def test_duplicate_sparse_entries_are_canonicalized(self):
        matrix = sparse.coo_matrix(([1, 2, 3], ([0, 0, 1], [0, 0, 1])), shape=(3, 2))
        expected = [roc_auc_score([1, 0, 0], matrix.toarray()[:, i]) for i in range(2)]
        np.testing.assert_allclose(d.sparse_column_auc(matrix, [1, 0, 0]), expected)


class DictionaryTests(unittest.TestCase):
    def test_only_train_selects_feature_and_qualification_names_it(self):
        rows, catalog, store, dictionary = fit_fixture()
        entry = next(e for e in dictionary["classes"] if e["class_id"] == "class_a")
        self.assertEqual(entry["feature_idx"], 7)
        self.assertEqual(entry["train_auc"], 1.0)
        self.assertEqual(entry["qualification_auc"], 1.0)
        self.assertTrue(entry["qualified"])
        self.assertEqual(entry["threshold"], 0)
        self.assertEqual(entry["association_scope"], "motifassociation_notverifiedactivity")
        self.assertEqual(entry["activity_claim"], "not established")
        self.assertFalse(dictionary["test_used_for_fitting_or_naming"])
        self.assertFalse(next(e for e in dictionary["classes"] if e["class_id"] == "absent")["qualified"])

    def test_changing_test_labels_cannot_change_dictionary_or_inference_cards(self):
        rows, catalog, store, dictionary = fit_fixture()
        test = rows[-1]
        h33, features = store.load("short", test["short_sequence"])
        card = d.build_card(test["sample_id"], test["short_sequence"], h33, features, dictionary)
        changed = copy.deepcopy(rows)
        changed[-1].update(class_id="absent", parent_motif_start=5, parent_motif_end=6,
                           short_motif_start=5, short_motif_end=6,
                           known_positive_spans=[{"class_id": "absent", "parent_start": 5,
                                                  "parent_end": 6, "short_start": 5, "short_end": 6}])
        # Separate store means source receipts include TRAIN/qualification only.
        store2 = FakeStore(rows)
        dictionary2 = d.fit_dictionary(changed, catalog, store2)
        self.assertEqual(dictionary, dictionary2)
        card2 = d.build_card(test["sample_id"], test["short_sequence"], h33, features, dictionary2)
        self.assertEqual(card, card2)
        self.assertNotIn("anchor_class_id", card)
        self.assertNotIn("parent_motif_start", card)
        self.assertNotIn("known_positive_spans", card)

    def test_qualification_cannot_choose_better_feature_or_relax_naming(self):
        rows, catalog, store, dictionary = fit_fixture()
        for row in rows:
            if row["split"] == "qualification":
                h33, matrix = store.data["full", row["parent_sequence"]]
                values = matrix.toarray()
                values[:, 7] = 0
                values[0, 7] = 10
                values[1:3, 8] = 100
                store.data["full", row["parent_sequence"]] = h33, sparse.csr_matrix(values)
        entry = next(e for e in d.fit_dictionary(rows, catalog, store)["classes"]
                     if e["class_id"] == "class_a")
        self.assertEqual(entry["feature_idx"], 7)
        self.assertFalse(entry["qualified"])
        self.assertLess(entry["qualification_auc"], 0.70)

    def test_same_parent_sites_are_unioned_without_duplicate_parent_weight(self):
        row = sample("p", "train", start=2, end=3)
        duplicate = dict(row, sample_id="site2", parent_motif_start=5, parent_motif_end=6)
        parents = d.class_parents([row, duplicate], "class_a")
        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0]["spans"], [(2, 3), (5, 6)])

    def test_unknown_features_retained_when_no_names_qualify(self):
        rows, _, store, dictionary = fit_fixture()
        for entry in dictionary["classes"]:
            entry["qualified"] = False
        h33, matrix = store.load("short", "ACDEFG")
        card = d.build_card("anonymous", "ACDEFG", h33, matrix, dictionary)
        self.assertEqual(card["associations"], [])
        self.assertEqual(card["unknown_active_top_features"][0]["feature_id"], 7)
        self.assertEqual(card["unknown_active_top_features"][0]["position"], 2)
        self.assertNotIn("description", card["unknown_active_top_features"][0])

    def test_cards_use_fixed_training_threshold_and_one_based_contiguous_region(self):
        rows, _, store, dictionary = fit_fixture()
        h33, matrix = store.load("short", "ACDEFG")
        card = d.build_card("anonymous", "ACDEFG", h33, matrix, dictionary)
        association = card["associations"][0]
        self.assertEqual((association["start"], association["end"], association["peak_position"]), (2, 3, 2))
        self.assertEqual(association["threshold"], 0)
        self.assertEqual(association["activity_claim"], "not established")
        self.assertIsNone(d.peak_region(np.zeros(4), 0))

    def test_raw_h33_control_exposes_no_motifae_feature_ids_or_activations(self):
        rows, _, store, dictionary = fit_fixture()
        h33, matrix = store.load("short", "ACDEFG")
        raw_card = d.build_card("anonymous", "ACDEFG", h33, matrix, dictionary, "raw_probe")
        self.assertEqual(raw_card["unknown_active_top_features"], [])
        for association in raw_card["associations"]:
            self.assertTrue(association["feature_id"].startswith("raw_h33_ridge:"))

    def test_frozen_write_refuses_score_driven_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dictionary.json"
            d.frozen_write(path, "original")
            d.frozen_write(path, "original")
            with self.assertRaises(RuntimeError):
                d.frozen_write(path, "replacement")


class RegexTests(unittest.TestCase):
    def test_anchors_are_respected_and_spans_one_based(self):
        self.assertEqual(d.regex_spans("ACDEFG", "^AC"), [(1, 2)])
        self.assertEqual(d.regex_spans("ACDEFG", "FG$"), [(5, 6)])
        self.assertEqual(d.regex_spans("XACDEFG", "^AC"), [])
        self.assertEqual(d.regex_spans("ACDEFGX", "FG$"), [])
        self.assertEqual(d.regex_spans("ACDEFG", "(?=C)"), [])

    def test_regex_evidence_does_not_claim_verified_activity(self):
        _, _, _, dictionary = fit_fixture()
        card = d.build_regex_card("x", "ACDEFG", dictionary)
        self.assertEqual(card["associations"][0]["start"], 2)
        self.assertEqual(card["associations"][0]["association_evidence"],
                         "unqualified catalog regex pattern match")
        self.assertEqual(card["associations"][0]["activity_claim"], "not established")
        json.dumps(card, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
