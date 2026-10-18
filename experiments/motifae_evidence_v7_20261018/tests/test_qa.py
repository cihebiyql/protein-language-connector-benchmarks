import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "src/qa_pilot.py"
spec = importlib.util.spec_from_file_location("qa_pilot_under_test", MODULE)
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


def card():
    return qa.evidence_card({"sample_id": "secret", "sequence_length": 15, "true_class": "secret_gold",
                             "associations": [{"class_id": "A", "description": "Candidate A", "feature_id": 10,
                                               "start": 3, "end": 7, "peak": 1., "threshold": .5,
                                               "gold_label": "must not pass"}],
                             "unknown_active_top_features": [{"feature_id": 20, "peak": .5, "position": 8}]})


class ParserTests(unittest.TestCase):
    def parse(self, raw, question="class"):
        return qa.parse_answer(raw, question, 15, {"A", "B"}, card())

    def test_strict_json_no_repair(self):
        for raw in ['```json\n{"class_id":"A"}\n```', 'answer {"class_id":"A"}',
                    '{"class_id":"A","extra":1}', '{"class_id":"A","class_id":"B"}',
                    '[{"class_id":"A"}]', '{"class_id":NaN}', '{"class_id":null}']:
            self.assertIsNone(self.parse(raw)[0], raw)
        self.assertEqual(self.parse('{"class_id":"A"}')[0], {"class_id": "A"})
        self.assertIsNone(self.parse('{"class_id":"not_in_catalog"}')[0])

    def test_positions_require_real_integers_and_full_bounds(self):
        for obj in [{"start": True, "end": 5}, {"start": "3", "end": 5}, {"start": 3., "end": 5},
                    {"start": 0, "end": 5}, {"start": 3, "end": 16}, {"start": 5, "end": 3},
                    {"start": None, "end": 3}]:
            self.assertIsNone(self.parse(json.dumps(obj), "position")[0])
        self.assertEqual(self.parse('{"start":null,"end":null}', "position")[0], {"start": None, "end": None})

    def test_evidence_allowed_identifiers_and_activity(self):
        self.assertEqual(self.parse('{"evidence_id":"E1"}', "evidence")[0], {"evidence_id": "E1"})
        self.assertIsNone(self.parse('{"evidence_id":"E2"}', "evidence")[0])
        self.assertIsNone(qa.parse_answer('{"evidence_id":"E1"}', "evidence", 15, {"A"}, None)[0])
        self.assertIsNone(self.parse('{"activity":"yes"}', "activity")[0])


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.catalog = [{"class_id": "A", "description": "Candidate A", "regex": "SECRET_REGEX"}]

    def test_prompt_does_not_include_identity_gold_or_regex_vocabulary(self):
        messages = qa.make_messages("class", self.catalog, card=card())
        text = json.dumps(messages)
        for forbidden in ["secret_gold", "sample_id", "gold_label", "SECRET_REGEX", '"split"', "parent_accession"]:
            self.assertNotIn(forbidden, text)
        self.assertIn("E1", text)

    def test_sequence_and_card_inputs_are_exclusive(self):
        sequence = "ACDEFGHIKL"
        self.assertIn(sequence, json.dumps(qa.make_messages("class", self.catalog, sequence=sequence)))
        self.assertNotIn(sequence, json.dumps(qa.make_messages("class", self.catalog, card=card())))
        with self.assertRaises(ValueError):
            qa.make_messages("class", self.catalog, sequence=sequence, card=card())

    def test_wrong_card_deranges_and_preserves_donor_coordinates(self):
        mapping = qa.derangement(["c", "b", "a"])
        self.assertEqual(mapping, qa.derangement(["a", "c", "b"]))
        self.assertEqual(set(mapping), set(mapping.values()))
        self.assertTrue(all(key != value for key, value in mapping.items()))
        shortened = qa.evidence_card(card(), 4)
        self.assertEqual(shortened["sequence_length"], 4)
        self.assertEqual(shortened["associations"][0]["end"], 7)
        self.assertIsNone(qa.parse_answer('{"start":3,"end":7}', "position", 4, {"A"}, shortened)[0])

    def test_template_not_language_model_and_empty_card_abstention(self):
        self.assertEqual(qa.template_answer("class", card()), {"class_id": "A"})
        empty = qa.evidence_card({"sequence_length": 15, "associations": []})
        self.assertEqual(qa.template_answer("class", empty), {"class_id": "unknown"})
        self.assertEqual(qa.template_answer("evidence", empty), {"evidence_id": None})
        self.assertEqual(qa.template_answer("activity", empty), {"activity": "not_established"})


class EvaluationTests(unittest.TestCase):
    def test_full_denominator_nonanchor_positive_and_invalid_not_rewarded(self):
        samples = {"a": {"class_id": "A", "short_motif_start": 3, "short_motif_end": 7,
                         "known_positive_spans": [{"class_id": "A", "short_start": 3, "short_end": 7, "fully_contained": True},
                                                  {"class_id": "B", "short_start": 10, "short_end": 12, "fully_contained": True}]},
                   "b": {"class_id": "A", "short_motif_start": 3, "short_motif_end": 7}}
        records = []
        for question, parsed in [("class", {"class_id": "B"}), ("position", {"start": 10, "end": 12}),
                                 ("activity", {"activity": "not_established"}), ("evidence", {"evidence_id": "E1"})]:
            records += [{"sample_id": "a", "arm": "motifae", "question": question, "parsed": parsed},
                        {"sample_id": "b", "arm": "motifae", "question": question, "parsed": None}]
        result, _ = qa.evaluate(records, samples, {"motifae": {"a": card(), "b": card()}})
        arm = result["arms"]["motifae"]
        self.assertEqual(arm["class"]["known_positive_recovered_rate"], .5)
        self.assertEqual(arm["class"]["unsupported_nonunknown_rate"], .5)
        self.assertEqual(arm["position"]["known_positive_overlap_rate"], .5)
        self.assertEqual(arm["position"]["anchor_overlap_rate"], 0)
        self.assertEqual(arm["position"]["mean_best_known_positive_iou_full_denominator"], .5)
        self.assertEqual(arm["activity"]["correct_not_established_rate"], .5)
        self.assertEqual(arm["activity"]["invalid"], 1)
        self.assertTrue(result["template_is_not_qwen"])

    def test_overclaim_rate_is_separate_from_invalid(self):
        samples = {"a": {"class_id": "A", "short_motif_start": 1, "short_motif_end": 2}}
        record = {"sample_id": "a", "arm": "sequence", "question": "activity", "parsed": {"activity": "established"}}
        result, _ = qa.evaluate([record], samples, {})
        self.assertEqual(result["arms"]["sequence"]["activity"]["activity_overclaim_rate"], 1.)


class FrozenContractTests(unittest.TestCase):
    def fixture(self, scope):
        (scope / "src").mkdir(); (scope / "data").mkdir()
        for rel in ["config.json", "src/qa_pilot.py", "data/panel.jsonl", "data/catalog.json"]:
            (scope / rel).write_text("{}", encoding="utf-8")
        (scope / "dictionary.json").write_text("{}", encoding="utf-8")
        (scope / "dictionary.sha256").write_text(qa.sha(scope / "dictionary.json") + "\n", encoding="utf-8")
        for rel in qa.CARD_PATHS.values():
            (scope / rel).write_text("", encoding="utf-8")
        pins = {rel: qa.sha(scope / rel) for rel in ["config.json", "src/qa_pilot.py", "data/panel.jsonl", "data/catalog.json"]}
        (scope / "frozen_manifest.json").write_text(json.dumps({"source_pins": pins}), encoding="utf-8")

    def test_requires_frozen_sources_and_final_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "requires"):
                qa.verify_freeze(scope)
            self.fixture(scope)
            receipt = qa.verify_freeze(scope)
            self.assertTrue(receipt["frozen_before_test_qa"])
            (scope / "dictionary.json").write_text('{"changed":true}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "dictionary"):
                qa.verify_freeze(scope)

    def test_changed_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            scope = Path(directory); self.fixture(scope)
            (scope / "config.json").write_text('{"changed":true}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                qa.verify_freeze(scope)

    def test_synthetic_smoke_never_reads_panel(self):
        class Mock:
            def generate(self, messages):
                text = messages[1]["content"]
                question = next(q for q, prompt in qa.QUESTION_TEXT.items() if prompt in json.loads(text)["question"])
                return json.dumps({"class": {"class_id": "synthetic_example"}, "position": {"start": 3, "end": 6},
                                   "activity": {"activity": "not_established"}, "evidence": {"evidence_id": "E1"}}[question])
        with tempfile.TemporaryDirectory() as directory:
            scope = Path(directory)
            (scope / "config.json").write_text("{}", encoding="utf-8")
            result = qa.run_smoke(scope, Mock())
            self.assertFalse(result["reads_real_test"])
            self.assertFalse(result["score_gate"])
            self.assertEqual(len(result["records"]), 4)
            self.assertTrue(all(r["parsed"] is not None for r in result["records"]))

    def test_full_mock_matrix_and_identical_input_technical_resume(self):
        class Mock:
            def __init__(self):
                self.calls = 0

            def generate(self, messages):
                self.calls += 1
                payload = json.loads(messages[1]["content"])
                self_question = next(q for q, prompt in qa.QUESTION_TEXT.items() if prompt == payload["question"])
                return json.dumps({"class": {"class_id": "unknown"}, "position": {"start": None, "end": None},
                                   "activity": {"activity": "not_established"}, "evidence": {"evidence_id": None}}[self_question])

        with tempfile.TemporaryDirectory() as directory:
            scope = Path(directory); self.fixture(scope)
            config = {"qa": {"arms": list(qa.ARMS), "questions": list(qa.QUESTIONS), "shuffle_seed": 991}}
            (scope / "config.json").write_text(json.dumps(config), encoding="utf-8")
            rows = [{"sample_id": f"private_{i}", "split": "test", "short_sequence": "ACDEFGHIKLMNPQR",
                     "class_id": "A", "short_motif_start": 3, "short_motif_end": 7} for i in range(67)]
            (scope / "data/panel.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            (scope / "data/catalog.json").write_text(json.dumps({"classes": [{"class_id": "A", "description": "Candidate A"}]}), encoding="utf-8")
            for rel in qa.CARD_PATHS.values():
                (scope / rel).write_text("".join(json.dumps({**card(), "sample_id": r["sample_id"]}) + "\n" for r in rows), encoding="utf-8")
            pins = {rel: qa.sha(scope / rel) for rel in ["config.json", "src/qa_pilot.py", "data/panel.jsonl", "data/catalog.json"]}
            (scope / "frozen_manifest.json").write_text(json.dumps({"source_pins": pins}), encoding="utf-8")
            mock = Mock(); result = qa.run_formal(scope, mock)
            self.assertEqual(mock.calls, 1340)
            self.assertEqual(result["observed_records"], 1608)
            self.assertEqual(result["arms"]["sequence"]["class"]["invalid"], 0)
            mock = Mock(); resumed = qa.run_formal(scope, mock, resume=True)
            self.assertEqual(mock.calls, 0)
            self.assertEqual(result, resumed)
            with self.assertRaisesRegex(RuntimeError, "already exist"):
                qa.run_formal(scope, Mock())
            (scope / "data/cards_motifae.jsonl").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "identical"):
                qa.run_formal(scope, Mock(), resume=True)


if __name__ == "__main__":
    unittest.main()
