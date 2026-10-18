import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import qa_train as train


def row(ident, split="train", value=1.0):
    return {"id": ident, "group_id": "group_" + ident, "split": split,
            "sequence": "ACDEFGHIK", "experiment": "cohort",
            "rt": value, "ccs": value + 10}


class ProtocolTests(unittest.TestCase):
    def test_prompt_copies_only_allowed_context(self):
        item = row("SECRET_IDENT", value=9876.54321)
        item["source"] = "SECRET_SOURCE"
        prompt = train.make_prompt(item, "both", {"rt": 2.5, "ccs": 20.5}, "M3")
        for forbidden in (item["id"], item["group_id"], item["sequence"], "SECRET_SOURCE", "9876.54321"):
            self.assertNotIn(forbidden, prompt)
        for expected in ("Charge: 2", "2.5 minutes", "20.5 square angstroms", "Question key: both"):
            self.assertIn(expected, prompt)

    def test_text_has_predicted_attributes_not_gold(self):
        item = row("x", value=9999)
        prompt = train.make_prompt(item, "rt", {"rt": 2, "ccs": 12}, "TEXT", [1.25, 16.75])
        self.assertIn("1.25 minutes", prompt)
        self.assertIn("16.75 square angstroms", prompt)
        self.assertIn("Sequence length: 9 residues", prompt)
        self.assertIn("A:0.111", prompt)
        self.assertNotIn("9999", prompt)
        self.assertNotIn(item["sequence"], prompt)

    def test_removed_text_has_no_attributes_or_length(self):
        prompt = train.make_prompt(row("x"), "rt", {"rt": 2, "ccs": 12}, "TEXT")
        self.assertIn("estimates unavailable", prompt)
        self.assertNotIn("Sequence length", prompt)

    def test_templates_held_out_and_seed_order_shared(self):
        rows = [row(str(i)) for i in range(40)]
        a = train.build_examples(rows, 17, 0)
        b = train.build_examples(rows, 17, 0)
        self.assertEqual([(x["id"], q) for x, q in a], [(x["id"], q) for x, q in b])
        self.assertEqual(len(a), len(rows) * 3)
        self.assertNotEqual([(x["id"], q) for x, q in a],
                            [(x["id"], q) for x, q in train.build_examples(rows, 42, 0)])
        self.assertEqual({train.template_index(str(i), "rt", 0) for i in range(40)}, {0, 1})
        held_out = train.make_prompt(rows[0], "rt", {"rt": 2, "ccs": 12}, "M0")
        self.assertIn("measured retention time above", held_out)
        for epoch in range(6):
            self.assertNotIn("measured retention time above", train.make_prompt(
                rows[0], "rt", {"rt": 2, "ccs": 12}, "M0", epoch=epoch))

    def test_contract_medians_and_group_overlap(self):
        rows = [row("a", value=1), row("b", value=3)]
        config = {"thresholds": {"rt": 2, "ccs": 12}}
        train.validate_contract(config, rows, [row("c", "validation")])
        bad = {"thresholds": {"rt": 100, "ccs": 12}}
        with self.assertRaisesRegex(ValueError, "TRAIN median"):
            train.validate_contract(bad, rows, [])
        item = row("c", "validation")
        item["group_id"] = rows[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "groups overlap"):
            train.validate_contract(config, rows, [item])


class AssetsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.scope = Path(self.tmp.name)
        (self.scope / "cache").mkdir()
        (self.scope / "data").mkdir()
        ids = np.array(["a", "b", "c", "d"])
        values = np.stack([np.full((8, 640), x, np.float32) for x in (1, 3, 1000, 2000)])
        np.savez(self.scope / "cache/features.npz", ids=ids, p15=values, p30=values * 2)
        np.savez(self.scope / "data/attributes.npz", ids=ids,
                 predictions=np.ones((4, 2)), split=np.array(["train", "train", "test", "test"]))
        train.atomic_json(self.scope / "data/control_mapping.json",
                          {"train": {"a": "b", "b": "a"}, "test": {"c": "d", "d": "c"}})

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_mean_excludes_test_and_mapping_is_frozen(self):
        assets = train.Assets(self.scope, {}, [row("a"), row("b")])
        self.assertTrue(np.all(assets.mean["p15"] == 2))
        self.assertTrue(np.all(assets.mean["p30"] == 4))
        self.assertEqual(assets.donors([row("c", "test"), row("d", "test")]),
                         {"c": "d", "d": "c"})

    def test_control_self_swap_and_cross_split_rejected(self):
        assets = train.Assets(self.scope, {}, [row("a"), row("b")])
        assets.control_mapping["test"]["c"] = "c"
        with self.assertRaisesRegex(ValueError, "derange"):
            assets.donors([row("c", "test"), row("d", "test")])
        assets.control_mapping["test"]["c"] = "a"
        with self.assertRaisesRegex(ValueError, "within split"):
            assets.donors([row("c", "test"), row("d", "test")])

    def test_identity_order_is_required(self):
        np.savez(self.scope / "data/attributes.npz", ids=np.array(["b", "a", "c", "d"]),
                 predictions=np.ones((4, 2)), split=np.array(["train", "train", "test", "test"]))
        with self.assertRaisesRegex(ValueError, "identity order"):
            train.Assets(self.scope, {}, [row("a"), row("b")])


class FrozenManifestTests(unittest.TestCase):
    def test_rejects_unpinned_and_changed_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            scope = Path(temporary)
            config_path = scope / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            data = scope / "dummy"
            data.write_text("original", encoding="utf-8")
            train.atomic_json(scope / "frozen_manifest.json", {"files": [{"path": "dummy", "sha256": train.sha256(data)}]})
            with self.assertRaisesRegex(ValueError, "not frozen"):
                train.verify_frozen(scope, config_path, {})
            data.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Frozen file changed"):
                train.verify_frozen(scope, config_path, {})


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch not available in this Python")
class FakeLMTests(unittest.TestCase):
    def test_greedy_has_no_candidate_constraint_and_stops_at_eos(self):
        import torch

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def apply_chat_template(self, messages, **kwargs):
                return [3] * (1 if messages[0]["content"] == "short" else 3)

            def decode(self, tokens, **kwargs):
                return "".join({1: "yes", 4: "maybe"}[token] for token in tokens)

        calls = []

        class FakeLM:
            def __call__(self, ids, mask, ends, memory, off=False):
                calls.append((ids.clone(), mask.clone(), ends.clone(), off))
                logits = torch.zeros((*ids.shape, 5))
                for i in range(ids.shape[0]):
                    logits[i, -1, 2 if ids[i, -1] in (1, 4) else (1 if i == 0 else 4)] = 10
                return {"logits": logits}

        answers = train.greedy_generate(FakeLM(), Tokenizer(), ["short", "long"], {}, "cpu", torch, 4)
        self.assertEqual(answers, ["yes", "maybe"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1].tolist(), [[0, 0, 1], [1, 1, 1]])
        self.assertEqual(calls[0][2].tolist(), [2, 2])
        self.assertEqual(calls[1][2].tolist(), [2, 2])

    def test_text_and_mean_receive_train_mean_bank(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cache").mkdir()
            (root / "data").mkdir()
            ids = np.array(["a", "b", "c", "d"])
            bank = np.stack([np.full((8, 640), x, np.float32) for x in (1, 3, 1000, 2000)])
            np.savez(root / "cache/features.npz", ids=ids, p15=bank, p30=bank)
            np.savez(root / "data/attributes.npz", ids=ids, predictions=np.ones((4, 2)),
                     split=np.array(["train", "train", "test", "test"]))
            train.atomic_json(root / "data/control_mapping.json", {"test": {"c": "d", "d": "c"}})
            assets = train.Assets(root, {}, [row("a"), row("b")])
            for arm, mode in (("TEXT", "normal"), ("M3", "mean")):
                memory = assets.memory(["c"], mode, arm, torch, "cpu")
                self.assertTrue(torch.all(memory["p15"] == 2))
            self.assertTrue(torch.all(assets.memory(["c"], "normal", "M3", torch, "cpu")["p15"] == 1000))

    def test_formal_six_epochs_receipt_before_test_and_checkpoint_only_adapter(self):
        import torch

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def apply_chat_template(self, messages, **kwargs):
                return [3, 3]

            def encode(self, answer, **kwargs):
                return [1 if answer == "yes" else 4]

            def decode(self, tokens, **kwargs):
                return "".join({1: "yes", 3: "unknown", 4: "no", 0: "pad"}.get(x, "?") for x in tokens)

        class FakeLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(5))
                self.base_lm = torch.nn.Linear(1, 1).requires_grad_(False)

            def forward(self, ids, mask, ends, memory, off=False):
                return {"logits": self.bias.view(1, 1, -1).expand(*ids.shape, 5),
                        "gate": self.bias[0].tanh(), "routes": None}

            def capacity_report(self):
                return {"trainable_parameters": self.bias.numel()}

            def trainable_state_dict(self):
                return {"bias": self.bias.detach().cpu().clone()}

            def load_trainable_state_dict(self, state):
                self.bias.data.copy_(state["bias"])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cache").mkdir()
            (root / "data").mkdir()
            all_rows = []
            mappings = {}
            for split in ("train", "validation", "test"):
                rows = [row(f"{split}_{i}", split, i) for i in range(4)]
                train.write_jsonl(root / "data" / (split + ".jsonl"), rows)
                all_rows += rows
                mappings[split] = {rows[i]["id"]: rows[(i + 1) % 4]["id"] for i in range(4)}
            train.atomic_json(root / "data/control_mapping.json", mappings)
            bank = np.ones((12, 8, 640), dtype=np.float32)
            np.savez(root / "cache/features.npz", ids=np.array([x["id"] for x in all_rows]), p15=bank, p30=bank)
            np.savez(root / "data/attributes.npz", ids=np.array([x["id"] for x in all_rows]),
                     predictions=np.ones((12, 2)), split=np.array([x["split"] for x in all_rows]))
            train.atomic_json(root / "config.json", {"thresholds": {"rt": 1.5, "ccs": 11.5},
                              "model_path": "fake-model", "seeds": [17]})
            original_read = train.read_jsonl
            test_opens = []

            def guarded_read(path):
                if Path(path).name == "test.jsonl":
                    receipt = root / "runs/M3__seed17/formal/selected_checkpoint.json"
                    selected = json.loads(receipt.read_text())
                    self.assertEqual(selected["epochs_completed"], 6)
                    self.assertFalse(selected["test_opened"])
                    self.assertEqual(selected["checkpoint_sha256"], train.sha256(receipt.with_name("best_adapter.pt")))
                    test_opens.append(True)
                return original_read(path)

            model = FakeLM()
            with patch.object(train, "verify_frozen", return_value={"config_sha256": "fake-test-digest"}), \
                 patch.object(train, "load_model", return_value=(model, Tokenizer())), \
                 patch.object(train, "read_jsonl", side_effect=guarded_read), \
                 patch.object(torch.cuda, "max_memory_allocated", return_value=0), \
                 patch.object(torch.cuda, "max_memory_reserved", return_value=0):
                result = train.run(root, "M3", 17, "formal")
            self.assertEqual(test_opens, [True])
            self.assertEqual(result["epochs_completed"], 6)
            self.assertEqual(result["training_examples_seen"], 4 * 3 * 6)
            self.assertEqual(result["optimizer_updates"], 2 * 6)
            run_dir = root / "runs/M3__seed17/formal"
            self.assertEqual(set(torch.load(run_dir / "best_adapter.pt", weights_only=True)), {"bias"})
            self.assertEqual(len(json.loads((run_dir / "training_history.json").read_text())), 6)
            for condition in ("normal", "mean", "shuffled", "off"):
                artifact = result["artifacts"][f"test_{condition}.jsonl"]
                self.assertEqual(artifact["rows"], 12)
                self.assertEqual(artifact["sha256"], train.sha256(run_dir / f"test_{condition}.jsonl"))
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.base_lm.parameters()))
            with self.assertRaises(FileExistsError):
                train.run(root, "M3", 17, "formal")


if __name__ == "__main__":
    unittest.main()
