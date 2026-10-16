"""Real CPU Qwen checks; random tiny weights, no downloads or GPU work."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from foundation_20261016.connectors import FrozenConnectorLM


METHODS = ("input_mlp", "qformer_style", "internal_cross_attention", "constant")


def fixture(method):
    torch.manual_seed(71)
    base = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=48, hidden_size=24, intermediate_size=40, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, attention_dropout=0.1,
        eos_token_id=47, pad_token_id=0)).eval()
    return FrozenConnectorLM(base, method, source_dim=8, memory_dim=12,
                             num_memory_tokens=3, insertion_layer=1)


def batch():
    torch.manual_seed(19)
    return dict(input_ids=torch.tensor([[0, 0, 3, 4, 5], [2, 3, 4, 5, 6]]),
                attention_mask=torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]),
                memory=torch.randn(2, 7, 8),
                memory_mask=torch.tensor([[1, 1, 0, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0]]),
                memory_type_ids=torch.tensor([[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0]]))


class FoundationConnectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_frozen_base_trainable_connector_and_causal_loss(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = fixture(method).train()
                inputs = batch()
                labels = inputs["input_ids"].clone()
                labels[:, :3] = -100
                output = model(**inputs, labels=labels)
                self.assertEqual(output.logits.shape, (2, 5, 48))
                self.assertTrue(torch.isfinite(output.loss))
                expected = torch.nn.functional.cross_entropy(
                    output.logits[:, :-1].reshape(-1, 48), labels[:, 1:].reshape(-1), ignore_index=-100)
                torch.testing.assert_close(output.loss, expected)
                output.loss.backward()
                self.assertFalse(model.base_lm.training)
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.base_lm.parameters()))
                gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
                report = model.capacity_report()
                self.assertEqual(report["trainable_parameters"], sum(
                    p.numel() for p in model.parameters() if p.requires_grad))
                self.assertTrue(report["frozen_parameters"] > report["trainable_parameters"])

    def test_off_is_exact_plain_lm_and_independent_of_memory(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = fixture(method)
                inputs = batch()
                mask = inputs["attention_mask"]
                reference = model.base_lm(input_ids=inputs["input_ids"], attention_mask=mask,
                    position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=False).logits
                actual = model(**inputs, message_mode="off")
                torch.testing.assert_close(actual.logits, reference, rtol=0, atol=0)
                actual.logits.sum().backward()
                self.assertEqual(sum(p.grad.abs().sum().item() for p in model.parameters()
                                     if p.requires_grad and p.grad is not None), 0)
                inputs["memory"] += 30
                torch.testing.assert_close(model(**inputs, message_mode="off").logits, reference,
                                           rtol=0, atol=0)

    def test_signal_changes_logits_but_constant_does_not(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = fixture(method).eval()
                inputs = batch()
                original = model(**inputs).logits
                inputs["memory"] = torch.randn_like(inputs["memory"]) * 3
                changed = model(**inputs).logits
                if method == "constant":
                    torch.testing.assert_close(original, changed, rtol=0, atol=0)
                else:
                    self.assertGreater((original - changed).abs().max().item(), 1e-7)

    def test_masked_residues_do_not_contribute(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = fixture(method).eval()
                inputs = batch()
                original = model(**inputs).logits
                inputs["memory"][~inputs["memory_mask"].bool()] = 10000
                torch.testing.assert_close(original, model(**inputs).logits, rtol=0, atol=0)

    def test_type_embedding_distinguishes_molecule_types(self):
        for method in METHODS[:-1]:
            with self.subTest(method=method):
                model = fixture(method).eval()
                inputs = batch()
                original = model(**inputs).logits
                inputs["memory_type_ids"] = 1 - inputs["memory_type_ids"]
                self.assertGreater((original - model(**inputs).logits).abs().max().item(), 1e-7)

    def test_internal_hooks_cleanup_after_success_and_failure(self):
        model = fixture("internal_cross_attention")
        layer = model.base_lm.model.layers[0]
        inputs = batch()
        before = len(layer._forward_hooks)
        first = model(**inputs).logits
        self.assertEqual(len(layer._forward_hooks), before)
        with patch.object(model.base_lm.lm_head, "forward", side_effect=RuntimeError("forced failure")):
            with self.assertRaisesRegex(RuntimeError, "forced failure"):
                model(**inputs)
        self.assertEqual(len(layer._forward_hooks), before)
        torch.testing.assert_close(first, model(**inputs).logits, rtol=0, atol=0)

    def test_generation_is_deterministic_and_left_padding_invariant(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = fixture(method).train()
                inputs = batch()
                a = model.greedy_generate(**inputs, max_new_tokens=4, eos_token_id=[])
                b = model.greedy_generate(**inputs, max_new_tokens=4, eos_token_id=[])
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertEqual(a.shape, (2, 4))
                self.assertTrue(model.training)
                single = {name: value[:1] for name, value in inputs.items()}
                single["input_ids"] = single["input_ids"][:, 2:]
                single["attention_mask"] = single["attention_mask"][:, 2:]
                torch.testing.assert_close(a[:1], model.greedy_generate(
                    **single, max_new_tokens=4, eos_token_id=[]), rtol=0, atol=0)

    def test_generation_eos_stops_and_bounds_are_checked(self):
        model = fixture("constant")
        with torch.no_grad():
            model.base_lm.lm_head.weight.zero_()
        self.assertEqual(model.greedy_generate(**batch(), eos_token_id=0).tolist(), [[0], [0]])
        for limit in (0, -1, 257, True):
            with self.assertRaises(ValueError):
                model.greedy_generate(**batch(), max_new_tokens=limit)

    def test_validation_rejects_nonfinite_empty_mask_and_invalid_types(self):
        model = fixture("input_mlp")
        for key, replacement in (
            ("memory", torch.full((2, 7, 8), float("nan"))),
            ("memory_mask", torch.zeros(2, 7)),
            ("memory_mask", torch.full((2, 7), 0.5)),
            ("memory_type_ids", torch.full((2, 7), 2)),
            ("attention_mask", torch.zeros(2, 5)),
            ("memory", torch.empty(2, 0, 8)),
            ("input_ids", torch.full((2, 5), 48)),
        ):
            with self.subTest(key=key, shape=replacement.shape):
                inputs = batch()
                inputs[key] = replacement
                with self.assertRaises(ValueError):
                    model(**inputs)
        with self.assertRaises(ValueError):
            model(**batch(), message_mode="mismatch")
        with self.assertRaises(ValueError):
            model(**batch(), labels=torch.full((2, 5), -100))
        with self.assertRaises(ValueError):
            FrozenConnectorLM(model.base_lm, "internal_cross_attention", insertion_layer=20)

    def test_aliases_and_default_type_ids(self):
        self.assertEqual(FrozenConnectorLM(fixture("constant").base_lm, "BASE04").method, "input_mlp")
        model = fixture("qformer_style")
        inputs = batch()
        inputs.pop("memory_type_ids")
        output = model(**inputs).logits
        inputs["memory_type_ids"] = torch.zeros_like(inputs["memory_mask"])
        torch.testing.assert_close(output, model(**inputs).logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
