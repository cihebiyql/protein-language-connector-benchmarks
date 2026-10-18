"""CPU tests of causal injection, frozen backbone, and exact adapter controls."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn


spec = importlib.util.spec_from_file_location(
    "qa_model", Path(__file__).resolve().parents[1] / "src" / "qa_model.py")
qa_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa_model)


class CausalBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mix = nn.Linear(width, width)

    def forward(self, x, mask):
        visible = x * mask[..., None]
        context = visible.cumsum(1) / mask.cumsum(1).clamp_min(1)[..., None]
        return (x + torch.tanh(self.mix(context)),)


class FakeLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(31, 12)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([CausalBlock(12) for _ in range(3)])
        self.head = nn.Linear(12, 31)
        self.is_gradient_checkpointing = False
        self.raise_after_hook = False

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, position_ids, use_cache, return_dict):
        assert use_cache is False and return_dict is True
        self.last_positions = position_ids.detach().clone()
        x = self.embedding(input_ids) + position_ids[..., None] * 0.01
        for index, layer in enumerate(self.model.layers):
            x = layer(x, attention_mask)[0]
            if index == 1 and self.raise_after_hook:
                raise RuntimeError("deliberate fake LM failure")
        return SimpleNamespace(logits=self.head(x))


class SharedMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = qa_model.SharedMemoryLM(FakeLM(), hidden_dim=12, esm_dim=6,
                                             memory_dim=8, num_heads=2, hook_layer=1)
        self.ids = torch.tensor([[0, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]])
        self.mask = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
        self.end = torch.tensor([3, 2])
        self.memory = {name: torch.randn(2, 8, 6) for name in ("p15", "p30")}
        with torch.no_grad():
            self.model.alpha.fill_(0.6)
            self.model.view_router.weight.normal_(0, 0.5)

    def call(self, **kwargs):
        args = dict(input_ids=self.ids, attention_mask=self.mask,
                    prompt_end_index=self.end, memory_inputs=self.memory)
        args.update(kwargs)
        return self.model(**args)

    def test_future_answers_do_not_change_prefix_or_prompt_routing(self):
        original = self.call()
        changed = self.ids.clone()
        changed[:, 4:] = torch.tensor([[20, 21], [22, 23]])
        perturbed = self.call(input_ids=changed)
        torch.testing.assert_close(original["routes"], perturbed["routes"], atol=0, rtol=0)
        torch.testing.assert_close(original["logits"][:, :4], perturbed["logits"][:, :4],
                                   atol=0, rtol=0)
        self.assertFalse(torch.equal(original["logits"][:, 4:], perturbed["logits"][:, 4:]))

    def test_future_question_tokens_do_not_leak_to_earlier_prefix(self):
        original = self.call()
        changed = self.ids.clone()
        changed[0, 3] = 25  # Prompt end changes router; earlier prefix must stay causal.
        perturbed = self.call(input_ids=changed)
        self.assertFalse(torch.equal(original["routes"][0], perturbed["routes"][0]))
        torch.testing.assert_close(original["logits"][0, :3], perturbed["logits"][0, :3],
                                   atol=0, rtol=0)

    def test_off_matches_unmodified_lm_and_hooks_cleaned(self):
        actual = self.call(off=True, memory_inputs=None)["logits"]
        expected = self.model.base_lm(self.ids, self.mask,
            (self.mask.cumsum(-1) - 1).clamp_min(0), False, True).logits
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.call()
        self.assertEqual(len(self.model.base_lm.model.layers[1]._forward_hooks), 0)
        self.assertFalse(self.model._in_forward)

    def test_failure_always_removes_hook(self):
        self.model.base_lm.raise_after_hook = True
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            self.call()
        self.assertFalse(self.model._in_forward)
        self.assertEqual(len(self.model.base_lm.model.layers[1]._forward_hooks), 0)
        self.model.base_lm.raise_after_hook = False
        self.call()

    def test_backbone_frozen_but_adapters_receive_answer_gradients(self):
        self.model.train()
        self.assertFalse(self.model.base_lm.training)
        logits = self.call()["logits"]
        loss = torch.nn.functional.cross_entropy(logits[:, 3:-1].reshape(-1, 31),
                                                 self.ids[:, 4:].reshape(-1))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in self.model.base_lm.parameters()))
        self.assertGreater(self.model.projections[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.view_router.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.alpha.grad.abs().item(), 0)

    def test_capacity_exact_across_three_arms(self):
        arms = [qa_model.SharedMemoryLM(FakeLM(), variant=arm, hidden_dim=12,
                    esm_dim=6, memory_dim=8, num_heads=2, hook_layer=1)
                for arm in ("M0", "M3", "TEXT")]
        reports = [model.capacity_report() for model in arms]
        self.assertEqual(reports[0]["trainable_by_name"], reports[1]["trainable_by_name"])
        self.assertEqual(reports[1]["trainable_by_name"], reports[2]["trainable_by_name"])
        self.assertEqual(reports[0]["logical_bank_payload_bytes_per_sample"], 512)
        self.assertTrue(all(report["task_specific_parameters"] == 0 for report in reports))

    def test_checkpoint_never_materializes_or_loads_backbone_state(self):
        with patch.object(self.model.base_lm, "state_dict", side_effect=AssertionError("LM state")):
            state = self.model.trainable_state_dict()
            self.assertFalse(any(key.startswith("base_lm.") for key in state))
            self.model.load_trainable_state_dict(state)
        with torch.no_grad():
            self.model.alpha.fill_(0.1)
        self.model.load_trainable_state_dict(state)
        torch.testing.assert_close(self.model.alpha.cpu(), state["alpha"])
        bad = dict(state)
        bad["alpha"] = torch.tensor(float("nan"))
        with self.assertRaises(ValueError):
            self.model.load_trainable_state_dict(bad)

    def test_left_padding_positions_and_memory_validation(self):
        self.call()
        torch.testing.assert_close(self.model.base_lm.last_positions,
            torch.tensor([[0, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]]))
        bad = dict(self.memory)
        bad["p15"] = bad["p15"].requires_grad_()
        with self.assertRaisesRegex(ValueError, "detached"):
            self.call(memory_inputs=bad)

    def test_zero_gate_exactly_reproduces_frozen_lm(self):
        with torch.no_grad():
            self.model.alpha.zero_()
        on = self.call()["logits"]
        off = self.call(off=True)["logits"]
        torch.testing.assert_close(on, off, atol=0, rtol=0)

    def test_m0_ignores_p15_and_uses_static_router(self):
        self.model.variant = "M0"
        original = self.call()
        changed_memory = dict(self.memory)
        changed_memory["p15"] = torch.randn_like(changed_memory["p15"]) * 20
        altered = self.call(memory_inputs=changed_memory)
        torch.testing.assert_close(original["logits"], altered["logits"], atol=0, rtol=0)
        changed_ids = self.ids.clone()
        changed_ids[:, 2:4] = 24
        question = self.call(input_ids=changed_ids)
        torch.testing.assert_close(original["routes"], question["routes"], atol=0, rtol=0)

    def test_m3_memory_changes_answer_logits_without_changing_prefix(self):
        original = self.call()
        changed = {name: value.flip(0) for name, value in self.memory.items()}
        swapped = self.call(memory_inputs=changed)
        torch.testing.assert_close(original["logits"][0, :3], swapped["logits"][0, :3],
                                   atol=0, rtol=0)
        self.assertFalse(torch.equal(original["logits"][:, 4:], swapped["logits"][:, 4:]))

    def test_tiny_real_qwen3_forward_and_backward(self):
        try:
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except ImportError:
            self.skipTest("Qwen3 Transformers classes unavailable")
        config = Qwen3Config(vocab_size=31, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=3, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=4,
                             max_position_embeddings=32, attention_dropout=0.0,
                             tie_word_embeddings=False, pad_token_id=0)
        model = qa_model.SharedMemoryLM(Qwen3ForCausalLM(config), hidden_dim=16,
                                        esm_dim=6, memory_dim=8, num_heads=2, hook_layer=1)
        with torch.no_grad():
            model.alpha.fill_(0.5)
        output = model(self.ids, self.mask, self.end, self.memory)
        self.assertEqual(output["logits"].shape, (2, 6, 31))
        output["logits"][:, 4:].square().mean().backward()
        self.assertIsNotNone(model.projections[0].weight.grad)
        self.assertTrue(all(p.grad is None for p in model.base_lm.parameters()))
        self.assertEqual(len(model.base_lm.model.layers[1]._forward_hooks), 0)

    def test_reentrant_forward_rejected_without_leaked_hook(self):
        self.model._in_forward = True
        with self.assertRaisesRegex(RuntimeError, "reentrant"):
            self.call()
        self.model._in_forward = False
        self.assertEqual(len(self.model.base_lm.model.layers[1]._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
