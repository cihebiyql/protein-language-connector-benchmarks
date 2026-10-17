"""CPU causal fixtures for single-sequence connector invariants."""
import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from src.mt_model import ESMTaskModel, SingleMemoryLM


class CausalBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, width)
        self.dropout = nn.Dropout(0.5)

    def forward(self, hidden, mask):
        values = hidden * mask[..., None]
        mean = values.cumsum(1) / mask.cumsum(1).clamp_min(1)[..., None]
        return (hidden + self.dropout(self.proj(mean).tanh()),)


class ToyLM(nn.Module):
    def __init__(self, width=16, blocks=3):
        super().__init__()
        self.embedding = nn.Embedding(37, width)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([CausalBlock(width) for _ in range(blocks)])
        self.model.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, 37, bias=False)
        self.is_gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, **kwargs):
        h = self.embedding(input_ids)
        for layer in self.model.layers:
            h = layer(h, attention_mask)[0]
        return SimpleNamespace(logits=self.lm_head(self.model.norm(h)))


def memory(batch=2, dim=12):
    return {"p15": torch.randn(batch, 8, dim), "p30": torch.randn(batch, 8, dim),
            "condition_ids": torch.arange(batch * 10).reshape(batch, 10),
            "observed": (torch.arange(batch * 10).reshape(batch, 10) % 2 == 0)}


class SingleMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.ids = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 5, 4, 3, 6, 7]])
        self.mask = torch.ones_like(self.ids)
        self.end = torch.tensor([2, 2])
        self.mem = memory()
        self.labels = self.ids.clone()
        self.labels[:, :3] = -100

    def model(self, variant="M3", task="rt", base=None):
        return SingleMemoryLM(base or ToyLM(), task, variant, hidden_dim=16,
            esm_dim=12, memory_dim=8, num_heads=2, hook_layer=1,
            candidate_vectors=torch.randn(31, 1536) if task == "elm" else None)

    def run_model(self, model, ids=None, **kwargs):
        return model(self.ids if ids is None else ids, self.mask, self.end, self.mem, **kwargs)

    def test_matched_active_capacity_and_only_pc_banks(self):
        for task in ("meltome", "rt", "elm"):
            a, b = self.model("M0", task), self.model("M3", task)
            self.assertEqual(a.capacity_report()["trainable_parameters"], b.capacity_report()["trainable_parameters"])
            self.assertEqual(a.capacity_report()["logical_distinct_slots"], 26)
            self.assertEqual(a.capacity_report()["extra_lm_positions"], 0)
            self.assertEqual(set(a.readers), {"p", "c"})
            self.assertFalse(any("relation" in n or "target" in n for n, _ in a.named_modules()))
            self.assertEqual(a.type_embedding.num_embeddings, 2)
            self.assertEqual(a.capacity_report()["logical_bank_payload_bytes_per_sample"], 26 * 8 * 4)

    def test_zero_gate_and_off_exactly_recover_base(self):
        model = self.model()
        baseline = model.base_lm(input_ids=self.ids, attention_mask=self.mask).logits
        self.assertTrue(torch.equal(self.run_model(model)["logits"], baseline))
        model.alpha.data.fill_(0.6)
        self.assertTrue(torch.equal(self.run_model(model, off=True)["logits"], baseline))
        self.assertFalse(torch.equal(self.run_model(model)["logits"], baseline))
        self.assertIsNone(self.run_model(model, off=True)["routes"])

    def test_no_future_answer_leak_for_both_arms_and_heads(self):
        changed = self.ids.clone()
        changed[:, 3:] = torch.tensor([[15, 16, 17], [18, 19, 20]])
        for task in ("rt", "elm"):
            for variant in ("M0", "M3"):
                model = self.model(variant, task)
                model.alpha.data.fill_(0.9)
                a, b = self.run_model(model), self.run_model(model, changed)
                for key in ("prompt_hidden", "numeric_z" if task == "rt" else "ranking_logits"):
                    self.assertTrue(torch.equal(a[key], b[key]), (task, variant, key))
                self.assertTrue(torch.equal(a["logits"][:, :3], b["logits"][:, :3]))

    def test_static_routing_independent_of_prompt_vs_conditional(self):
        for variant in ("M0", "M3"):
            model = self.model(variant)
            nn.init.normal_(model.view_router.weight)
            nn.init.normal_(model.bank_router.weight)
            a = self.run_model(model)["routes"]
            changed = self.ids.clone()
            changed[:, :3] = 21
            b = self.run_model(model, changed)["routes"]
            self.assertTrue((a["view"] > 0).all())
            self.assertTrue((a["bank"] > 0).all())
            self.assertEqual(torch.equal(a["view"], b["view"]), variant == "M0")

    def test_m0_ignores_layer15_m3_uses_it(self):
        for variant in ("M0", "M3"):
            model = self.model(variant)
            model.alpha.data.fill_(0.9)
            before = self.run_model(model)["numeric_z"]
            changed = {**self.mem, "p15": torch.randn_like(self.mem["p15"])}
            after = model(self.ids, self.mask, self.end, changed)["numeric_z"]
            self.assertEqual(torch.equal(before, after), variant == "M0")

    def test_real_ce_gradients_post_gate_step_all_registered_trainables_used(self):
        for variant in ("M0", "M3"):
            model = self.model(variant).train()
            optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.2)
            output = self.run_model(model, labels=self.labels)
            expected = F.cross_entropy(output["logits"][:, :-1].flatten(0, 1), self.labels[:, 1:].flatten(), ignore_index=-100)
            self.assertTrue(torch.equal(output["loss"], expected))
            loss = output["loss"] + output["numeric_z"].square().mean()
            loss.backward()
            self.assertGreater(model.alpha.grad.abs().item(), 0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            output = self.run_model(model, labels=self.labels)
            (output["loss"] + output["numeric_z"].square().mean()).backward()
            for name, param in model.named_parameters():
                if name.startswith("base_lm."):
                    self.assertFalse(param.requires_grad)
                    self.assertIsNone(param.grad)
                else:
                    self.assertEqual(param.dtype, torch.float32)
                    self.assertIsNotNone(param.grad, name)
                    self.assertTrue(torch.isfinite(param.grad).all(), name)
                    self.assertGreater(param.grad.abs().sum().item(), 0, name)
            self.assertFalse(model.base_lm.training)

    def test_semantic_31_candidate_ranking_and_description_permutation(self):
        model = self.model(task="elm")
        scores = self.run_model(model)["ranking_logits"]
        self.assertEqual(scores.shape, (2, 31))
        self.assertTrue(torch.isfinite(scores).all())
        order = torch.arange(30, -1, -1)
        model.ranking_head.candidate_vectors.copy_(model.ranking_head.candidate_vectors[order].clone())
        self.assertTrue(torch.equal(self.run_model(model)["ranking_logits"], scores[:, order]))
        self.assertNotIn("numeric_head", dict(model.named_modules()))
        self.run_model(model)["ranking_logits"].square().mean().backward()
        self.assertIsNotNone(model.ranking_head.query_projection.weight.grad)
        self.assertIsNotNone(model.ranking_head.candidate_projection.weight.grad)
        self.assertFalse(model.ranking_head.candidate_vectors.requires_grad)

    def test_baseline_both_layers_conditions_and_actual_capacity(self):
        for task in ("meltome", "rt", "elm"):
            model = ESMTaskModel(task, esm_dim=12, memory_dim=8,
                candidate_vectors=torch.randn(31, 1536) if task == "elm" else None)
            key = "ranking_logits" if task == "elm" else "numeric_z"
            before = model(self.mem)[key]
            for field in ("p15", "p30"):
                changed = {**self.mem, field: torch.randn_like(self.mem[field])}
                self.assertFalse(torch.equal(before, model(changed)[key]))
            changed = {**self.mem, "observed": ~self.mem["observed"]}
            self.assertFalse(torch.equal(before, model(changed)[key]))
            before.square().mean().backward()
            for name, param in model.named_parameters():
                self.assertIsNotNone(param.grad, name)
                self.assertGreater(param.grad.abs().sum().item(), 0, name)
            self.assertEqual(model.capacity_report()["trainable_parameters"], sum(p.numel() for p in model.parameters()))
            self.assertFalse(model.capacity_report()["online_lm"])

    def test_checkpoint_exact_roundtrip_and_reject_bad_state(self):
        model = self.model(task="elm")
        state = model.trainable_state_dict()
        self.assertFalse(any(k.startswith("base_lm.") for k in state))
        before = self.run_model(model)["ranking_logits"]
        model.ranking_head.query_projection.weight.data.add_(2)
        model.load_trainable_state_dict(state)
        self.assertTrue(torch.equal(before, self.run_model(model)["ranking_logits"]))
        bad = copy.deepcopy(state)
        bad["alpha"] = torch.tensor(float("nan"))
        with self.assertRaises(ValueError):
            model.load_trainable_state_dict(bad)

    def test_input_and_supervision_validation_hook_cleanup(self):
        model = self.model()
        with self.assertRaises(ValueError):
            self.run_model(model, labels=self.ids)
        bad = {**self.mem, "sample_id": torch.ones(2)}
        with self.assertRaises(ValueError):
            model(self.ids, self.mask, self.end, bad)
        bad = {**self.mem, "p15": torch.full_like(self.mem["p15"], float("nan"))}
        with self.assertRaises(ValueError):
            model(self.ids, self.mask, self.end, bad)
        self.assertEqual(len(model.base_lm.model.norm._forward_hooks), 0)
        self.assertEqual(len(model.base_lm.model.layers[1]._forward_hooks), 0)
        self.assertTrue(torch.isfinite(self.run_model(model)["numeric_z"]).all())

    def test_growing_context_preserves_prompt_anchor_and_off(self):
        model = self.model()
        model.alpha.data.fill_(0.6)
        before = self.run_model(model)["prompt_hidden"]
        extended = torch.cat((self.ids, torch.tensor([[20], [21]])), dim=1)
        after = model(extended, torch.ones_like(extended), self.end, self.mem)["prompt_hidden"]
        self.assertTrue(torch.equal(before, after))

    def test_bfloat16_frozen_base_fp32_adapter_backward(self):
        model = self.model(base=ToyLM().to(torch.bfloat16))
        model.alpha.data.fill_(0.6)
        output = self.run_model(model, labels=self.labels)
        (output["loss"] + output["numeric_z"].square().mean()).backward()
        self.assertEqual(output["logits"].dtype, torch.bfloat16)
        self.assertEqual(output["numeric_z"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(model.alpha.grad))


if __name__ == "__main__":
    unittest.main()
