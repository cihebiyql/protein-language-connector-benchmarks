"""CPU-only model contracts: no pretrained model or external weights."""
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mm_model import MolecularMemoryLM, VARIANTS


class TinyCausalBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, hidden):
        n = torch.arange(1, hidden.shape[1] + 1, device=hidden.device)[None, :, None]
        return hidden + torch.tanh(self.linear(hidden.cumsum(1) / n))


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(23, 12)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyCausalBlock(12) for _ in range(3)])
        self.model.norm = nn.LayerNorm(12)
        self.lm_head = nn.Linear(12, 23)
        self.fail = False
        self.last_use_cache = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, position_ids, use_cache, return_dict):
        self.last_use_cache = use_cache
        hidden = self.embedding(input_ids) * attention_mask[..., None]
        for layer in self.model.layers:
            hidden = layer(hidden)
        if self.fail:
            raise RuntimeError("injected tiny LM failure")
        hidden = self.model.norm(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def fixture(variant="M3"):
    torch.manual_seed(31)
    model = MolecularMemoryLM(TinyLM(), variant, hidden_dim=12, esm_dim=6,
                              memory_dim=8, num_heads=2, hook_layer=1)
    ids = torch.tensor([[0, 3, 4, 5, 6], [2, 3, 4, 7, 8]])
    mask = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    end = torch.tensor([2, 2])
    memory = {name: torch.randn(2, 8, 6) for name in ("p15", "p30", "t15", "t30")}
    memory["condition_ids"] = torch.arange(20).reshape(2, 10)
    memory["observed"] = torch.tensor([[True, False] * 5, [False, True] * 5])
    return model, {"input_ids": ids, "attention_mask": mask,
                   "prompt_end_index": end, "memory_inputs": memory}


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_variants_same_parameter_graph_and_uniform_initial_routes(self):
        counts, states = [], []
        for variant in VARIANTS:
            model, inputs = fixture(variant)
            out = model(**inputs)
            self.assertEqual(out["logits"].shape, (2, 5, 23))
            self.assertEqual(out["numeric"].shape, (2,))
            self.assertIs(out["numeric"], out["numeric_z"])
            self.assertEqual(out["gate"].item(), 0.)
            torch.testing.assert_close(out["routes"]["view"], torch.full((2, 2), .5))
            torch.testing.assert_close(out["routes"]["bank"], torch.full((2, 4), .25))
            counts.append(model.capacity_report()["trainable_parameters"])
            states.append({k: tuple(v.shape) for k, v in model.trainable_state_dict().items()})
        self.assertEqual(len(set(counts)), 1)
        self.assertTrue(all(s == states[0] for s in states))

    def test_off_is_exact_base_and_does_not_read_memory(self):
        model, inputs = fixture()
        with torch.no_grad():
            model.alpha.fill_(1.)
        model.train()
        self.assertFalse(model.base_lm.training)
        off = model(**{**inputs, "memory_inputs": {"invalid": None}}, off=True)
        mask = inputs["attention_mask"]
        base = model.base_lm(input_ids=inputs["input_ids"], attention_mask=mask,
                             position_ids=(mask.cumsum(-1) - 1).clamp_min(0),
                             use_cache=False, return_dict=True)
        self.assertTrue(torch.equal(off["logits"], base.logits))
        self.assertIsNone(off["routes"])
        self.assertEqual(model.diagnostics()["reader_calls"], 0)
        self.assertFalse(model.base_lm.last_use_cache)

    def test_numeric_and_prompt_routes_ignore_arbitrary_teacher_suffix(self):
        for variant in VARIANTS:
            model, inputs = fixture(variant)
            with torch.no_grad():
                model.alpha.fill_(.6)
                model.view_router.weight.normal_()
                model.bank_router.weight.normal_()
            first = model(**inputs)
            changed = inputs["input_ids"].clone()
            changed[:, 3:] = torch.tensor([[17, 18], [19, 20]])
            changed = torch.cat((changed, torch.tensor([[21], [22]])), 1)
            second = model(**{**inputs, "input_ids": changed,
                              "attention_mask": torch.cat((inputs["attention_mask"], torch.ones(2, 1, dtype=torch.long)), 1)})
            torch.testing.assert_close(first["numeric"], second["numeric"], rtol=0, atol=0)
            for key in ("view", "bank"):
                torch.testing.assert_close(first["routes"][key], second["routes"][key], rtol=0, atol=0)

    def test_frozen_base_and_open_gate_task_gradients_every_module(self):
        for variant in VARIANTS:
            model, inputs = fixture(variant)
            with torch.no_grad():
                model.alpha.fill_(.5)
            labels = torch.full_like(inputs["input_ids"], -100)
            labels[:, 3:] = inputs["input_ids"][:, 3:]
            out = model(**inputs, labels=labels)
            (out["loss"] + out["numeric"].square().mean()).backward()
            self.assertTrue(all(p.grad is None for p in model.base_lm.parameters()))
            for name, module in (("proj0", model.projections[0]), ("proj1", model.projections[1]),
                                 ("rel0", model.relations[0]), ("rel1", model.relations[1]),
                                 *model.readers.items(), ("condition", model.condition_embedding),
                                 ("missing", model.missing_embedding), ("type", model.type_embedding),
                                 ("source", model.source_embedding), ("query", model.query_projection),
                                 ("view", model.view_router), ("bank", model.bank_router),
                                 ("output", model.output_projection), ("numeric", model.numeric_head)):
                grads = [p.grad for p in module.parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads), (variant, name))
                self.assertGreater(sum(g.abs().sum().item() for g in grads), 0., (variant, name))

    def test_zero_gate_has_signal_without_claiming_branch_nonzero_grad(self):
        model, inputs = fixture()
        out = model(**inputs)
        out["numeric"].square().mean().backward()
        self.assertGreater(model.alpha.grad.abs().item(), 0.)
        self.assertEqual(model.projections[0].weight.grad.abs().sum().item(), 0.)

    def test_safe_tensor_state_roundtrip_includes_buffers_excludes_base(self):
        model, inputs = fixture()
        with torch.no_grad():
            model.alpha.fill_(.7)
        expected = model(**inputs)["numeric"].detach()
        state = model.trainable_state_dict()
        self.assertIn("static_anchor", state)
        self.assertIn("slot_positions", state)
        self.assertFalse(any(k.startswith("base_lm.") for k in state))
        serialized = io.BytesIO()
        torch.save(state, serialized)
        serialized.seek(0)
        restored = torch.load(serialized, map_location="cpu", weights_only=True)
        fresh, _ = fixture()
        fresh.load_trainable_state_dict(restored)
        torch.testing.assert_close(fresh(**inputs)["numeric"], expected, rtol=0, atol=0)
        invalid = dict(restored)
        invalid["alpha"] = torch.tensor(float("nan"))
        with self.assertRaises(ValueError):
            fresh.load_trainable_state_dict(invalid)

    def test_hooks_removed_after_success_and_exception(self):
        model, inputs = fixture()
        layer, norm = model.base_lm.model.layers[1], model.base_lm.model.norm
        before = (len(layer._forward_hooks), len(norm._forward_hooks))
        model(**inputs)
        self.assertEqual(before, (len(layer._forward_hooks), len(norm._forward_hooks)))
        model.base_lm.fail = True
        with self.assertRaisesRegex(RuntimeError, "injected tiny"):
            model(**inputs)
        self.assertEqual(before, (len(layer._forward_hooks), len(norm._forward_hooks)))
        self.assertFalse(model._in_forward)
        model.base_lm.fail = False
        self.assertTrue(torch.isfinite(model(**inputs)["numeric"]).all())

    def test_input_contract_and_prompt_supervision_rejected(self):
        model, inputs = fixture()
        with self.assertRaises(ValueError):
            model(**{**inputs, "prompt_end_index": torch.tensor([0, 2])})
        with self.assertRaises(ValueError):
            model(**inputs, labels=inputs["input_ids"])
        invalid = dict(inputs["memory_inputs"])
        invalid["observed"] = invalid["observed"].long()
        with self.assertRaises(ValueError):
            model(**{**inputs, "memory_inputs": invalid})
        invalid = dict(inputs["memory_inputs"])
        invalid["p15"] = invalid["p15"].requires_grad_()
        with self.assertRaises(ValueError):
            model(**{**inputs, "memory_inputs": invalid})

    def test_condition_missing_ignores_hashed_value_but_observed_does_not(self):
        model, inputs = fixture()
        views, c = model._banks(inputs["memory_inputs"])
        changed = dict(inputs["memory_inputs"])
        changed["condition_ids"] = (changed["condition_ids"] + 100) % 4096
        _, c2 = model._banks(changed)
        missing = ~changed["observed"]
        self.assertTrue(torch.equal(c[missing], c2[missing]))
        self.assertFalse(torch.equal(c[~missing], c2[~missing]))

    def test_actual_reader_calls_and_communication_ledger(self):
        model, inputs = fixture()
        calls = {name: 0 for name in ("p", "t", "c", "r", "relation0", "relation1")}
        def counter(name):
            def hook(*_args):
                calls[name] += 1
            return hook
        handles = [module.register_forward_hook(counter(name)) for name, module in model.readers.items()]
        handles += [module.attention.register_forward_hook(counter("relation" + str(i)))
                    for i, module in enumerate(model.relations)]
        model(**inputs)
        for handle in handles:
            handle.remove()
        self.assertEqual(calls, {"p": 2, "t": 2, "c": 1, "r": 2, "relation0": 1, "relation1": 1})
        budget = model.communication_report(2, 5)
        ledger = model.diagnostics()
        for name, value in budget.items():
            if name in ledger:
                self.assertEqual(ledger[name], value, name)
        self.assertEqual(ledger["active_text_positions"], 9)
        self.assertEqual(ledger["reader_calls"], 7)
        self.assertEqual(ledger["relation_calls"], 2)
        model.diagnostics(reset=True)
        self.assertEqual(model.diagnostics()["forward_calls"], 0)

    def test_input_selection_self_cross_and_static_query_controls(self):
        for variant, (intermediate, conditional, cross) in VARIANTS.items():
            model, inputs = fixture(variant)
            memory = inputs["memory_inputs"]
            views, _ = model._banks(memory)
            changed = {**memory, "p15": memory["p15"] + torch.randn_like(memory["p15"]),
                       "t15": memory["t15"] + torch.randn_like(memory["t15"])}
            altered, _ = model._banks(changed)
            self.assertEqual(torch.equal(views[0]["p"], altered[0]["p"]), not intermediate)
            target_changed = {**memory, "t15": torch.randn_like(memory["t15"]),
                              "t30": torch.randn_like(memory["t30"])}
            target_views, _ = model._banks(target_changed)
            self.assertEqual(torch.equal(views[0]["r"], target_views[0]["r"]), not cross)
            with torch.no_grad():
                model.view_router.weight.normal_()
                model.bank_router.weight.normal_()
            first = model(**inputs)
            ids = inputs["input_ids"].clone()
            ids[:, 2] = torch.tensor([19, 21])
            second = model(**{**inputs, "input_ids": ids})
            self.assertEqual(torch.equal(first["routes"]["view"], second["routes"]["view"]), not conditional)


if __name__ == "__main__":
    unittest.main()
