"""CPU-only fixed-message and actual tiny-Qwen checks, no downloaded model."""
import inspect
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from matched_budget_20261016 import (
    FixedMessageSender, MatchedConnectorLM, MatchedReader, expand_neutral, neutral_message)


def fixture(placement, gate_init="zero"):
    torch.manual_seed(81)
    base = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=48, hidden_size=24, intermediate_size=40, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, attention_dropout=.2,
        eos_token_id=47, pad_token_id=0))
    return MatchedConnectorLM(base, placement=placement, insertion_layer=1, gate_init=gate_init)


def batch():
    generator = torch.Generator().manual_seed(19)
    return {"input_ids": torch.tensor([[0, 0, 3, 4, 5], [2, 3, 4, 5, 6]]),
            "attention_mask": torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]),
            "memory": torch.randn(2, 16, 128, generator=generator),
            "memory_mask": torch.ones(2, 16, dtype=torch.bool)}


class MatchedCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_reader_parameters_gate_and_placement_initialization(self):
        reader = MatchedReader(hidden_dim=1536)
        self.assertEqual(sum(p.numel() for p in reader.parameters()), 464001)
        self.assertEqual(float(reader.alpha.detach()), 0.)
        self.assertEqual(float(MatchedReader(gate_init="near_unit").alpha.detach()), 3.)
        early, mid = fixture("early"), fixture("mid")
        for key in early.reader_state():
            torch.testing.assert_close(early.reader_state()[key], mid.reader_state()[key], rtol=0, atol=0)
        self.assertEqual(early.capacity_report()["trainable_parameters"], mid.capacity_report()["trainable_parameters"])
        self.assertEqual(early.capacity_report()["extra_lm_positions"], 0)

    def test_state_loading_is_strict_and_includes_gate(self):
        early, mid = fixture("early", "near_unit"), fixture("mid")
        state = early.reader_state()
        self.assertIn("alpha", state)
        mid.load_reader_state(state)
        for key in state:
            torch.testing.assert_close(mid.reader_state()[key], state[key], rtol=0, atol=0)
        bad = dict(state); bad.pop("alpha")
        with self.assertRaises(ValueError):
            mid.load_reader_state(bad)
        bad = dict(state); bad["alpha"] = torch.tensor(float("nan"))
        with self.assertRaises(ValueError):
            mid.load_reader_state(bad)
        bad = dict(state); bad["alpha"] = torch.zeros(2)
        with self.assertRaises(ValueError):
            mid.load_reader_state(bad)

    def test_zero_gate_and_off_exactly_equal_plain_lm_and_positions(self):
        inputs = batch(); mask = inputs["attention_mask"]
        for placement in ("early", "mid"):
            model = fixture(placement).train()
            reference = model.base_lm(input_ids=inputs["input_ids"], attention_mask=mask,
                position_ids=(mask.cumsum(-1)-1).clamp_min(0), use_cache=False).logits
            with patch.object(model.base_lm, "forward", wraps=model.base_lm.forward) as forward:
                actual = model(**inputs).logits
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            torch.testing.assert_close(forward.call_args.kwargs["position_ids"], (mask.cumsum(-1)-1).clamp_min(0))
            self.assertNotIn("inputs_embeds", forward.call_args.kwargs)
            self.assertFalse(forward.call_args.kwargs["use_cache"])
            self.assertEqual(actual.shape, (2, 5, 48))
            torch.testing.assert_close(model(**inputs, message_mode="off").logits, reference, rtol=0, atol=0)
            self.assertFalse(model.base_lm.training)

    def test_zero_gate_first_gradient_and_backbone_freeze(self):
        for placement in ("early", "mid"):
            model = fixture(placement).train(); inputs = batch()
            labels = inputs["input_ids"].clone(); labels[:, :3] = -100
            output = model(**inputs, labels=labels)
            expected = F.cross_entropy(output.logits[:, :-1].reshape(-1,48), labels[:,1:].reshape(-1), ignore_index=-100)
            torch.testing.assert_close(output.loss, expected)
            output.loss.backward()
            self.assertGreater(float(model.reader.alpha.grad.abs()), 0.)
            for name,p in model.reader.named_parameters():
                if name != "alpha":
                    self.assertIsNotNone(p.grad)
                    self.assertEqual(float(p.grad.abs().sum()), 0.)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.base_lm.parameters()))

    def test_near_unit_uses_content_and_gradients(self):
        for placement in ("early", "mid"):
            model = fixture(placement, "near_unit").train(); inputs = batch()
            original = model(**inputs).logits
            changed = dict(inputs); changed["memory"] = inputs["memory"] + .7
            self.assertGreater(float((original-model(**changed).logits).detach().abs().max()), 1e-7)
            labels = inputs["input_ids"].clone(); labels[:,:3] = -100
            model(**inputs, labels=labels).loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.reader.parameters()))
            self.assertGreater(float(model.reader.query_projection.weight.grad.abs().sum()), 0.)
            self.assertTrue(all(p.grad is None for p in model.base_lm.parameters()))

    def test_future_text_does_not_leak_to_earlier_logits(self):
        for placement in ("early", "mid"):
            model = fixture(placement, "near_unit"); inputs = batch()
            original = model(**inputs).logits
            inputs["input_ids"][:,-1] = 19
            torch.testing.assert_close(original[:,:-1], model(**inputs).logits[:,:-1], rtol=0, atol=0)

    def test_hooks_removed_after_success_reader_failure_and_later_failure(self):
        for placement in ("early", "mid"):
            model = fixture(placement, "near_unit"); layer = model.base_lm.model.layers[0]
            initial = (len(layer._forward_pre_hooks),len(layer._forward_hooks))
            reference = model(**batch()).logits
            for target in (model.reader, model.base_lm.lm_head):
                with patch.object(target, "forward", side_effect=RuntimeError("forced failure")):
                    with self.assertRaisesRegex(RuntimeError, "forced failure"):
                        model(**batch())
                self.assertEqual((len(layer._forward_pre_hooks),len(layer._forward_hooks)), initial)
            torch.testing.assert_close(model(**batch()).logits, reference, rtol=0, atol=0)

    def test_reused_greedy_determinism_left_padding_and_eos(self):
        for placement in ("early", "mid"):
            model = fixture(placement, "near_unit").train(); inputs = batch()
            a = model.greedy_generate(**inputs,max_new_tokens=4,eos_token_id=[])
            b = model.greedy_generate(**inputs,max_new_tokens=4,eos_token_id=[])
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            single = {k:v[:1] for k,v in inputs.items()}
            single["input_ids"] = single["input_ids"][:,2:]
            single["attention_mask"] = single["attention_mask"][:,2:]
            torch.testing.assert_close(a[:1],model.greedy_generate(**single,max_new_tokens=4,eos_token_id=[]),rtol=0,atol=0)
            self.assertTrue(model.training)
            with torch.no_grad(): model.base_lm.lm_head.weight.zero_()
            self.assertEqual(model.greedy_generate(**inputs,eos_token_id=0).tolist(),[[0],[0]])
            for cap in (0,-1,257,True):
                with self.assertRaises(ValueError):model.greedy_generate(**inputs,max_new_tokens=cap)

    def test_invalid_masks_dtype_shapes_nonfinite_types_and_labels_rejected(self):
        model = fixture("early")
        for key,value in (("memory",torch.zeros(2,16,640)),("memory",torch.zeros(2,15,128)),
            ("memory",torch.zeros(2,16,128,dtype=torch.float16)),("memory",torch.full((2,16,128),float("nan"))),
            ("memory",torch.zeros(2,16,128,requires_grad=True)),("memory_mask",torch.zeros(2,16)),
            ("memory_mask",torch.full((2,16),.5)),("attention_mask",torch.zeros(2,5)),
            ("input_ids",torch.ones(2,5)),("memory_type_ids",torch.zeros(2,16,dtype=torch.long)),
            ("labels",torch.full((2,5),-100)),("labels",torch.full((2,5),50))):
            with self.subTest(key=key,shape=value.shape):
                inputs = batch();inputs[key]=value
                with self.assertRaises(ValueError):model(**inputs)
        with self.assertRaises(ValueError):model(**batch(),message_mode="shuffle")
        with self.assertRaises(ValueError):fixture("bad")

    def test_diagnostics_and_main_attention_weight_path(self):
        model = fixture("mid", "near_unit"); inputs = batch()
        with patch.object(model.reader.cross_attention,"forward",wraps=model.reader.cross_attention.forward) as attention:
            model(**inputs)
            self.assertFalse(attention.call_args.kwargs["need_weights"])
        d = model.diagnostics(reset=True)
        self.assertEqual(d["reader_forward_calls"],1)
        self.assertEqual(d["active_text_positions"],8)
        self.assertEqual(d["realized_padded_text_queries"],10)
        self.assertEqual(d["realized_logical_message_payload_bytes"],2*8192)
        self.assertEqual(d["realized_written_delta_bytes"],2*5*24*4)
        self.assertGreater(d["gated_delta_l2_per_token_rms"],0)
        self.assertGreater(d["raw_delta_l2_per_token_rms"],0)
        self.assertGreater(d["hidden_l2_per_token_rms"],0)
        self.assertEqual(model.diagnostics()["reader_forward_calls"],0)
        hidden = torch.randn(2,5,24)
        diagnostic = model.reader.attention_diagnostic(hidden,inputs["memory"],inputs["memory_mask"])
        self.assertEqual(diagnostic["attention_weights"].shape,(2,4,5,16))
        receipt = model.communication_report(2,5)
        self.assertEqual(receipt["message_payload_bytes"],2*16*128*4)
        self.assertEqual(receipt["attention_edges_including_heads"],2*4*5*16)
        self.assertEqual(receipt["written_delta_bytes"],2*5*24*4)

    def test_sender_no_parameters_fixed_rng_and_exact_documented_order(self):
        torch.manual_seed(912);before=torch.get_rng_state().clone()
        sender = FixedMessageSender();after=torch.get_rng_state()
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in sender.parameters()),0)
        expected = torch.randn(640,128,generator=torch.Generator().manual_seed(314159))/math.sqrt(640)
        torch.testing.assert_close(sender.projection,expected,rtol=0,atol=0)
        peptide,target = torch.randn(5,640),torch.randn(23,640)
        normalized = [F.layer_norm(x,(640,)) for x in (peptide,target)]
        pooled = torch.cat([F.adaptive_avg_pool1d(x.T[None],8)[0].T for x in normalized])
        expected = F.layer_norm(pooled @ sender.projection,(128,))
        torch.testing.assert_close(sender.encode(peptide,target),expected,rtol=0,atol=0)
        self.assertEqual(sender.capacity_report()["message_bytes"],8192)
        self.assertFalse(sender.encode(peptide.requires_grad_(),target).requires_grad)

    def test_balanced_molecule_separation_and_global_extension(self):
        sender = FixedMessageSender();peptide,target = torch.randn(7,640),torch.randn(27,640)
        before = sender.encode(peptide,target);after=sender.encode(peptide,target.flip(1))
        torch.testing.assert_close(before[:8],after[:8],rtol=0,atol=0)
        self.assertGreater(float((before[8:]-after[8:]).abs().max()),.01)
        joined=torch.cat([F.layer_norm(x,(640,))for x in(peptide,target)])
        pooled=F.adaptive_avg_pool1d(joined.T[None],16)[0].T
        expected=F.layer_norm(pooled @ sender.projection,(128,))
        torch.testing.assert_close(sender.encode(peptide,target,"global16"),expected,rtol=0,atol=0)
        self.assertNotIn("gold",inspect.signature(sender.encode).parameters)
        with self.assertRaises(TypeError):sender.encode(peptide,target,gold=4.)

    def test_neutral_is_fixed_slotwise_train_mean_no_sample_side_channel(self):
        messages=torch.randn(7,16,128)
        neutral=neutral_message(messages)
        torch.testing.assert_close(neutral,messages.mean(0),rtol=0,atol=0)
        expanded=expand_neutral(neutral,3)
        torch.testing.assert_close(expanded[0],expanded[2],rtol=0,atol=0)
        expanded[0].zero_()
        torch.testing.assert_close(expanded[1],neutral,rtol=0,atol=0)
        for invalid in (torch.zeros(0,16,128),torch.zeros(2,16,640),torch.full((2,16,128),float("inf"))):
            with self.assertRaises(ValueError):neutral_message(invalid)
        with self.assertRaises(ValueError):expand_neutral(neutral,0)

    def test_sender_shape_and_finite_guards_and_backbone_mutation_guard(self):
        sender=FixedMessageSender();target=torch.randn(9,640)
        for invalid in (torch.empty(0,640),torch.zeros(4,128),torch.ones(4,640,dtype=torch.long),
                        torch.full((4,640),float("nan"))):
            with self.assertRaises(ValueError):sender.encode(invalid,target)
        with self.assertRaises(ValueError):sender.encode(target,target,"unregistered")
        model=fixture("early")
        next(model.base_lm.parameters()).requires_grad_(True)
        with self.assertRaisesRegex(ValueError,"Backbone must remain frozen"):
            model(**batch())

    def test_sender_and_reader_precision_survives_outer_autocast(self):
        sender=FixedMessageSender();peptide,target=torch.randn(5,640),torch.randn(19,640)
        expected=sender.encode(peptide,target)
        reader=MatchedReader(hidden_dim=24,gate_init="near_unit")
        inputs=batch();hidden=torch.randn(2,5,24)
        expected_delta=reader(hidden,inputs["memory"],inputs["memory_mask"])
        with torch.autocast(device_type="cpu",dtype=torch.bfloat16):
            actual=sender.encode(peptide,target)
            actual_delta=reader(hidden,inputs["memory"],inputs["memory_mask"])
        self.assertEqual(actual.dtype,torch.float32)
        self.assertEqual(actual_delta.dtype,torch.float32)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(actual_delta,expected_delta,rtol=0,atol=0)


if __name__ == "__main__":
    unittest.main()
