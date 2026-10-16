"""CPU checks for efficient frozen-LM scoring and answer-only gradients."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from transformers import Qwen2Config,Qwen2ForCausalLM

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bridge import ResidueBridge,add_message,inject_after
from real_bridge_model import FrozenBridgeScorer

def fixture(layer):
    torch.manual_seed(17)
    scorer=FrozenBridgeScorer.__new__(FrozenBridgeScorer)
    nn.Module.__init__(scorer)
    scorer.qwen=Qwen2ForCausalLM(Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2)).eval().requires_grad_(False)
    scorer.adapter=ResidueBridge(16,32,8)
    scorer.insertion_layer=layer
    scorer.tokenizer=SimpleNamespace(pad_token_id=0)
    scorer.alias_token_ids=list(range(2,36))
    return scorer

class EfficientReadoutTests(unittest.TestCase):
    def test_candidate_logits_are_exact_frozen_lm_head_rows(self):
        for layer in (0,1,2):
            scorer=fixture(layer)
            features=torch.randn(2,3,16)
            mask=torch.ones(2,3)
            prompts=[[1,2,3,4,5,6,7],[1,2,3,4,5,6,7,8,9]]
            full=scorer(features,mask,prompts,2,full_vocab=True)
            candidates=scorer(features,mask,prompts,2)
            torch.testing.assert_close(candidates,full[:,scorer.alias_token_ids])

    def test_efficient_readout_matches_original_full_sequence_logits(self):
        scorer=fixture(2)
        features=torch.randn(2,3,16)
        mask=torch.ones(2,3)
        prompts=[[1,2,3,4,5,6,7]]*2
        actual=scorer(features,mask,prompts,2,full_vocab=True)
        ids=torch.tensor(prompts)
        message=scorer.adapter(features)
        with inject_after(scorer.qwen,2,message,2):
            expected=scorer.qwen(input_ids=ids,use_cache=False).logits[:,-1].float()
        torch.testing.assert_close(actual,expected)

    def test_full_vocab_loss_weights_biological_samples_equally(self):
        scorer=fixture(0)
        logits=torch.randn(2,64,requires_grad=True)
        losses=scorer.soft_positive_loss(logits,[[0],[1,2]])
        lp=logits.log_softmax(-1)
        torch.testing.assert_close(losses,torch.stack([-lp[0,2],-(lp[1,3]+lp[1,4])/2]))
        losses.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_message_off_preserves_positions_and_blocks_adapter_signal(self):
        scorer=fixture(1)
        features=torch.randn(2,3,16)
        prompts=[[1,2,3,4,5,6,7]]*2
        full=scorer(features,torch.ones(2,3),prompts,2,message_mode='off',full_vocab=True)
        reference=scorer.qwen(input_ids=torch.tensor(prompts),use_cache=False).logits[:,-1].float()
        torch.testing.assert_close(full,reference)
        scorer.soft_positive_loss(full,[[0],[1]]).mean().backward()
        self.assertEqual(sum(p.grad.abs().sum().item() for p in scorer.adapter.parameters()),0)
        self.assertFalse(any(p.grad is not None for p in scorer.qwen.parameters()))

if __name__=='__main__':
    unittest.main()
