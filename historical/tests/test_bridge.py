import sys
from pathlib import Path
import unittest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from bridge import ResidueBridge, add_message, bridge_forward


class BridgeTests(unittest.TestCase):
    def test_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in ResidueBridge().parameters()), 563201)

    def test_slots_preserve_text_and_gradient(self):
        hidden = torch.randn(2, 9, 8)
        message = torch.randn(2, 3, 8, requires_grad=True)
        result = add_message(hidden, message, 2)
        torch.testing.assert_close(result[:, :2], hidden[:, :2])
        torch.testing.assert_close(result[:, 5:], hidden[:, 5:])
        result.sum().backward()
        torch.testing.assert_close(message.grad, torch.ones_like(message))
        with self.assertRaises(ValueError):
            add_message(hidden, message, 8)

    def test_zero_message_and_backward_all_interfaces(self):
        torch.manual_seed(17)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32,
            intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2)).eval().requires_grad_(False)
        ids = torch.randint(0, 64, (2, 12))
        mask = torch.ones_like(ids)
        labels = ids.clone()
        labels[:, :8] = -100
        reference = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        for layer in (0, 1, 2):
            zero = torch.zeros(2, 3, 32)
            output = bridge_forward(model, ids, mask, zero, 2, layer)
            torch.testing.assert_close(output.logits, reference)
            message = torch.randn(2, 3, 32, requires_grad=True)
            output = bridge_forward(model, ids, mask, message, 2, layer, labels)
            output.loss.backward()
            self.assertTrue(torch.isfinite(message.grad).all())
            self.assertGreater(message.grad.abs().sum().item(), 0)
            self.assertFalse(any(p.grad is not None for p in model.parameters()))
            self.assertTrue(all(len(block._forward_hooks) == 0 for block in model.model.layers))

        # After the final block there is no later attention to carry slot information to answer tokens.
        message = torch.randn(2, 3, 32, requires_grad=True)
        bridge_forward(model, ids, mask, message, 2, 3, labels).loss.backward()
        self.assertEqual(message.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
