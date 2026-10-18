"""Task-free protein-memory injection into a frozen causal language model.

The caller constructs ordinary question text and answer-only language-model
targets. TEXT uses the identical M3 module with TRAIN-mean memory supplied by
the caller; this module never computes molecular attributes or fitted means.
"""
from collections import OrderedDict
import math

import torch
from torch import nn
from torch.nn import functional as F


def _positions(count, width):
    positions = torch.arange(count, dtype=torch.float32)[:, None]
    scales = torch.exp(torch.arange(0, width, 2, dtype=torch.float32)
                       * (-math.log(10000.0) / width))
    result = torch.zeros(count, width, dtype=torch.float32)
    result[:, 0::2] = torch.sin(positions * scales)
    result[:, 1::2] = torch.cos(positions * scales[:width // 2])
    return result


class SharedMemoryLM(nn.Module):
    """Shared generative decoder with two eight-slot protein memory views."""

    def __init__(self, base_lm, variant="M3", hidden_dim=2560, esm_dim=640,
                 memory_dim=128, num_heads=4, hook_layer=17):
        super().__init__()
        if variant not in ("M0", "M3", "TEXT"):
            raise ValueError("variant must be M0, M3, or TEXT")
        layers = getattr(getattr(base_lm, "model", None), "layers", None)
        if layers is None or not 0 <= hook_layer < len(layers):
            raise ValueError("LM must expose model.layers[hook_layer]")
        if getattr(base_lm, "is_gradient_checkpointing", False):
            raise ValueError("Scoped-hook training requires checkpointing disabled")
        if min(hidden_dim, esm_dim, memory_dim, num_heads) < 1 or memory_dim % num_heads:
            raise ValueError("Invalid dimensions or attention head count")
        if base_lm.get_input_embeddings().weight.shape[-1] != hidden_dim:
            raise ValueError("hidden_dim does not match LM embeddings")
        self.base_lm = base_lm.eval().requires_grad_(False)
        self.variant = variant
        self.hidden_dim, self.esm_dim, self.memory_dim = hidden_dim, esm_dim, memory_dim
        self.num_heads, self.hook_layer = num_heads, hook_layer
        self.projections = nn.ModuleList([
            nn.Linear(esm_dim, memory_dim, dtype=torch.float32) for _ in range(2)])
        self.source_embedding = nn.Embedding(2, memory_dim, dtype=torch.float32)
        self.query_norm = nn.LayerNorm(hidden_dim, dtype=torch.float32)
        self.query_projection = nn.Linear(hidden_dim, memory_dim, dtype=torch.float32)
        self.reader = nn.MultiheadAttention(memory_dim, num_heads, dropout=0,
                                            batch_first=True, dtype=torch.float32)
        self.view_router = nn.Linear(memory_dim, 2, dtype=torch.float32)
        nn.init.zeros_(self.view_router.weight)
        nn.init.zeros_(self.view_router.bias)
        self.output_projection = nn.Linear(memory_dim, hidden_dim, dtype=torch.float32)
        self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.register_buffer("slot_positions", _positions(8, memory_dim))
        self.register_buffer("static_anchor", torch.linspace(-1., 1., hidden_dim))
        self._in_forward = False

    def train(self, mode=True):
        super().train(mode)
        self.base_lm.eval()
        return self

    def _own_tensors(self):
        """Yield adapter parameters/buffers without materializing an LM state."""
        for name, tensor in self.named_parameters():
            if not name.startswith("base_lm."):
                yield name, tensor
        for name, tensor in self.named_buffers():
            if not name.startswith("base_lm."):
                yield name, tensor

    def trainable_state_dict(self):
        return OrderedDict((name, tensor.detach().cpu().clone())
                           for name, tensor in self._own_tensors())

    def load_trainable_state_dict(self, state):
        expected = dict(self._own_tensors())
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError("Adapter checkpoint tensor names differ")
        for name, tensor in expected.items():
            value = state[name]
            if (not isinstance(value, torch.Tensor) or value.shape != tensor.shape
                    or value.dtype != tensor.dtype or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid adapter tensor: {name}")
        with torch.no_grad():
            for name, tensor in expected.items():
                tensor.copy_(state[name].to(device=tensor.device))

    def capacity_report(self):
        counts = {name: p.numel() for name, p in self.named_parameters() if p.requires_grad}
        return {"variant": self.variant, "trainable_parameters": sum(counts.values()),
                "trainable_by_name": counts, "dtype": "torch.float32",
                "logical_distinct_slots": 16,
                "logical_bank_payload_bytes_per_sample": 16 * self.memory_dim * 4,
                "extra_lm_positions": 0, "task_specific_parameters": 0,
                "hook_layer_zero_based": self.hook_layer,
                "hook_after_block_one_based": self.hook_layer + 1}

    def _validate(self, ids, mask, end, memory, off):
        if ids.ndim != 2 or ids.dtype != torch.long or min(ids.shape) < 1:
            raise ValueError("input_ids must be nonempty int64 [B,T]")
        batch, length = ids.shape
        if (mask.shape != ids.shape or mask.device != ids.device
                or not torch.all((mask == 0) | (mask == 1))):
            raise ValueError("attention_mask must be binary [B,T] on text device")
        # The public interface supports left padding and an active contiguous suffix.
        if torch.any(mask[:, 1:].long() < mask[:, :-1].long()):
            raise ValueError("Only left-padded contiguous text is supported")
        if (end.shape != (batch,) or end.dtype != torch.long or end.device != ids.device
                or torch.any((end < 0) | (end >= length))):
            raise ValueError("prompt_end_index must be int64 [B] within text")
        if not mask[torch.arange(batch, device=ids.device), end].bool().all():
            raise ValueError("prompt_end_index must identify an active token")
        if getattr(self.base_lm, "is_gradient_checkpointing", False):
            raise ValueError("Gradient checkpointing must stay disabled")
        if any(p.requires_grad or p.grad is not None for p in self.base_lm.parameters()):
            raise ValueError("Backbone must remain frozen and gradient-free")
        if any(p.dtype != torch.float32 for n, p in self.named_parameters()
               if not n.startswith("base_lm.")):
            raise ValueError("Adapter parameters must stay float32")
        if off:
            return
        if not isinstance(memory, dict) or set(memory) != {"p15", "p30"}:
            raise ValueError("memory_inputs must contain exactly p15 and p30")
        for name in ("p15", "p30"):
            value = memory[name]
            if (not isinstance(value, torch.Tensor) or value.shape != (batch, 8, self.esm_dim)
                    or not value.is_floating_point() or value.device != ids.device
                    or value.requires_grad or not torch.isfinite(value).all()):
                raise ValueError(f"{name} must be detached finite [B,8,esm_dim] on text device")

    def _inject(self, hidden, memory, mask, end):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            query = self.query_projection(self.query_norm(hidden.float()))
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            anchor = (self.query_projection(self.query_norm(self.static_anchor))
                      .expand(hidden.shape[0], -1) if self.variant == "M0"
                      else query[rows, end])
            weights = self.view_router(anchor).softmax(-1)
            sources = ("p30", "p30") if self.variant == "M0" else ("p15", "p30")
            message = torch.zeros_like(query)
            for index, name in enumerate(sources):
                slots = self.projections[index](F.layer_norm(memory[name].detach().float(),
                                                            (self.esm_dim,)))
                slots = slots + self.source_embedding.weight[index] + self.slot_positions
                read = self.reader(query, slots, slots, need_weights=False)[0]
                message = message + read * weights[:, index, None, None]
            delta = (self.alpha.tanh() * self.output_projection(message)).to(hidden.dtype)
            # Routing reads the prompt end. Applying its result to earlier prompt
            # positions would leak future tokens through subsequent causal layers.
            active = mask.bool() & (torch.arange(hidden.shape[1], device=hidden.device)
                                    [None, :] >= end[:, None])
            updated = hidden + delta * active[..., None].to(delta.dtype)
        if not torch.isfinite(updated).all():
            raise ValueError("Memory produced nonfinite hidden states")
        return updated, weights

    def forward(self, input_ids, attention_mask, prompt_end_index, memory_inputs=None,
                off=False):
        if self._in_forward:
            raise RuntimeError("Concurrent/reentrant scoped-hook forwards unsupported")
        self._validate(input_ids, attention_mask, prompt_end_index, memory_inputs, off)
        self.base_lm.eval()
        captured = {}
        handle = None
        self._in_forward = True
        try:
            if not off:
                def inject(_module, _args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    updated, weights = self._inject(hidden, memory_inputs,
                                                    attention_mask, prompt_end_index)
                    captured["routes"] = weights
                    return (updated,) + output[1:] if isinstance(output, tuple) else updated
                handle = self.base_lm.model.layers[self.hook_layer].register_forward_hook(inject)
            position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
            output = self.base_lm(input_ids=input_ids, attention_mask=attention_mask,
                                  position_ids=position_ids, use_cache=False, return_dict=True)
            if not off and "routes" not in captured:
                raise RuntimeError("LM did not execute configured memory layer")
            return {"logits": output.logits, "routes": captured.get("routes"),
                    "gate": self.alpha.tanh()}
        finally:
            if handle is not None:
                handle.remove()
            self._in_forward = False
