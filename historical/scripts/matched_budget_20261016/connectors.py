"""Same finite-message reader at early or middle frozen-LM states.

No full-residue input or added LM position is supported. The old validated
greedy decoder is reused verbatim; memory now means exactly 16 x 128 float32.
"""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from foundation_20261016.connectors import FrozenConnectorLM as _ValidatedDecoder


class MatchedReader(nn.Module):
    def __init__(self, hidden_dim=1536, memory_dim=128, gate_init="zero"):
        super().__init__()
        if (not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim < 1
                or memory_dim != 128):
            raise ValueError("Reader needs positive hidden_dim and fixed memory_dim128")
        if gate_init not in ("zero", "near_unit"):
            raise ValueError("gate_init must be zero or near_unit")
        self.hidden_dim, self.memory_dim = hidden_dim, memory_dim
        self.query_norm = nn.LayerNorm(hidden_dim, dtype=torch.float32)
        self.query_projection = nn.Linear(hidden_dim, memory_dim, dtype=torch.float32)
        self.cross_attention = nn.MultiheadAttention(memory_dim, 4, dropout=0,
                                                    batch_first=True, dtype=torch.float32)
        self.output_projection = nn.Linear(memory_dim, hidden_dim, dtype=torch.float32)
        self.alpha = nn.Parameter(torch.tensor(0. if gate_init == "zero" else 3., dtype=torch.float32))

    def _read(self, hidden, memory, memory_mask, need_weights=False):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            query = self.query_projection(self.query_norm(hidden.float()))
            message, weights = self.cross_attention(query, memory, memory,
                key_padding_mask=~memory_mask.bool(), need_weights=need_weights,
                average_attn_weights=False)
            return self.output_projection(message), weights

    def forward(self, hidden, memory, memory_mask):
        raw_delta, _ = self._read(hidden, memory, memory_mask, need_weights=False)
        self._raw_delta_sq_by_position = raw_delta.detach().float().square().sum(-1)
        return raw_delta * self.alpha.tanh()

    @torch.no_grad()
    def attention_diagnostic(self, hidden, memory, memory_mask):
        """Separate diagnostic path; training/main generation never requests weights."""
        raw_delta, weights = self._read(hidden, memory, memory_mask, need_weights=True)
        return {"attention_weights": weights.detach(), "raw_delta": raw_delta.detach(),
                "gated_delta": (raw_delta * self.alpha.tanh()).detach()}


class MatchedConnectorLM(nn.Module):
    def __init__(self, base_lm, placement="early", memory_dim=128,
                 num_memory_tokens=16, insertion_layer=14, gate_init="zero"):
        super().__init__()
        if placement not in ("early", "mid"):
            raise ValueError("placement must be early or mid")
        if memory_dim != 128 or num_memory_tokens != 16:
            raise ValueError("Matched core accepts exactly 16 x 128 messages")
        layers = getattr(getattr(base_lm, "model", None), "layers", None)
        if layers is None or not len(layers):
            raise ValueError("Reader requires base_lm.model.layers")
        if (not isinstance(insertion_layer, int) or isinstance(insertion_layer, bool)
                or not 1 <= insertion_layer <= len(layers)):
            raise ValueError("insertion_layer is a one-based valid decoder block")
        if getattr(base_lm, "is_gradient_checkpointing", False):
            raise ValueError("Hook insertion requires gradient checkpointing disabled")
        self.base_lm = base_lm.eval().requires_grad_(False)
        self.placement, self.insertion_layer = placement, insertion_layer
        self.memory_dim, self.num_memory_tokens = memory_dim, num_memory_tokens
        self.reader = MatchedReader(base_lm.get_input_embeddings().weight.shape[-1],
                                    memory_dim, gate_init)
        self._diagnostic_sums = None
        self._diagnostic_forward_calls = 0
        self._physical_queries = 0
        self._message_payload_bytes = 0
        self._written_delta_bytes = 0

    def train(self, mode=True):
        super().train(mode)
        self.base_lm.eval()
        return self

    def reader_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.reader.state_dict().items()}

    def load_reader_state(self, state):
        expected = self.reader.state_dict()
        if set(state) != set(expected):
            raise ValueError("Reader state tensor names differ")
        for key, value in state.items():
            if (not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                    or value.dtype != torch.float32 or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid reader state tensor: {key}")
        self.reader.load_state_dict(state, strict=True)

    def capacity_report(self):
        counts = {name: p.numel() for name, p in self.named_parameters() if p.requires_grad}
        return {"method": "matched_fixed_message_reader", "placement": self.placement,
                "insertion_layer": self.insertion_layer if self.placement == "mid" else 0,
                "num_memory_tokens": 16, "memory_dim": 128,
                "trainable_parameters": sum(counts.values()), "trainable_by_name": counts,
                "trainable_dtype": "torch.float32", "sender_trainable_parameters": 0,
                "frozen_parameters": sum(p.numel() for p in self.parameters() if not p.requires_grad),
                "gate": "tanh(alpha)", "extra_lm_positions": 0,
                "off_control": "remove_intervention"}

    def communication_report(self, batch_size, text_tokens):
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
               for v in (batch_size, text_tokens)):
            raise ValueError("batch_size/text_tokens must be positive padded tensor dimensions")
        element_size = self.base_lm.get_input_embeddings().weight.element_size()
        return {"placement": self.placement, "batch_size": batch_size,
                "padded_text_tokens_per_row": text_tokens, "message_slots": 16,
                "message_width": 128, "message_dtype": "torch.float32",
                "message_payload_bytes": batch_size * 16 * 128 * 4,
                "logical_projected_kv_bytes": batch_size * 2 * 16 * 128 * 4,
                "reader_layers": 1, "reader_calls_per_forward": 1,
                "queries_per_forward": batch_size * text_tokens,
                "attention_edges_including_heads": batch_size * 4 * text_tokens * 16,
                "written_delta_dtype": str(self.base_lm.get_input_embeddings().weight.dtype),
                "written_delta_bytes": batch_size * text_tokens * self.reader.hidden_dim * element_size,
                "extra_lm_positions": 0, "kv_cache": False,
                "meaning": "logical padded tensor sizes per forward; not measured device bus traffic"}

    def _validate(self, input_ids, attention_mask, memory, memory_mask, memory_type_ids, labels):
        if (getattr(self.base_lm, "is_gradient_checkpointing", False)
                or any(p.requires_grad or p.grad is not None for p in self.base_lm.parameters())):
            raise ValueError("Backbone must remain frozen, gradient-free, and without gradient checkpointing")
        if (input_ids.ndim != 2 or min(input_ids.shape) < 1 or input_ids.dtype != torch.long):
            raise ValueError("input_ids must be nonempty int64 [batch,text]")
        text_mask = _ValidatedDecoder._binary_mask(attention_mask, input_ids.shape, "attention_mask")
        if not text_mask.any(dim=1).all():
            raise ValueError("Every row needs an active text token")
        vocab = self.base_lm.get_input_embeddings().weight.shape[0]
        if torch.any((input_ids < 0) | (input_ids >= vocab)):
            raise ValueError("Text token outside vocabulary")
        if (memory.shape != (input_ids.shape[0], 16, 128) or memory.dtype != torch.float32
                or memory.requires_grad or not torch.isfinite(memory).all()):
            raise ValueError("memory must be frozen finite float32 [batch,16,128]; raw residues forbidden")
        mask = _ValidatedDecoder._binary_mask(memory_mask, memory.shape[:2], "memory_mask")
        if not mask.all():
            raise ValueError("All sixteen fixed message slots must be active")
        if memory_type_ids is not None:
            raise ValueError("No residue type IDs are accepted after the fixed sender")
        if any(t.device != input_ids.device for t in (attention_mask, memory, memory_mask)):
            raise ValueError("All inputs must share a device")
        if any(p.dtype != torch.float32 for p in self.reader.parameters()):
            raise ValueError("Reader parameters must remain float32")
        if labels is not None and (labels.shape != input_ids.shape or labels.dtype != torch.long
                or labels.device != input_ids.device
                or torch.any((labels != -100) & ((labels < 0) | (labels >= vocab)))):
            raise ValueError("labels must be int64 text-shaped vocabulary IDs or -100")
        return text_mask, mask

    def _inject(self, hidden, memory, memory_mask, text_mask):
        delta = self.reader(hidden, memory, memory_mask).to(hidden.dtype)
        delta = delta * text_mask.unsqueeze(-1).to(hidden.dtype)
        updated = hidden + delta
        if not torch.isfinite(updated).all():
            raise ValueError("Reader produced nonfinite hidden states")
        with torch.no_grad():
            mask = text_mask.unsqueeze(-1)
            # Scalar sums remain detached on-device until explicitly requested.
            values = torch.stack((hidden.detach().float().square().masked_fill(~mask, 0).sum(),
                                  delta.detach().float().square().sum(),
                                  self.reader._raw_delta_sq_by_position.masked_fill(~text_mask, 0).sum(),
                                  text_mask.sum().float()))
            self._diagnostic_sums = values if self._diagnostic_sums is None else self._diagnostic_sums + values
            self._diagnostic_forward_calls += 1
            self._physical_queries += hidden.shape[0] * hidden.shape[1]
            self._message_payload_bytes += hidden.shape[0] * 16 * 128 * 4
            self._written_delta_bytes += hidden.numel() * hidden.element_size()
        return updated

    def diagnostics(self, reset=False):
        result = {"placement": self.placement, "alpha": float(self.reader.alpha.detach()),
                  "gate_tanh_alpha": float(self.reader.alpha.detach().tanh()),
                  "reader_forward_calls": self._diagnostic_forward_calls,
                  "realized_padded_text_queries": self._physical_queries,
                  "realized_logical_message_payload_bytes": self._message_payload_bytes,
                  "realized_attention_edges_including_heads": self._physical_queries * 4 * 16,
                  "realized_written_delta_bytes": self._written_delta_bytes,
                  "byte_semantics": "module boundary logical tensors; not measured PCI/network bytes or Shannon information"}
        if self._diagnostic_sums is not None:
            hidden_sq, delta_sq, raw_delta_sq, n = self._diagnostic_sums.detach().cpu().tolist()
            result.update({"active_text_positions": int(n),
                "hidden_l2_per_token_rms": (hidden_sq / n) ** .5,
                "gated_delta_l2_per_token_rms": (delta_sq / n) ** .5,
                "raw_delta_l2_per_token_rms": (raw_delta_sq / n) ** .5,
                "raw_delta_over_hidden_l2": (raw_delta_sq / hidden_sq) ** .5 if hidden_sq > 0 else None,
                "gated_delta_over_hidden_l2": (delta_sq / hidden_sq) ** .5 if hidden_sq > 0 else None})
        if reset:
            self._diagnostic_sums = None
            self._diagnostic_forward_calls = 0
            self._physical_queries = 0
            self._message_payload_bytes = 0
            self._written_delta_bytes = 0
        return result

    @contextmanager
    def _hook(self, memory, memory_mask, text_mask):
        if self.placement == "early":
            layer = self.base_lm.model.layers[0]

            def before(_module, args, kwargs):
                if args:
                    return (self._inject(args[0], memory, memory_mask, text_mask),) + args[1:], kwargs
                if "hidden_states" not in kwargs:
                    raise ValueError("Decoder does not expose hidden_states")
                return args, {**kwargs, "hidden_states": self._inject(
                    kwargs["hidden_states"], memory, memory_mask, text_mask)}

            handle = layer.register_forward_pre_hook(before, with_kwargs=True)
        else:
            layer = self.base_lm.model.layers[self.insertion_layer - 1]

            def after(_module, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                updated = self._inject(hidden, memory, memory_mask, text_mask)
                return (updated,) + output[1:] if isinstance(output, tuple) else updated

            handle = layer.register_forward_hook(after)
        try:
            yield
        finally:
            handle.remove()

    def forward(self, input_ids, attention_mask, memory, memory_mask,
                memory_type_ids=None, labels=None, message_mode="correct"):
        if message_mode not in ("correct", "off"):
            raise ValueError("message_mode must be correct or off")
        text_mask, source_mask = self._validate(input_ids, attention_mask, memory,
                                               memory_mask, memory_type_ids, labels)
        self.base_lm.eval()
        kwargs = {"input_ids": input_ids, "attention_mask": text_mask.long(),
                  "position_ids": (text_mask.long().cumsum(-1) - 1).clamp_min(0),
                  "use_cache": False, "return_dict": True}
        if message_mode == "correct":
            with self._hook(memory, source_mask, text_mask):
                output = self.base_lm(**kwargs)
        else:
            output = self.base_lm(**kwargs)
        logits = output.logits
        if not torch.isfinite(logits).all():
            raise ValueError("Language model produced nonfinite logits")
        if message_mode == "off":
            logits = logits + self.reader.alpha.to(logits.dtype) * 0
        loss = None
        if labels is not None:
            shifted = labels[:, 1:].masked_fill(~text_mask[:, 1:], -100)
            if not torch.any(shifted != -100):
                raise ValueError("labels require an active causal target after first token")
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                   shifted.reshape(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def greedy_generate(self, input_ids, attention_mask, memory, memory_mask,
                        memory_type_ids=None, max_new_tokens=24, eos_token_id=None,
                        pad_token_id=None, message_mode="correct"):
        return _ValidatedDecoder.greedy_generate(self, input_ids, attention_mask, memory,
            memory_mask, memory_type_ids, max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id, pad_token_id=pad_token_id, message_mode=message_mode)
