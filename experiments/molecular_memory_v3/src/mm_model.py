"""Trainable molecular memory around a frozen causal LM; no model loading here."""
from collections import OrderedDict
import math

import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = {
    "M0": (False, False, False),
    "M1": (True, False, False),
    "M2": (True, True, False),
    "M3": (True, True, True),
    "A_L": (False, True, True),
    "A_Q": (True, False, True),
}


def _positions(length, width):
    position = torch.arange(length, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, width, 2, dtype=torch.float32)
                          * (-math.log(10000.0) / width))
    result = torch.zeros(length, width, dtype=torch.float32)
    result[:, 0::2] = torch.sin(position * frequency)
    result[:, 1::2] = torch.cos(position * frequency[:width // 2])
    return result


class RelationBank(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=0,
                                               batch_first=True, dtype=torch.float32)
        self.mlp = nn.Sequential(nn.Linear(3 * width, width, dtype=torch.float32), nn.GELU(),
                                 nn.Linear(width, width, dtype=torch.float32))

    def forward(self, peptide, partner):
        related, _ = self.attention(peptide, partner, partner, need_weights=False)
        return self.mlp(torch.cat((peptide, related, peptide * related), dim=-1))


class MolecularMemoryLM(nn.Module):
    """Six content controls share the same fully registered parameter graph.

    memory_inputs contains p15/p30/t15/t30 [B,8,esm_dim], condition_ids
    int64 [B,10], observed bool [B,10]. Missing conditions are real tokens,
    not padding. All tensors share the input_ids device. The raw ESM tensors
    are frozen inputs. There is no per-example sequence truncation here.
    """
    def __init__(self, base_lm, variant="M3", hidden_dim=1536, esm_dim=640,
                 memory_dim=128, num_heads=4, hook_layer=13):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("Unknown molecular memory variant")
        layers = getattr(getattr(base_lm, "model", None), "layers", None)
        norm = getattr(getattr(base_lm, "model", None), "norm", None)
        if layers is None or norm is None or not 0 <= hook_layer < len(layers):
            raise ValueError("LM needs model.layers[hook_layer] and final model.norm")
        if getattr(base_lm, "is_gradient_checkpointing", False):
            raise ValueError("Hook-based training requires gradient checkpointing disabled")
        if hidden_dim != base_lm.get_input_embeddings().weight.shape[-1]:
            raise ValueError("hidden_dim does not match LM embeddings")
        if memory_dim % num_heads or min(hidden_dim, esm_dim, memory_dim, num_heads) < 1:
            raise ValueError("Invalid dimensions or head count")
        self.base_lm = base_lm.eval().requires_grad_(False)
        self.variant = variant
        self.hidden_dim, self.esm_dim, self.memory_dim = hidden_dim, esm_dim, memory_dim
        self.num_heads, self.hook_layer = num_heads, hook_layer
        self.projections = nn.ModuleList([nn.Linear(esm_dim, memory_dim, dtype=torch.float32) for _ in range(2)])
        self.type_embedding = nn.Embedding(4, memory_dim, dtype=torch.float32)  # p/t/c/r
        self.source_embedding = nn.Embedding(2, memory_dim, dtype=torch.float32)  # logical view
        self.condition_embedding = nn.Embedding(4096, memory_dim, dtype=torch.float32)
        self.missing_embedding = nn.Embedding(10, memory_dim, dtype=torch.float32)
        self.query_norm = nn.LayerNorm(hidden_dim, dtype=torch.float32)
        self.query_projection = nn.Linear(hidden_dim, memory_dim, dtype=torch.float32)
        self.relations = nn.ModuleList([RelationBank(memory_dim, num_heads) for _ in range(2)])
        self.readers = nn.ModuleDict({name: nn.MultiheadAttention(
            memory_dim, num_heads, dropout=0, batch_first=True, dtype=torch.float32)
            for name in ("p", "t", "c", "r")})
        self.view_router = nn.Linear(memory_dim, 2, dtype=torch.float32)
        self.bank_router = nn.Linear(memory_dim, 4, dtype=torch.float32)
        nn.init.zeros_(self.view_router.weight)
        nn.init.zeros_(self.view_router.bias)
        nn.init.zeros_(self.bank_router.weight)
        nn.init.zeros_(self.bank_router.bias)
        self.output_projection = nn.Linear(memory_dim, hidden_dim, dtype=torch.float32)
        self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.numeric_head = nn.Linear(hidden_dim, 1, dtype=torch.float32)
        self.register_buffer("slot_positions", _positions(10, memory_dim))
        # Nonconstant before LN: unlike all-ones, this cannot normalize to zero.
        self.register_buffer("static_anchor", torch.linspace(-1., 1., hidden_dim))
        self._latest_routes = None
        self._ledger = self._empty_ledger()
        self._in_forward = False

    @staticmethod
    def _empty_ledger():
        return {"forward_calls": 0, "off_forward_calls": 0, "padded_text_queries": 0,
                "active_text_positions": 0, "logical_bank_payload_bytes": 0,
                "logical_selected_raw_fp32_bytes": 0,
                "logical_reader_kv_bytes": 0, "logical_relation_kv_bytes": 0,
                "logical_reader_query_bytes": 0, "logical_relation_query_bytes": 0,
                "reader_calls": 0, "relation_calls": 0,
                "reader_attention_edges_including_heads": 0,
                "relation_attention_edges_including_heads": 0,
                "written_delta_bytes": 0}

    def train(self, mode=True):
        super().train(mode)
        self.base_lm.eval()
        return self

    def trainable_state_dict(self):
        """All non-backbone parameters AND buffers, safe for tensor-only checkpointing."""
        return OrderedDict((name, value.detach().cpu().clone())
                           for name, value in self.state_dict().items()
                           if not name.startswith("base_lm."))

    def load_trainable_state_dict(self, state):
        expected = self.trainable_state_dict()
        if set(state) != set(expected):
            raise ValueError("Non-backbone checkpoint tensor names differ")
        for name, value in state.items():
            if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                    or value.dtype != expected[name].dtype or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid non-backbone tensor: {name}")
        self.load_state_dict({**self.state_dict(), **state}, strict=True)

    def capacity_report(self):
        counts = {name: value.numel() for name, value in self.named_parameters()
                  if value.requires_grad}
        return {"variant": self.variant, "trainable_parameters": sum(counts.values()),
                "trainable_by_name": counts, "dtype": "torch.float32",
                "bank_slots_per_view": {"p": 8, "t": 8, "r": 8},
                "shared_condition_slots": 10, "logical_distinct_slots": 58,
                "extra_lm_positions": 0, "hook_after_block_one_based": self.hook_layer + 1}

    def communication_report(self, batch_size, text_tokens, hidden_element_bytes=4):
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 1
               for x in (batch_size, text_tokens, hidden_element_bytes)):
            raise ValueError("Communication dimensions must be positive integers")
        b, t, d, h = batch_size, text_tokens, self.memory_dim, self.num_heads
        return {"logical_selected_raw_fp32_bytes": b * 32 * self.esm_dim * 4,
                "logical_bank_payload_bytes": b * 58 * d * 4,
                "logical_reader_kv_bytes": b * 2 * 58 * d * 4,
                "logical_relation_kv_bytes": b * 2 * 16 * d * 4,
                "logical_reader_query_bytes": b * 7 * t * d * 4,
                "logical_relation_query_bytes": b * 16 * d * 4,
                "reader_calls": 7, "relation_calls": 2,
                "reader_attention_edges_including_heads": b * h * t * 58,
                "relation_attention_edges_including_heads": b * h * 2 * 8 * 8,
                "written_delta_bytes": b * t * self.hidden_dim * hidden_element_bytes,
                "extra_lm_positions": 0, "kv_cache": False,
                "semantics": "Logical module-boundary tensor sizes, not device traffic or FLOPs"}

    def diagnostics(self, reset=False):
        report = {**self._ledger, "variant": self.variant,
                  "alpha": float(self.alpha.detach().cpu()),
                  "gate_tanh_alpha": float(self.alpha.detach().tanh().cpu()),
                  "routes": None if self._latest_routes is None else {
                      name: value.cpu().tolist() for name, value in self._latest_routes.items()},
                  "communication_semantics": "padded logical tensors; not measured bus traffic"}
        report["active_text_positions"] = int(report["active_text_positions"])
        if reset:
            self._ledger = self._empty_ledger()
            self._latest_routes = None
        return report

    def _validate(self, input_ids, attention_mask, prompt_end_index, memory_inputs, labels, off):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or min(input_ids.shape) < 1:
            raise ValueError("input_ids must be nonempty int64 [B,T]")
        b, t = input_ids.shape
        if (attention_mask.shape != input_ids.shape or attention_mask.device != input_ids.device
                or not torch.all((attention_mask == 0) | (attention_mask == 1))):
            raise ValueError("attention_mask must be binary and text-shaped on same device")
        if (prompt_end_index.shape != (b,) or prompt_end_index.dtype != torch.long
                or prompt_end_index.device != input_ids.device
                or torch.any((prompt_end_index < 0) | (prompt_end_index >= t))):
            raise ValueError("Explicit prompt_end_index must be int64 [B] within text")
        if not attention_mask[torch.arange(b, device=input_ids.device), prompt_end_index].bool().all():
            raise ValueError("prompt_end_index must identify an active text token")
        if getattr(self.base_lm, "is_gradient_checkpointing", False):
            raise ValueError("Gradient checkpointing must stay disabled")
        if any(p.requires_grad or p.grad is not None for p in self.base_lm.parameters()):
            raise ValueError("Backbone must stay frozen and gradient-free")
        if any(p.dtype != torch.float32 for n, p in self.named_parameters()
               if not n.startswith("base_lm.")):
            raise ValueError("All trainable parameters must stay float32")
        if labels is not None:
            if (labels.shape != input_ids.shape or labels.dtype != torch.long
                    or labels.device != input_ids.device):
                raise ValueError("labels must be int64 [B,T] on same device")
            prefix = torch.arange(t, device=input_ids.device)[None, :] <= prompt_end_index[:, None]
            if torch.any((labels != -100) & (prefix | ~attention_mask.bool())):
                raise ValueError("Answer labels may supervise only active post-prompt positions")
        if off:
            return
        required = {"p15", "p30", "t15", "t30", "condition_ids", "observed"}
        if not isinstance(memory_inputs, dict) or set(memory_inputs) != required:
            raise ValueError("memory_inputs must contain exactly the six input-only tensors")
        for name in ("p15", "p30", "t15", "t30"):
            value = memory_inputs[name]
            if (value.shape != (b, 8, self.esm_dim) or not value.is_floating_point()
                    or value.device != input_ids.device or value.requires_grad
                    or not torch.isfinite(value).all()):
                raise ValueError(f"{name} must be frozen finite [B,8,esm_dim] on same device")
        ids, observed = memory_inputs["condition_ids"], memory_inputs["observed"]
        if (ids.shape != (b, 10) or ids.dtype != torch.long or ids.device != input_ids.device
                or torch.any((ids < 0) | (ids >= 4096))):
            raise ValueError("condition_ids must be int64 [B,10] in [0,4096)")
        if (observed.shape != (b, 10) or observed.dtype != torch.bool
                or observed.device != input_ids.device):
            raise ValueError("observed must be bool [B,10]")

    def _banks(self, memory):
        intermediate, _, cross = VARIANTS[self.variant]
        first = "15" if intermediate else "30"
        views = []
        for index, layer in enumerate((first, "30")):
            source = self.source_embedding.weight[index]
            position = self.slot_positions[:8]
            p = self.projections[index](F.layer_norm(memory["p" + layer].float(), (self.esm_dim,)))
            t = self.projections[index](F.layer_norm(memory["t" + layer].float(), (self.esm_dim,)))
            p = p + source + position + self.type_embedding.weight[0]
            t = t + source + position + self.type_embedding.weight[1]
            r = self.relations[index](p, t if cross else p)
            r = r + source + position + self.type_embedding.weight[3]
            views.append({"p": p, "t": t, "r": r})
        ids, observed = memory["condition_ids"], memory["observed"]
        c = torch.where(observed[..., None], self.condition_embedding(ids),
                        self.missing_embedding.weight[None, :, :])
        c = c + self.slot_positions + self.type_embedding.weight[2]
        return views, c

    def _inject(self, hidden, memory, text_mask, prompt_end_index):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            query = self.query_projection(self.query_norm(hidden.float()))
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            if VARIANTS[self.variant][1]:
                anchor = query[rows, prompt_end_index]
            else:
                anchor = self.query_projection(self.query_norm(self.static_anchor)).expand(hidden.shape[0], -1)
            view_weights = self.view_router(anchor).softmax(-1)
            bank_weights = self.bank_router(anchor).softmax(-1)
            views, condition = self._banks(memory)
            messages = []
            for name in ("p", "t", "c", "r"):
                if name == "c":
                    message, _ = self.readers[name](query, condition, condition, need_weights=False)
                else:
                    reads = [self.readers[name](query, view[name], view[name], need_weights=False)[0]
                             for view in views]
                    message = sum(read * view_weights[:, i, None, None] for i, read in enumerate(reads))
                messages.append(message)
            combined = sum(message * bank_weights[:, i, None, None]
                           for i, message in enumerate(messages))
            raw_delta = self.output_projection(combined)
            delta = (self.alpha.tanh() * raw_delta).to(hidden.dtype)
            delta = delta * text_mask[..., None].to(delta.dtype)
            updated = hidden + delta
        if not torch.isfinite(updated).all():
            raise ValueError("Molecular memory produced nonfinite hidden states")
        self._latest_routes = {"view": view_weights.detach(), "bank": bank_weights.detach()}
        self._ledger["forward_calls"] += 1
        self._ledger["padded_text_queries"] += hidden.shape[0] * hidden.shape[1]
        self._ledger["active_text_positions"] += text_mask.detach().sum()
        budget = self.communication_report(hidden.shape[0], hidden.shape[1], hidden.element_size())
        for name in self._ledger:
            if name in budget:
                self._ledger[name] += budget[name]
        return updated, {"view": view_weights, "bank": bank_weights}

    def forward(self, input_ids, attention_mask, prompt_end_index, memory_inputs=None,
                labels=None, off=False):
        if self._in_forward:
            raise RuntimeError("Concurrent/reentrant forwards are unsupported by scoped hooks")
        self._validate(input_ids, attention_mask, prompt_end_index, memory_inputs, labels, off)
        self.base_lm.eval()
        captured, handles = {}, []
        self._in_forward = True
        try:
            def capture_norm(_module, _args, output):
                captured["final"] = output

            handles.append(self.base_lm.model.norm.register_forward_hook(capture_norm))
            if not off:
                def inject(_module, _args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    updated, routes = self._inject(hidden, memory_inputs, attention_mask.bool(), prompt_end_index)
                    captured["routes"] = routes
                    return (updated,) + output[1:] if isinstance(output, tuple) else updated
                handles.append(self.base_lm.model.layers[self.hook_layer].register_forward_hook(inject))
            else:
                self._ledger["off_forward_calls"] += 1
            position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
            output = self.base_lm(input_ids=input_ids, attention_mask=attention_mask,
                                  position_ids=position_ids, use_cache=False, return_dict=True)
            if "final" not in captured:
                raise RuntimeError("LM did not execute the final model.norm")
            rows = torch.arange(input_ids.shape[0], device=input_ids.device)
            with torch.autocast(device_type=input_ids.device.type, enabled=False):
                numeric = self.numeric_head(captured["final"][rows, prompt_end_index].float()).squeeze(-1)
            logits = output.logits
            loss = None
            if labels is not None:
                shifted = labels[:, 1:]
                count = (shifted != -100).sum()
                if not count:
                    raise ValueError("No supervised answer tokens")
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                       shifted.reshape(-1), ignore_index=-100, reduction="sum") / count
            return {"loss": loss, "logits": logits, "numeric": numeric, "numeric_z": numeric,
                    "routes": captured.get("routes"), "gate": self.alpha.tanh()}
        finally:
            for handle in reversed(handles):
                handle.remove()
            self._in_forward = False
