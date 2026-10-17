"""Single-sequence P/C memory adapters and frozen-ESM task controls.

Model loading, labels, task losses other than answer CE, and data fitting belong
to the trainer. No candidate IDs, split identifiers, or targets enter memory.
"""
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .v4_model import _positions


TASKS = {"meltome", "elm", "rt"}


class _CheckpointMixin:
    def trainable_state_dict(self):
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


class SemanticHead(nn.Module):
    """Semantic ranking against the same frozen descriptor vectors in every arm."""
    def __init__(self, query_dim, candidate_vectors, width=128):
        super().__init__()
        vectors = torch.as_tensor(candidate_vectors, dtype=torch.float32).detach().clone()
        if vectors.shape != (31, 1536) or not torch.isfinite(vectors).all():
            raise ValueError("candidate_vectors must be finite float32 [31,1536]")
        self.register_buffer("candidate_vectors", vectors)
        self.query_projection = nn.Linear(query_dim, width, dtype=torch.float32)
        self.candidate_projection = nn.Linear(1536, width, dtype=torch.float32)
        self.temperature = 0.1

    def forward(self, query):
        q = F.normalize(self.query_projection(query.float()), dim=-1)
        c = F.normalize(self.candidate_projection(self.candidate_vectors), dim=-1)
        return q @ c.T / self.temperature


def _memory_validate(memory, batch, esm_dim, device):
    required = {"p15", "p30", "condition_ids", "observed"}
    if not isinstance(memory, dict) or set(memory) != required:
        raise ValueError("memory_inputs must contain exactly p15,p30,condition_ids,observed")
    for name in ("p15", "p30"):
        value = memory[name]
        if (value.shape != (batch, 8, esm_dim) or not value.is_floating_point()
                or value.device != device or value.requires_grad or not torch.isfinite(value).all()):
            raise ValueError(f"{name} must be frozen finite [B,8,esm_dim] on same device")
    ids, observed = memory["condition_ids"], memory["observed"]
    if (ids.shape != (batch, 10) or ids.dtype != torch.long or ids.device != device
            or torch.any((ids < 0) | (ids >= 4096))):
        raise ValueError("condition_ids must be int64 [B,10] in [0,4096)")
    if observed.shape != (batch, 10) or observed.dtype != torch.bool or observed.device != device:
        raise ValueError("observed must be bool [B,10] on same device")


class SingleMemoryLM(_CheckpointMixin, nn.Module):
    def __init__(self, base_lm, task, variant="M3", hidden_dim=1536, esm_dim=640,
                 memory_dim=128, num_heads=4, hook_layer=13, candidate_vectors=None):
        super().__init__()
        if task not in TASKS or variant not in ("M0", "M3"):
            raise ValueError("Unknown task or single-sequence variant")
        layers = getattr(getattr(base_lm, "model", None), "layers", None)
        norm = getattr(getattr(base_lm, "model", None), "norm", None)
        if layers is None or norm is None or not 0 <= hook_layer < len(layers):
            raise ValueError("LM needs model.layers[hook_layer] and final model.norm")
        if getattr(base_lm, "is_gradient_checkpointing", False):
            raise ValueError("Hook training requires gradient checkpointing disabled")
        if hidden_dim != base_lm.get_input_embeddings().weight.shape[-1]:
            raise ValueError("hidden_dim does not match LM embeddings")
        if min(hidden_dim, esm_dim, memory_dim, num_heads) < 1 or memory_dim % num_heads:
            raise ValueError("Invalid dimensions or head count")
        self.base_lm = base_lm.eval().requires_grad_(False)
        self.task, self.variant = task, variant
        self.hidden_dim, self.esm_dim, self.memory_dim = hidden_dim, esm_dim, memory_dim
        self.hook_layer, self.num_heads = hook_layer, num_heads
        self.projections = nn.ModuleList([nn.Linear(esm_dim, memory_dim, dtype=torch.float32)
                                          for _ in range(2)])
        self.type_embedding = nn.Embedding(2, memory_dim, dtype=torch.float32)
        self.source_embedding = nn.Embedding(2, memory_dim, dtype=torch.float32)
        self.condition_embedding = nn.Embedding(4096, memory_dim, dtype=torch.float32)
        self.missing_embedding = nn.Embedding(10, memory_dim, dtype=torch.float32)
        self.query_norm = nn.LayerNorm(hidden_dim, dtype=torch.float32)
        self.query_projection = nn.Linear(hidden_dim, memory_dim, dtype=torch.float32)
        self.readers = nn.ModuleDict({name: nn.MultiheadAttention(memory_dim, num_heads,
            dropout=0, batch_first=True, dtype=torch.float32) for name in ("p", "c")})
        self.view_router = nn.Linear(memory_dim, 2, dtype=torch.float32)
        self.bank_router = nn.Linear(memory_dim, 2, dtype=torch.float32)
        for router in (self.view_router, self.bank_router):
            nn.init.zeros_(router.weight)
            nn.init.zeros_(router.bias)
        self.output_projection = nn.Linear(memory_dim, hidden_dim, dtype=torch.float32)
        self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32))
        if task == "elm":
            self.ranking_head = SemanticHead(hidden_dim, candidate_vectors, memory_dim)
        else:
            self.numeric_head = nn.Linear(hidden_dim, 1, dtype=torch.float32)
        self.register_buffer("slot_positions", _positions(10, memory_dim))
        self.register_buffer("static_anchor", torch.linspace(-1., 1., hidden_dim))
        self._in_forward = False

    def train(self, mode=True):
        super().train(mode)
        self.base_lm.eval()
        return self

    def capacity_report(self):
        counts = {name: p.numel() for name, p in self.named_parameters() if p.requires_grad}
        return {"task": self.task, "variant": self.variant,
                "trainable_parameters": sum(counts.values()), "trainable_by_name": counts,
                "dtype": "torch.float32", "bank_slots_per_view": {"p": 8},
                "shared_condition_slots": 10, "logical_distinct_slots": 26,
                "logical_bank_payload_bytes_per_sample": 26 * self.memory_dim * 4,
                "extra_lm_positions": 0, "hook_layer_zero_based": self.hook_layer,
                "hook_after_block_one_based": self.hook_layer + 1,
                "frozen_description_count": 31 if self.task == "elm" else 0}

    def _validate(self, ids, mask, end, memory, labels, off):
        if ids.ndim != 2 or ids.dtype != torch.long or min(ids.shape) < 1:
            raise ValueError("input_ids must be nonempty int64 [B,T]")
        b, t = ids.shape
        if (mask.shape != ids.shape or mask.device != ids.device
                or not torch.all((mask == 0) | (mask == 1))):
            raise ValueError("attention_mask must be binary and text-shaped")
        if (end.shape != (b,) or end.dtype != torch.long or end.device != ids.device
                or torch.any((end < 0) | (end >= t))):
            raise ValueError("prompt_end_index must be int64 [B] within text")
        if not mask[torch.arange(b, device=ids.device), end].bool().all():
            raise ValueError("prompt_end_index must identify an active token")
        if getattr(self.base_lm, "is_gradient_checkpointing", False):
            raise ValueError("Gradient checkpointing must stay disabled")
        if any(p.requires_grad or p.grad is not None for p in self.base_lm.parameters()):
            raise ValueError("Backbone must stay frozen and gradient-free")
        if any(p.dtype != torch.float32 for n, p in self.named_parameters()
               if not n.startswith("base_lm.")):
            raise ValueError("Adapter and head parameters must stay float32")
        if labels is not None:
            if labels.shape != ids.shape or labels.dtype != torch.long or labels.device != ids.device:
                raise ValueError("labels must be int64 [B,T] on same device")
            prefix = torch.arange(t, device=ids.device)[None, :] <= end[:, None]
            if torch.any((labels != -100) & (prefix | ~mask.bool())):
                raise ValueError("Labels may supervise only active post-prompt positions")
        if not off:
            _memory_validate(memory, b, self.esm_dim, ids.device)

    def _inject(self, hidden, memory, text_mask, end):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            query = self.query_projection(self.query_norm(hidden.float()))
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            anchor = (query[rows, end] if self.variant == "M3" else
                      self.query_projection(self.query_norm(self.static_anchor)).expand(hidden.shape[0], -1))
            view_weights = self.view_router(anchor).softmax(-1)
            bank_weights = self.bank_router(anchor).softmax(-1)
            first = "p15" if self.variant == "M3" else "p30"
            peptide = []
            for i, name in enumerate((first, "p30")):
                p = self.projections[i](F.layer_norm(memory[name].float(), (self.esm_dim,)))
                p = p + self.source_embedding.weight[i] + self.slot_positions[:8] + self.type_embedding.weight[0]
                peptide.append(self.readers["p"](query, p, p, need_weights=False)[0])
            p_message = sum(read * view_weights[:, i, None, None] for i, read in enumerate(peptide))
            condition = torch.where(memory["observed"][..., None],
                self.condition_embedding(memory["condition_ids"]), self.missing_embedding.weight[None])
            condition = condition + self.slot_positions + self.type_embedding.weight[1]
            c_message = self.readers["c"](query, condition, condition, need_weights=False)[0]
            message = (p_message * bank_weights[:, 0, None, None]
                       + c_message * bank_weights[:, 1, None, None])
            delta = (self.alpha.tanh() * self.output_projection(message)).to(hidden.dtype)
            updated = hidden + delta * text_mask[..., None].to(delta.dtype)
        if not torch.isfinite(updated).all():
            raise ValueError("Memory produced nonfinite hidden states")
        return updated, {"view": view_weights, "bank": bank_weights}

    def forward(self, input_ids, attention_mask, prompt_end_index, memory_inputs=None,
                labels=None, off=False):
        if self._in_forward:
            raise RuntimeError("Concurrent/reentrant scoped-hook forwards unsupported")
        self._validate(input_ids, attention_mask, prompt_end_index, memory_inputs, labels, off)
        self.base_lm.eval()
        captured, handles = {}, []
        self._in_forward = True
        try:
            def capture(_module, _args, output):
                captured["final"] = output
            handles.append(self.base_lm.model.norm.register_forward_hook(capture))
            if not off:
                def inject(_module, _args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    updated, routes = self._inject(hidden, memory_inputs, attention_mask.bool(), prompt_end_index)
                    captured["routes"] = routes
                    return (updated,) + output[1:] if isinstance(output, tuple) else updated
                handles.append(self.base_lm.model.layers[self.hook_layer].register_forward_hook(inject))
            position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
            output = self.base_lm(input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, use_cache=False, return_dict=True)
            if "final" not in captured:
                raise RuntimeError("LM did not execute final model.norm")
            rows = torch.arange(input_ids.shape[0], device=input_ids.device)
            prompt_hidden = captured["final"][rows, prompt_end_index].float()
            result = {"logits": output.logits, "prompt_hidden": prompt_hidden,
                      "routes": captured.get("routes"), "gate": self.alpha.tanh(), "loss": None}
            with torch.autocast(device_type=input_ids.device.type, enabled=False):
                if self.task == "elm":
                    result["ranking_logits"] = self.ranking_head(prompt_hidden)
                else:
                    result["numeric_z"] = self.numeric_head(prompt_hidden).squeeze(-1)
            if labels is not None:
                shifted = labels[:, 1:]
                count = (shifted != -100).sum()
                if not count:
                    raise ValueError("No supervised answer tokens")
                result["loss"] = F.cross_entropy(output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]),
                    shifted.reshape(-1), ignore_index=-100, reduction="sum") / count
            return result
        finally:
            for handle in reversed(handles):
                handle.remove()
            self._in_forward = False


class ESMTaskModel(_CheckpointMixin, nn.Module):
    def __init__(self, task, esm_dim=640, memory_dim=128, candidate_vectors=None):
        super().__init__()
        if task not in TASKS or min(esm_dim, memory_dim) < 1:
            raise ValueError("Unknown task or invalid dimensions")
        self.task, self.esm_dim, self.memory_dim = task, esm_dim, memory_dim
        self.condition_embedding = nn.Embedding(4096, memory_dim, dtype=torch.float32)
        self.missing_embedding = nn.Embedding(10, memory_dim, dtype=torch.float32)
        self.mlp = nn.Sequential(nn.Linear(16 * esm_dim + 10 * memory_dim, memory_dim,
            dtype=torch.float32), nn.GELU())
        if task == "elm":
            self.ranking_head = SemanticHead(memory_dim, candidate_vectors, memory_dim)
        else:
            self.numeric_head = nn.Linear(memory_dim, 1, dtype=torch.float32)

    def capacity_report(self):
        counts = {name: p.numel() for name, p in self.named_parameters() if p.requires_grad}
        return {"task": self.task, "variant": "ESM", "trainable_parameters": sum(counts.values()),
                "trainable_by_name": counts, "dtype": "torch.float32", "online_lm": False,
                "uses_esm_layers": [15, 30], "condition_slots": 10,
                "capacity_matching": "actual baseline capacity; not matched to connector",
                "frozen_description_count": 31 if self.task == "elm" else 0}

    def forward(self, memory_inputs):
        if not isinstance(memory_inputs, dict) or "p15" not in memory_inputs:
            raise ValueError("memory_inputs required")
        p15 = memory_inputs["p15"]
        _memory_validate(memory_inputs, p15.shape[0], self.esm_dim, p15.device)
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError("Baseline parameters must stay float32")
        with torch.autocast(device_type=p15.device.type, enabled=False):
            c = torch.where(memory_inputs["observed"][..., None],
                self.condition_embedding(memory_inputs["condition_ids"]), self.missing_embedding.weight[None])
            x = torch.cat((memory_inputs["p15"].float().flatten(1),
                           memory_inputs["p30"].float().flatten(1), c.flatten(1)), dim=-1)
            hidden = self.mlp(x)
            if self.task == "elm":
                return {"ranking_logits": self.ranking_head(hidden)}
            return {"numeric_z": self.numeric_head(hidden).squeeze(-1)}
