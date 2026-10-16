"""Trainable residue connectors around a frozen causal language model.

These are controlled architecture baselines, not full reproductions of ProtT3.
Input connectors prepend memory embeddings; internal attention adds a residual
after the selected decoder block. ``off`` removes either intervention entirely,
so positions and logits equal the ordinary frozen text model. No KV cache is used.
"""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast


class FrozenConnectorLM(nn.Module):
    """Frozen LM plus one independently trainable connector.

    ``insertion_layer`` is one-based: 14 inserts after decoder block 13.
    Callers supply ordinary text IDs, never reserved or overwritten text slots.
    Memory types are 0 (peptide) and 1 (target); omitted types default to 0.
    Returned logits always have the original text length. Labels use ordinary
    causal shifting on those logits, with -100 and masked text excluded.
    """

    METHODS = {"input_mlp", "qformer_style", "internal_cross_attention", "constant"}
    ALIASES = {"BASE04": "input_mlp", "BASE05": "qformer_style",
               "BASE06": "internal_cross_attention"}

    def __init__(self, base_lm, method, source_dim=640, memory_dim=128,
                 num_memory_tokens=16, insertion_layer=14):
        super().__init__()
        method = self.ALIASES.get(method, method)
        if method not in self.METHODS:
            raise ValueError(f"Unknown connector method: {method}")
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1
               for n in (source_dim, memory_dim, num_memory_tokens)):
            raise ValueError("Connector dimensions must be positive integers")
        self.base_lm = base_lm.eval().requires_grad_(False)
        self.method = method
        self.source_dim = source_dim
        self.memory_dim = memory_dim
        self.num_memory_tokens = num_memory_tokens
        self.insertion_layer = insertion_layer
        hidden = base_lm.get_input_embeddings().weight.shape[-1]
        # Every module below is connector-owned and remains trainable.
        if method == "constant":
            self.constant_tokens = nn.Parameter(torch.randn(num_memory_tokens, hidden) * 0.02)
        else:
            self.type_embedding = nn.Embedding(2, source_dim)
            self.source_norm = nn.LayerNorm(source_dim)
            if method == "input_mlp":
                self.projection = nn.Sequential(nn.Linear(source_dim, memory_dim),
                                                nn.GELU(), nn.Linear(memory_dim, hidden))
            else:
                heads = next(h for h in (4, 2, 1) if memory_dim % h == 0)
                self.source_projection = nn.Linear(source_dim, memory_dim)
                self.cross_attention = nn.MultiheadAttention(memory_dim, heads, dropout=0,
                                                             batch_first=True)
                self.output_projection = nn.Linear(memory_dim, hidden)
                if method == "qformer_style":
                    self.queries = nn.Parameter(torch.randn(num_memory_tokens, memory_dim) * 0.02)
                else:
                    self.query_norm = nn.LayerNorm(hidden)
                    self.query_projection = nn.Linear(hidden, memory_dim)
                    self._decoder_layer()  # Fail at construction on unsupported LM/layout.

    def _decoder_layer(self):
        layers = getattr(getattr(self.base_lm, "model", None), "layers", None)
        if layers is None:
            raise ValueError("Internal attention requires base_lm.model.layers")
        if (not isinstance(self.insertion_layer, int) or isinstance(self.insertion_layer, bool)
                or not 1 <= self.insertion_layer <= len(layers)):
            raise ValueError("Internal insertion layer out of bounds (one-based)")
        return layers[self.insertion_layer - 1]

    def train(self, mode=True):
        super().train(mode)
        self.base_lm.eval()  # Frozen dropout must never change the comparison.
        return self

    def capacity_report(self):
        trainable = {name: p.numel() for name, p in self.named_parameters() if p.requires_grad}
        return {"method": self.method, "source_dim": self.source_dim,
                "memory_dim": self.memory_dim, "num_memory_tokens": self.num_memory_tokens,
                "insertion_layer": self.insertion_layer if self.method == "internal_cross_attention" else None,
                "trainable_parameters": sum(trainable.values()),
                "frozen_parameters": sum(p.numel() for p in self.parameters() if not p.requires_grad),
                "trainable_by_name": trainable, "off_control": "remove_intervention"}

    @staticmethod
    def _binary_mask(mask, shape, name):
        if mask.shape != shape or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{name} must have shape {tuple(shape)} and contain only 0/1")
        return mask.bool()

    def _validate(self, input_ids, attention_mask, memory, memory_mask, memory_type_ids, labels):
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a nonempty [batch, text] tensor")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must have dtype int64")
        text_mask = self._binary_mask(attention_mask, input_ids.shape, "attention_mask")
        if not torch.all(text_mask.any(dim=1)):
            raise ValueError("Every sample needs an unmasked text token")
        vocab = self.base_lm.get_input_embeddings().weight.shape[0]
        if torch.any((input_ids < 0) | (input_ids >= vocab)):
            raise ValueError("Text token outside vocabulary")
        if (memory.ndim != 3 or memory.shape[0] != input_ids.shape[0]
                or memory.shape[1] == 0 or memory.shape[2] != self.source_dim):
            raise ValueError("memory must have shape [batch, nonempty residues, source_dim]")
        if not memory.is_floating_point() or not torch.isfinite(memory).all():
            raise ValueError("memory must be finite floating point")
        source_mask = self._binary_mask(memory_mask, memory.shape[:2], "memory_mask")
        if not torch.all(source_mask.any(dim=1)):
            raise ValueError("Every sample needs an unmasked memory residue")
        if memory_type_ids is None:
            memory_type_ids = torch.zeros_like(memory_mask, dtype=torch.long)
        if (memory_type_ids.shape != memory.shape[:2] or memory_type_ids.dtype != torch.long
                or torch.any((memory_type_ids < 0) | (memory_type_ids > 1))):
            raise ValueError("memory_type_ids must be int64 [batch, residues] containing 0/1")
        tensors = (attention_mask, memory, memory_mask, memory_type_ids)
        if any(t.device != input_ids.device for t in tensors):
            raise ValueError("All inputs must share a device")
        if labels is not None:
            if (labels.shape != input_ids.shape or labels.dtype != torch.long
                    or labels.device != input_ids.device
                    or torch.any((labels != -100) & ((labels < 0) | (labels >= vocab)))):
                raise ValueError("labels must be int64 text-shaped token IDs or -100")
        return text_mask, source_mask, memory_type_ids

    def _source(self, memory, types):
        memory = memory.to(self.type_embedding.weight.dtype)
        return self.source_norm(memory + self.type_embedding(types))

    def _masked_pool(self, source, mask):
        # Adaptive average pooling on each sample's valid ordered residues.
        # No pooling parameters and no contribution from padding, even in holes.
        return torch.stack([F.adaptive_avg_pool1d(row[valid].T.unsqueeze(0), self.num_memory_tokens)
                            .squeeze(0).T for row, valid in zip(source, mask)])

    def _prefix(self, memory, mask, types):
        if self.method == "constant":
            return self.constant_tokens.unsqueeze(0).expand(memory.shape[0], -1, -1)
        source = self._source(memory, types)
        if self.method == "input_mlp":
            return self.projection(self._masked_pool(source, mask))
        keys = self.source_projection(source)
        queries = self.queries.unsqueeze(0).expand(memory.shape[0], -1, -1)
        message, _ = self.cross_attention(queries, keys, keys, key_padding_mask=~mask,
                                          need_weights=False)
        return self.output_projection(message)

    @contextmanager
    def _internal_hook(self, memory, source_mask, types, text_mask):
        keys = self.source_projection(self._source(memory, types))

        def inject(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            query = self.query_projection(self.query_norm(hidden.to(self.query_norm.weight.dtype)))
            message, _ = self.cross_attention(query, keys, keys, key_padding_mask=~source_mask,
                                              need_weights=False)
            delta = self.output_projection(message).to(hidden.dtype)
            updated = hidden + delta * text_mask.unsqueeze(-1).to(hidden.dtype)
            if not torch.isfinite(updated).all():
                raise ValueError("Internal connector produced nonfinite hidden states")
            return (updated,) + output[1:] if isinstance(output, tuple) else updated

        handle = self._decoder_layer().register_forward_hook(inject)
        try:
            yield
        finally:
            handle.remove()

    def forward(self, input_ids, attention_mask, memory, memory_mask,
                memory_type_ids=None, labels=None, message_mode="correct"):
        if message_mode not in ("correct", "off"):
            raise ValueError("message_mode must be correct or off")
        text_mask, source_mask, types = self._validate(
            input_ids, attention_mask, memory, memory_mask, memory_type_ids, labels)
        self.base_lm.eval()
        prefix_length = 0
        if message_mode == "correct" and self.method != "internal_cross_attention":
            prefix = self._prefix(memory, source_mask, types)
            embeds = self.base_lm.get_input_embeddings()(input_ids)
            prefix_length = prefix.shape[1]
            full_mask = torch.cat((torch.ones_like(text_mask[:, :1]).expand(-1, prefix_length), text_mask), 1)
            kwargs = {"inputs_embeds": torch.cat((prefix.to(embeds.dtype), embeds), 1),
                      "attention_mask": full_mask.long()}
        else:
            kwargs = {"input_ids": input_ids, "attention_mask": text_mask.long()}
        kwargs.update(position_ids=(kwargs["attention_mask"].cumsum(-1) - 1).clamp_min(0),
                      use_cache=False, return_dict=True)
        if message_mode == "correct" and self.method == "internal_cross_attention":
            with self._internal_hook(memory, source_mask, types, text_mask):
                output = self.base_lm(**kwargs)
        else:
            output = self.base_lm(**kwargs)
        logits = output.logits[:, prefix_length:]
        if not torch.isfinite(logits).all():
            raise ValueError("Language model produced nonfinite logits")
        # Keep a zero-gradient graph for off-mode backward without touching base.
        if message_mode == "off":
            parameter = next(p for p in self.parameters() if p.requires_grad)
            logits = logits + parameter.reshape(-1)[0].to(logits.dtype) * 0
        loss = None
        if labels is not None:
            shifted = labels[:, 1:].masked_fill(~text_mask[:, 1:], -100)
            if not torch.any(shifted != -100):
                raise ValueError("labels need at least one unmasked causal target after the first token")
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                   shifted.reshape(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    @torch.no_grad()
    def greedy_generate(self, input_ids, attention_mask, memory, memory_mask,
                        memory_type_ids=None, max_new_tokens=16, eos_token_id=None,
                        pad_token_id=None, message_mode="correct"):
        """Deterministic greedy text tokens only, with bounded no-cache decoding.

        Batch rows that finish early return padding after EOS. Both left and
        right padded prompts use the last active token and cumulative positions.
        """
        if (not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool)
                or not 1 <= max_new_tokens <= 256):
            raise ValueError("max_new_tokens must be an integer in [1, 256]")
        config = self.base_lm.generation_config
        if eos_token_id is None:
            eos_token_id = config.eos_token_id
        eos_values = [] if eos_token_id is None else (
            list(eos_token_id) if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
        if pad_token_id is None:
            pad_token_id = config.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_values[0] if eos_values else 0
        vocab = self.base_lm.get_input_embeddings().weight.shape[0]
        if any(not isinstance(i, int) or not 0 <= i < vocab for i in eos_values + [pad_token_id]):
            raise ValueError("EOS/padding IDs must be valid vocabulary integers")
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        ids, mask = input_ids, attention_mask
        generated = []
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                output = self(ids, mask, memory, memory_mask, memory_type_ids,
                              message_mode=message_mode)
                positions = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
                last = positions.masked_fill(~mask.bool(), -1).max(-1).values
                token = output.logits[torch.arange(ids.shape[0], device=ids.device), last].argmax(-1)
                token = torch.where(finished, torch.full_like(token, pad_token_id), token)
                generated.append(token)
                active = ~finished
                for eos in eos_values:
                    finished |= active & (token == eos)
                if finished.all():
                    break
                ids = torch.cat((ids, token[:, None]), 1)
                mask = torch.cat((mask, active[:, None].to(mask.dtype)), 1)
        finally:
            self.train(was_training)
        return torch.stack(generated, 1)
