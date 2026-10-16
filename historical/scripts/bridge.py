"""Small residue-level adapter; insertion layers are numbered after blocks."""
from contextlib import contextmanager
import torch
from torch import nn


class ResidueBridge(nn.Module):
    def __init__(self, source_dim=640, target_dim=1536, bottleneck=256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(source_dim), nn.Linear(source_dim, bottleneck),
            nn.GELU(), nn.Linear(bottleneck, target_dim), nn.LayerNorm(target_dim),
        )
        self.gate = nn.Parameter(torch.ones(()))

    def forward(self, residues):
        return self.gate * self.projection(residues)


def add_message(hidden, message, start):
    """Add only to the reserved positions, without in-place autograd edits."""
    stop = start + message.shape[1]
    if hidden.shape[0] != message.shape[0] or hidden.shape[2] != message.shape[2]:
        raise ValueError('Batch and hidden dimensions must match')
    if start < 0 or stop > hidden.shape[1]:
        raise ValueError('Molecular slots exceed sequence bounds')
    delta = torch.nn.functional.pad(message.to(hidden.dtype), (0, 0, start, hidden.shape[1] - stop))
    return hidden + delta


@contextmanager
def inject_after(model, layer, message, start):
    """Install a temporary hook; layer=7 means after model.layers[6]."""
    if layer < 1 or layer > len(model.model.layers):
        raise ValueError('Internal insertion layer out of bounds')

    def hook(_module, _args, output):
        if isinstance(output, tuple):
            return (add_message(output[0], message, start),) + output[1:]
        return add_message(output, message, start)

    handle = model.model.layers[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def bridge_forward(model, input_ids, attention_mask, message, start, insertion_layer, labels=None):
    embeds = model.get_input_embeddings()(input_ids)
    kwargs = dict(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels, use_cache=False)
    if insertion_layer == 0:
        return model(**{**kwargs, 'inputs_embeds': add_message(embeds, message, start)})
    with inject_after(model, insertion_layer, message, start):
        return model(**kwargs)
