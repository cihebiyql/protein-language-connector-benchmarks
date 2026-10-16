"""Parameter-free, label-free fixed 16 x 128 protein messages.

Only residue tensors enter this sender. Molecule-specific normalization precedes
pooling, then the pinned random projection and slot normalization are applied.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class FixedMessageSender(nn.Module):
    def __init__(self, seed=314159, source_dim=640, memory_dim=128,
                 slots_per_molecule=8):
        super().__init__()
        if (source_dim, memory_dim, slots_per_molecule) != (640, 128, 8):
            raise ValueError("The frozen sender requires source640, message128 and 8+8 slots")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("Projection seed must be an integer")
        self.seed = seed
        self.source_dim, self.memory_dim = source_dim, memory_dim
        self.slots_per_molecule = slots_per_molecule
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.register_buffer("projection", torch.randn(source_dim, memory_dim,
            generator=generator, dtype=torch.float32) / math.sqrt(source_dim))

    def _residues(self, value):
        if (not isinstance(value, torch.Tensor) or value.ndim != 2
                or value.shape[0] < 1 or value.shape[1] != self.source_dim
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError("Each molecule requires finite floating [nonempty residues,640]")
        if value.device != self.projection.device:
            raise ValueError("Residues and fixed projection must share a device")
        return F.layer_norm(value.float(), (self.source_dim,))

    @staticmethod
    def _pool(value, count):
        return F.adaptive_avg_pool1d(value.T.unsqueeze(0), count).squeeze(0).T

    @torch.no_grad()
    def encode(self, peptide_residues, target_residues, sender_mode="balanced"):
        peptide, target = self._residues(peptide_residues), self._residues(target_residues)
        if sender_mode == "balanced":
            pooled = torch.cat((self._pool(peptide, 8), self._pool(target, 8)), dim=0)
        elif sender_mode == "global16":
            pooled = self._pool(torch.cat((peptide, target), dim=0), 16)
        else:
            raise ValueError("sender_mode must be balanced or global16")
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            message = F.layer_norm(pooled @ self.projection, (self.memory_dim,))
        if message.shape != (16, 128) or not torch.isfinite(message).all():
            raise ValueError("Invalid fixed message")
        return message.contiguous()

    def forward(self, peptide_residues, target_residues, sender_mode="balanced"):
        return self.encode(peptide_residues, target_residues, sender_mode)

    def capacity_report(self):
        return {"trainable_parameters": 0, "projection_seed": self.seed,
                "projection_shape": [640, 128], "projection_dtype": str(self.projection.dtype),
                "projection_std_rule": "1/sqrt(640)", "message_shape": [16, 128],
                "message_dtype": "torch.float32", "message_bytes": 8192,
                "normalization": "per-residue LN640 -> pool -> R -> per-slot LN128; affine=False eps=1e-5",
                "gold_or_metadata_input": False}


def _message_tensor(messages, batched):
    shape = (16, 128)
    if (not isinstance(messages, torch.Tensor) or messages.dtype != torch.float32
            or messages.ndim != (3 if batched else 2)
            or tuple(messages.shape[-2:]) != shape or not torch.isfinite(messages).all()
            or (batched and messages.shape[0] == 0)):
        raise ValueError("Messages must be finite float32 with trailing shape [16,128]")


@torch.no_grad()
def neutral_message(train_messages):
    """Caller must supply TRAIN-only messages; this function never reads rows/labels."""
    _message_tensor(train_messages, batched=True)
    return train_messages.mean(dim=0).contiguous()


@torch.no_grad()
def expand_neutral(neutral, batch_size):
    _message_tensor(neutral, batched=False)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    return neutral.unsqueeze(0).expand(batch_size, -1, -1).clone()
