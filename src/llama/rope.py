"""Rotary Position Embeddings (Su et al., RoFormer) from first principles.

For channel pair l with frequency w_l = base^(-2l/d_rot), position m rotates
(x_{2l}, x_{2l+1}) by angle m*w_l.  Writing R_m = blockdiag(R(m w_l)),
R_m^T R_n = R_{n-m} because 2D rotations commute and R(a)^T = R(-a), so

    (R_m q)^T (R_n k) = q^T R_m^T R_n k = q^T R_{n-m} k,

i.e. the attention logit depends on positions only through n - m.

Two memory layouts of the "pairs" are supported:
  * "interleaved": pairs (0,1),(2,3),...  (Meta's reference llama code, complex view)
  * "half":        pairs (i, i + d_rot/2) (Hugging Face `rotate_half` convention)
They are equivalent up to a fixed permutation of the head dimensions, which
is tested in tests/test_rope.py.  Partial rotary (rotary_dim < head_dim)
rotates only the first `rotary_dim` channels.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def inverse_frequencies(rotary_dim: int, base: float, scaling: Optional[dict] = None) -> torch.Tensor:
    inv = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim))
    if scaling is None:
        return inv.float()
    kind = scaling.get("type")
    if kind == "linear":
        return (inv / float(scaling["factor"])).float()
    if kind == "llama3":
        # LLaMA 3.1 frequency-dependent scaling.
        factor = float(scaling.get("factor", 8.0))
        low = float(scaling.get("low_freq_factor", 1.0))
        high = float(scaling.get("high_freq_factor", 4.0))
        orig = float(scaling.get("original_max_position_embeddings", 8192))
        low_wl, high_wl = orig / low, orig / high
        wavelen = 2 * math.pi / inv
        out = torch.where(wavelen > low_wl, inv / factor, inv)
        smooth = (orig / wavelen - low) / (high - low)
        mid = (wavelen <= low_wl) & (wavelen >= high_wl)
        out = torch.where(mid, (1 - smooth) * inv / factor + smooth * inv, out)
        return out.float()
    raise ValueError(f"unknown rope scaling type {kind}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, layout: str = "half") -> torch.Tensor:
    """Rotate the first 2*cos.shape[-1] channels of x.

    x: [..., T, head_dim]; cos/sin: broadcastable to [..., T, rotary_dim/2].
    """
    rd = 2 * cos.shape[-1]
    x_rot, x_pass = x[..., :rd], x[..., rd:]
    if layout == "half":
        c = torch.cat((cos, cos), dim=-1)
        s = torch.cat((sin, sin), dim=-1)
        out = x_rot * c + rotate_half(x_rot) * s
    elif layout == "interleaved":
        xe, xo = x_rot[..., 0::2], x_rot[..., 1::2]
        re = xe * cos - xo * sin
        ro = xe * sin + xo * cos
        out = torch.stack((re, ro), dim=-1).flatten(-2)
    else:
        raise ValueError(layout)
    return torch.cat((out.to(x.dtype), x_pass), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, base: float = 10_000.0, scaling: Optional[dict] = None, layout: str = "half") -> None:
        super().__init__()
        self.rotary_dim = rotary_dim
        self.layout = layout
        self.register_buffer("inv_freq", inverse_frequencies(rotary_dim, base, scaling), persistent=False)

    def angles(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids: [B, T] (integers, may include offsets) -> cos, sin [B, 1, T, rd/2]."""
        theta = position_ids.float()[..., None] * self.inv_freq  # [B, T, rd/2]
        return theta.cos()[:, None], theta.sin()[:, None]

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.angles(position_ids)
        return apply_rotary(q, cos, sin, self.layout), apply_rotary(k, cos, sin, self.layout)
