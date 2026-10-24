"""RMSNorm (Zhang & Sennrich, 2019) as used in LLaMA.

y = x / sqrt(mean(x^2, -1) + eps) * g

Normalization is over the last (feature) dimension. Like the LLaMA reference,
statistics are computed in float32 and the result is cast back to the input
dtype before scaling, which keeps bf16/fp16 training stable.
"""
from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.to(torch.promote_types(dtype, torch.float32))  # never downcast (e.g. float64)
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x32.to(dtype)
