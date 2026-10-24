"""SwiGLU FFN (Shazeer, 2020): W_down( SiLU(W_gate x) * (W_up x) ). No biases, as in LLaMA."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))
