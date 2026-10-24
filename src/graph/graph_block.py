"""LLaMA-style graph Transformer block.

    h   = x + Dropout( Attn_G( Norm(x) ) )
    out = h + Dropout( FFN( Norm(h) ) )

Differences from the LM decoder block (src/llama/decoder.py):
  1. attention is over graph neighbourhoods (sparse, segment softmax), not a causal sequence;
     no causal mask is applied (no temporal interpretation for ordinary graph tasks);
  2. positions are graph-aware phases (or none), not token indices;
  3. dropout is used (small labelled sets such as Cora overfit quickly);
  4. Norm / FFN are switchable (RMSNorm|LayerNorm|none, SwiGLU|GELU-MLP) for ablation.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..llama.rmsnorm import RMSNorm
from ..llama.swiglu import SwiGLU
from .grouped_attention import GraphGQAttention


def make_norm(kind: str, dim: int) -> nn.Module:
    return {"rms": lambda: RMSNorm(dim), "layer": lambda: nn.LayerNorm(dim), "none": nn.Identity}[kind]()


class GeluMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1, self.fc2, self.dropout = nn.Linear(dim, hidden), nn.Linear(hidden, dim), nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class GraphLlamaBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, ffn_mult: float = 2.0, norm: str = "rms",
                 ffn: str = "swiglu", rotary: Optional[nn.Module] = None, dropout: float = 0.0,
                 attn_dropout: float = 0.0, struct_bias_dim: int = 0, edge_bias_dim: int = 0) -> None:
        super().__init__()
        self.attn_norm = make_norm(norm, dim)
        self.attn = GraphGQAttention(dim, num_heads, num_kv_heads, rotary=rotary, attn_dropout=attn_dropout,
                                     struct_bias_dim=struct_bias_dim, edge_bias_dim=edge_bias_dim)
        self.ffn_norm = make_norm(norm, dim)
        hidden = int(ffn_mult * dim)
        # parameter parity: SwiGLU has 3*d*hidden weights, GELU-MLP 2*d*hidden' -> hidden' = 1.5*hidden
        self.ffn = SwiGLU(dim, hidden) if ffn == "swiglu" else GeluMLP(dim, int(1.5 * hidden))
        self.dropout = dropout

    def forward(self, x, edge_index, ctx=None, edge_attr=None):
        x = x + F.dropout(self.attn(self.attn_norm(x), edge_index, ctx, edge_attr), self.dropout, self.training)
        return x + F.dropout(self.ffn(self.ffn_norm(x)), self.dropout, self.training)
