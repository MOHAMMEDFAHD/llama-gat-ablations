"""Sparse one-hop graph attention with MHA/GQA/MQA and optional graph rotary.

    score_ij^h = (R_i^{g(h)} q_i^h)^T (R_ij^{g(h)} k_j^{g(h)}) / sqrt(d_h) + b_ij^h
    alpha_ij^h = softmax_{j in N(i)} score_ij^h
    m_i^h      = sum_{j in N(i)} alpha_ij^h v_j^{g(h)}

g(h) = h // (H / H_kv).  K and V are projected and *gathered per edge* at H_kv
heads ([E, H_kv, d_h]); queries are gathered at H heads.  So GQA reduces
  * K/V projection parameters:  2 d H d_h  ->  2 d H_kv d_h
  * gathered K/V edge tensors:   2 E H d_h  ->  2 E H_kv d_h
but NOT the gathered queries (E H d_h), scores (E H), or the weighted-message
tensor (E H d_h).  The net memory effect must be measured, not assumed.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .graph_rope import NodeRotary, rotate_pairs
from .sparse_ops import segment_softmax


class GraphGQAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: Optional[int] = None, head_dim: Optional[int] = None,
                 rotary: Optional[nn.Module] = None, attn_dropout: float = 0.0, edge_bias_dim: int = 0,
                 struct_bias_dim: int = 0) -> None:
        super().__init__()
        self.h = num_heads
        self.hkv = num_kv_heads or num_heads
        if self.h % self.hkv:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.g = self.h // self.hkv
        self.d = head_dim or dim // num_heads
        self.q_proj = nn.Linear(dim, self.h * self.d, bias=False)
        self.k_proj = nn.Linear(dim, self.hkv * self.d, bias=False)
        self.v_proj = nn.Linear(dim, self.hkv * self.d, bias=False)
        self.o_proj = nn.Linear(self.h * self.d, dim, bias=False)
        self.rotary = rotary
        self.attn_dropout = attn_dropout
        self.edge_bias = nn.Linear(edge_bias_dim, self.h) if edge_bias_dim else None
        self.struct_bias = (nn.Sequential(nn.Linear(2 * struct_bias_dim, 32), nn.SiLU(), nn.Linear(32, self.h))
                            if struct_bias_dim else None)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, ctx: Optional[dict] = None,
                edge_attr: Optional[torch.Tensor] = None, return_attention: bool = False):
        n = x.shape[0]
        ctx = ctx or {}
        src, dst = edge_index
        q = self.q_proj(x).view(n, self.h, self.d)
        k = self.k_proj(x).view(n, self.hkv, self.d)
        v = self.v_proj(x).view(n, self.hkv, self.d)
        if isinstance(self.rotary, NodeRotary):  # rotate at node level: O(N d), before gathering
            ph = self.rotary.phases({**ctx, "num_nodes": n})            # [N, H_kv, rd/2]
            k = rotate_pairs(k, ph)
            q = rotate_pairs(q, ph.repeat_interleave(self.g, dim=1))     # query head h uses phase of g(h)
        qe = q[dst].view(-1, self.hkv, self.g, self.d)                   # [E, H_kv, G, d]
        ke = k[src]                                                       # [E, H_kv, d]
        if self.rotary is not None and getattr(self.rotary, "kind", "") == "edge":
            ke = rotate_pairs(ke, self.rotary.phases(x, edge_index))
        scores = torch.einsum("ekgd,ekd->ekg", qe, ke).reshape(-1, self.h) / math.sqrt(self.d)
        if self.edge_bias is not None and edge_attr is not None:
            scores = scores + self.edge_bias(edge_attr)
        if self.struct_bias is not None:
            s = ctx["rwse"]
            scores = scores + self.struct_bias(torch.cat([(s[src] - s[dst]).abs(), s[src] * s[dst]], -1))
        alpha = segment_softmax(scores.to(torch.promote_types(scores.dtype, torch.float32)), dst, n)                # [E, H]; fp32 softmax under AMP
        a = F.dropout(alpha, self.attn_dropout, self.training)
        msg = a.to(v.dtype).view(-1, self.hkv, self.g, 1) * v[src].unsqueeze(2)  # [E, H_kv, G, d]
        out = torch.zeros(n, self.hkv, self.g, self.d, dtype=msg.dtype, device=x.device).index_add_(0, dst, msg)
        out = self.o_proj(out.reshape(n, self.h * self.d))               # concat heads -> output projection
        return (out, alpha) if return_attention else out
