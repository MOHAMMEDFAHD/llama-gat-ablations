"""Causal self-attention with MHA / GQA / MQA, RoPE inside attention, and a KV cache.

GQA head sharing: with H query heads and H_kv key/value heads, G = H / H_kv.
Query head h uses key/value head g(h) = floor(h / G); i.e. query heads
[g*G, ..., (g+1)*G - 1] share KV head g.  MHA is H_kv = H, MQA is H_kv = 1.
The cache stores K/V at H_kv heads (that is the memory saving); heads are
expanded with `repeat_kv` only for the score computation.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import LlamaConfig
from .rope import RotaryEmbedding


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, H_kv, T, D] -> [B, H_kv * n_rep, T, D] with head order g*n_rep + r."""
    if n_rep == 1:
        return x
    b, hkv, t, d = x.shape
    return x[:, :, None].expand(b, hkv, n_rep, t, d).reshape(b, hkv * n_rep, t, d)


def kv_head_for_query_head(h: int, num_heads: int, num_kv_heads: int) -> int:
    return h // (num_heads // num_kv_heads)


def build_attention_bias(
    q_len: int,
    k_len: int,
    past_len: int,
    attention_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
    causal: bool = True,
) -> torch.Tensor:
    """Additive mask [B or 1, 1, Tq, Tk]. Uses finfo.min (not -inf) so fully masked
    rows (e.g. left-padding queries) give a uniform distribution instead of NaN."""
    neg = torch.finfo(dtype).min
    allowed = torch.ones(q_len, k_len, dtype=torch.bool, device=device)
    if causal:
        qpos = torch.arange(q_len, device=device)[:, None] + past_len
        kpos = torch.arange(k_len, device=device)[None, :]
        allowed = kpos <= qpos
    allowed = allowed[None, None]
    if attention_mask is not None:  # [B, Tk] with 1 = real token
        allowed = allowed & attention_mask[:, None, None, :k_len].bool()
    bias = torch.zeros(allowed.shape, dtype=dtype, device=device)
    return bias.masked_fill(~allowed, neg)


def manual_attention(q, k, v, bias, dropout_p: float, training: bool):
    """softmax(QK^T / sqrt(d) + M) V, computed explicitly. Returns (out, weights)."""
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    scores = scores + bias
    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    weights = F.dropout(weights, p=dropout_p, training=training)
    return torch.matmul(weights, v), weights


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.h = config.num_attention_heads
        self.hkv = config.num_key_value_heads
        self.d = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.h * self.d, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.hkv * self.d, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.hkv * self.d, bias=False)
        self.o_proj = nn.Linear(self.h * self.d, config.hidden_size, bias=False)
        self.rope = RotaryEmbedding(config.rotary_dim, config.rope_theta, config.rope_scaling, config.rope_layout)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        backend: Optional[str] = None,
    ):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.h, self.d).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.hkv, self.d).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.hkv, self.d).transpose(1, 2)
        q, k = self.rope(q, k, position_ids)  # RoPE applied inside attention, to Q and K only
        past_len = 0
        if past_kv is not None:
            past_len = past_kv[0].shape[2]
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v) if use_cache else None
        k_rep, v_rep = repeat_kv(k, self.h // self.hkv), repeat_kv(v, self.h // self.hkv)
        bias = build_attention_bias(t, k.shape[2], past_len, attention_mask, q.dtype, x.device)
        backend = backend or self.config.attention_backend
        p = self.config.attention_dropout if self.training else 0.0
        weights = None
        if backend == "manual" or output_attentions:
            out, weights = manual_attention(q, k_rep, v_rep, bias, p, self.training)
        elif backend == "sdpa":
            out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=bias, dropout_p=p)
        else:
            raise ValueError(backend)
        out = out.transpose(1, 2).reshape(b, t, self.h * self.d)  # concatenate heads
        return self.o_proj(out), present, weights
