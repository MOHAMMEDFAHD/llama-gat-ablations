"""GraphLlama: stacked LLaMA-style graph blocks with node / graph / link heads.

Ablation arms (see configs/experiments.yaml) are all instances of this class:
  rotary in {none, index, spectral, rwse, edge}, pe in {none, lap, rwse} (absolute, added to input),
  num_kv_heads (MHA/GQA/MQA), norm, ffn, struct_bias.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .graph_block import GraphLlamaBlock, make_norm
from .graph_rope import build_rotary
from .sparse_ops import apply_self_loop_policy, scatter_mean


class GraphLlama(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dim: int = 64, num_layers: int = 2, num_heads: int = 8,
                 num_kv_heads: Optional[int] = None, rotary: Optional[str] = None, rotary_dim: Optional[int] = None,
                 rotary_init_scale: float = 1.0, rotary_learnable: bool = True, pe: str = "none", lap_k: int = 16, rwse_dim: int = 16,
                 norm: str = "rms", ffn: str = "swiglu", ffn_mult: float = 2.0, dropout: float = 0.5,
                 input_dropout: float = 0.5, attn_dropout: float = 0.1, struct_bias: bool = False,
                 edge_dim: int = 0, task: str = "node", self_loops: str = "add", lap_sign_flip: bool = True) -> None:
        super().__init__()
        num_kv_heads = num_kv_heads or num_heads
        head_dim = dim // num_heads
        rotary_dim = rotary_dim or head_dim
        self.task, self.pe, self.self_loops, self.lap_sign_flip = task, pe, self_loops, lap_sign_flip
        self.rotary_kind = rotary
        self.input_dropout = input_dropout
        self.inp = nn.Linear(in_dim, dim)
        self.pe_proj = nn.Linear(lap_k if pe == "lap" else rwse_dim, dim) if pe != "none" else None
        # one rotary module per layer (independent learnable frequencies / scorers)
        self.blocks = nn.ModuleList([
            GraphLlamaBlock(dim, num_heads, num_kv_heads, ffn_mult, norm, ffn,
                            build_rotary(rotary, dim, num_kv_heads, rotary_dim, lap_k, rwse_dim, rotary_init_scale, rotary_learnable),
                            dropout, attn_dropout, rwse_dim if struct_bias else 0, edge_dim)
            for _ in range(num_layers)])
        self.final_norm = make_norm(norm, dim)
        self.head = nn.Linear(dim, out_dim) if task in ("node", "graph") else nn.Identity()

    def _ctx(self, struct: Optional[dict], n: int) -> dict:
        ctx = dict(struct or {})
        if "lap_vecs" in ctx and self.training and self.lap_sign_flip:
            # random sign flips per eigenvector: equivariance *in distribution* only (Dwivedi & Bresson)
            s = (torch.randint(0, 2, (ctx["lap_vecs"].shape[1],), device=ctx["lap_vecs"].device) * 2 - 1).float()
            ctx["lap_vecs"] = ctx["lap_vecs"] * s
        return ctx

    def embed(self, x, edge_index, struct=None, edge_attr=None):
        n = x.shape[0]
        ei = apply_self_loop_policy(edge_index, n, self.self_loops) if edge_attr is None else edge_index
        ctx = self._ctx(struct, n)
        h = self.inp(F.dropout(x, self.input_dropout, self.training))
        if self.pe_proj is not None:
            h = h + self.pe_proj(ctx["lap_vecs"] * (n ** 0.5) if self.pe == "lap" else ctx["rwse"])
        for blk in self.blocks:
            h = blk(h, ei, ctx, edge_attr)
        return self.final_norm(h)

    def forward(self, x, edge_index, struct=None, batch=None, num_graphs=None, edge_attr=None):
        h = self.embed(x, edge_index, struct, edge_attr)
        if self.task == "node":
            return self.head(h)
        if self.task == "graph":
            if batch is None:
                batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            return self.head(scatter_mean(h, batch, int(num_graphs or int(batch.max()) + 1)))
        return h  # link prediction: node embeddings

    @staticmethod
    def link_logits(h: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        return (h[pairs[0]] * h[pairs[1]]).sum(-1)
