"""Original GAT (Velickovic et al., 2018), implemented manually.

    e_ij    = LeakyReLU( a^T [W h_i || W h_j] )      (additive attention)
    alpha_ij = softmax_{j in N(i)} e_ij
    h'_i    = sum_j alpha_ij W h_j                 (per head; heads concatenated or averaged)

Additive vs dot-product: the score decomposes as LeakyReLU(a_1^T W h_i + a_2^T W h_j).
Because LeakyReLU is monotone and the query term a_1^T W h_i is shared by all
neighbours of i, the *ranking* of neighbours is the same for every query node
("static attention", Brody et al. 2022).  Dot-product attention q_i^T k_j is a
bilinear form whose ranking depends on the query.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .sparse_ops import apply_self_loop_policy, segment_softmax


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 8, concat: bool = True, negative_slope: float = 0.2,
                 dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.heads, self.out_dim, self.concat = heads, out_dim, concat
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))  # a_2 (applied to neighbour j)
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))  # a_1 (applied to target i)
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim)) if bias else None
        self.negative_slope, self.dropout = negative_slope, dropout
        nn.init.xavier_uniform_(self.lin.weight); nn.init.xavier_uniform_(self.att_src); nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edge_index, return_attention: bool = False):
        n = x.shape[0]
        h = self.lin(x).view(n, self.heads, self.out_dim)
        src, dst = edge_index
        e = (h * self.att_src).sum(-1)[src] + (h * self.att_dst).sum(-1)[dst]   # [E, H]
        alpha = segment_softmax(F.leaky_relu(e, self.negative_slope), dst, n)
        alpha_d = F.dropout(alpha, self.dropout, self.training)
        out = torch.zeros_like(h).index_add_(0, dst, h[src] * alpha_d.unsqueeze(-1))
        out = out.reshape(n, -1) if self.concat else out.mean(1)
        if self.bias is not None:
            out = out + self.bias
        return (out, alpha) if return_attention else out


class GAT(nn.Module):
    """2-layer GAT with the Cora hyper-parameters of the original paper (8 heads x 8 hidden, dropout 0.6)."""

    def __init__(self, in_dim, num_classes, hidden=8, heads=8, out_heads=1, dropout=0.6, self_loops="add"):
        super().__init__()
        self.l1 = GATLayer(in_dim, hidden, heads, True, dropout=dropout)
        self.l2 = GATLayer(hidden * heads, num_classes, out_heads, False, dropout=dropout)
        self.dropout, self.self_loops = dropout, self_loops

    def forward(self, x, edge_index, **_):
        ei = apply_self_loop_policy(edge_index, x.shape[0], self.self_loops)
        x = F.dropout(x, self.dropout, self.training)
        x = F.elu(self.l1(x, ei))
        x = F.dropout(x, self.dropout, self.training)
        return self.l2(x, ei)
