"""Baselines with a uniform interface: forward(x, edge_index, struct=None) -> logits.

GCN, GraphSAGE, GIN, GATv2, GraphGPS use PyTorch Geometric layers (verified library code).
Original GAT is our manual implementation (src/graph/gat.py), cross-checked against PyG GATConv in tests.
GT-LapPE: *global* dense softmax attention over all nodes with LapPE added to the input
(Dwivedi & Bresson-style PE with a Graphormer-style global receptive field). O(N^2) memory.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64, dropout=0.5):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.c1, self.c2, self.dropout = GCNConv(in_dim, hidden), GCNConv(hidden, out_dim), dropout

    def forward(self, x, edge_index, **_):
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.c1(x, edge_index))
        return self.c2(F.dropout(x, self.dropout, self.training), edge_index)


class SAGE(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64, dropout=0.5):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.c1, self.c2, self.dropout = SAGEConv(in_dim, hidden), SAGEConv(hidden, out_dim), dropout

    def forward(self, x, edge_index, **_):
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.c1(x, edge_index))
        return self.c2(F.dropout(x, self.dropout, self.training), edge_index)


class GIN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64, dropout=0.5):
        super().__init__()
        from torch_geometric.nn import GINConv
        mlp = lambda i, o: nn.Sequential(nn.Linear(i, hidden), nn.ReLU(), nn.Linear(hidden, o))
        self.c1, self.c2, self.dropout = GINConv(mlp(in_dim, hidden)), GINConv(mlp(hidden, out_dim)), dropout

    def forward(self, x, edge_index, **_):
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.c1(x, edge_index))
        return self.c2(F.dropout(x, self.dropout, self.training), edge_index)


class GATv2(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=8, heads=8, dropout=0.6):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        self.c1 = GATv2Conv(in_dim, hidden, heads=heads, dropout=dropout)
        self.c2 = GATv2Conv(hidden * heads, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, **_):
        x = F.dropout(x, self.dropout, self.training)
        x = F.elu(self.c1(x, edge_index))
        return self.c2(F.dropout(x, self.dropout, self.training), edge_index)


class GTLapPE(nn.Module):
    def __init__(self, in_dim, out_dim, dim=64, heads=4, layers=2, lap_k=16, dropout=0.5):
        super().__init__()
        self.inp, self.pe = nn.Linear(in_dim, dim), nn.Linear(lap_k, dim)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout, batch_first=True, norm_first=True)
                                     for _ in range(layers)])
        self.head, self.dropout = nn.Linear(dim, out_dim), dropout

    def forward(self, x, edge_index, struct=None, **_):
        pe = struct["lap_vecs"] * (x.shape[0] ** 0.5)
        if self.training:
            pe = pe * (torch.randint(0, 2, (pe.shape[1],)) * 2 - 1).float()
        h = (self.inp(F.dropout(x, self.dropout, self.training)) + self.pe(pe))[None]
        for l in self.layers:
            h = l(h)
        return self.head(h[0])


class GraphGPS(nn.Module):
    """GPSConv (Rampasek et al. 2022): GIN local MPNN + global multi-head attention, RWSE input encoding."""

    def __init__(self, in_dim, out_dim, dim=64, heads=4, layers=2, rwse_dim=16, dropout=0.5):
        super().__init__()
        from torch_geometric.nn import GINConv, GPSConv
        self.inp, self.pe = nn.Linear(in_dim, dim), nn.Linear(rwse_dim, dim)
        mk = lambda: GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.convs = nn.ModuleList([GPSConv(dim, mk(), heads=heads, dropout=dropout, attn_type="multihead",
                                            attn_kwargs={"dropout": 0.1}) for _ in range(layers)])
        self.head, self.dropout = nn.Linear(dim, out_dim), dropout

    def forward(self, x, edge_index, struct=None, **_):
        h = self.inp(F.dropout(x, self.dropout, self.training)) + self.pe(struct["rwse"])
        for c in self.convs:
            h = c(h, edge_index)
        return self.head(h)
