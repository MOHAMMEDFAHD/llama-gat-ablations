from __future__ import annotations

from .baselines import GCN, GIN, SAGE, GATv2, GraphGPS, GTLapPE
from .gat import GAT
from .model import GraphLlama

BASELINES = {"gcn": GCN, "sage": SAGE, "gin": GIN, "gat": GAT, "gatv2": GATv2, "gt_lappe": GTLapPE, "gps": GraphGPS}


def build_model(name: str, in_dim: int, out_dim: int, **kw):
    if name in BASELINES:
        return BASELINES[name](in_dim, out_dim, **kw)
    if name == "graphllama":
        return GraphLlama(in_dim, out_dim, **kw)
    raise KeyError(name)
