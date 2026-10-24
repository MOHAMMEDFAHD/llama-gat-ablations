"""Neighbourhood-masked scaled dot-product attention (baseline A2) = GraphLlama with no rotary/PE.
Kept as a named factory for clarity of the experiment table."""
from .model import GraphLlama


def dot_product_graph_attention(in_dim, out_dim, **kw):
    kw.setdefault("rotary", None); kw.setdefault("pe", "none")
    return GraphLlama(in_dim, out_dim, **kw)
