"""Sparse neighbourhood primitives (no dense N x N tensors).

Convention: edge_index[0] = source j, edge_index[1] = target i; messages flow
j -> i and the softmax is over incoming edges of each target i, i.e. N(i) =
{j : (j, i) in E}.  This matches PyTorch Geometric's `flow="source_to_target"`.
"""
from __future__ import annotations

from typing import Literal, Optional

import torch


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax of `scores` [E, ...] grouped by `index` [E] (the target node).

    Max-subtraction per group for stability. Nodes with no incoming edges simply
    receive no mass (their aggregated message is 0), never NaN.
    """
    shape = (num_nodes,) + scores.shape[1:]
    idx = index.view(-1, *([1] * (scores.dim() - 1))).expand_as(scores)
    smax = torch.full(shape, float("-inf"), dtype=scores.dtype, device=scores.device)
    smax = smax.scatter_reduce(0, idx, scores, reduce="amax", include_self=True)
    ex = (scores - smax.gather(0, idx).detach()).exp()
    denom = torch.zeros(shape, dtype=scores.dtype, device=scores.device).index_add_(0, index, ex)
    return ex / denom.gather(0, idx).clamp_min(torch.finfo(scores.dtype).tiny)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    out = torch.zeros((num_nodes,) + src.shape[1:], dtype=src.dtype, device=src.device)
    return out.index_add_(0, index, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    s = scatter_sum(src, index, num_nodes)
    cnt = scatter_sum(torch.ones_like(index, dtype=src.dtype), index, num_nodes).clamp_min(1)
    return s / cnt.view(-1, *([1] * (src.dim() - 1)))


def coalesce(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Remove duplicate edges (keeps direction)."""
    key = edge_index[0] * num_nodes + edge_index[1]
    key = torch.unique(key)
    return torch.stack([key // num_nodes, key % num_nodes])


def apply_self_loop_policy(edge_index: torch.Tensor, num_nodes: int,
                           policy: Literal["add", "remove", "keep"] = "add") -> torch.Tensor:
    """'add': remove existing loops then add exactly one per node (so isolated nodes attend to themselves).
    'remove': drop all loops. 'keep': unchanged (after de-duplication)."""
    ei = coalesce(edge_index, num_nodes)
    if policy == "keep":
        return ei
    ei = ei[:, ei[0] != ei[1]]
    if policy == "remove":
        return ei
    loops = torch.arange(num_nodes, device=ei.device)
    return torch.cat([ei, torch.stack([loops, loops])], 1)


def to_undirected(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return coalesce(torch.cat([edge_index, edge_index.flip(0)], 1), num_nodes)


def in_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return torch.bincount(edge_index[1], minlength=num_nodes)


def batch_graphs(graphs: list[dict]) -> dict:
    """Concatenate graphs {x, edge_index, (y), (edge_attr)} into one disjoint graph with a batch vector."""
    xs, eis, ys, bs, eas = [], [], [], [], []
    off = 0
    for g, gr in enumerate(graphs):
        n = gr["x"].shape[0]
        xs.append(gr["x"]); eis.append(gr["edge_index"] + off); bs.append(torch.full((n,), g, dtype=torch.long))
        if "y" in gr: ys.append(gr["y"].view(1, -1) if gr["y"].dim() <= 1 else gr["y"])
        if "edge_attr" in gr: eas.append(gr["edge_attr"])
        off += n
    out = {"x": torch.cat(xs), "edge_index": torch.cat(eis, 1), "batch": torch.cat(bs), "num_graphs": len(graphs)}
    if ys: out["y"] = torch.cat(ys).squeeze(-1)
    if eas: out["edge_attr"] = torch.cat(eas)
    return out
