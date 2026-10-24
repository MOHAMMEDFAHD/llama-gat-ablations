from __future__ import annotations

import math
import resource
from typing import Iterable

import numpy as np
import torch


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    seen = {}
    for p in model.parameters():
        if (not trainable_only) or p.requires_grad:
            seen[id(p)] = p.numel()
    return int(sum(seen.values()))


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(-1) == y).float().mean())


def macro_f1(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    pred = logits.argmax(-1)
    f1s = []
    for c in range(num_classes):
        tp = int(((pred == c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum()); fn = int(((pred != c) & (y == c)).sum())
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Mann-Whitney AUC with tie handling (average ranks)."""
    s = scores.detach().double().cpu().numpy(); l = labels.detach().cpu().numpy().astype(bool)
    order = s.argsort(kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ties
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    npos, nneg = l.sum(), (~l).sum()
    return float((ranks[l].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    s = scores.detach().double().cpu().numpy(); l = labels.detach().cpu().numpy().astype(bool)
    order = np.argsort(-s, kind="mergesort"); l = l[order]
    tp = np.cumsum(l); prec = tp / np.arange(1, len(l) + 1)
    return float((prec * l).sum() / max(l.sum(), 1))


def mean_std_ci(values: Iterable[float], conf: float = 0.95) -> dict:
    v = np.asarray(list(values), dtype=float)
    n = len(v)
    mean = float(v.mean()) if n else float("nan")
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        from scipy import stats
        h = float(stats.t.ppf(0.5 + conf / 2, n - 1) * std / math.sqrt(n))
    else:
        h = float("nan")
    return {"n": n, "mean": mean, "std": std, "ci95_low": mean - h, "ci95_high": mean + h}


def peak_memory_mb(device: torch.device) -> dict:
    """CUDA: exact allocator peak. CPU: process max RSS (a coarse, process-wide upper bound)."""
    out = {"cpu_max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}
    if device.type == "cuda":
        out["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    return out
