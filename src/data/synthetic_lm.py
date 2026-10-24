"""Tiny learnable LM task: arithmetic progressions mod V.

Each sequence is x_t = (a + t*s) mod V with random a, s (s in [1, max_step]).
After two tokens the continuation is deterministic, so a correct model can
reach ~0 loss on positions >= 2, and greedy generation is checkable exactly.
"""
from __future__ import annotations

import torch


def progression_batch(batch: int, seq_len: int, vocab: int, max_step: int = 7, generator: torch.Generator | None = None) -> torch.Tensor:
    a = torch.randint(0, vocab, (batch, 1), generator=generator)
    s = torch.randint(1, max_step + 1, (batch, 1), generator=generator)
    t = torch.arange(seq_len)[None]
    return (a + t * s) % vocab


def progression_labels(ids: torch.Tensor) -> torch.Tensor:
    """Ignore the loss on predicting token 1 (not determined by token 0)."""
    labels = ids.clone()
    labels[:, :2] = -100
    return labels
