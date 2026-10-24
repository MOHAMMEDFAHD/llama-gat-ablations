from __future__ import annotations

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """X = Embedding(input_ids). Weight may be tied to the LM head by the parent model."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(input_ids, self.weight)
