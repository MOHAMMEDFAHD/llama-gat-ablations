from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import LlamaConfig
from .decoder import LlamaDecoderLayer
from .embeddings import TokenEmbedding
from .rmsnorm import RMSNorm


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    past_key_values: Optional[list] = None
    attentions: Optional[list] = None


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.reset_parameters()
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def reset_parameters(self) -> None:
        # Matches the HF LLaMA convention: N(0, initializer_range) for all matrices, RMSNorm gains = 1.
        std = self.config.initializer_range
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, std)
            elif isinstance(m, TokenEmbedding):
                nn.init.normal_(m.weight, 0.0, std)
            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> CausalLMOutput:
        b, t = input_ids.shape
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        if position_ids is None:
            if attention_mask is not None:
                # positions count only real tokens (handles left padding)
                pos = attention_mask.long().cumsum(-1) - 1
                position_ids = pos.clamp(min=0)[:, past_len : past_len + t]
            else:
                position_ids = torch.arange(past_len, past_len + t, device=input_ids.device)[None].expand(b, t)
        x = self.embed_tokens(input_ids)
        presents, attns = [], []
        for i, layer in enumerate(self.layers):
            pkv = None if past_key_values is None else past_key_values[i]
            x, present, w = layer(x, position_ids, attention_mask, pkv, use_cache, output_attentions)
            presents.append(present)
            attns.append(w)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(), labels[:, 1:].reshape(-1), ignore_index=-100
            )
        return CausalLMOutput(logits, loss, presents if use_cache else None, attns if output_attentions else None)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())
