"""Autoregressive generation: greedy, temperature, top-k, top-p, with or without KV cache."""
from __future__ import annotations

from typing import Optional

import torch

from .model import LlamaForCausalLM


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None,
                top_p: Optional[float] = None, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """logits [B, V] -> token ids [B]. temperature == 0 means greedy."""
    if temperature == 0.0:
        return logits.argmax(-1)
    logits = logits.float() / temperature
    if top_k is not None and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(-1)
        cum = probs.cumsum(-1)
        remove = (cum - probs) > top_p  # keep the smallest prefix whose mass exceeds top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, sorted_logits)
    probs = logits.softmax(-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(model: LlamaForCausalLM, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.0,
             top_k: Optional[int] = None, top_p: Optional[float] = None, use_cache: bool = True,
             eos_token_id: Optional[int] = None, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    model.eval()
    ids = input_ids
    past = None
    finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    for _ in range(max_new_tokens):
        if use_cache:
            step_in = ids if past is None else ids[:, -1:]
            out = model(step_in, past_key_values=past, use_cache=True)
            past = out.past_key_values
        else:
            out = model(ids)
        nxt = sample_next(out.logits[:, -1], temperature, top_k, top_p, generator)
        if eos_token_id is not None:
            nxt = torch.where(finished, torch.full_like(nxt, eos_token_id), nxt)
            finished |= nxt == eos_token_id
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        if eos_token_id is not None and bool(finished.all()):
            break
    return ids
