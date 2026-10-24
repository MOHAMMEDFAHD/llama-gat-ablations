"""Pre-norm LLaMA decoder block:
    h = x + Attn(RMSNorm(x));  out = h + SwiGLU(RMSNorm(h))
"""
from __future__ import annotations

from torch import nn

from .attention import LlamaAttention
from .config import LlamaConfig
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = LlamaAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(self, x, position_ids, attention_mask=None, past_kv=None, use_cache=False, output_attentions=False):
        a, present, w = self.self_attn(self.input_layernorm(x), position_ids, attention_mask, past_kv, use_cache, output_attentions)
        h = x + a
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out, present, w
