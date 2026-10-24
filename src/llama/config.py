"""Configuration for the LLaMA-style decoder.

Architecture-generation notes (documented, not assumed):
  * LLaMA 1 (Touvron et al., 2023a): RMSNorm pre-norm, SwiGLU, RoPE base 1e4, MHA for all sizes.
  * LLaMA 2 (2023b): same block; GQA used for the 70B model only; context 4096.
  * LLaMA 3 (2024): GQA (8 KV heads) for all sizes, RoPE base 5e5, 128k vocab.
  * LLaMA 3.1: adds "llama3" RoPE frequency scaling for long context.
Presets below are tiny and only mimic the *choices* of each generation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional


@dataclass
class LlamaConfig:
    vocab_size: int = 256
    hidden_size: int = 128
    intermediate_size: Optional[int] = None  # None -> LLaMA rule (see __post_init__)
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: Optional[int] = None  # None -> MHA
    max_position_embeddings: int = 256
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    rotary_dim: Optional[int] = None  # None -> full head_dim
    rope_layout: Literal["half", "interleaved"] = "half"
    rope_scaling: Optional[dict] = None  # {"type": "linear"|"llama3", ...}; only if explicitly set
    attention_dropout: float = 0.0
    attention_backend: Literal["manual", "sdpa"] = "sdpa"
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02
    multiple_of: int = 32

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.intermediate_size is None:
            # LLaMA reference: hidden = int(2 * 4d / 3), rounded up to `multiple_of`.
            h = int(2 * 4 * self.hidden_size / 3)
            self.intermediate_size = self.multiple_of * ((h + self.multiple_of - 1) // self.multiple_of)
        if self.rotary_dim is None:
            self.rotary_dim = self.head_dim
        if self.rotary_dim % 2 != 0 or self.rotary_dim > self.head_dim:
            raise ValueError("rotary_dim must be even and <= head_dim")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def to_dict(self) -> dict:
        return asdict(self)


def preset(name: str, **overrides) -> LlamaConfig:
    """Tiny configs reflecting generation-specific *choices* (not sizes)."""
    base = dict(vocab_size=256, hidden_size=128, num_hidden_layers=2, num_attention_heads=4)
    table = {
        "llama1-tiny": dict(num_key_value_heads=4, rope_theta=1e4),
        "llama2-tiny": dict(num_key_value_heads=4, rope_theta=1e4),  # GQA only at 70B in LLaMA 2
        "llama2-70b-style-tiny": dict(num_key_value_heads=1, rope_theta=1e4),
        "llama3-tiny": dict(num_key_value_heads=2, rope_theta=5e5),
    }
    if name not in table:
        raise KeyError(f"unknown preset {name}")
    base.update(table[name])
    base.update(overrides)
    return LlamaConfig(**base)
