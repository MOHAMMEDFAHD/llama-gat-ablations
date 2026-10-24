import math

import pytest
import torch

from src.llama.attention import LlamaAttention, build_attention_bias, kv_head_for_query_head, manual_attention, repeat_kv
from src.llama.config import LlamaConfig
from src.llama.rmsnorm import RMSNorm
from src.llama.rope import RotaryEmbedding, apply_rotary, inverse_frequencies
from src.llama.swiglu import SwiGLU

torch.manual_seed(0)


# ---------------- RMSNorm ----------------
def test_rmsnorm_matches_torch_reference():
    x = torch.randn(4, 7, 32, dtype=torch.float64, requires_grad=True)
    ours = RMSNorm(32, eps=1e-6).double(); ref = torch.nn.RMSNorm(32, eps=1e-6).double()
    with torch.no_grad():
        w = torch.randn(32, dtype=torch.float64); ours.weight.copy_(w); ref.weight.copy_(w)
    y1, y2 = ours(x), ref(x)
    assert torch.allclose(y1, y2, atol=1e-12)
    g1, = torch.autograd.grad(y1.sum(), x); g2, = torch.autograd.grad(y2.sum(), x)
    assert torch.allclose(g1, g2, atol=1e-10)


def test_rmsnorm_gradcheck_and_dim():
    n = RMSNorm(6).double()
    x = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: n(t), (x,))
    y = n(x)  # unit gain -> rms over last dim is ~1
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(3, dtype=torch.float64), atol=1e-4)


def test_rmsnorm_bf16_stable():
    x = (torch.randn(2, 5, 64) * 1e3).bfloat16()
    y = RMSNorm(64)(x)
    assert y.dtype == torch.float32 or y.dtype == torch.bfloat16
    assert torch.isfinite(y.float()).all()
    ref = RMSNorm(64)(x.float())
    assert (y.float() - ref).abs().max() < 2e-2


# ---------------- SwiGLU ----------------
def test_swiglu_formula():
    m = SwiGLU(8, 16)
    x = torch.randn(3, 8)
    ref = m.down_proj(torch.nn.functional.silu(m.gate_proj(x)) * m.up_proj(x))
    assert torch.allclose(m(x), ref)
    assert m(x).shape == (3, 8)


# ---------------- RoPE ----------------
def meta_reference_rope(x, positions, base):
    """Independent reference: Meta llama `precompute_freqs_cis` + complex multiply (interleaved)."""
    d = x.shape[-1]
    freqs = 1.0 / (base ** (torch.arange(0, d, 2)[: d // 2].double() / d))
    ang = torch.outer(positions.double(), freqs)
    cis = torch.polar(torch.ones_like(ang), ang)
    xc = torch.view_as_complex(x.double().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * cis).flatten(-2)


def test_rope_interleaved_matches_meta_reference():
    x = torch.randn(1, 2, 9, 16, dtype=torch.float64)
    pos = torch.arange(3, 12)
    rope = RotaryEmbedding(16, 10_000.0, layout="interleaved")
    cos, sin = rope.angles(pos[None])
    ours = apply_rotary(x, cos.double(), sin.double(), "interleaved")
    ref = meta_reference_rope(x, pos, 10_000.0)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_rope_half_equals_interleaved_up_to_permutation():
    d = 16
    perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])  # interleaved -> half layout
    x = torch.randn(2, 3, 5, d)
    pos = torch.arange(5)[None].expand(2, 5)
    rope_i = RotaryEmbedding(d, layout="interleaved"); rope_h = RotaryEmbedding(d, layout="half")
    yi = apply_rotary(x, *rope_i.angles(pos), "interleaved")
    yh = apply_rotary(x[..., perm], *rope_h.angles(pos), "half")
    assert torch.allclose(yi[..., perm], yh, atol=1e-6)


@pytest.mark.parametrize("layout", ["half", "interleaved"])
def test_rope_norm_preservation_and_relative(layout):
    d = 32
    rope = RotaryEmbedding(d, layout=layout)
    q = torch.randn(1, 1, 1, d, dtype=torch.float64); k = torch.randn(1, 1, 1, d, dtype=torch.float64)

    def rot(v, p):
        c, s = rope.angles(torch.tensor([[p]]))
        return apply_rotary(v, c.double(), s.double(), layout)

    assert torch.allclose(rot(q, 17).norm(), q.norm())
    s1 = (rot(q, 5) * rot(k, 12)).sum(); s2 = (rot(q, 105) * rot(k, 112)).sum(); s3 = (rot(q, 0) * rot(k, 7)).sum()
    # positions enter only through n - m (float32 angles -> ~1e-4 tolerance at position 112)
    assert abs(float(s1 - s2)) < 1e-3 and abs(float(s1 - s3)) < 1e-3


def test_partial_rotary_leaves_tail_untouched_and_grads():
    x = torch.randn(1, 1, 4, 16, requires_grad=True)
    rope = RotaryEmbedding(8)
    y = apply_rotary(x, *rope.angles(torch.arange(4)[None]))
    assert torch.equal(y[..., 8:], x[..., 8:])
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_rope_scaling_only_when_selected():
    base = inverse_frequencies(16, 1e4, None)
    lin = inverse_frequencies(16, 1e4, {"type": "linear", "factor": 4})
    l3 = inverse_frequencies(16, 5e5, {"type": "llama3", "factor": 8, "low_freq_factor": 1, "high_freq_factor": 4,
                                       "original_max_position_embeddings": 8192})
    assert torch.allclose(lin * 4, base)
    raw = inverse_frequencies(16, 5e5)
    assert torch.allclose(l3[0], raw[0])  # highest frequency untouched
    assert torch.allclose(l3[-1], raw[-1] / 8)  # lowest frequency divided by factor


# ---------------- attention / GQA ----------------
def test_config_validation():
    with pytest.raises(ValueError):
        LlamaConfig(hidden_size=30, num_attention_heads=4)
    with pytest.raises(ValueError):
        LlamaConfig(hidden_size=32, num_attention_heads=4, num_key_value_heads=3)


def test_gqa_head_mapping():
    assert [kv_head_for_query_head(h, 8, 2) for h in range(8)] == [0, 0, 0, 0, 1, 1, 1, 1]
    kv = torch.arange(2).float().view(1, 2, 1, 1)
    assert repeat_kv(kv, 4).flatten().tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_gqa_equals_mha_with_duplicated_kv_weights():
    cg = LlamaConfig(hidden_size=64, num_attention_heads=8, num_key_value_heads=2, attention_backend="manual")
    cm = LlamaConfig(hidden_size=64, num_attention_heads=8, num_key_value_heads=8, attention_backend="manual")
    ag, am = LlamaAttention(cg), LlamaAttention(cm)
    d = cg.head_dim
    with torch.no_grad():
        am.q_proj.weight.copy_(ag.q_proj.weight); am.o_proj.weight.copy_(ag.o_proj.weight)
        for name in ("k_proj", "v_proj"):
            w = getattr(ag, name).weight.view(2, d, 64)
            getattr(am, name).weight.copy_(w.repeat_interleave(4, dim=0).reshape(8 * d, 64))
    x = torch.randn(2, 6, 64); pos = torch.arange(6)[None].expand(2, 6)
    assert torch.allclose(ag(x, pos)[0], am(x, pos)[0], atol=1e-5)
    # and GQA really has fewer K/V parameters
    assert ag.k_proj.weight.numel() * 4 == am.k_proj.weight.numel()


@pytest.mark.parametrize("hkv", [4, 2, 1])
def test_sdpa_matches_manual(hkv):
    c = LlamaConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=hkv)
    a = LlamaAttention(c).eval()
    x = torch.randn(3, 9, 64); pos = torch.arange(9)[None].expand(3, 9)
    mask = torch.ones(3, 9, dtype=torch.long); mask[1, :3] = 0
    o1 = a(x, pos, mask, backend="manual")[0]; o2 = a(x, pos, mask, backend="sdpa")[0]
    assert torch.allclose(o1, o2, atol=1e-5)


def test_fully_masked_rows_no_nan():
    bias = build_attention_bias(3, 3, 0, torch.tensor([[0, 0, 1]]), torch.float32, torch.device("cpu"))
    q = torch.randn(1, 1, 3, 4)
    out, w = manual_attention(q, q, q, bias, 0.0, False)
    assert torch.isfinite(out).all() and torch.allclose(w.sum(-1), torch.ones(1, 1, 3))


def test_attention_weights_are_causal():
    c = LlamaConfig(hidden_size=32, num_attention_heads=2)
    a = LlamaAttention(c)
    _, _, w = a(torch.randn(1, 5, 32), torch.arange(5)[None], output_attentions=True)
    assert torch.all(torch.triu(w[0, 0], diagonal=1) == 0)
