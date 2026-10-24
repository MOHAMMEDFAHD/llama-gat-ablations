"""Graph-compatible rotary transformations for one-hop sparse attention.

Notation: a node rotation is a block-diagonal R(phi_u) acting on channel pairs.
For node-wise phases, (R(phi_i) q)^T (R(phi_j) k) = q^T R(phi_j - phi_i) k, so
the score depends on the *relative* phase of the endpoints (telescoping; zero
circulation on cycles - cf. AW-RoPE Prop. B.2).

GQA tie: phases are defined per **key/value head** and shared by the query heads
of that group.  If query head h and its KV head g(h) used different phase
functions, the relative-phase identity would not hold.

Implemented (all re-implementations by us, *not* the authors' code):
  * IndexRotary      - standard 1D RoPE on the arbitrary node index. INVALID for graphs
                       (breaks permutation equivariance); kept as a negative control.
  * SpectralRotary   - WIRE-style: phi_u = nu^T r_u with r_u = LapPE coordinates
                       (optionally eigenvalue-scaled), nu learnable.  [Reid et al.]
  * RWSERotary       - phi_u = MLP(RWSE_u): learned map from sign-free random-walk
                       features to phases (WIRE-family parameterisation; cf. WIRE's RWPE variant).
  * EdgeRotary       - one-hop simplification of AW-RoPE's antisymmetric edge field:
                       score = q_i^T R(omega * a_ji) k_j with a_ji = -a_ij. No walk sum (L=1 on scores),
                       so it is NOT AW-RoPE; it isolates "edge vs node phases" in local attention.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from ..llama.rope import apply_rotary, inverse_frequencies


def rotate_pairs(x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """x [..., H, d]; phase [..., H, rd/2] -> rotate first rd channels ('half' layout)."""
    return apply_rotary(x, phase.cos(), phase.sin(), "half")


class NodeRotary(nn.Module):
    kind = "node"

    def phases(self, ctx: dict) -> torch.Tensor:  # [N, H_kv, rd/2]
        raise NotImplementedError


class IndexRotary(NodeRotary):
    def __init__(self, num_kv_heads: int, rotary_dim: int, base: float = 10_000.0):
        super().__init__()
        self.hkv = num_kv_heads
        self.register_buffer("inv_freq", inverse_frequencies(rotary_dim, base), persistent=False)

    def phases(self, ctx):
        n = ctx["num_nodes"]
        ph = torch.arange(n, device=self.inv_freq.device).float()[:, None] * self.inv_freq
        return ph[:, None].expand(n, self.hkv, ph.shape[-1])


class SpectralRotary(NodeRotary):
    def __init__(self, num_kv_heads: int, rotary_dim: int, num_eigs: int, init_scale: float = 1.0,
                 eigval_scaling: bool = False, learnable: bool = True):
        super().__init__()
        self.hkv, self.half = num_kv_heads, rotary_dim // 2
        nu = torch.randn(num_kv_heads, self.half, num_eigs) * init_scale
        self.nu = nn.Parameter(nu, requires_grad=learnable)
        self.eigval_scaling = eigval_scaling

    def phases(self, ctx):
        r = ctx["lap_vecs"]  # [N, m]; sign flips handled by the model in training if enabled
        if self.eigval_scaling:
            r = r * ctx["lap_vals"].clamp_min(1e-6).rsqrt()[None]
        # Eigenvectors have unit 2-norm, entries ~ 1/sqrt(N); rescale so phases are O(1).
        r = r * math.sqrt(r.shape[0])
        return torch.einsum("nm,hlm->nhl", r, self.nu)


class RWSERotary(NodeRotary):
    def __init__(self, num_kv_heads: int, rotary_dim: int, rwse_dim: int, hidden: int = 32, init_scale: float = 1.0):
        super().__init__()
        self.hkv, self.half = num_kv_heads, rotary_dim // 2
        self.mlp = nn.Sequential(nn.Linear(rwse_dim, hidden), nn.SiLU(), nn.Linear(hidden, num_kv_heads * self.half))
        nn.init.normal_(self.mlp[-1].weight, std=init_scale / math.sqrt(hidden)); nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("mu", torch.zeros(rwse_dim)); self.register_buffer("sd", torch.ones(rwse_dim))
        self._fitted = False

    def phases(self, ctx):
        z = ctx["rwse"]
        if not self._fitted:  # standardize once using the (transductive) graph's statistics; no labels used
            self.mu.copy_(z.mean(0)); self.sd.copy_(z.std(0).clamp_min(1e-6)); self._fitted = True
        return self.mlp((z - self.mu) / self.sd).view(-1, self.hkv, self.half)


class EdgeRotary(nn.Module):
    kind = "edge"

    def __init__(self, dim: int, num_kv_heads: int, rotary_dim: int, hidden: int = 32, a_max: float = math.pi,
                 base: float = 10_000.0):
        super().__init__()
        self.hkv, self.a_max = num_kv_heads, a_max
        self.scorer = nn.Sequential(nn.Linear(2 * dim, hidden), nn.SiLU(), nn.Linear(hidden, num_kv_heads))
        # AW-RoPE initialises omega with the RoPE schedule; we rescale so the top frequency is 1.
        inv = inverse_frequencies(rotary_dim, base)
        self.omega = nn.Parameter(inv / inv.max())

    def phases(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """[E, H_kv, rd/2] phase applied to k_j for edge j -> i (antisymmetric in (i, j))."""
        src, dst = edge_index
        s_ij = self.scorer(torch.cat([h[dst], h[src]], -1))
        s_ji = self.scorer(torch.cat([h[src], h[dst]], -1))
        a = self.a_max * torch.tanh((s_ij - s_ji) / self.a_max)  # odd -> a_ji = -a_ij ; self-loops -> 0
        return a[..., None] * self.omega


def build_rotary(kind: Optional[str], dim: int, num_kv_heads: int, rotary_dim: int, lap_k: int, rwse_dim: int,
                 init_scale: float = 1.0, learnable: bool = True) -> Optional[nn.Module]:
    if kind in (None, "none"):
        return None
    if kind == "index":
        return IndexRotary(num_kv_heads, rotary_dim)
    if kind == "spectral":
        return SpectralRotary(num_kv_heads, rotary_dim, lap_k, init_scale, learnable=learnable)
    if kind == "rwse":
        return RWSERotary(num_kv_heads, rotary_dim, rwse_dim, init_scale=init_scale)
    if kind == "edge":
        return EdgeRotary(dim, num_kv_heads, rotary_dim)
    raise ValueError(kind)
