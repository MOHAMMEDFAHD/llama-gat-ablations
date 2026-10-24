"""Structural / positional encodings with explicit ambiguity diagnostics.

Laplacian eigenvectors (LapPE)
  * Sign ambiguity: v and -v are both eigenvectors.
  * Basis ambiguity: for an eigenvalue of multiplicity m > 1, any orthonormal basis of
    the eigenspace is valid.  On a disconnected graph the eigenvalue 0 of the
    normalized Laplacian has multiplicity = #components (Cora: 78).
  => any function of *individual* eigenvectors is equivariant only given a
    consistently permuted eigenbasis, which an eigensolver does not guarantee.
Random-walk structural encoding (RWSE)
  * rwse_k(i) = (P^k)_ii with P = D^-1 A.  diag(f(P)) is a matrix function, so
    RWSE is exactly permutation-equivariant and free of sign/basis ambiguity.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import torch


def _adjacency(edge_index: torch.Tensor, n: int) -> sp.csr_matrix:
    ei = edge_index.cpu().numpy()
    a = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(np.float64)  # symmetrize, binary
    a.setdiag(0); a.eliminate_zeros()
    return a


@dataclass
class SpectralInfo:
    vecs: torch.Tensor                 # [N, k] selected eigenvectors
    vals: torch.Tensor                 # [k]
    all_vals: np.ndarray
    num_components: int
    selected_multiplicities: list = field(default_factory=list)
    min_gap_selected: float = float("nan")
    seconds: float = 0.0


def normalized_laplacian_eigs(edge_index: torch.Tensor, n: int, k: int, skip_null: bool = True,
                              sign_canonicalize: bool = False, tol: float = 1e-6) -> SpectralInfo:
    """Dense eigendecomposition (O(N^3) time, O(N^2) memory) - acceptable for Cora (N=2708).

    skip_null: drop the null space (multiplicity = #components), whose basis is arbitrary.
    """
    t0 = time.perf_counter()
    a = _adjacency(edge_index, n)
    ncomp, _ = csg.connected_components(a, directed=False)
    deg = np.asarray(a.sum(1)).ravel()
    dinv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    lap = np.eye(n) - (sp.diags(dinv) @ a @ sp.diags(dinv)).toarray()
    lap[deg == 0, deg == 0] = 0.0  # isolated nodes: L_ii = 0 (part of the null space)
    vals, vecs = np.linalg.eigh(lap)
    start = int((vals < tol).sum()) if skip_null else 0
    sel_vals, sel_vecs = vals[start : start + k], vecs[:, start : start + k].copy()
    if sign_canonicalize:  # heuristic: largest-|entry| positive. NOT robust to ties or to basis ambiguity.
        idx = np.abs(sel_vecs).argmax(0)
        sel_vecs *= np.sign(sel_vecs[idx, np.arange(sel_vecs.shape[1])])[None]
    mult = [int((np.abs(vals - v) < tol).sum()) for v in sel_vals]
    gaps = np.diff(vals[max(start - 1, 0) : start + k + 1])
    if len(sel_vecs[0]) < k:  # tiny graphs: pad with zeros
        pad = np.zeros((n, k - sel_vecs.shape[1])); sel_vecs = np.concatenate([sel_vecs, pad], 1)
        sel_vals = np.concatenate([sel_vals, np.zeros(k - len(sel_vals))])
    return SpectralInfo(torch.tensor(sel_vecs, dtype=torch.float32), torch.tensor(sel_vals, dtype=torch.float32),
                        vals, int(ncomp), mult, float(gaps.min()) if len(gaps) else float("nan"),
                        time.perf_counter() - t0)


def rwse(edge_index: torch.Tensor, n: int, steps: int = 16, method: str = "auto") -> torch.Tensor:
    """[N, steps] return probabilities (P^k)_ii, k = 1..steps.

    'eig': (P^k)_ii = (S^k)_ii with S = D^-1/2 A D^-1/2 = U diag(mu) U^T, so (P^k)_ii = sum_m U_im^2 mu_m^k.
           Exact and basis-invariant within eigenspaces. O(N^3).
    'power': sparse repeated products (exact, slower when P^k fills in); used to cross-check.
    """
    a = _adjacency(edge_index, n)
    deg = np.asarray(a.sum(1)).ravel()
    if method == "auto":
        method = "eig" if n <= 6000 else "power"
    if method == "eig":
        dinv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
        s = (sp.diags(dinv) @ a @ sp.diags(dinv)).toarray()
        mu, u = np.linalg.eigh(s)
        u2 = u**2
        out = np.stack([u2 @ (mu**k) for k in range(1, steps + 1)], 1)
    else:
        dinv = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-12), 0.0)
        p = sp.diags(dinv) @ a
        m = p.copy(); cols = []
        for _ in range(steps):
            cols.append(m.diagonal()); m = m @ p
        out = np.stack(cols, 1)
    return torch.tensor(np.clip(out, 0.0, 1.0), dtype=torch.float32)


def compute_structural(edge_index: torch.Tensor, n: int, lap_k: int = 16, rwse_steps: int = 16) -> dict:
    t0 = time.perf_counter(); spec = normalized_laplacian_eigs(edge_index, n, lap_k); t1 = time.perf_counter()
    rw = rwse(edge_index, n, rwse_steps); t2 = time.perf_counter()
    return {"lap_vecs": spec.vecs, "lap_vals": spec.vals, "rwse": rw,
            "diagnostics": {"num_components": spec.num_components, "lap_selected_multiplicities": spec.selected_multiplicities,
                            "lap_min_gap_selected": spec.min_gap_selected,
                            "null_space_dim": int((spec.all_vals < 1e-6).sum()),
                            "lap_seconds": t1 - t0, "rwse_seconds": t2 - t1}}
