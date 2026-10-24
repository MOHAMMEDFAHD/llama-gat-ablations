# Mathematical Formulation and Analysis

Statuses: **Proposition** = proved here; **Known result** = from cited work; **Conjecture / Open** = unproved.

## 1. LLaMA attention with RoPE

- Rotation for channel pair ℓ, with θ_ℓ = b^{−2ℓ/d_r}: R_m = ⊕_ℓ R(m θ_ℓ), where R(a) = [[cos a, −sin a], [sin a, cos a]].
- **Proposition 1 (relative position).** (R_m q)ᵀ(R_n k) = qᵀ R_{n−m} k.
  - *Proof.* R(a)ᵀ = R(−a), and planar rotations compose additively. So R_mᵀ R_n = ⊕_ℓ R((n−m)θ_ℓ) = R_{n−m}. ∎
  - Each R_m is orthogonal, so ‖R_m q‖ = ‖q‖ (tested in `tests/test_llama_components.py`).
- **Head grouping.**
  - GQA: g(h) = ⌊h / (H/H_kv)⌋. Score: s_{tu}^h = (R_t q_t^h)ᵀ(R_u k_u^{g(h)}) / √d_h + M_{tu}.
  - **Proposition 2.** GQA equals MHA whose K/V weights are duplicated within each group. Hence GQA is MHA with a parameter-tying constraint, and its function class is a subset of MHA's. *Proof:* substitute the duplicated weights into the MHA scores. ∎ (Numerically tested.)

## 2. Graph attention (one-hop sparse)

- Notation: N(i) = {j : (j,i) ∈ E}, after the self-loop policy is applied. q_i = W_Q h_i, k_j = W_K h_j, v_j = W_V h_j.
- Score, attention weight, and message:

```
s_ij^h = (R_i^{g(h)} q_i^h)ᵀ (R_ij^{g(h)} k_j^{g(h)}) / √d_h + b_ij^h
α_ij^h = exp(s_ij^h) / Σ_{j'∈N(i)} exp(s_ij'^h)
m_i^h  = Σ_{j∈N(i)} α_ij^h v_j^{g(h)}
```

- Rotations depend on the arm:
  - **node-wise arms:** R_i = R(φ_i) and R_ij = R(φ_j);
  - **edge arm:** R_i = I and R_ij = R(ω a_ji).
- **Proposition 3 (GQA phase tie).**
  - Node-wise rotations reduce the score to q_iᵀ R(φ_j − φ_i) k_j only if query head h and KV head g(h) use the same phase function. If query and key phases differ, you get q_iᵀ R(φ_j^{(k)} − φ_i^{(q)}) k_j, which is no longer a function of a difference of one potential.
  - So phases are defined per KV head and shared within a group.
  - *Consequence:* with GQA, the number of independent phase functions per layer drops from H to H_kv. This is an expressiveness cost specific to rotary + GQA.

## 3. Permutation equivariance

- Let P be a permutation, X′ = PX, A′ = PAPᵀ, and let every structural input be transformed consistently. We want f(X′, A′) = P f(X, A).
- **Proposition 4 (backbone).** The one-hop attention layer with no rotary (or with node phases given as an equivariant function Φ(X, A), i.e. Φ(PX, PAPᵀ) = PΦ), plus node-wise RMSNorm / SwiGLU / residual / linear layers, is permutation-equivariant.
  - *Proof:*
    - Projections, norms and FFNs act row-wise, so they commute with P.
    - The edge set is relabelled bijectively.
    - Each score s_ij depends only on (h_i, h_j, φ_i, φ_j). These are relabelled consistently, so each multiset {s_ij}_{j∈N(i)} maps to the multiset of π(i).
    - Segment softmax and summation are invariant to the order within a neighbourhood. ∎
- **Proposition 5 (RWSE phases are exactly equivariant).**
  - RWSE_k(i) = (P_rw^k)_{ii}, with P_rw = D⁻¹A. Relabelling gives P_rw′ = P P_rw Pᵀ, hence diag(P_rw′^k) = P diag(P_rw^k).
  - φ = MLP(RWSE) is applied row-wise, so Φ is equivariant. Prop. 4 then applies.
  - No sign or basis choice is involved.
  - Our "eig" implementation computes Σ_m U_im² μ_m^k. This value is invariant to the eigenbasis chosen inside each eigenspace, because Σ_{m∈eigenspace} U_im² is the diagonal of the eigenspace projector. ∎
- **Proposition 6 (edge phases are exactly equivariant).**
  - a_ji = a_max tanh((s([h_i,h_j]) − s([h_j,h_i]))/a_max) uses a scorer shared across edges, so it is relabelled with the edge.
  - It is antisymmetric (tanh is odd) and zero on self-loops. ∎
  - This matches AW-RoPE Lemma B.1 (Known result).
- **Proposition 7 (LapPE phases are not canonical).**
  - φ_i = νᵀ r_i with r_i = (u_1(i), …, u_m(i)). The map is equivariant *given* the eigenvector matrix U′ = PU.
  - But an eigensolver applied to A′ may return U′ = PUS, with S block-orthogonal on each eigenspace (S = diag(±1) when eigenvalues are simple). Then φ′ = PUSν ≠ PUν in general.
  - Training with random sign flips makes the *distribution* of outputs invariant to S ∈ {±1}^m. It gives no guarantee for repeated eigenvalues.
  - On Cora:
    - the null space has dimension 78 (= #components). We **skip** it, since its basis is arbitrary;
    - the 16 selected eigenvalues are simple, but the smallest gap is 2.2·10⁻⁴. By Davis–Kahan, eigenvectors with tiny gaps are unstable under perturbation (cf. Huang et al., arXiv 2310.02579).
  - **Status:** Proposition (non-canonicity) proved by the sign counterexample; the stability discussion is a known result.
  - Tested: projectors are equivariant on C₈, which has doubly degenerate eigenvalues. Model equivariance holds when U is permuted consistently.
- **Negative control.**
  - Index-RoPE, φ_i = i·θ, is not equivariant: φ_{π(i)} − φ_{π(j)} ≠ φ_i − φ_j. This is verified numerically.
  - This is why the prompt's warning ("do not assign a rotation based on node ID") is correct.

## 4. What node-wise phases can express in one-hop attention

- **Proposition 8 (telescoping).** For node-wise phases, the product of relative rotations along any closed walk is the identity (AW-RoPE Prop. B.2, Known result).
- **Remark (one-hop).** In one-hop attention each score uses a single edge, so circulation never enters a single layer's computation. The node-vs-edge distinction appears only through *which functions of (i, j) are representable*:
  - Node-wise phases give R(φ_j − φ_i): a gradient field.
  - Edge phases give R(ω a_ji): any antisymmetric field. This is strictly larger whenever the graph has cycles (Prop. B.11 of AW-RoPE).
- **Conjecture 1 (H6).** For homophilous graphs and low-frequency Laplacian coordinates, |φ_j − φ_i| over edges is small relative to random pairs. One-hop spectral RoPE then acts close to the identity, and A6 ≈ A2.
  - Measured by `phase_diagnostics` in `experiments/run_cora.py`.
  - Unproved in general. Heuristic: for an eigenvector u with eigenvalue λ, Σ_{(i,j)∈E} (u_i/√d_i − u_j/√d_j)² = λ, so small-λ coordinates vary slowly along edges.

## 5. Complexity (one layer, N nodes, E edges incl. self-loops, H, H_kv, head dim d_h, model dim d)

| Part | Time | Memory (activations) |
|---|---|---|
| Q projection | O(N d H d_h) | N H d_h |
| K, V projections | O(2 N d H_kv d_h) | 2 N H_kv d_h |
| node rotary | O(N H_kv d_h) (+ MLP for RWSE) | N H_kv d_h/2 phases |
| edge rotary | O(E d · hidden) scorer + O(E H_kv d_h) | E H_kv d_h/2 |
| gather q_i, k_j, v_j | – | **E H d_h** + **2 E H_kv d_h** |
| scores + segment softmax | O(E H d_h) | E H |
| weighted messages + scatter | O(E H d_h) | E H d_h |
| output projection | O(N H d_h d) | N d |

- GQA therefore reduces the gathered-K/V term only. The E H d_h terms (queries, messages) stay.
- **Prediction:** the per-layer activation saving is less than (1 − H_kv/H). It is measured in `results/gqa_efficiency.json`.
- **Structural preprocessing (one-off, per graph):**
  - LapPE (dense eigh): O(N³) time, O(N²) memory.
  - RWSE via eig: O(N³). RWSE via sparse powers: O(K · nnz(P^k)), which can fill in.
  - For Cora, N³ ≈ 2·10¹⁰ flops, about 3 s measured.
  - **The models are not linear-time end-to-end when LapPE is used.** The RWSE MLP-phase arm can use sparse or sampled estimators on large graphs (not implemented).

## 6. Over-smoothing / over-squashing (qualitative, not proved)

- All arms are one-hop, 2-layer, so the receptive field is 2 hops. Rotary phases change neighbour weights but not the receptive field, so they cannot by themselves address over-squashing (a topological bottleneck).
- RMSNorm rescales but does not re-center. Its interaction with over-smoothing (convergence of h_i to a common direction) is an **open question**, not addressed by these experiments.
