# Literature Review — LLaMA-Inspired Graph Attention

Date of investigation: 2026-10-24.

## Verification levels

Every entry carries one of two tags. Do not cite an entry in a paper without upgrading it to **[V-fetched]**.

- **[V-fetched]** — the source was retrieved during this investigation (arXiv abstract page, PDF, official repo, or proceedings listing) and the claims below were checked against it.
- **[V-prior]** — a well-established paper described from the team's background knowledge. Title/authors/venue are standard, but formulas and numbers were *not* re-checked this session. Re-verify before citing.

## 1. The three priority papers (all [V-fetched])

### 1.1 Rotary Position Encodings for Graphs (WIRE)
- **Authors:** Isaac Reid, Arijit Sehanobish, Cederik Höfs, Bruno Mlodozeniec, Leonhard Vulpius, Federico Barbero, Adrian Weller, Krzysztof Choromanski, Richard E. Turner, Petar Veličković.
- **Venue/year:** arXiv:2509.22259 (v1 Sep 2025, v5 Aug 2026); ICML 2026 spotlight per the official repo `github.com/cederikhoefs/Graph-RoPE`.
- **Name:** WIRE = *Wave-Induced Rotary Encodings*. (A third-party paper calls it "Wavelet-Induced"; that is a misnomer.)
- **Problem:** Can RoPE, as used in LLMs and ViTs, be applied to graphs?
- **Formulation:** Each node u gets spectral coordinates r_u, e.g. selected Laplacian eigenvector components with optional eigenvalue scaling. Channel pair ℓ gets phase φ_{uℓ} = ν_ℓᵀ r_u. Queries and keys are rotated by R(φ), so (R(φ_u)q)ᵀ(R(φ_v)k) = qᵀR(φ_v − φ_u)k. This is the node-wise extension of RoPE.
- **Properties claimed:**
  - recovers standard RoPE on grids;
  - asymptotically depends on graph effective resistance;
  - compatible with linear attention, unlike bias-based relative encodings.
- **Other details:**
  - RoPE: yes, node-wise. GQA/MQA: not reported.
  - Attention: evaluated inside GraphGPS-style models with linear (ReLU / Performer) global attention, per AW-RoPE's description of the protocol.
  - Also has an RWPE-coordinate variant that reads return probabilities, per AW-RoPE §2.
- **Overlap with us:** very high. "Graph-aware RoPE with node-wise spectral or structural phases" is WIRE.

### 1.2 Analytic-Walk Rotary Positional Encodings for Graphs (AW-RoPE)
- **Authors:** Jiaqing Xie, Yuxin Wang, Xipeng Qiu (Shanghai Innovation Institute / Fudan).
- **Source:** arXiv:2609.32178 (2026). Code: `github.com/jiaqingxie/AW-RoPE`.
- **Core argument:** With node-wise phases, route phases telescope to φ_v − φ_u. So all routes between the same endpoints get identical accumulated phase, and cycle circulation is always zero (their Prop. B.2).
- **Formulation:**
  - A shared scorer gives an antisymmetric bounded edge displacement a_uv = a_max·tanh((s([h_u,h_v]) − s([h_v,h_u]))/a_max).
  - One-step transport is T_uv = P_uv·e^{iω a_uv}.
  - Exact variant: queries and keys are transformed by the resolvent (I − zT)⁻¹. Sparse variant: truncated sum Σ_{k≤L} z^k T^k.
- **Complexity:** Exact is O(N³d) time and O(N²d) memory. Sparse is O(L(N+M)d), excluding the scorer and the attention backbone.
- **Theory:** permutation equivariance (Prop. B.6); gradient fields recover node-wise RoPE (Thm. B.7); truncation bounds for the forward pass and the parameter derivative.
- **Evidence (as reported by the authors, not re-run by us):**
  - 15–58% nRMSE reduction on five synthetic tasks;
  - best mean on 12/13 datasets with a GIN backbone and on all Performer tasks;
  - phase utility is task-dependent (MolHIV: zero field beats learned field).
- **Overlap with us: decisive.** The prompt's candidates "pairwise edge-conditioned rotations" and "path-sensitive rotary mechanisms" (§4.3 G, §4.4) are exactly AW-RoPE's contribution. They cannot be claimed as ours.

### 1.3 Weaving Graph over Tokens: Contextualizing Structured Sequences for LLMs (Weaver)
- **Venue:** ICML 2026 (OpenReview listing, found via a third-party index).
  - **Author list NOT verified** — retrieve from OpenReview before citing.
  - **arXiv id not found** in our searches.
- **Setting:** decoder-only LLMs over serialized graphs (tokens grouped into nodes). It does not address GNN node classification.
- **Mechanisms:**
  - graph-causal masking: causal within a node, bidirectional between nodes;
  - Graph-over-Tokens RoPE (GoRoPE), which maps graph distances into rotary positions.
- **Reported result:** +13.2% SceneGraph QA over the base model.
- **Overlap:** conceptual (graph distance → RoPE). The setting differs: a token-level LLM, not a node-level GNN.

## 2. Established background ([V-prior] unless marked)

| # | Paper | Attention form | PE / SE | Local / global | Notes relevant here |
|---|---|---|---|---|---|
| 1 | GAT — Veličković et al., ICLR 2018 | additive: LeakyReLU(aᵀ[Wh_i‖Wh_j]) | none | local | *static* attention (see GATv2) |
| 2 | GATv2 — Brody, Alon, Yahav, ICLR 2022 [V-fetched: arXiv 2105.14491] | aᵀLeakyReLU(W[h_i‖h_j]) | none | local | proves GAT static. Also notes Transformer-style dot-product graph attention ("DPGAT") is *strictly weaker* than dynamic attention in their sense — relevant to our A2 baseline. |
| 3 | Dwivedi & Bresson, "A generalization of transformer networks to graphs", 2020 | dot-product, neighborhood-masked | Laplacian eigenvectors (LapPE), random sign flip | local | origin of LapPE in graph Transformers |
| 4 | SAN — Kreuzer et al., NeurIPS 2021 | full + sparse | learned spectral attention over eigenpairs | global | |
| 5 | Graphormer — Ying et al., NeurIPS 2021 | dot-product + additive bias b_φ(SPD) + edge-path bias | centrality + SPD bias | global, O(N²) | bias-based relative PE |
| 6 | GraphGPS — Rampášek et al., NeurIPS 2022 | MPNN + global attention per layer | LapPE / RWSE / ... | hybrid | standard hybrid baseline |
| 7 | LSPE / RWSE — Dwivedi et al., ICLR 2022 | – | random-walk return probabilities (P^k)_ii | – | sign-free, permutation-equivariant |
| 8 | SignNet/BasisNet — Lim et al., ICLR 2023 | – | sign- and basis-invariant eigenvector processing | – | addresses eigenvector ambiguity |
| 9 | Stability of PEs — Huang et al. ("On the stability of expressive PEs for graphs", arXiv 2310.02579) | – | SPE | – | eigenvalue-gap instability |
| 10 | RoFormer — Su et al., 2021 (Neurocomputing 2024) | dot-product with rotated q,k | RoPE | – | (R_m q)ᵀ(R_n k) = qᵀR_{n−m}k |
| 11 | MQA — Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", 2019 | 1 KV head | – | – | targets decode memory bandwidth |
| 12 | GQA — Ainslie et al., EMNLP 2023 | H_kv groups | – | – | used in LLaMA 2 (70B) and LLaMA 3 |
| 13 | Exphormer — Shirzad et al., ICML 2023 | sparse (graph + expander + virtual nodes) | – | sparse global | efficient graph transformer |
| 14 | GRIT — Ma et al., ICML 2023 | MPNN-free, learned RRWP | relative random-walk PE | global | |
| 15 | Over-squashing — Alon & Yahav, ICLR 2021; Topping et al., ICLR 2022 (curvature) | – | – | – | bottlenecks / long range |
| 16 | Over-smoothing — Li, Han, Wu AAAI 2018; Oono & Suzuki ICLR 2020 | – | – | – | depth collapse |
| 17 | LRGB — Dwivedi et al., NeurIPS 2022 D&B [cited in AW-RoPE] | – | – | – | long-range benchmark |
| 18 | MagNet — Zhang et al., NeurIPS 2021 [cited in AW-RoPE] | – | magnetic Laplacian phases | – | fixed U(1) edge phases from direction |
| 19 | Neural Sheaf Diffusion — Bodnar et al., NeurIPS 2022 | – | learned restriction maps | – | learned edge transport |
| 20 | RotatE — Sun et al., ICLR 2019 | – | relation = rotation in ℂ | – | edge-type rotation precedent in KG embeddings |
| 21 | GraphSAGE (Hamilton et al. 2017), GCN (Kipf & Welling 2017), GIN (Xu et al. 2019) | – | – | local | baselines |
| 22 | LLaMA (Touvron et al. 2023a), LLaMA 2 (2023b), LLaMA 3 "The Llama 3 Herd of Models" (2024) | GQA (L2-70B, all L3) | RoPE, θ=10⁴ (L1/L2), 5·10⁵ (L3); L3.1 adds frequency scaling | causal | reference architecture |

**GQA/MQA in graph attention:** our searches found no paper that studies GQA as a design axis for sparse graph neighborhood attention. This is an absence of evidence from a limited search, not proof of absence.

## 3. Conclusions for the research plan

1. **Node-wise graph RoPE is established (WIRE).** Spectral and RWPE coordinates are both covered.
2. **Edge-conditioned, path-sensitive rotary transport is established (AW-RoPE).** It comes with proofs of equivariance and of the node-wise obstruction.
3. **The provisional SAG-RoGAT idea, "rotation parameters conditioned on structural features", is therefore largely pre-empted.**
   - Node-conditioned rotations are WIRE-like: a learned map from structural features to phases is a parameterization of WIRE's coordinates.
   - Edge-conditioned rotations are AW-RoPE-like: an L=1 truncation.
4. **What remains defensibly open (and modest):**
   - **(a)** A controlled study of graph RoPE inside *local, one-hop sparse softmax attention* (a GAT-replacement setting), versus global/linear attention in WIRE/AW-RoPE. Open question: does the relative phase φ_j − φ_i between *adjacent* nodes carry usable signal when low-frequency eigenvectors vary slowly along edges? This is falsifiable.
   - **(b)** Exact permutation-equivariance analysis of spectral phases on *disconnected* graphs (e.g. Cora, 78 components) — repeated zero eigenvalues make the eigenbasis arbitrary.
   - **(c)** GQA in sparse graph attention: what it actually saves (K/V projection parameters, and gathered per-edge K/V tensors E·H_kv·d instead of E·H·d), measured rather than assumed.
   - **(d)** A LLaMA-block graph Transformer (RMSNorm + SwiGLU + GQA + graph RoPE) as an *engineering adaptation*, with clean ablations.

None of (a)–(d) is a new attention mechanism. The revised hypotheses are in `novelty_matrix.md`.
