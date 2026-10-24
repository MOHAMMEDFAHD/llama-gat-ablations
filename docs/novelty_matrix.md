# Novelty Matrix

Legend: ✔ = present in the work, ✘ = absent, ~ = partial or different setting.

| Component | GAT | GATv2 | Graphormer | GraphGPS | WIRE (2025/26) | AW-RoPE (2026) | Weaver (2026) | **This project** | Status for us |
|---|---|---|---|---|---|---|---|---|---|
| Neighborhood (local) softmax attention | ✔ additive | ✔ dynamic additive | ✘ (global) | ✔ (MPNN part) | ✘ evaluated global/linear | ✘ global/linear + GIN | ✘ | ✔ dot-product | Established |
| Dot-product Q/K graph attention | ✘ | ✘ (DPGAT analysed) | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | Established |
| Node-wise RoPE from spectral coords | ✘ | ✘ | ✘ | ✘ | ✔ | (baseline) | ~ GoRoPE on distances | A6/A7 | **Established (WIRE)** |
| Node-wise RoPE from RW / structural coords | ✘ | ✘ | ✘ | ✘ | ✔ RWPE variant | ✘ | ✘ | A8/A9 (learned MLP(RWSE)→phase) | **Largely established**; learned-MLP parameterization is a minor variant |
| Edge-conditioned / path-sensitive rotation | ✘ | ✘ | ✘ | ✘ | ✘ | ✔ | ✘ | A10-edge (one-hop, L=1 simplification) | **Established (AW-RoPE)** |
| Structural attention bias b_ij | ✘ | ✘ | ✔ SPD | ~ | ✘ (contrasted) | ✘ | ✘ | optional | Established |
| GQA / MQA in graph attention | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ~ (inherits LLM's) | ✔ measured | **Not found in our search**; engineering adaptation |
| RMSNorm + SwiGLU pre-norm graph block | ✘ | ✘ | ✘ (LayerNorm) | ✘ (BN/LN) | ✘ | ✘ | ~ (LLM) | ✔ | Engineering adaptation |
| Equivariance analysis on disconnected graphs (eigenspace ambiguity) | – | – | – | – | partial (sign handling unverified by us) | ✔ for AW | – | ✔ numerical + analytic notes | Analysis contribution, not mechanism |

## Classification of what this project can claim

| Category | Items |
|---|---|
| Previously established ideas | RoPE; GQA/MQA; RMSNorm; SwiGLU; GAT/GATv2; dot-product graph attention; LapPE; RWSE; node-wise graph RoPE (WIRE); edge-rotary walk transport (AW-RoPE); Graphormer biases |
| Engineering adaptations | LLaMA-style graph block; GQA in sparse neighborhood attention with K/V gathered at H_kv heads; RWSE-MLP phase parameterization |
| Potentially novel (weak) | An empirical finding about graph RoPE in *one-hop local* softmax attention on homophilous citation graphs. This would only count if it holds across seeds, and the evidence would be a measurement rather than a mechanism. |
| Requires theoretical proof | Exact equivariance of each rotary variant (done analytically for RWSE and edge variants in `mathematics.md`). LapPE variant: equivariant *only* given a consistent eigenbasis, which cannot be canonical under repeated eigenvalues. |
| Requires experimental validation | H1–H5 below |

## Revised hypotheses (replacing the provisional H1–H5)

The original H2 ("our structure-adaptive rotation beats existing graph RoPE") is **withdrawn as a novelty claim**. Our mechanism is a WIRE-family parameterization, so H2′ below is a comparison *within* the WIRE family.

| ID | Hypothesis | Falsified if |
|---|---|---|
| H1 | Node-wise graph RoPE (LapPE phases) improves one-hop dot-product attention (A6 vs A2) on Cora. | 95% CI of the accuracy difference (5 seeds) includes 0 or is negative |
| H2′ | RWSE-conditioned learned phases (A8) match or beat LapPE phases (A6) in one-hop attention, while being exactly permutation-equivariant. | A8 significantly below A6 |
| H3 | GQA (H_kv = H/4) reduces K/V parameters and gathered K/V memory by ≈4× with < 1 pt accuracy loss vs MHA (A7 vs A6, A9 vs A8). | accuracy drop ≥ 1 pt, or memory saving not observed |
| H4 | (Long-range) | **Not testable on Cora** — Cora is homophilous with short-range dependencies; deferred to LRGB/synthetic |
| H5 | RWSE-phase and edge-phase layers are exactly node-permutation equivariant; LapPE-phase layers are not equivariant to the *eigensolver's* basis choice. | numerical tests contradict the analytic claims |
| H6 (new) | Local graph RoPE is near-identity on homophilous graphs: adjacent-node phase differences |φ_j − φ_i| are small for low-frequency eigenvectors, so A6 ≈ A2. | measured phase differences are large and A6 ≫ A2 |
