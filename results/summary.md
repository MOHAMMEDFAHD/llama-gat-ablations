# Cora results (public Planetoid split)

Mean ± sample std over seeds; 95% CI is a t-interval. Model selection on validation accuracy only.

| Arm | Description | n | Test acc | 95% CI | Val acc | Test macro-F1 | Params | s/epoch (CPU, 1 thread) |
|---|---|---|---|---|---|---|---|---|
| A0 | GAT (manual, additive) | 5 | 82.50 ± 0.71 | [81.62, 83.38] | 81.16 | 81.32 | 92,373 | 0.102 |
| A1 | GATv2 (PyG) | 5 | 81.86 ± 0.99 | [80.63, 83.09] | 81.64 | 80.67 | 184,604 | 0.126 |
| A2 | dot-product, no PE (LLaMA block) | 5 | 78.64 ± 1.95 | [76.22, 81.06] | 77.96 | 77.64 | 174,471 | 0.178 |
| A3 | index-RoPE on node ids (INVALID control) | 5 | 79.14 ± 1.15 | [77.71, 80.57] | 78.80 | 78.13 | 174,471 | 0.205 |
| A4 | absolute LapPE (sign-flip aug.) | 5 | 74.34 ± 2.00 | [71.85, 76.83] | 74.88 | 72.33 | 175,559 | 0.181 |
| A5 | WIRE-style spectral RoPE, fixed random freqs | 5 | 78.16 ± 0.61 | [77.40, 78.92] | 78.08 | 77.01 | 174,471 | 0.191 |
| A6 | spectral RoPE, learnable freqs, MHA | 5 | 77.78 ± 1.31 | [76.16, 79.40] | 78.20 | 76.67 | 175,495 | 0.195 |
| A7 | spectral RoPE, GQA (H_kv=2) | 5 | 77.32 ± 1.53 | [75.42, 79.22] | 78.28 | 76.37 | 162,439 | 0.181 |
| A8 | RWSE-MLP phases, MHA | 5 | 78.96 ± 1.63 | [76.93, 80.99] | 78.36 | 77.75 | 177,671 | 0.198 |
| A9 | RWSE-MLP phases, GQA (H_kv=2) | 5 | 76.76 ± 1.21 | [75.26, 78.26] | 77.96 | 75.80 | 163,799 | 0.176 |
| A10 | A9 + RWSE structural bias | 5 | 78.00 ± 1.27 | [76.42, 79.58] | 77.52 | 76.83 | 166,439 | 0.200 |
| A10e | edge (antisymmetric) phases, MHA | 5 | 78.84 ± 1.09 | [77.49, 80.19] | 78.40 | 77.54 | 183,263 | 0.275 |
| B_gcn | GCN (PyG) | 5 | 81.74 ± 0.46 | [81.17, 82.31] | 80.16 | 80.77 | 92,231 | 0.093 |
| B_gin | GIN (PyG) | 5 | 77.06 ± 1.57 | [75.11, 79.01] | 77.28 | 76.48 | 100,551 | 0.180 |
| B_sage | GraphSAGE (PyG) | 5 | 78.42 ± 0.26 | [78.10, 78.74] | 79.04 | 77.68 | 184,391 | 0.184 |
| C_3layers | A2 with 3 layers | 3 | 77.37 ± 1.80 | [72.89, 81.85] | 78.00 | 76.19 | 215,559 | 0.227 |
| C_gelu | A2 with GELU-MLP (param-matched) | 3 | 77.17 ± 2.59 | [70.74, 83.60] | 77.93 | 76.28 | 174,983 | 0.173 |
| C_layernorm | A2 with LayerNorm | 3 | 76.70 ± 1.11 | [73.93, 79.47] | 77.80 | 75.57 | 174,791 | 0.165 |
| C_mqa | A2 with MQA (H_kv=1) | 3 | 74.87 ± 2.12 | [69.60, 80.14] | 76.20 | 73.92 | 160,135 | 0.157 |

## Paired comparisons (test accuracy, percentage points, same seeds)

| Comparison | a − b | 95% CI (paired t) | p (paired t) | n |
|---|---|---|---|---|
| H1: spectral RoPE vs plain dot-product (A6 − A2) | -0.86 | [-4.44, +2.72] | 0.541 | 5 |
| H1': WIRE-style fixed-freq vs plain (A5 − A2) | -0.48 | [-3.61, +2.65] | 0.693 | 5 |
| H2': RWSE-phase vs LapPE-phase (A8 − A6) | +1.18 | [-0.34, +2.70] | 0.098 | 5 |
| H3: GQA vs MHA (spectral) (A7 − A6) | -0.46 | [-3.19, +2.27] | 0.664 | 5 |
| H3: GQA vs MHA (RWSE) (A9 − A8) | -2.20 | [-4.57, +0.17] | 0.061 | 5 |
| A10: +struct bias vs A9 (A10 − A9) | +1.24 | [-0.57, +3.05] | 0.130 | 5 |
| edge-phase vs node-phase (RWSE) (A10e − A8) | -0.12 | [-2.28, +2.04] | 0.885 | 5 |
| abs LapPE vs plain (A4 − A2) | -4.30 | [-7.84, -0.76] | 0.028 | 5 |
| index-RoPE (invalid) vs plain (A3 − A2) | +0.50 | [-1.26, +2.26] | 0.474 | 5 |
| GraphLlama-plain vs GAT (A2 − A0) | -3.86 | [-5.87, -1.85] | 0.006 | 5 |
| GATv2 vs GAT (A1 − A0) | -0.64 | [-2.47, +1.19] | 0.388 | 5 |
| LayerNorm vs RMSNorm (C_layernorm − A2) | -0.93 | [-2.94, +1.07] | 0.184 | 3 |
| GELU-MLP vs SwiGLU (C_gelu − A2) | -0.47 | [-10.29, +9.35] | 0.857 | 3 |
| MQA vs MHA (plain) (C_mqa − A2) | -2.77 | [-11.59, +6.05] | 0.310 | 3 |
| 3 layers vs 2 (C_3layers − A2) | -0.27 | [-3.09, +2.56] | 0.724 | 3 |

## H6 phase diagnostic: mean |sin(Δφ/2)| (edges vs random node pairs), trained models

| Arm | Layer | Edges | Random pairs | Ratio |
|---|---|---|---|---|
| A10 | layer0 | 0.1093 | 0.1716 | 0.637 |
| A10 | layer1 | 0.0897 | 0.1561 | 0.575 |
| A3 | layer0 | 0.5592 | 0.5811 | 0.962 |
| A3 | layer1 | 0.5592 | 0.5811 | 0.962 |
| A5 | layer0 | 0.3654 | 0.5364 | 0.681 |
| A5 | layer1 | 0.3645 | 0.5373 | 0.678 |
| A6 | layer0 | 0.3656 | 0.5367 | 0.681 |
| A6 | layer1 | 0.3642 | 0.5372 | 0.678 |
| A7 | layer0 | 0.3785 | 0.5482 | 0.690 |
| A7 | layer1 | 0.3393 | 0.5074 | 0.669 |
| A8 | layer0 | 0.1008 | 0.1656 | 0.609 |
| A8 | layer1 | 0.0838 | 0.1421 | 0.590 |
| A9 | layer0 | 0.1003 | 0.1655 | 0.606 |
| A9 | layer1 | 0.1469 | 0.1999 | 0.735 |
