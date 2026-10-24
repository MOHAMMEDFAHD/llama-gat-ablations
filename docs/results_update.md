# Results update (supersedes PENDING items in the draft report)

All numbers come from `results/summary.md` and `results/gqa_efficiency.json`, produced by the scripts in this repo.

## Baselines (5 seeds, test accuracy %)
| Model | Test acc | Params |
|---|---|---|
| GAT (manual) | 82.50 ± 0.71 | 92,373 |
| GCN | 81.74 ± 0.46 | 92,231 |
| GATv2 | 81.86 ± 0.99 | 184,604 |
| GraphSAGE | 78.42 ± 0.26 | 184,391 |
| GIN | 77.06 ± 1.57 | 100,551 |
| GraphLlama A2 (dot-product, no PE) | 78.64 ± 1.95 | 174,471 |

GT-LapPE and GraphGPS (global O(N^2) attention) were **not run** (too costly on 1 CPU core).

## Component ablations on A2 (3 seeds; paired vs A2 on seeds 0–2)
| Variant | Test acc | Δ vs A2 (pp) | 95% CI | p |
|---|---|---|---|---|
| LayerNorm instead of RMSNorm | 76.70 ± 1.11 | −0.93 | [−2.94, +1.07] | 0.18 |
| GELU-MLP instead of SwiGLU (param-matched) | 77.17 ± 2.59 | −0.47 | [−10.29, +9.35] | 0.86 |
| MQA (H_kv = 1) | 74.87 ± 2.12 | −2.77 | [−11.59, +6.05] | 0.31 |
| 3 layers | 77.37 ± 1.80 | −0.27 | [−3.09, +2.56] | 0.72 |

None of these differences is significant with 3 seeds. The rotary-dimension ablation (C_rotdim4) was **not run**.

## GQA efficiency: one attention layer, H = 8, dim 64, forward + backward
| Graph | H_kv | K/V params | Autograd-saved MB | Saving vs MHA | fwd+bwd ms |
|---|---|---|---|---|---|
| Cora (E = 13,264), no rotary | 8 | 8,192 | 16.77 | – | 25.6 |
| | 4 | 4,096 | 13.53 | 19% | 21.7 |
| | 2 | 2,048 | 11.91 | 29% | 16.3 |
| | 1 | 1,024 | 11.10 | 34% | 18.1 |
| Synthetic (N = 20k, E = 419,802) | 8 | 8,192 | 492.0 | – | 1048 |
| | 4 | 4,096 | 389.5 | 21% | 924 |
| | 2 | 2,048 | 338.3 | 31% | 780 |
| | 1 | 1,024 | 312.7 | 36% | 775 |

These results confirm the analytical prediction in `mathematics.md` §5. GQA cuts K/V parameters by H/H_kv, but activation memory falls only 29–31% at H_kv = 2, versus the naive 75%. That is because gathered queries and weighted messages stay at E·H·d_h.

Timings are CPU, single-threaded, and may have overlapped with other jobs, so treat them as indicative only.

## Revised H3 verdict
- **Memory:** a real saving, but far smaller than the parameter ratio.
- **Accuracy cost:** −0.5 pp (spectral) to −2.2 pp (RWSE) at H_kv = 2, and −2.8 pp for MQA. None is significant.
- **Conclusion:** H3 is only partially supported.
