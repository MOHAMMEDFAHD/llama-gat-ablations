# PROGRESS

## Completed
- [x] Environment inspection: 1 CPU core, 3 GB RAM, no GPU; PyTorch 2.14.1 + CUDA runtime wheels (CPU use only); PyG.
- [x] Literature / novelty investigation (`docs/literature_review.md`, `docs/novelty_matrix.md`).
  - WIRE (node-wise spectral RoPE) and AW-RoPE (edge/walk rotary) pre-empt the provisional SAG-RoGAT idea.
  - Hypotheses revised; H2 withdrawn as a novelty claim; H6 added.
- [x] LLaMA reference implementation + 31 tests:
  - RMSNorm, RoPE (2 layouts, partial, scaling), GQA/MQA, SDPA/manual, SwiGLU, KV cache, sampling;
  - synthetic training/generation, checkpoint resume.
- [x] Graph baselines:
  - manual GAT, cross-checked against PyG GATConv;
  - GATv2, GCN, SAGE, GIN, GT-LapPE, GraphGPS (PyG).
- [x] Graph rotary variants (index control, spectral/WIRE-style, RWSE-MLP, edge), sparse GQA attention, LLaMA-style graph block, node/graph/link heads.
- [x] Graph tests (31): sparse = dense, GQA = duplicated-MHA, permutation equivariance per arm, edge cases, batching, resume, AMP, tasks.
- [x] Bugs found by tests and fixed:
  - float64 down-cast in RMSNorm and in the graph softmax;
  - bf16 scatter dtype mismatch;
  - default-dtype leak between test modules.
- [x] Validation-only backbone tuning (8 configs × 2 tuning seeds; flat landscape 0.785–0.792).
- [x] Cora experiment runner (resumable) + paired-statistics summarizer.

## Experimental status
- Done: A0–A10e (5 seeds), GCN/SAGE/GIN (5 seeds), C_layernorm/C_gelu/C_mqa/C_3layers (3 seeds), GQA benchmark.
- Not run: C_rotdim4, B_gt, B_gps (global O(N^2) baselines; too slow on 1 CPU core). Resume with `python experiments/run_cora.py`.
- See `results/summary.md` and `docs/results_update.md`.

## Failed / not verified
- PyG's Planetoid downloader fails behind the sandbox proxy (aiohttp header size) → `scripts/get_cora.sh`.
- Background processes do not survive between sessions; the runner is resumable to compensate.
- The WIRE-style and AW-style arms are our simplified re-implementations; official code was not run.

## Open research questions
- Does any graph RoPE help *one-hop* attention on a homophilous graph (H1, H6)?
- GQA + rotary: phases are tied per KV group, so GQA reduces the number of independent phase functions. Is that the cost?
- Long-range tasks (LRGB, synthetic SPD) — where rotary/structural encodings are reported to matter — untested here.

## Next actions
1. Heterophilous graphs (Chameleon/Actor) and LRGB Peptides with the same arms.
2. Run official WIRE and AW-RoPE code as baselines under our protocol.
3. Multi-hop / global attention variants, where relative phases between distant nodes are non-trivial.
