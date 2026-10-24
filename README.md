# llama-graph-attention

A research codebase for one question: **do LLaMA-style attention components — RMSNorm, SwiGLU, GQA, rotary encodings — help graph attention networks?**

It contains two independent implementations:

1. **`src/llama/`** — a from-scratch, configurable LLaMA-style decoder.
   - Includes RMSNorm, RoPE (two layouts, partial rotary, linear / LLaMA-3.1 scaling), MHA/GQA/MQA, and SDPA plus a manual reference backend.
   - Includes SwiGLU, a KV cache, and greedy / temperature / top-k / top-p decoding.
   - Verified against a Meta-style complex RoPE reference and `torch.nn.RMSNorm`. Trains on a synthetic task and generates correctly.
2. **`src/graph/`** — sparse one-hop graph attention in a LLaMA-style block.
   - Rotary options: none, **index** (invalid control), **spectral** (WIRE-style), **RWSE-MLP**, and **edge** (antisymmetric, AW-RoPE-inspired).
   - Also includes MHA/GQA/MQA and optional structural bias.
   - Heads for node classification, graph classification and link prediction.
   - Baselines: manual GAT (checked equal to PyG `GATConv`), GATv2, GCN, GraphSAGE, GIN, a global Graph Transformer with LapPE, and GraphGPS.

**Read first:** `docs/literature_review.md` and `docs/novelty_matrix.md`.

- The literature check found the core "graph-aware RoPE" idea already published.
  - Node-wise spectral phases: **WIRE**, Reid et al., ICML 2026.
  - Edge- and path-conditioned rotary transport: **AW-RoPE**, Xie et al., 2026.
- The hypotheses were revised accordingly. This repository does **not** claim a new attention mechanism.

## Status (see `PROGRESS.md`)
- Tests: `python -m pytest tests -q` → 62 passed (CPU).
- Cora experiments: `results/summary.md` (real runs; numbers are not edited by hand).
- Not done: LRGB / ZINC / heterophilous benchmarks; official WIRE / AW-RoPE code comparisons; GPU runs.

## Quick start
```bash
pip install -r requirements.txt
bash scripts/get_cora.sh
python -m pytest tests -q
python experiments/run_cora.py && python experiments/summarize.py
```

## Layout
```
src/llama      LLaMA reference (config, rmsnorm, rope, attention, swiglu, decoder, model, generation)
src/graph      sparse_ops, gat (manual), baselines (PyG), graph_rope, grouped_attention, graph_block, model
src/data       cora loader, structural encodings (LapPE w/ ambiguity diagnostics, RWSE), synthetic LM task
src/training   graph trainer (early stop, resume, AMP), LM trainer (checkpointing)
experiments    run_cora.py (resumable ablation runner), summarize.py (paired stats)
scripts        tune_backbone_val.py (validation-only), bench_gqa.py, get_cora.sh, smoke_cora.py
docs           literature_review, novelty_matrix, mathematics, architecture, reproducibility, research_report
results        runs/*.json (one per arm x seed), summary.md/json, gqa_efficiency.json, backbone_tuning_val.json
```
