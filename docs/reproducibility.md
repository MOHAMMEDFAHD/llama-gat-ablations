# Reproducibility

## Environment
- Python 3.12, PyTorch 2.14.1 (PyPI wheel), torch_geometric 2.x, numpy 2.4, scipy 1.17.
- Hardware used for reported numbers: **1 CPU core, 3 GB RAM, no GPU**. `torch.set_num_threads(1)`.
- The PyPI Linux torch wheel needs the CUDA runtime wheels at import time, even on CPU.
  - `pip install -r requirements.txt` resolves them.
  - Alternatively, a CPU-only wheel from `download.pytorch.org/whl/cpu` avoids them. That index was unreachable in our sandbox.

## Data
- Cora (Planetoid; Yang et al. 2016), public split: 140 / 500 / 1000.
- Raw files come from `raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.cora.*`. The PyG downloader failed through our proxy, so `scripts/get_cora.sh` fetches them directly.
- SHA-256 prefix of the concatenated raw files is recorded in every result JSON (`data_meta.raw_sha256_16`).
- Features are row-normalised (`NormalizeFeatures`).
- Structural encodings use the graph only (no labels) and are cached in `data_cache/`. The transductive setting means test nodes' *structure* (not labels) is visible, which is standard for Planetoid.

## Protocol
1. Hyper-parameter search touches validation accuracy only (`scripts/tune_backbone_val.py`).
   - Tuning seeds are 100 and 101; evaluation seeds are 0–4.
   - The shared GraphLlama backbone was selected **once on arm A2** and frozen for all GraphLlama arms. Rotary arms received **no extra tuning**.
   - This favours none of the rotary variants, but may under-tune them.
2. Baselines use literature defaults (`configs/graph_baselines.yaml`).
3. Model selection: best validation accuracy, with ties broken by lower validation loss. Test is evaluated once, with the selected weights.
4. Determinism: `set_seed` seeds python, numpy and torch, and enables `torch.use_deterministic_algorithms(warn_only=True)`. A CPU rerun with the same seed reproduces bit-identical training losses (tested in `test_trainer_resume_is_bitwise_identical` and `test_reproducible_under_seed`).

## Commands
```bash
pip install -r requirements.txt
bash scripts/get_cora.sh
python -m pytest tests -q                     # 62 tests
python scripts/tune_backbone_val.py           # validation-only
python experiments/run_cora.py                # all arms x seeds (resumable)
python experiments/summarize.py               # results/summary.md
python scripts/bench_gqa.py                   # results/gqa_efficiency.json
```

## Memory measurement caveat
- On CPU, `cpu_max_rss_mb` is the process-wide maximum RSS. It is coarse, and it is monotone across runs within one process.
- For GQA we therefore report **bytes saved by autograd for backward**. It is exact, deterministic and device-independent.
- On CUDA, `torch.cuda.max_memory_allocated` is also recorded.
