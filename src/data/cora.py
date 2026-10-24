"""Cora (Planetoid; Yang, Cohen & Salakhutdinov 2016) with the public split:
20 labelled nodes / class for training (140), 500 validation, 1000 test.
Features are row-normalised (NormalizeFeatures), as in GCN/GAT.
Structural encodings are computed from the graph only (no labels) and cached.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from .structural import compute_structural

ROOT = Path(__file__).resolve().parents[2] / "data_cache"


def load_cora(root: Path | str = ROOT, lap_k: int = 16, rwse_steps: int = 16) -> dict:
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T

    ds = Planetoid(str(root), "Cora", split="public", transform=T.NormalizeFeatures())
    d = ds[0]
    cache = Path(root) / f"cora_struct_k{lap_k}_rw{rwse_steps}.pt"
    if cache.exists():
        struct = torch.load(cache, weights_only=False)
    else:
        struct = compute_structural(d.edge_index, d.num_nodes, lap_k, rwse_steps)
        torch.save(struct, cache)
    raw = Path(root) / "Cora" / "raw"
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(raw.glob("*")))).hexdigest()[:16]
    meta = {"dataset": "Cora (Planetoid, public split)", "num_nodes": d.num_nodes,
            "num_edges_directed": int(d.edge_index.shape[1]), "num_features": ds.num_features,
            "num_classes": ds.num_classes, "train/val/test": [int(d.train_mask.sum()), int(d.val_mask.sum()), int(d.test_mask.sum())],
            "raw_sha256_16": digest, "structural": struct["diagnostics"]}
    return {"x": d.x, "y": d.y, "edge_index": d.edge_index, "train_mask": d.train_mask, "val_mask": d.val_mask,
            "test_mask": d.test_mask, "num_classes": ds.num_classes,
            "struct": {k: struct[k] for k in ("lap_vecs", "lap_vals", "rwse")}, "meta": meta}


if __name__ == "__main__":
    t = time.time(); data = load_cora(); print(json.dumps(data["meta"], indent=1, default=str)); print("seconds", time.time() - t)
