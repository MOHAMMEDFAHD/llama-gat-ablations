"""Validation-only grid for the shared GraphLlama backbone (arm A2, no positional info).
The selected setting is frozen for ALL GraphLlama arms. Test accuracy is never computed here."""
import itertools, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from src.data.cora import load_cora
from src.graph.model import GraphLlama
from src.training.graph_trainer import train_node_classifier
from src.utils.seed import set_seed

torch.set_num_threads(1)
data = load_cora()
grid = {"dropout": [0.6, 0.8], "weight_decay": [5e-4, 5e-3], "lr": [5e-3, 1e-2]}
out = []
for d, wd, lr in itertools.product(*grid.values()):
    vals = []
    for seed in (100, 101):  # tuning seeds disjoint from evaluation seeds 0..4
        set_seed(seed)
        m = GraphLlama(data["x"].shape[1], data["num_classes"], dropout=d, input_dropout=d, attn_dropout=0.3)
        r = train_node_classifier(m, data, lr=lr, weight_decay=wd, epochs=300, patience=100)
        vals.append(r["val_acc"])
    rec = {"dropout": d, "weight_decay": wd, "lr": lr, "val_mean": float(np.mean(vals)), "vals": vals}
    out.append(rec); print(json.dumps(rec), flush=True)
best = max(out, key=lambda r: r["val_mean"])
Path("results").mkdir(exist_ok=True)
Path("results/backbone_tuning_val.json").write_text(json.dumps({"grid": out, "selected": best}, indent=1))
print("SELECTED", best)
