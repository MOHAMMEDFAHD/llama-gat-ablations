"""Quick timing/sanity run (validation accuracy only is printed by default)."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.data.cora import load_cora
from src.graph.registry import build_model
from src.training.graph_trainer import train_node_classifier
from src.utils.seed import set_seed

torch.set_num_threads(1)
data = load_cora()
arms = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [["gat", {}], ["graphllama", {}], ["graphllama", {"rotary": "spectral"}]]
for name, kw in arms:
    set_seed(0)
    m = build_model(name, data["x"].shape[1], data["num_classes"], **kw)
    t = time.time(); r = train_node_classifier(m, data, epochs=200, patience=100)
    print(name, kw, "val=%.3f" % r["val_acc"], "best_ep", r["best_epoch"], "s/ep=%.3f" % r["seconds_per_epoch"], "params", r["num_parameters"], flush=True)
