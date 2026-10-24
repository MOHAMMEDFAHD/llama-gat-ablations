"""Run ablation arms on Cora over multiple seeds. Resumable: finished (arm, seed) files are skipped.

usage: python experiments/run_cora.py --config configs/experiments.yaml [--arms A2 A6] [--seeds 0 1 2 3 4]
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.data.cora import load_cora  # noqa: E402
from src.graph.graph_rope import NodeRotary  # noqa: E402
from src.graph.registry import build_model  # noqa: E402
from src.training.graph_trainer import train_node_classifier  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def phase_diagnostics(model, data) -> dict | None:
    """H6: mean |sin(dphi/2)| over graph edges vs random node pairs, per layer (eval mode)."""
    if not hasattr(model, "blocks"):
        return None
    out = {}
    ei = data["edge_index"]; n = data["x"].shape[0]
    g = torch.Generator().manual_seed(0)
    rnd = torch.randint(0, n, (2, ei.shape[1]), generator=g)
    model.eval()
    with torch.no_grad():
        for li, blk in enumerate(model.blocks):
            rot = blk.attn.rotary
            if isinstance(rot, NodeRotary):
                ph = rot.phases({**data["struct"], "num_nodes": n})
                f = lambda p: float((torch.sin((ph[p[1]] - ph[p[0]]) / 2)).abs().mean())
                out[f"layer{li}"] = {"edge_mean_abs_sin_half_dphi": f(ei), "random_pair_mean_abs_sin_half_dphi": f(rnd)}
    return out or None


def meta() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None
    return {"torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(),
            "threads": torch.get_num_threads(), "cuda": torch.cuda.is_available(), "git_commit": commit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/experiments.yaml"))
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--out", default=str(ROOT / "results/runs"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(cfg.get("threads", 1))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", True) else "cpu")
    data = load_cora()
    data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}
    data["struct"] = {k: v.to(device) for k, v in data["struct"].items()}
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    arms = cfg["arms"]
    names = args.arms or list(arms)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for name in names:
        arm = arms[name]
        for seed in (args.seeds if args.seeds is not None else arm.get("seeds", seeds)):
            f = out / f"{name}_seed{seed}.json"
            if f.exists():
                continue
            set_seed(seed)
            kw = {**cfg.get("graphllama_backbone", {}), **arm.get("model_kwargs", {})} if arm["model"] == "graphllama" else arm.get("model_kwargs", {})
            model = build_model(arm["model"], data["x"].shape[1], data["num_classes"], **kw).to(device)
            tr = {**cfg["train"], **arm.get("train", {})}
            t0 = time.time()
            res = train_node_classifier(model, data, **tr)
            res.pop("history")
            rec = {"arm": name, "description": arm.get("description", ""), "seed": seed, "model": arm["model"],
                   "model_kwargs": kw, "train_cfg": tr, "wall_seconds": time.time() - t0,
                   "phase_diagnostics": phase_diagnostics(model, data), "data_meta": data["meta"], "env": meta(), **res}
            f.write_text(json.dumps(rec, indent=1, default=str))
            print(f"{name} seed={seed} val={res['val_acc']:.3f} test={res['test_acc']:.3f} s={rec['wall_seconds']:.0f}", flush=True)


if __name__ == "__main__":
    main()
