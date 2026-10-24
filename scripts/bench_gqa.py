"""Measure what GQA saves in sparse graph attention (one GraphGQAttention layer, forward + backward).

Metrics:
  * K/V projection parameters;
  * bytes of tensors saved by autograd for backward (activation memory, exact and deterministic);
  * wall-clock forward+backward (median of repeats, CPU 1 thread).
Graphs: Cora (N=2708, E=13264 with self-loops) and a synthetic random graph (N=20k, avg deg 20).
"""
import json, statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.data.cora import load_cora
from src.graph.grouped_attention import GraphGQAttention
from src.graph.graph_rope import RWSERotary
from src.graph.sparse_ops import apply_self_loop_policy

torch.set_num_threads(1)


def saved_bytes(fn):
    total = [0]; seen = set()
    def pack(t):
        if t.data_ptr() not in seen:
            seen.add(t.data_ptr()); total[0] += t.numel() * t.element_size()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, total[0]


def bench(n, ei, dim=64, heads=8, reps=5, struct=None):
    rows = []
    x = torch.randn(n, dim)
    for hkv in (8, 4, 2, 1):
        torch.manual_seed(0)
        rot = RWSERotary(hkv, dim // heads, struct["rwse"].shape[1]) if struct is not None else None
        att = GraphGQAttention(dim, heads, hkv, rotary=rot)
        ctx = struct or {}
        out, sb = saved_bytes(lambda: att(x, ei, ctx))
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); o = att(x, ei, ctx); o.sum().backward(); ts.append(time.perf_counter() - t0)
        rows.append({"H": heads, "H_kv": hkv, "kv_proj_params": att.k_proj.weight.numel() + att.v_proj.weight.numel(),
                     "total_params": sum(p.numel() for p in att.parameters()),
                     "autograd_saved_MB": sb / 2**20, "fwd_bwd_ms_median": 1e3 * statistics.median(ts)})
    return rows


if __name__ == "__main__":
    data = load_cora()
    ei = apply_self_loop_policy(data["edge_index"], data["x"].shape[0], "add")
    res = {"cora_rwse_rotary": bench(data["x"].shape[0], ei, struct=data["struct"]), "cora_no_rotary": bench(data["x"].shape[0], ei)}
    g = torch.Generator().manual_seed(0); n = 20000
    src = torch.randint(0, n, (n * 10,), generator=g); dst = torch.randint(0, n, (n * 10,), generator=g)
    big = apply_self_loop_policy(torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], 1), n, "add")
    res["synthetic_N20k_E400k"] = bench(n, big, reps=3)
    res["edges"] = {"cora": int(ei.shape[1]), "synthetic": int(big.shape[1])}
    Path("results").mkdir(exist_ok=True)
    Path("results/gqa_efficiency.json").write_text(json.dumps(res, indent=1))
    for k, rows in res.items():
        if k == "edges": continue
        print(k)
        for r in rows: print("  ", r)
