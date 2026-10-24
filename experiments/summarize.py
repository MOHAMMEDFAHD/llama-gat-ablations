"""Aggregate results/runs/*.json -> results/summary.md and results/summary.json.
Paired comparisons (same seeds) use a paired t interval on the per-seed difference."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.metrics import mean_std_ci  # noqa: E402

PAIRS = [  # (hypothesis, arm_a, arm_b): reports a - b
    ("H1: spectral RoPE vs plain dot-product", "A6", "A2"),
    ("H1': WIRE-style fixed-freq vs plain", "A5", "A2"),
    ("H2': RWSE-phase vs LapPE-phase", "A8", "A6"),
    ("H3: GQA vs MHA (spectral)", "A7", "A6"),
    ("H3: GQA vs MHA (RWSE)", "A9", "A8"),
    ("A10: +struct bias vs A9", "A10", "A9"),
    ("edge-phase vs node-phase (RWSE)", "A10e", "A8"),
    ("abs LapPE vs plain", "A4", "A2"),
    ("index-RoPE (invalid) vs plain", "A3", "A2"),
    ("GraphLlama-plain vs GAT", "A2", "A0"),
    ("GATv2 vs GAT", "A1", "A0"),
    ("LayerNorm vs RMSNorm", "C_layernorm", "A2"),
    ("GELU-MLP vs SwiGLU", "C_gelu", "A2"),
    ("MQA vs MHA (plain)", "C_mqa", "A2"),
    ("3 layers vs 2", "C_3layers", "A2"),
    ("rotary_dim=4 vs full (spectral)", "C_rotdim4", "A6"),
]


def main():
    runs = defaultdict(dict)
    for f in sorted((ROOT / "results/runs").glob("*.json")):
        r = json.loads(f.read_text()); runs[r["arm"]][r["seed"]] = r
    rows, summary = [], {}
    for arm, by_seed in runs.items():
        vals = list(by_seed.values())
        s = {k: mean_std_ci([v[k] for v in vals]) for k in ("test_acc", "val_acc", "test_macro_f1")}
        s["num_parameters"] = vals[0]["num_parameters"]
        s["seconds_per_epoch"] = float(np.mean([v["seconds_per_epoch"] for v in vals]))
        s["inference_seconds"] = float(np.mean([v["inference_seconds"] for v in vals]))
        s["best_epoch"] = float(np.mean([v["best_epoch"] for v in vals]))
        s["description"] = vals[0]["description"]
        pd = [v["phase_diagnostics"] for v in vals if v.get("phase_diagnostics")]
        if pd:
            s["phase_edge_vs_random"] = {
                k: [float(np.mean([p[k]["edge_mean_abs_sin_half_dphi"] for p in pd])),
                    float(np.mean([p[k]["random_pair_mean_abs_sin_half_dphi"] for p in pd]))] for k in pd[0]}
        summary[arm] = s
    order = sorted(summary, key=lambda a: (a[0] != "A", int(''.join(c for c in a[1:] if c.isdigit()) or 99) if a[0] == "A" else 0, a))
    lines = ["# Cora results (public Planetoid split)", "",
             "Mean ± sample std over seeds; 95% CI is a t-interval. Model selection on validation accuracy only.", "",
             "| Arm | Description | n | Test acc | 95% CI | Val acc | Test macro-F1 | Params | s/epoch (CPU, 1 thread) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for a in order:
        s = summary[a]; t = s["test_acc"]
        lines.append(f"| {a} | {s['description']} | {t['n']} | {100*t['mean']:.2f} ± {100*t['std']:.2f} | [{100*t['ci95_low']:.2f}, {100*t['ci95_high']:.2f}] | "
                     f"{100*s['val_acc']['mean']:.2f} | {100*s['test_macro_f1']['mean']:.2f} | {s['num_parameters']:,} | {s['seconds_per_epoch']:.3f} |")
    lines += ["", "## Paired comparisons (test accuracy, percentage points, same seeds)", "",
              "| Comparison | a − b | 95% CI (paired t) | p (paired t) | n |", "|---|---|---|---|---|"]
    paired = {}
    for name, a, b in PAIRS:
        if a not in runs or b not in runs:
            continue
        common = sorted(set(runs[a]) & set(runs[b]))
        if len(common) < 2:
            continue
        d = np.array([runs[a][s]["test_acc"] - runs[b][s]["test_acc"] for s in common]) * 100
        m, sd = d.mean(), d.std(ddof=1)
        h = stats.t.ppf(0.975, len(d) - 1) * sd / math.sqrt(len(d)) if sd > 0 else 0.0
        p = float(stats.ttest_rel([runs[a][s]["test_acc"] for s in common], [runs[b][s]["test_acc"] for s in common]).pvalue) if sd > 0 else float("nan")
        paired[name] = {"diff_pp": m, "ci": [m - h, m + h], "p": p, "n": len(d)}
        lines.append(f"| {name} ({a} − {b}) | {m:+.2f} | [{m-h:+.2f}, {m+h:+.2f}] | {p:.3f} | {len(d)} |")
    ph = {a: s["phase_edge_vs_random"] for a, s in summary.items() if "phase_edge_vs_random" in s}
    if ph:
        lines += ["", "## H6 phase diagnostic: mean |sin(Δφ/2)| (edges vs random node pairs), trained models", "",
                  "| Arm | Layer | Edges | Random pairs | Ratio |", "|---|---|---|---|---|"]
        for a, d in ph.items():
            for layer, (e, r) in d.items():
                lines.append(f"| {a} | {layer} | {e:.4f} | {r:.4f} | {e / max(r, 1e-12):.3f} |")
    (ROOT / "results/summary.md").write_text("\n".join(lines) + "\n")
    (ROOT / "results/summary.json").write_text(json.dumps({"arms": summary, "paired": paired}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
