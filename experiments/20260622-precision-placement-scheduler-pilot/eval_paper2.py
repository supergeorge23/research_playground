#!/usr/bin/env python3
"""Paper 2 evaluation: does precision-as-a-load-balancing-knob cut the EP straggler?

Consumes Paper 0 outputs (routing_<model>.json + sensitivity_<model>.json) and,
per MoE layer, compares the synchronous-layer straggler under:
  - rr   : round-robin placement, bf16 (naive)
  - lpt  : load-balanced placement, bf16 (the placement-only baseline to beat)
  - joint: joint precision x placement (our scheduler) at an accuracy budget
Sweeps the accuracy budget and reports the mean straggler reduction vs lpt.

Pure-python (src.scheduler + src.profiling); runs locally, no GPU. --dry-run
synthesizes a favorable-but-honest case (hot experts mildly LESS sensitive, per
MoPEQ) so the pipeline is testable with no Paper 0 data.

  python3 eval_paper2.py --results-dir <paper0-outputs> --model mixtral-8x7b
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.scheduler import precision_placement as pp  # noqa: E402

MODEL_SHAPES = {
    "mixtral-8x7b": dict(hidden=4096, intermediate=14336, n_experts=8),
    "qwen-moe-a2.7b": dict(hidden=2048, intermediate=1408, n_experts=60),
    "deepseek-v2-lite": dict(hidden=2048, intermediate=1408, n_experts=64),
}


def _find(results_dir, prefix, model):
    for p in glob.glob(str(Path(results_dir) / f"{prefix}_*.json")):
        d = json.loads(Path(p).read_text())
        m = d.get("config", {}).get("model", "")
        if model in m or model == m or any(k in m.lower() for k in [model]):
            return d
    return None


def _load_real(results_dir, model):
    rt = _find(results_dir, "routing", model)
    se = _find(results_dir, "sensitivity", model)
    if rt is None:
        raise SystemExit(f"no routing_*.json for {model} in {results_dir}")
    layers = [(p["layer"], p["counts"]) for p in rt["per_layer"]]
    sens = {}
    if se is not None:
        for e in se["per_expert"]:
            sens[(e["layer"], e["expert"])] = e["sensitivity"]
    return layers, sens


def _synth(n_layers, n_experts, seed=0):
    import random
    rng = random.Random(seed)
    layers, sens = [], {}
    for L in range(n_layers):
        w = [1.0 / (i + 1) ** 1.3 for i in range(n_experts)]
        rng.shuffle(w)
        counts = [0] * n_experts
        for _ in range(8192):
            r = rng.random() * sum(w)
            acc = 0.0
            for e, wi in enumerate(w):
                acc += wi
                if r <= acc:
                    counts[e] += 1
                    break
        layers.append((L, counts))
        mx = max(counts) or 1
        for e in range(n_experts):
            # hot experts mildly LESS sensitive (favorable, MoPEQ-style) + noise
            sens[(L, e)] = round(1.0 - 0.6 * counts[e] / mx + rng.uniform(0, 0.2), 4)
    return layers, sens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="outputs")
    ap.add_argument("--model", default="mixtral-8x7b")
    ap.add_argument("--gpu", default="a100-80-sxm")
    ap.add_argument("--ep-size", type=int, default=8)
    ap.add_argument("--budgets", default="0,0.05,0.1,0.2,0.5",
                    help="accuracy budgets as fraction of per-layer total sensitivity")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-experts", type=int, default=8)
    args = ap.parse_args()

    if args.dry_run:
        shape = MODEL_SHAPES.get(args.model, MODEL_SHAPES["mixtral-8x7b"])
        layers, sens = _synth(args.n_layers, args.n_experts)
        n_experts = args.n_experts
    else:
        shape = MODEL_SHAPES[args.model]
        layers, sens = _load_real(args.results_dir, args.model)
        n_experts = shape["n_experts"]

    hidden, intermediate = shape["hidden"], shape["intermediate"]
    budgets = [float(x) for x in args.budgets.split(",")]
    big = max([v for v in sens.values()], default=1.0) * 10 + 1.0

    rows = []
    for f in budgets:
        ratios, accs, nlows, rr_red = [], [], [], []
        for (L, counts) in layers:
            sv = [sens.get((L, e), big) for e in range(len(counts))]
            budget = f * sum(s for s in sv if s < big)   # downcastable mass only
            r = pp.evaluate_layer(counts, sv, args.ep_size, hidden, intermediate,
                                  args.gpu, budget)
            ratios.append(r["joint_vs_lpt"])
            accs.append(r["acc_cost"])
            nlows.append(r["n_low"])
            rr_red.append(1.0 - r["lpt_s"] / r["rr_s"] if r["rr_s"] else 0.0)
        n = len(layers)
        rows.append({
            "budget_frac": f,
            "mean_straggler_cut_vs_lpt_pct": round(100 * (1 - sum(ratios) / n), 2),
            "mean_acc_cost": round(sum(accs) / n, 4),
            "mean_experts_downcast": round(sum(nlows) / n, 2),
            "lpt_cut_vs_rr_pct": round(100 * sum(rr_red) / n, 2),
        })

    print(f"=== Paper 2: precision-as-load-balancing ({args.model}, EP={args.ep_size}, {args.gpu}) ===")
    print(f"{'budget':>8} {'straggler_cut_vs_lpt':>22} {'acc_cost':>10} {'#downcast':>10}")
    for r in rows:
        print(f"{r['budget_frac']:>8} {r['mean_straggler_cut_vs_lpt_pct']:>20}% "
              f"{r['mean_acc_cost']:>10} {r['mean_experts_downcast']:>10}")
    print(f"(placement-only LPT already cuts the naive round-robin straggler by "
          f"~{rows[0]['lpt_cut_vs_rr_pct']}% before any precision change.)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"model": args.model, "gpu": args.gpu,
                                              "ep_size": args.ep_size, "rows": rows}, indent=2))
        csv_path = Path(args.out).with_suffix(".csv")
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {args.out} and {csv_path}")


if __name__ == "__main__":
    main()
