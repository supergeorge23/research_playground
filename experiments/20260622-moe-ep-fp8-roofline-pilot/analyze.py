#!/usr/bin/env python3
"""Aggregate Paper 0 results into the figures/tables the agenda calls for.

Consumes a results dir of bench_*.json, routing_*.json and sensitivity_*.json
(any subset) and:
  - tabulates serving throughput / TPOT per config; computes EP-vs-TP deltas;
  - summarizes routing skew + EP straggler per model;
  - runs the roofline classifier for each model's expert GEMMs at decode vs
    prefill, reporting the realized int8 speedup on the run GPU (A100) AND the
    fp8 ceiling on H100 (projection);
  - if sensitivity is present, reports the hotness-vs-sensitivity correlation
    over (layer, expert) pairs -- the headline "is hot == high precision?" test.

Pure-python (stdlib + src.profiling); runs locally, no GPU. Writes summary.csv.

  python3 analyze.py --results-dir outputs
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
from src.profiling import roofline as rl       # noqa: E402
from src.profiling import router_stats as rs    # noqa: E402

# (hidden, intermediate, n_experts, top_k) for the roofline pass.
MODEL_SHAPES = {
    "mixtral-8x7b": dict(hidden=4096, intermediate=14336, n_experts=8, top_k=2),
    "qwen-moe-a2.7b": dict(hidden=2048, intermediate=1408, n_experts=60, top_k=4),
    "deepseek-v2-lite": dict(hidden=2048, intermediate=1408, n_experts=64, top_k=6),
}


def load(results_dir, prefix):
    out = []
    for p in sorted(glob.glob(str(Path(results_dir) / f"{prefix}_*.json"))):
        out.append(json.loads(Path(p).read_text()))
    return out


def short_name(model):
    m = model.lower()
    for k in MODEL_SHAPES:
        if k in m:
            return k
    return m


def bench_table(benches):
    rows = []
    for b in benches:
        c, m = b["config"], b["metrics"]
        rows.append({
            "model": short_name(c["model"]), "parallel": c["parallel"], "dtype": c["dtype"],
            "in": c["input_len"], "out": c["output_len"], "conc": c["concurrency"],
            "tok_per_s": round(m["throughput_tokens_per_s"], 1),
            "tpot_p50_ms": m["tpot_ms"]["p50"], "tpot_p95_ms": m["tpot_ms"]["p95"],
            "ttft_p50_ms": m["ttft_ms"]["p50"],
        })
    return rows


def ep_vs_tp(rows):
    idx = {}
    for r in rows:
        key = (r["model"], r["dtype"], r["in"], r["out"], r["conc"])
        idx.setdefault(key, {})[r["parallel"]] = r["tok_per_s"]
    out = []
    for key, d in idx.items():
        tp = next((v for k, v in d.items() if k.startswith("tp") and "ep" not in k), None)
        ep = next((v for k, v in d.items() if k.startswith("ep") or "ep" in k), None)
        if tp and ep:
            out.append({"key": key, "tp": tp, "ep": ep,
                        "ep_speedup": round(ep / tp, 3) if tp else None})
    return out


def roofline_pass(model_key, gpu="a100-80-sxm", decode_m=1, prefill_m=4096):
    s = MODEL_SHAPES.get(model_key)
    if not s:
        return None
    rep = []
    for tag, m in (("decode", decode_m), ("prefill", prefill_m)):
        bf16 = rl.classify_gemm(m, s["hidden"], s["intermediate"], "bf16", gpu)
        int8 = rl.classify_gemm(m, s["hidden"], s["intermediate"], "int8", gpu)
        h_bf16 = rl.classify_gemm(m, s["hidden"], s["intermediate"], "bf16", "h100-80-sxm")
        h_fp8 = rl.classify_gemm(m, s["hidden"], s["intermediate"], "fp8", "h100-80-sxm")
        i8 = int8.speedup_vs(bf16)
        f8 = h_fp8.speedup_vs(h_bf16)
        rep.append({"regime": tag, "m": m, "bound": bf16.bound,
                    "int8_realized": round(i8, 2) if i8 else None,
                    "fp8_h100": round(f8, 2) if f8 else None})
    return rep


def sensitivity_map(sens_files, override_path):
    """model -> {(layer, expert): sensitivity}."""
    out = {}
    files = list(sens_files)
    if override_path:
        files.append(json.loads(Path(override_path).read_text()))
    for sf in files:
        model = sf["config"]["model"]
        d = out.setdefault(model, {})
        for e in sf["per_expert"]:
            d[(e["layer"], e["expert"])] = e["sensitivity"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="outputs")
    ap.add_argument("--gpu", default="a100-80-sxm")
    ap.add_argument("--sensitivity", default=None, help="optional extra sensitivity json")
    args = ap.parse_args()

    benches = load(args.results_dir, "bench")
    routings = load(args.results_dir, "routing")
    sens = sensitivity_map(load(args.results_dir, "sensitivity"), args.sensitivity)

    print("=== Serving sweep ===")
    rows = bench_table(benches)
    for r in rows:
        print(f"  {r['model']:<16} {r['parallel']:<8} {r['dtype']:<5} "
              f"in{r['in']}/out{r['out']} c{r['conc']}: {r['tok_per_s']:>10} tok/s  "
              f"TPOT p50={r['tpot_p50_ms']}ms")
    print("\n=== EP vs TP (tokens/s) ===")
    for d in ep_vs_tp(rows):
        print(f"  {d['key']}: TP={d['tp']} EP={d['ep']} -> EP x{d['ep_speedup']}")

    print("\n=== Routing skew ===")
    for rt in routings:
        s = rt["summary"]
        print(f"  {short_name(rt['config']['model'])}: mean gini={s['mean_gini']} "
              f"mean EP({s['ep_size']}) straggler={s['mean_straggler']}")

    print(f"\n=== Roofline (representative up-proj; run GPU = {args.gpu}) ===")
    seen = {b["config"]["model"] for b in benches} or {rt["config"]["model"] for rt in routings} or set(MODEL_SHAPES)
    for mdl in sorted(seen):
        key = short_name(mdl)
        rep = roofline_pass(key, gpu=args.gpu)
        if rep:
            for row in rep:
                print(f"  {key:<16} {row['regime']:<7} m={row['m']:<5} bound={row['bound']:<8} "
                      f"int8@{args.gpu}={row['int8_realized']}x  fp8@H100(proj)={row['fp8_h100']}x")

    print("\n=== Hotness vs quant-sensitivity (the headline test) ===")
    if not sens:
        print("  (no sensitivity_*.json found -- run sensitivity.py to close the loop)")
    for rt in routings:
        model = rt["config"]["model"]
        smap = sens.get(model) or sens.get(short_name(model))
        if not smap:
            continue
        hot, sv = [], []
        for layer in rt["per_layer"]:
            L = layer["layer"]
            for e, c in enumerate(layer["counts"]):
                if (L, e) in smap:
                    hot.append(c)
                    sv.append(smap[(L, e)])
        if len(hot) >= 3:
            r = rs.correlation(hot, sv)
            verdict = ("NEGATIVE: cold experts more fragile -> 'hot=high precision' is WRONG"
                       if r < -0.2 else
                       "POSITIVE: hot experts more fragile" if r > 0.2 else
                       "~0: sensitivity and hotness are independent axes")
            print(f"  {short_name(model)} (n={len(hot)} layer-expert pairs): "
                  f"pearson r = {r:+.3f}  [{verdict}]")

    out_csv = Path(args.results_dir) / "summary.csv"
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
