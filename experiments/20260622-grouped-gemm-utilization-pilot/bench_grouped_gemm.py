#!/usr/bin/env python3
"""Paper 1 microbench: realized utilization of the MoE expert grouped GEMM.

Sweeps {uniform, skewed} token distributions x {bf16, int8} x {loop, padded} and
reports achieved TFLOP/s and utilization (vs roofline peak). The headline is the
utilization collapse under skew / small per-expert tokens -- the gap a
utilization-aware kernel (Paper 1) must close.

Correct torch code on a CUDA box (bf16 cuBLAS, int8 torch._int_mm). --dry-run
synthesizes times from the roofline + an inefficiency model so the reporting path
is testable with no GPU.

  python3 bench_grouped_gemm.py --n-experts 64 --hidden 2048 --intermediate 1408 \
      --total-tokens 4096 --gpu a100-80-sxm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.kernels import grouped_gemm as gg  # noqa: E402


def build_counts(n, total, dist, seed=0):
    if dist == "uniform":
        base = total // n
        cc = [base] * n
        for i in range(total - base * n):
            cc[i] += 1
        return cc
    import random
    rng = random.Random(seed)
    w = [1.0 / (i + 1) ** 1.3 for i in range(n)]
    rng.shuffle(w)
    s = sum(w)
    return [max(1, int(total * wi / s)) for wi in w]


def synth_dict(counts, hidden, intermediate, dtype, gpu, method):
    pred = gg.predicted_time_s(counts, hidden, intermediate, dtype, gpu)
    if method == "padded":
        waste = (max(counts) * len(counts)) / max(sum(counts), 1)
        dt = pred * max(1.0, waste)
    else:  # loop: per-expert launch overhead, worse with many small experts
        dt = pred * (1.2 + 2e-5 * sum(1 for c in counts if c > 0))
    flops = gg.up_proj_flops(counts, hidden, intermediate)
    return {
        "method": method, "dtype": dtype, "time_ms": round(dt * 1e3, 4),
        "achieved_tflops": round(flops / dt / 1e12, 2),
        "utilization": round(gg.utilization(counts, hidden, intermediate, dtype, gpu, dt), 4),
        "roofline_time_ms": round(pred * 1e3, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-experts", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--intermediate", type=int, default=1408)
    ap.add_argument("--total-tokens", type=int, default=4096)
    ap.add_argument("--gpu", default="a100-80-sxm")
    ap.add_argument("--dtypes", default="bf16,int8")
    ap.add_argument("--dists", default="uniform,skew")
    ap.add_argument("--methods", default="loop,padded")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = []
    for dist in args.dists.split(","):
        counts = build_counts(args.n_experts, args.total_tokens, dist)
        for dtype in args.dtypes.split(","):
            for method in args.methods.split(","):
                if method == "padded" and dtype == "int8":
                    continue  # int8 padded bmm unsupported here
                if args.dry_run:
                    res = synth_dict(counts, args.hidden, args.intermediate, dtype, args.gpu, method)
                else:
                    res = gg.measure(counts, args.hidden, args.intermediate, dtype,
                                     args.gpu, method=method, iters=args.iters)
                res.update(dist=dist, n_experts=args.n_experts,
                           tokens_per_expert=round(args.total_tokens / args.n_experts, 1))
                rows.append(res)

    print(f"=== Paper 1: grouped-GEMM utilization ({args.gpu}, "
          f"{args.n_experts} experts, {args.total_tokens} tokens) ===")
    print(f"{'dist':>8} {'dtype':>6} {'method':>7} {'TFLOP/s':>9} {'util':>7} {'time_ms':>9}")
    for r in rows:
        print(f"{r['dist']:>8} {r['dtype']:>6} {r['method']:>7} "
              f"{r['achieved_tflops']:>9} {r['utilization']:>7} {r['time_ms']:>9}")
    print("Look for: utilization dropping from uniform->skew and for small tokens/expert "
          "(decode); that gap is what a utilization-aware kernel must recover.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"gpu": args.gpu, "rows": rows}, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
