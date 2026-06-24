#!/usr/bin/env python3
"""No-GPU self-test for the Paper 0 pure-python core.

Run this locally BEFORE shipping anything to the 8xA100 box -- it validates the
roofline and router-stats logic on synthetic data so you don't debug on the
expensive machine. Exits non-zero on any failed assertion.

Usage:  python3 experiments/20260622-moe-ep-fp8-roofline-pilot/selftest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.profiling import roofline as rl          # noqa: E402
from src.profiling import router_stats as rs       # noqa: E402


def test_decode_expert_is_memory_bound():
    # Mixtral-ish expert: hidden=4096, intermediate=14336. Decode: 1 token.
    r = rl.classify_gemm(m=1, k=4096, n=14336, dtype="bf16", gpu="a100-80-sxm")
    assert r.bound == "memory", r
    # Prefill-ish: 8192 tokens through the same expert -> compute-bound.
    r2 = rl.classify_gemm(m=8192, k=4096, n=14336, dtype="bf16", gpu="a100-80-sxm")
    assert r2.bound == "compute", r2
    print(f"  decode AI={r.arithmetic_intensity:.2f} (<{r.machine_balance:.1f}) -> {r.bound}")
    print(f"  prefill AI={r2.arithmetic_intensity:.1f} (>={r2.machine_balance:.1f}) -> {r2.bound}")


def test_fp8_unsupported_on_a100_but_ok_on_h100():
    a = rl.classify_gemm(1, 4096, 14336, "fp8", "a100-80-sxm")
    assert a.supported is False and a.t_pred_s is None, a
    h = rl.classify_gemm(1, 4096, 14336, "fp8", "h100-80-sxm")
    assert h.supported is True and h.t_pred_s is not None, h
    print(f"  fp8@A100 supported={a.supported}; fp8@H100 supported={h.supported}")


def test_fp8_weight_traffic_helps_memory_bound_on_h100():
    bf16 = rl.classify_gemm(1, 4096, 14336, "bf16", "h100-80-sxm")
    fp8 = rl.classify_gemm(1, 4096, 14336, "fp8", "h100-80-sxm")
    sp = fp8.speedup_vs(bf16)
    # memory-bound decode: ~halved weight bytes -> ~2x, never below 1x.
    assert sp is not None and 1.5 <= sp <= 2.2, sp
    print(f"  fp8 vs bf16 (decode, mem-bound) predicted speedup ~ {sp:.2f}x")


def test_skew_and_straggler():
    balanced = [100] * 8
    skewed = [400, 200, 50, 50, 40, 30, 20, 10]
    mb = rs.skew_metrics(balanced)
    ms = rs.skew_metrics(skewed)
    assert mb.gini < 0.05 and mb.norm_entropy > 0.99, mb
    assert ms.gini > mb.gini and ms.max_over_mean > 1.5, ms
    # 8 experts over ep_size=4, round-robin: skewed should straggle more.
    sf_bal = rs.ep_straggler_factor(balanced, ep_size=4)
    sf_skew = rs.ep_straggler_factor(skewed, ep_size=4)
    assert abs(sf_bal - 1.0) < 1e-9, sf_bal
    assert sf_skew > sf_bal, (sf_bal, sf_skew)
    print(f"  balanced gini={mb.gini:.3f} straggler={sf_bal:.2f}; "
          f"skewed gini={ms.gini:.3f} straggler={sf_skew:.2f}")


def test_precision_as_balancing_knob():
    # Down-casting the hottest expert (cost 0.5) should cut the straggler.
    skewed = [400, 200, 50, 50, 40, 30, 20, 10]
    base = rs.ep_straggler_factor(skewed, ep_size=4)
    cost = [1.0] * 8
    cost[0] = 0.5                      # hottest expert at half cost (e.g. fp8/int8)
    knob = rs.ep_straggler_factor(skewed, ep_size=4, cost_per_token=cost)
    assert knob < base, (base, knob)
    print(f"  straggler {base:.2f} -> {knob:.2f} after down-casting hottest expert")


def test_correlation_signal():
    hot = [400, 200, 50, 50, 40, 30, 20, 10]
    # anti-correlated sensitivity (cold experts more fragile) -> negative r
    sens = [0.1, 0.2, 0.6, 0.6, 0.7, 0.8, 0.9, 1.0]
    r = rs.correlation(hot, sens)
    assert r < -0.5, r
    print(f"  hotness vs (anti-correlated) sensitivity: pearson r = {r:.2f}")


def main() -> int:
    tests = [
        ("roofline: decode mem-bound / prefill compute-bound", test_decode_expert_is_memory_bound),
        ("roofline: fp8 unsupported on A100, ok on H100", test_fp8_unsupported_on_a100_but_ok_on_h100),
        ("roofline: fp8 weight-traffic speedup on H100", test_fp8_weight_traffic_helps_memory_bound_on_h100),
        ("router: skew + EP straggler", test_skew_and_straggler),
        ("router: precision as balancing knob", test_precision_as_balancing_knob),
        ("router: hotness/sensitivity correlation", test_correlation_signal),
    ]
    for name, fn in tests:
        print(f"[ RUN  ] {name}")
        fn()
        print(f"[  OK  ] {name}")
    print(f"\nAll {len(tests)} self-tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
