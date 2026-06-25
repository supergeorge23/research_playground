"""Grouped-GEMM runners + realized-utilization accounting (Paper 1 baseline).

The expert up-projection is X_e[tokens_e, hidden] @ W_e[hidden, intermediate].
We measure how realized TFLOP/s (vs the roofline peak) collapses when per-expert
token counts are small/skewed -- the motivation for a utilization-aware kernel.

bf16/fp16 use cuBLAS (real tensor cores). int8 uses torch._int_mm (A100 SM80 INT8
tensor cores). fp8 is left to H100 (torch._scaled_mm). Pure-accounting helpers are
GPU-free and unit-testable; the runners need a CUDA device.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.profiling import roofline as rl  # noqa: E402


def up_proj_flops(token_counts, hidden: int, intermediate: int) -> float:
    """Total FLOPs of the per-expert up-projection grouped GEMM (2*m*k*n)."""
    return sum(2.0 * t * hidden * intermediate for t in token_counts if t > 0)


def predicted_time_s(token_counts, hidden, intermediate, dtype, gpu) -> float:
    """Ideal per-expert-loop time from the roofline (sum of per-expert t_pred)."""
    t = 0.0
    for m in token_counts:
        if m <= 0:
            continue
        r = rl.classify_gemm(m, hidden, intermediate, dtype, gpu)
        t += r.t_pred_s if r.t_pred_s is not None else r.t_memory_s
    return t


def utilization(token_counts, hidden, intermediate, dtype, gpu, time_s) -> float:
    """Realized fraction of peak: achieved_flops_per_s / peak_flops."""
    peak = rl.peak_flops(gpu, dtype)
    if not peak or time_s <= 0:
        return 0.0
    return (up_proj_flops(token_counts, hidden, intermediate) / time_s) / peak


# --------------------------- GPU runners (need CUDA) ------------------------ #

_TMAP = {"bf16": "bfloat16", "fp16": "float16"}


def _make(token_counts, hidden, intermediate, dtype, device):
    import torch
    xs, ws = [], []
    for t in token_counts:
        m = max(int(t), 0)
        if dtype in _TMAP:
            td = getattr(torch, _TMAP[dtype])
            xs.append(torch.randn(m, hidden, device=device, dtype=td))
            ws.append(torch.randn(hidden, intermediate, device=device, dtype=td))
        elif dtype == "int8":
            xs.append(torch.randint(-8, 8, (m, hidden), device=device, dtype=torch.int8))
            ws.append(torch.randint(-8, 8, (hidden, intermediate), device=device, dtype=torch.int8))
        else:
            raise ValueError(f"unsupported dtype for runner: {dtype}")
    return xs, ws


def _run_loop(xs, ws, dtype):
    import torch
    for x, w in zip(xs, ws):
        if x.shape[0] == 0:
            continue
        if dtype == "int8":
            torch._int_mm(x, w)            # int8 x int8 -> int32 (SM80+)
        else:
            torch.matmul(x, w)


def _run_padded(xs, ws, dtype):
    """Pad every expert to max tokens and batch via bmm -- wastes compute on skew."""
    import torch
    if dtype == "int8":
        raise ValueError("padded method supports bf16/fp16 only")
    maxm = max((x.shape[0] for x in xs), default=0)
    if maxm == 0:
        return
    E = len(xs)
    hidden = xs[0].shape[1] if E else 0
    td = xs[0].dtype
    X = torch.zeros(E, maxm, hidden, device=xs[0].device, dtype=td)
    for i, x in enumerate(xs):
        if x.shape[0]:
            X[i, : x.shape[0]] = x
    W = torch.stack(ws, dim=0)
    torch.bmm(X, W)


def measure(token_counts, hidden, intermediate, dtype, gpu, device="cuda",
            method="loop", iters=50, warmup=10):
    """Time the grouped GEMM and report realized utilization. Needs a CUDA device."""
    import torch
    xs, ws = _make(token_counts, hidden, intermediate, dtype, device)
    run = _run_padded if method == "padded" else _run_loop
    for _ in range(warmup):
        run(xs, ws, dtype)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run(xs, ws, dtype)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    flops = up_proj_flops(token_counts, hidden, intermediate)
    return {
        "method": method, "dtype": dtype, "time_ms": round(dt * 1e3, 4),
        "achieved_tflops": round(flops / dt / 1e12, 2),
        "utilization": round(utilization(token_counts, hidden, intermediate, dtype, gpu, dt), 4),
        "roofline_time_ms": round(predicted_time_s(token_counts, hidden, intermediate, dtype, gpu) * 1e3, 4),
    }
