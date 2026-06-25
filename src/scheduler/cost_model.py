"""Per-expert cost model for the precision x placement scheduler (Paper 2).

Turns (tokens, precision) into a GEMM time via the Paper 0 roofline, and exposes
the gpu-appropriate low-precision dtype. The low-precision lever is INT8 on A100
and FP8 on H100 (see the track's hardware note).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.profiling import roofline as rl  # noqa: E402


def low_dtype(gpu: str) -> str:
    """The on-device low-precision compute lever for `gpu`."""
    return "int8" if gpu.startswith("a100") else "fp8"


def expert_time_s(tokens: int, dtype: str, hidden: int, intermediate: int,
                  gpu: str, gated: bool = True) -> float:
    """Summed FFN GEMM time for one expert processing `tokens` tokens at `dtype`.

    If `dtype` has no tensor-core support on `gpu` (e.g. fp8 on A100), we credit
    only the memory time (no compute speedup) so the cost model never over-credits
    an unsupported precision.
    """
    if tokens <= 0:
        return 0.0
    t = 0.0
    for _name, k, n in rl.expert_gemm_shapes(hidden, intermediate, gated):
        r = rl.classify_gemm(tokens, k, n, dtype, gpu)
        t += r.t_pred_s if r.t_pred_s is not None else r.t_memory_s
    return t
