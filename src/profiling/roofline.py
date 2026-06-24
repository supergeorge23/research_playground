"""Roofline classification for (grouped) GEMMs used in MoE inference.

Pure-python, no GPU required. Paper 0 uses this to label each expert GEMM as
compute- vs memory-bound and to bound the low-precision speedup -- the core
claim that "utilization is the missing axis".

Peaks are DENSE tensor-core numbers (sparsity OFF), from NVIDIA datasheets.
Key fact baked in: A100 (Ampere) has NO FP8 tensor cores; FP8/FP4 compute
returns `supported=False` so we never over-credit FP8 speedup on A100.
"""

from __future__ import annotations

from dataclasses import dataclass

# name -> peak compute (TFLOP/s for float, TOP/s for int) + HBM bandwidth (TB/s)
GPU_SPECS = {
    "a100-80-sxm": {
        "bf16_tflops": 312.0, "fp16_tflops": 312.0, "int8_tops": 624.0,
        "fp8_tflops": None, "fp4_tflops": None,          # Ampere: no FP8/FP4 TC
        "mem_bw_tbs": 2.039, "mem_gb": 80.0,
    },
    "a100-40-sxm": {
        "bf16_tflops": 312.0, "fp16_tflops": 312.0, "int8_tops": 624.0,
        "fp8_tflops": None, "fp4_tflops": None,
        "mem_bw_tbs": 1.555, "mem_gb": 40.0,
    },
    "h100-80-sxm": {
        "bf16_tflops": 989.0, "fp16_tflops": 989.0, "int8_tops": 1979.0,
        "fp8_tflops": 1979.0, "fp4_tflops": None,        # Hopper: FP8 yes, FP4 no
        "mem_bw_tbs": 3.350, "mem_gb": 80.0,
    },
}

DTYPE_BYTES = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0, "fp4": 0.5}

_PEAK_KEY = {
    "bf16": "bf16_tflops", "fp16": "fp16_tflops", "fp8": "fp8_tflops",
    "int8": "int8_tops", "fp4": "fp4_tflops",
}


def peak_flops(gpu: str, dtype: str):
    """Peak ops/s for dtype on gpu, or None if not hardware-accelerated."""
    val = GPU_SPECS[gpu].get(_PEAK_KEY[dtype])
    return None if val is None else val * 1e12


@dataclass
class RooflineResult:
    m: int
    k: int
    n: int
    dtype: str
    gpu: str
    flops: float
    bytes: float
    arithmetic_intensity: float
    machine_balance: float
    bound: str                 # "compute" | "memory" | "unsupported-compute"
    t_compute_s: float | None
    t_memory_s: float
    t_pred_s: float | None
    supported: bool

    def speedup_vs(self, other: "RooflineResult") -> float | None:
        """Predicted speedup of self over other (e.g. fp8 vs bf16)."""
        if self.t_pred_s is None or other.t_pred_s in (None, 0):
            return None
        return other.t_pred_s / self.t_pred_s


def classify_gemm(m: int, k: int, n: int, dtype: str, gpu: str) -> RooflineResult:
    """Classify one GEMM A[m,k] @ B[k,n] -> C[m,n] on `gpu` in `dtype`.

    Models cold HBM traffic (read A and B, write C). For a single expert in
    decode, m (tokens routed) is small while B (k*n weights) is loaded in full,
    so the op is memory-bound and low precision helps only via weight traffic --
    exactly the asymmetry Paper 0 wants to expose.
    """
    db = DTYPE_BYTES[dtype]
    flops = 2.0 * m * n * k
    bytes_ = (m * k + k * n + m * n) * db
    bw = GPU_SPECS[gpu]["mem_bw_tbs"] * 1e12
    ai = flops / bytes_ if bytes_ else float("inf")
    t_mem = bytes_ / bw

    pf = peak_flops(gpu, dtype)
    if pf is None:
        # dtype has no tensor-core support here (e.g. fp8 on A100): report
        # the bf16 machine balance for context but refuse a compute estimate.
        bf16_pf = peak_flops(gpu, "bf16")
        return RooflineResult(
            m, k, n, dtype, gpu, flops, bytes_, ai,
            (bf16_pf / bw) if bf16_pf else 0.0,
            "unsupported-compute", None, t_mem, None, False,
        )

    machine_balance = pf / bw
    t_comp = flops / pf
    bound = "compute" if ai >= machine_balance else "memory"
    return RooflineResult(
        m, k, n, dtype, gpu, flops, bytes_, ai, machine_balance,
        bound, t_comp, t_mem, max(t_comp, t_mem), True,
    )


def expert_gemm_shapes(hidden: int, intermediate: int, gated: bool = True):
    """Return the (k, n) shapes of one expert's FFN GEMMs (per token-block).

    gated=True for SwiGLU-style experts (Mixtral/Qwen/DeepSeek): up & gate are
    [hidden->intermediate], down is [intermediate->hidden].
    """
    shapes = [("up", hidden, intermediate)]
    if gated:
        shapes.append(("gate", hidden, intermediate))
    shapes.append(("down", intermediate, hidden))
    return shapes
