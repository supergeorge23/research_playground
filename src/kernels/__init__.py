"""Paper 1: grouped-GEMM utilization microbenchmark for MoE expert layers.

Measures REALIZED tensor-core utilization of the per-expert (up-projection)
grouped GEMM as a function of per-expert token count, precision, and execution
method (per-expert loop vs padded batched), and contrasts uniform vs skewed
routing. This establishes the gap Paper 1's utilization-aware kernel must close;
the optimized kernel itself is the research contribution.

Correct torch code (cuBLAS for bf16/fp16, torch._int_mm for int8 on A100). No
custom kernel here -- this is the baseline + the measurement to beat.
"""

from . import grouped_gemm  # noqa: F401

__all__ = ["grouped_gemm"]
