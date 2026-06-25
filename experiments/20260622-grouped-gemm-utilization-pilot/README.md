# Paper 1 Pilot: Grouped-GEMM Utilization for MoE Experts

Experiment ID: `20260622-grouped-gemm-utilization-pilot`

Track: Utilization-Aware Precision × Parallelism Co-Design for MoE Inference
(`docs/research/moe-precision-parallelism-agenda.md`). This is **Paper 1's
baseline + measurement**; the utilization-aware kernel is the contribution and is
the TODO below.

## Status

Planned.

## Owner

Zhengqi Wang

## Hypothesis

Realized tensor-core utilization of the per-expert grouped GEMM collapses under
small / skewed per-expert token counts (the decode regime), and low precision's
benefit is regime-dependent (compute-bound vs memory-bound). A utilization-aware
kernel can recover the lost utilization.

## Method

`bench_grouped_gemm.py` sweeps {uniform, skew} × {bf16, int8} × {loop, padded}
and reports achieved TFLOP/s and utilization vs the roofline peak. bf16 uses
cuBLAS; int8 uses `torch._int_mm` (real A100 INT8 tensor cores); fp8 waits for
H100. Correct torch code — no custom kernel here.

## Metrics

- utilization (achieved / peak) and achieved TFLOP/s;
- measured / roofline time;
- the uniform→skew utilization gap, and the small-tokens/expert (decode) gap.

## Deliverable

The utilization gap that motivates the kernel, plus baseline numbers (per-expert
loop, padded batched) for the kernel to beat.

## Next (the actual Paper 1 research)

A **utilization-aware grouped-GEMM kernel** — heterogeneous per-expert precision,
padding-free for skewed token counts — in Triton/CUTLASS, added under this folder
and benchmarked with the same harness. Builds on the Paper 0 roofline + this gap.

## Tracking Files

config.yaml / run.md / results.md / artifacts.yaml
