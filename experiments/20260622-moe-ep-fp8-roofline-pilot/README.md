# MoE EP × FP8 Roofline Profiling Pilot

Experiment ID: `20260622-moe-ep-fp8-roofline-pilot`

Track: Utilization-Aware Precision × Parallelism Co-Design for MoE Inference
(`docs/research/moe-precision-parallelism-agenda.md`). This is **Paper 0**.

## Status

Planned.

## Owner

Zhengqi Wang

## Hypothesis

1. Whether EP inference improves throughput/latency is regime-dependent and
   predictable from a roofline model.
2. Per-expert quantization sensitivity is not monotone in activation frequency.
3. FP8's realized speedup depends on each expert's GEMM roofline regime
   (compute- vs memory-bound), not just on quantization error.

## Questions This Pilot Answers

- How does EP degree change decode throughput / p95 latency vs TP/DP, across
  NVLink vs PCIe, for a fixed MoE model and workload?
- Can we decompose any EP win/loss into memory-enabled batch, all-to-all cost,
  and load skew?
- Per expert: token-count distribution, GEMM roofline regime, and FP8/FP4
  sensitivity (accuracy delta). Is sensitivity correlated with hotness?

## Setup

- Models: start with Mixtral-8x7B; add Qwen-MoE / DeepSeek-MoE for fine-grained.
- Hardware: 1-2 on-demand GPUs; measure NVLink and PCIe paths where available.
- Harness: vLLM / SGLang FP8 MoE path for serving measurements.
- Git commit / data snapshot / exact config: see `config.yaml` and `run.md`.

## Metrics

- Throughput: decode tokens/s. Latency: p50/p95 per token.
- Utilization: FP8 tensor-core utilization; achieved vs peak TFLOPs per expert.
- Straggler: per-GPU finish-time gap within the MoE layer.
- Accuracy: task delta under FP8/FP4, per expert and per layer.

## Deliverable

A reproducible profiling harness + plots establishing "utilization is the missing
axis", feeding Paper 1 (kernel) and Paper 2 (scheduler).

## Tracking Files

- Config: `config.yaml`
- Run log: `run.md`
- Results: `results.md`
- Artifact pointers: `artifacts.yaml`
