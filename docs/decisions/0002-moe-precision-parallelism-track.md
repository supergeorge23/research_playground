# Decision 0002: Add Research Track — Utilization-Aware Precision × Parallelism Co-Design for MoE Inference

Date: 2026-06-22

Owner: Zhengqi Wang

## Context

Two recurring problems from industrial LLM serving motivate a new track:

1. Validating whether expert-parallel (EP) inference for MoE actually improves
   throughput/latency, and under what regime.
2. Raising the *realized* low-precision utilization of LLM matmuls — INT8 on the
   available A100, FP8 on H100 — especially MoE grouped GEMMs, and generalizing
   the method across models.

A survey of 2025-2026 work shows the generic framings are crowded. EP load
balancing (MoETuner, EPS-MoE, PROBE, AEP, ReaLB) and MoE mixed-precision
quantization (MoPEQ, MxMoE, MC-MoE, HOBBIT, ScaleBITS) are actively published.
Notably, MoPEQ finds per-expert *sensitivity* matters more than activation
*frequency*, which contradicts a naive "hot experts -> high precision" rule.

The open seam is the boundary these two lines leave fixed: quantization work
assumes expert GEMMs are balanced and deployable, while EP-scheduling work
ignores numeric precision. In a real EP deployment the expert computation *is* a
skewed, low-precision grouped GEMM, so the EP load distribution determines
whether low precision actually delivers speedup, and precision is an unused knob
for balancing the synchronous-layer straggler.

## Decision

Open a research track: **Utilization-Aware Precision × Parallelism Co-Design for
MoE Inference** (`docs/research/moe-precision-parallelism-agenda.md`). Structure
it as a risk-increasing, mutually de-risking paper sequence:

- **Paper 0 (measurement):** characterize when EP inference wins and the roofline
  regimes of MoE GEMMs across interconnects; establish utilization as the missing
  axis. First experiment: `20260622-moe-ep-fp8-roofline-pilot`.
- **Paper 1 (kernel):** utilization-aware low precision (INT8 W8A8 on A100, FP8 on
  H100) — a heterogeneous-precision grouped GEMM that realizes speedup under
  small/skewed per-expert token counts.
- **Paper 2 (system):** roofline-driven joint precision-and-placement scheduler
  that uses precision as a load-balancing dimension to cut the synchronous EP
  straggler (sensitivity-aware, utilization-aware, cross-GPU).

This track maps to 2026 H2 OKR Objective 3 (publishable AI-infra contributions).
It does not replace the MoE-in-Rec scaling-law track; because the owner's personal
research focus is LLM-only, this becomes the primary personal track.

## Alternatives Considered

- Generic EP load balancer or generic auto mixed-precision: rejected as crowded.
- Recommendation-specific MoE systems (earlier idea): dropped; personal research
  is LLM-only.
- Training-side attention-MoE overlap (FoldMoE / MegaScale-MoE line): parked; it
  needs cluster scale that is not reliably available under on-demand cloud.

## Consequences

- The first concrete work item is a low-cost profiling study (Paper 0), runnable
  on 1-2 on-demand GPUs.
- The differentiator vs. the field is industrial EP + low-precision (FP8/INT8)
  experience plus the utilization/roofline framing; the track must claim a precise
  niche rather than the broad "mixed-precision experts" headline, and move fast.
- Aligns with external systems-for-MoE research directions we are exploring for
  collaboration, supplying the quantization/kernel capability those groups lack.
- Industrial low-precision recipes carry IP: reproduce only on open models; commit
  no internal data, weights, or hostnames (see `docs/versioning/artifact-policy.md`).
- Hardware reality: the available box is 8x A100 (no FP8 tensor cores), so the
  on-device low-precision lever is INT8 (W8A8); FP8 is the H100 target, reported
  as a roofline projection. BF16 is the baseline, not a precision lever.
