# MoE Recommendation Scaling Law Pilot

Experiment ID: `20260621-moe-rec-scaling-pilot`

## Status

Planned.

## Hypothesis

MoE recommenders follow a predictable scaling curve when quality is modeled as a
function of data scale, total parameters, activated parameters, and compute. The
curve should extrapolate to at least one larger held-out scale better than a
naive dense-parameter-only fit.

## Initial Matrix

| Family | Data scales | Model scales | Expert configs | Compute control |
| --- | --- | --- | --- | --- |
| Dense Rec | TBD | small / medium / large | N/A | matched GPU-hours |
| MoE Rec | TBD | small / medium / large | experts/top-k TBD | matched GPU-hours |

## Metrics

- Primary offline quality: TBD.
- Secondary recommendation metrics: TBD.
- Scaling fit: residuals, extrapolation error, confidence intervals.
- Infra metrics: GPU utilization, memory, communication time, p50/p95 latency.

## Required Next Inputs

- exact current MoE Rec architecture;
- data slice definitions;
- metric definitions;
- available GPU budget;
- baseline dense model family.
