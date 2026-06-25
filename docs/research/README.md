# Research Tracks

This directory contains active and proposed research tracks. A track should
define its research question, hypotheses, experiment plan, expected outputs,
and links to code, artifacts, literature, and paper drafts.

## Active Tracks

| Track | Status | Entry Point |
| --- | --- | --- |
| MoE recommendation scaling laws | Active | `moe-rec-scaling-law-agenda.md` |
| Utilization-aware precision × parallelism for MoE inference | Active | `moe-precision-parallelism-agenda.md` |

## Active Experiments (precision × parallelism track)

| Paper | Experiment ID | Role |
| --- | --- | --- |
| Paper 0 | `20260622-moe-ep-fp8-roofline-pilot` | measure: EP-vs-TP, roofline regimes, routing skew, sensitivity |
| Paper 1 | `20260622-grouped-gemm-utilization-pilot` | measure: grouped-GEMM utilization gap (baseline; kernel is the contribution) |
| Paper 2 | `20260622-precision-placement-scheduler-pilot` | simulate: precision-as-load-balancing straggler cut on Paper 0 data |

## Proposed Tracks

- LLM4Rec training and recommendation infrastructure
- MoE model infrastructure
- FlashAttention and attention-kernel industrialization

New tracks should be added through an OKR update or decision record before they
start accumulating experiment folders and artifact manifests.
