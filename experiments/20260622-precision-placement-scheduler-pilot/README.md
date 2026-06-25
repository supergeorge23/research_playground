# Paper 2 Pilot: Precision × Placement Scheduling for MoE EP

Experiment ID: `20260622-precision-placement-scheduler-pilot`

Track: Utilization-Aware Precision × Parallelism Co-Design for MoE Inference
(`docs/research/moe-precision-parallelism-agenda.md`). This is **Paper 2**, the
scheduler — evaluated here as a *simulation driven by Paper 0 data*, no kernel
needed.

## Status

Planned.

## Owner

Zhengqi Wang

## Hypothesis

Precision is a load-balancing dimension: down-casting the overloaded,
non-sensitive experts equalizes per-GPU finish times and cuts the synchronous EP
straggler beyond what placement (LPT) alone achieves, at a controllable accuracy
cost.

## Method

`eval_paper2.py` consumes Paper 0 `routing_<model>.json` + `sensitivity_<model>.json`
and, per MoE layer, compares the synchronous-layer straggler under:

- `rr` — round-robin placement, bf16 (naive);
- `lpt` — load-balanced placement, bf16 (the placement-only baseline to beat);
- `joint` — joint precision × placement (our scheduler) at an accuracy budget,

sweeping the accuracy budget. Straggler time comes from the roofline cost model
(`src/scheduler/cost_model.py`); accuracy cost = summed sensitivity of down-cast
experts.

## Inputs

Paper 0 outputs for the same model (routing counts + per-expert sensitivity),
from `20260622-moe-ep-fp8-roofline-pilot`.

## Metrics

- mean straggler cut vs LPT (%) per accuracy budget (the headline);
- accuracy cost (summed sensitivity of down-cast experts);
- #experts down-cast; (reference) LPT cut vs round-robin.

## Deliverable

The Paper 2 headline curve — straggler reduction vs accuracy budget on real
routing+sensitivity — separating placement gains from precision gains. Real
serving-engine integration is the follow-on.

## Tracking Files

config.yaml / run.md / results.md / artifacts.yaml
