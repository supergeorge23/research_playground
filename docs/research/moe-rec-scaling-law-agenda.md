# Research Agenda: Scaling Laws for MoE Recommender Systems

## Starting Observation

In recent recommendation-system experiments, MoE models appear to exhibit
scaling-law behavior. This is strategically interesting because MoE decouples
total parameters from activated parameters, while recommender systems are
constrained by online data drift, sparse user-item signals, large embeddings,
and strict latency budgets.

## Central Research Question

Can we establish a practical scaling law for MoE recommender systems that
predicts quality under fixed data, compute, memory, and latency constraints?

## Candidate Claims

1. MoE recommenders have a measurable scaling law, but the relevant capacity
   variable is not only total parameter count.
2. Activated parameters, expert count, router entropy, expert load balance, and
   data scale jointly determine the effective scaling frontier.
3. MoE can keep improving after dense parameter scaling saturates, but only if
   the infrastructure controls routing imbalance, communication overhead, and
   serving cost.
4. Recommendation workloads expose different MoE scaling behavior from language
   modeling because examples are sparse, multi-objective, non-stationary, and
   latency constrained.

## Experiment Axes

| Axis | Examples |
| --- | --- |
| Data scale | sampled days, user/event count, item coverage, multimodal feature coverage |
| Model scale | dense width/depth, total MoE params, activated MoE params |
| Expert structure | number of experts, top-k, shared experts, expert granularity |
| Router behavior | entropy, load balance, specialization, drop rate |
| Compute budget | train FLOPs, GPU-hours, memory, network bandwidth |
| Serving budget | p50/p95 latency, QPS, active params/request, memory residency |
| Objective | CTR, CVR, watch time, retention, multi-task weighted utility |

## Baseline Protocol

1. Start with the current validated internal MoE recommendation setup.
2. Freeze data slices and feature definitions for the first scaling-law fit.
3. Run dense and MoE model families under matched training compute.
4. Fit curves for quality as a function of data, total parameters, activated
   parameters, and compute.
5. Reserve at least one larger scale as an extrapolation test.
6. Report negative findings: non-monotonic regions, expert collapse, router
   instability, and infra bottlenecks.

## Systems Angle

The strongest publishable systems direction is not merely "MoE improves Rec",
but:

- what scaling variable actually predicts quality for MoE recommenders;
- where the system bottleneck moves as MoE scales;
- how to schedule, shard, route, cache, or serve experts to improve the
  quality-cost frontier.

Potential systems topics:

- expert-parallel training for sparse recommendation workloads;
- routing-aware placement and caching for online recommendation serving;
- communication-computation overlap for MoE recommenders;
- FlashAttention-style kernel optimization as one component of the larger
  attention-plus-expert pipeline;
- LLM4Rec/MoE hybrid models under industrial latency constraints.

## Paper Skeleton

Working title: `Scaling Laws for Mixture-of-Experts Recommender Systems`

1. Motivation: dense scaling limits and MoE opportunity in recommendation.
2. Measurement: controlled scaling-law evidence from production-like workloads.
3. Analysis: which variables explain quality and which variables fail.
4. Systems bottlenecks: router imbalance, expert communication, memory, serving.
5. Prototype: infra improvement that shifts the quality-cost frontier.
6. Evaluation: offline metrics, online-proxy metrics, cost, latency, stability.
7. Implications: planning model size, data scale, GPU budget, and serving stack.

## Open Questions

- Does quality scale better with total params, activated params, or an
  interaction term?
- Does the optimal active-expert ratio become sparser as total model size grows?
- Does scaling behavior differ across recommendation surfaces or objectives?
- Are gains coming from specialization, capacity, regularization, or data
  partitioning effects?
- At what point do all-to-all communication and expert imbalance erase model
  quality gains?
