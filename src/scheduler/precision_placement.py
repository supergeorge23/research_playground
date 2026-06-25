"""Joint precision x placement scheduling for one MoE layer (Paper 2 core).

Decide, per expert: which GPU it lives on (placement) and whether to run it at
HIGH precision (bf16) or LOW precision (int8 on A100 / fp8 on H100), to minimize
the synchronous-layer straggler = max over GPUs of summed expert time, subject to
a total accuracy budget (sum of sensitivity of down-cast experts <= budget).

The central Paper 2 claim is that precision is a *load-balancing* dimension:
down-casting the overloaded, non-sensitive experts equalizes per-GPU finish times
without moving tokens or experts. This module quantifies that vs placement-only
and uniform baselines on real Paper 0 data.

Pure-python; no GPU. Greedy is used because the per-layer problem is small
(n_experts up to a few hundred) and the schedule is recomputed per routing window.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import cost_model as cm


@dataclass
class Plan:
    placement: list      # expert -> gpu index
    low: list            # expert -> bool (down-cast to low precision?)
    straggler_s: float   # synchronous-layer time = max GPU load
    acc_cost: float      # sum of sensitivity of down-cast experts
    n_low: int


def _gpu_loads(times, placement, ep_size):
    loads = [0.0] * ep_size
    for e, t in enumerate(times):
        loads[placement[e]] += t
    return loads


def _lpt_placement(times, ep_size):
    """Longest-processing-time greedy bin-packing -> balanced placement."""
    placement = [0] * len(times)
    loads = [0.0] * ep_size
    for e in sorted(range(len(times)), key=lambda i: -times[i]):
        g = min(range(ep_size), key=lambda i: loads[i])
        placement[e] = g
        loads[g] += times[e]
    return placement


def _rr_placement(n, ep_size):
    return [e % ep_size for e in range(n)]


def _times(counts, dtype, hidden, intermediate, gpu, gated):
    return [cm.expert_time_s(counts[e], dtype, hidden, intermediate, gpu, gated)
            for e in range(len(counts))]


def schedule_layer(counts, sens, ep_size, hidden, intermediate, gpu,
                   budget, gated=True) -> Plan:
    """Joint precision+placement greedy: bf16+LPT, then down-cast the best expert
    on the straggler GPU (max time-saved per unit sensitivity) until no strict
    improvement or the accuracy budget is exhausted."""
    n = len(counts)
    low_dt = cm.low_dtype(gpu)
    t_high = _times(counts, "bf16", hidden, intermediate, gpu, gated)
    t_low = _times(counts, low_dt, hidden, intermediate, gpu, gated)

    low = [False] * n
    times = list(t_high)
    placement = _lpt_placement(times, ep_size)
    acc = 0.0
    prev = max(_gpu_loads(times, placement, ep_size))

    while acc < budget:
        loads = _gpu_loads(times, placement, ep_size)
        sg = max(range(ep_size), key=lambda i: loads[i])
        cands = [e for e in range(n)
                 if placement[e] == sg and not low[e] and counts[e] > 0
                 and acc + sens[e] <= budget and (t_high[e] - t_low[e]) > 0]
        if not cands:
            break
        e = max(cands, key=lambda x: (t_high[x] - t_low[x]) / max(sens[x], 1e-9))
        # tentatively down-cast and re-balance
        low[e] = True
        times[e] = t_low[e]
        acc += sens[e]
        placement = _lpt_placement(times, ep_size)
        cur = max(_gpu_loads(times, placement, ep_size))
        if cur >= prev - 1e-15:        # spent budget without helping -> revert, stop
            low[e] = False
            times[e] = t_high[e]
            acc -= sens[e]
            placement = _lpt_placement(times, ep_size)
            break
        prev = cur

    return Plan(placement, low, max(_gpu_loads(times, placement, ep_size)), acc,
                sum(low))


def baseline_straggler(counts, ep_size, hidden, intermediate, gpu, mode,
                       gated=True):
    """Reference stragglers (all bf16): 'rr' = round-robin placement, 'lpt' =
    load-balanced placement (the placement-only baseline Paper 2 must beat)."""
    times = _times(counts, "bf16", hidden, intermediate, gpu, gated)
    placement = _rr_placement(len(counts), ep_size) if mode == "rr" \
        else _lpt_placement(times, ep_size)
    return max(_gpu_loads(times, placement, ep_size))


def evaluate_layer(counts, sens, ep_size, hidden, intermediate, gpu, budget,
                   gated=True):
    """Straggler for rr / lpt(placement-only) / joint, on one layer."""
    rr = baseline_straggler(counts, ep_size, hidden, intermediate, gpu, "rr", gated)
    lpt = baseline_straggler(counts, ep_size, hidden, intermediate, gpu, "lpt", gated)
    plan = schedule_layer(counts, sens, ep_size, hidden, intermediate, gpu, budget, gated)
    return {
        "rr_s": rr, "lpt_s": lpt, "joint_s": plan.straggler_s,
        "acc_cost": plan.acc_cost, "n_low": plan.n_low,
        "joint_vs_lpt": (plan.straggler_s / lpt) if lpt else 1.0,
    }
