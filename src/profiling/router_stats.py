"""Per-expert hotness / load-skew metrics for MoE routing.

Pure-python. Feed it per-expert token counts (one list per MoE layer, collected
by the HF forward-hook collector). Produces the skew signals Paper 0 needs and
an EP straggler model that already accepts a per-expert cost so Paper 2 can
later simulate "precision as a load-balancing knob".
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def gini(xs) -> float:
    xs = sorted(float(x) for x in xs)
    n = len(xs)
    s = sum(xs)
    if n == 0 or s == 0:
        return 0.0
    cum = sum(i * x for i, x in enumerate(xs, start=1))
    return (2.0 * cum) / (n * s) - (n + 1.0) / n


def norm_entropy(counts) -> float:
    """Routing entropy normalized to [0,1]; 1.0 = perfectly balanced."""
    s = sum(counts)
    if s == 0:
        return 0.0
    ps = [c / s for c in counts if c > 0]
    h = -sum(p * math.log(p) for p in ps)
    return h / math.log(len(counts)) if len(counts) > 1 else 0.0


@dataclass
class SkewMetrics:
    n_experts: int
    total_tokens: int
    max_count: int
    mean_count: float
    max_over_mean: float      # straggler factor for the single hottest expert
    gini: float
    norm_entropy: float
    frac_active: float        # fraction of experts that saw >=1 token


def skew_metrics(counts) -> SkewMetrics:
    n = len(counts)
    total = int(sum(counts))
    mx = max(counts) if counts else 0
    mean = total / n if n else 0.0
    active = sum(1 for c in counts if c > 0)
    return SkewMetrics(
        n_experts=n,
        total_tokens=total,
        max_count=int(mx),
        mean_count=mean,
        max_over_mean=(mx / mean) if mean else 0.0,
        gini=gini(counts),
        norm_entropy=norm_entropy(counts),
        frac_active=(active / n) if n else 0.0,
    )


def ep_straggler_factor(counts, ep_size, placement=None, cost_per_token=None) -> float:
    """Synchronous MoE-layer time ~ max GPU load / mean GPU load under EP.

    counts: per-expert token counts.
    placement: expert_idx -> gpu_idx (default round-robin across ep_size).
    cost_per_token: per-expert relative cost (default 1.0). Set < 1.0 for a
      down-cast expert to model precision as a balancing knob.
    Returns 1.0 for perfect balance; larger = worse straggler.
    """
    n = len(counts)
    if ep_size <= 0:
        raise ValueError("ep_size must be positive")
    if cost_per_token is None:
        cost_per_token = [1.0] * n
    if placement is None:
        placement = [e % ep_size for e in range(n)]
    loads = [0.0] * ep_size
    for e in range(n):
        loads[placement[e]] += counts[e] * cost_per_token[e]
    mean = sum(loads) / ep_size
    return (max(loads) / mean) if mean else 0.0


def correlation(xs, ys) -> float:
    """Pearson correlation; used to test 'is sensitivity correlated with hotness?'"""
    n = len(xs)
    if n == 0 or n != len(ys):
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return (cov / (vx * vy)) if vx and vy else 0.0
