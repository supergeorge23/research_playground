"""Reusable profiling utilities for the MoE precision x parallelism track.

Pure-python core (no GPU, no heavy deps):
- roofline: classify (grouped) GEMMs as compute- vs memory-bound and predict
  the low-precision speedup ceiling on a given GPU.
- router_stats: per-expert hotness / load-skew metrics and an EP straggler model.

GPU-dependent collectors live under the experiment folder and import from here.
"""

from . import roofline, router_stats  # noqa: F401

__all__ = ["roofline", "router_stats"]
