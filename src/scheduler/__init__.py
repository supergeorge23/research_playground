"""Paper 2: roofline-driven joint precision x placement scheduling for MoE EP.

Pure-python (uses src.profiling). Turns Paper 0 measurements (per-expert routing
counts + quantization sensitivity) into a scheduling decision that uses precision
as a load-balancing knob to cut the synchronous-layer EP straggler, and compares
it against placement-only and uniform baselines.
"""

from . import cost_model, precision_placement  # noqa: F401

__all__ = ["cost_model", "precision_placement"]
