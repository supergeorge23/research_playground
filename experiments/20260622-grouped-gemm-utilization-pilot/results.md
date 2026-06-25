# Results

Experiment ID: `20260622-grouped-gemm-utilization-pilot`

## Metrics

TBD (utilization + achieved TFLOP/s per dist × dtype × method × token-regime).

## Interpretation

TBD (how far below peak is the skewed/decode regime? how much does int8 help in
each regime? how much does padding waste cost under skew?).

## Decision

TBD (what utilization headroom justifies building the custom kernel, and which
regime to target first?).
