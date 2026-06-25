# Run Log

Experiment ID: `20260622-precision-placement-scheduler-pilot`

## Local validation (no GPU, no data)

```bash
python3 experiments/20260622-precision-placement-scheduler-pilot/eval_paper2.py \
    --dry-run --model qwen-moe-a2.7b --n-layers 6 --n-experts 60
python3 experiments/20260622-precision-placement-scheduler-pilot/eval_paper2.py \
    --dry-run --model mixtral-8x7b --n-layers 8 --n-experts 8
```

## Real run (after Paper 0 produced routing + sensitivity)

```bash
P0=experiments/20260622-moe-ep-fp8-roofline-pilot/outputs
python3 experiments/20260622-precision-placement-scheduler-pilot/eval_paper2.py \
    --results-dir $P0 --model mixtral-8x7b --gpu a100-80-sxm --ep-size 8 \
    --out experiments/20260622-precision-placement-scheduler-pilot/outputs/paper2_mixtral.json
# repeat with --model qwen-moe-a2.7b
```

This reads Paper 0 `routing_<model>.json` + `sensitivity_<model>.json`. With no
sensitivity file, experts default to "highly sensitive" (won't be down-cast), so
run Paper 0's sensitivity pass first for a meaningful curve.

## What to look for

- `straggler_cut_vs_lpt`: how much precision-balancing beats placement-only LPT
  at each accuracy budget. For coarse MoE (n_experts == ep_size) placement can't
  rebalance, so precision is the *only* lever; for fine-grained MoE it stacks on
  top of LPT.
- `acc_cost` / `#downcast`: the price paid (summed sensitivity) for that cut.

## Git Commit / Environment

TBD.

## Notes

Output lands in the git-ignored `outputs/`. Copy the headline curve into
`results.md`.
