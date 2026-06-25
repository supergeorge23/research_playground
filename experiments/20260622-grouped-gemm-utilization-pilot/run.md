# Run Log

Experiment ID: `20260622-grouped-gemm-utilization-pilot`

## Local validation (no GPU)

```bash
python3 experiments/20260622-grouped-gemm-utilization-pilot/bench_grouped_gemm.py \
    --dry-run --n-experts 64 --hidden 2048 --intermediate 1408 --total-tokens 4096
```

## On the A100 box

```bash
E=experiments/20260622-grouped-gemm-utilization-pilot
# decode -> prefill regimes for the fine-grained (Qwen) shape:
for T in 64 512 4096 32768; do
  python3 $E/bench_grouped_gemm.py --n-experts 64 --hidden 2048 --intermediate 1408 \
      --total-tokens $T --gpu a100-80-sxm --dtypes bf16,int8 --dists uniform,skew \
      --out $E/outputs/gg_qwen_T$T.json
done
# coarse (Mixtral) shape:
python3 $E/bench_grouped_gemm.py --n-experts 8 --hidden 4096 --intermediate 14336 \
    --total-tokens 4096 --gpu a100-80-sxm --out $E/outputs/gg_mixtral.json
```

## What to look for

- `utilization` falling from uniform→skew and at small `tokens/expert` (decode):
  that is the gap a utilization-aware kernel must recover.
- int8 vs bf16 achieved TFLOP/s in each regime (int8 should ~2x when compute-bound;
  in memory-bound decode the win is the halved/quartered weight traffic).
- `padded` method utilization vs `loop` under skew (padding waste).

## Git Commit / Environment

TBD — record torch / CUDA versions and the exact GPU.

## Notes

`torch._int_mm` needs SM80+ (A100 OK). fp8 (`torch._scaled_mm`) is H100-only.
Outputs land in the git-ignored `outputs/`.
