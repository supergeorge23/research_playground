# Run Log

Experiment ID: `20260622-moe-ep-fp8-roofline-pilot`

## Code

- `src/profiling/roofline.py`, `src/profiling/router_stats.py` — reusable pure-python core.
- `selftest.py` — no-GPU validation of the core (run before booking the box).
- `profile_routing.py` — per-expert routing counts via HF hooks (model-intrinsic).
- `sensitivity.py` — per-expert quant sensitivity (Δnll); closes the hotness↔sensitivity loop.
- `bench_serving.py` — OpenAI-compatible load client (throughput + TTFT/TPOT).
- `run_paper0.sh` — unattended sweep orchestrator for the 8xA100 box.
- `analyze.py` — aggregates results into tables + roofline + the correlation.
- `setup_env.sh`, `requirements.txt` — environment for the box.

## Precision axis on A100 (read before booking)

A100 (Ampere) has **no FP8 tensor cores**, so the precision axis is:

- **bf16** — baseline (312 TFLOPS TC, 2 bytes). Default serving point.
- **int8 (W8A8)** — hardware-capable on A100 (624 TOPS TC + 1-byte weights ≈ 2×
  compute and 2× weight traffic), the faithful stand-in for FP8-on-H100 — BUT
  vLLM's W8A8 *serving* path is Hopper/Ada-only, so you can't serve a W8A8 MoE on
  A100 today. `sensitivity.py` fake-quantizes ourselves, so its int8 sensitivity
  numbers run on A100 regardless — that is where the int8 precision signal lives.
- **w4a16 (GPTQ/AWQ, weight-only 4-bit)** — the runnable A100 low-precision
  *serving* point (real weight-traffic win in memory-bound decode via Marlin).
  Default Qwen ckpt: `Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4` (override `W4A16_QWEN`);
  add `w4a16` to `DTYPES`. **Best-effort**: quantized-MoE on A100 (Marlin) may hit
  vLLM #35922 — keep bf16 as the guaranteed path; don't burn hours on it.
- **fp8** — NOT used on A100 (would be emulated/weight-only). Kept for a future
  H100 run; `analyze.py` reports the **fp8@H100 ceiling as a projection** regardless.

Do not use FP16 vs BF16 as a *speed* axis — they are identical in throughput and
bytes on A100 (they differ only in numerical accuracy). FP32 is not a realistic
serving point.

## Local validation first (no GPU)

```bash
python3 experiments/20260622-moe-ep-fp8-roofline-pilot/selftest.py
E=experiments/20260622-moe-ep-fp8-roofline-pilot; R=/tmp/p0; mkdir -p "$R"
python3 $E/bench_serving.py   --dry-run --model mixtral-8x7b --parallel ep8 --dtype bf16 \
    --input-len 1024 --output-len 128 --concurrency 16 --num-prompts 64 --out $R/bench_demo.json
python3 $E/profile_routing.py --dry-run --model mixtral-8x7b --n-layers 4 --n-experts 8 --out $R/routing_mixtral-8x7b.json
python3 $E/sensitivity.py     --dry-run --quant int8 --model mixtral-8x7b --n-layers 4 --n-experts 8 --out $R/sensitivity_mixtral-8x7b.json
python3 $E/analyze.py --results-dir $R
```

## On the 8xA100 box

```bash
bash experiments/20260622-moe-ep-fp8-roofline-pilot/setup_env.sh
source .venv-paper0/bin/activate
huggingface-cli login          # Mixtral is gated: log in and accept its license

# cheap first real run: smallest open model, instant sensitivity proxy, small grid
MODELS=qwen-moe-a2.7b DTYPES=bf16 CONCS="1 16" SENS_METHOD=proxy \
  bash experiments/20260622-moe-ep-fp8-roofline-pilot/run_paper0.sh

# add the runnable A100 low-precision point (W4A16; Qwen GPTQ-Int4 is the default):
#   first: pip install -U "git+https://github.com/huggingface/transformers"  # qwen2_moe
export CALIB_FILE=/path/to/calib.txt                   # optional, for paper-grade ppl
DTYPES="bf16 w4a16" bash experiments/20260622-moe-ep-fp8-roofline-pilot/run_paper0.sh
# best-effort: if the Marlin-MoE A100 path crashes (vLLM #35922), fall back to
# DTYPES=bf16 and rely on the sensitivity pass; real low-prec serving = H100.
python3 experiments/20260622-moe-ep-fp8-roofline-pilot/analyze.py \
    --results-dir experiments/20260622-moe-ep-fp8-roofline-pilot/outputs
```

Grid knobs (env): `MODELS PARALLELS DTYPES INLENS OUTLENS CONCS ENGINE GPU`, and
for sensitivity `RUN_SENSITIVITY SENS_METHOD SENS_MAX_LAYERS SENS_QUANT CALIB_FILE`.

`analyze.py` auto-discovers `sensitivity_*.json` and prints the headline test:
the (layer, expert) Pearson correlation between hotness and quant-sensitivity. A
negative r confirms "cold experts are more fragile → hot=high-precision is wrong".

## Cost notes (avoid wasted GPU hours)

- Do the cheap first run above before the full sweep — it confirms the engine,
  client, and IO all work on real hardware in minutes.
- Sensitivity `direct` runs ~1 forward/expert; keep `SENS_MAX_LAYERS` small (8) or
  use `SENS_METHOD=proxy` (instant) for a first cut, then `direct` on a subset.
- routing + sensitivity are independent of parallel/dtype, so they run once per model.

## Git Commit / Environment

TBD — record on the box: git commit, GPU type, CUDA / NCCL / torch / vllm versions.

## Notes

Raw `server.log` and large traces stay under the git-ignored `outputs/`. Copy the
headline numbers into `results.md` and link big traces from `artifacts.yaml`.
