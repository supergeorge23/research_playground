#!/usr/bin/env bash
# One-shot runner for the precision x parallelism track (Papers 0/1/2).
# Run from the repo root.
#
#   bash scripts/run_all_precision_parallelism.sh             # full run, on the A100 box
#   bash scripts/run_all_precision_parallelism.sh --dry-run   # local smoke test, NO GPU
#
# Chain: P0 (measure: routing/sensitivity/serving) -> P1 (grouped-GEMM utilization)
#        -> P2 (precision x placement straggler simulation, reads P0 outputs).
# Knobs (env): GPU, EP_SIZE, MODELS, STAGES (subset of "0 1 2").
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
DRY=""; [[ "${1:-}" == "--dry-run" ]] && DRY=1

GPU="${GPU:-a100-80-sxm}"
EP_SIZE="${EP_SIZE:-8}"
MODELS="${MODELS:-mixtral-8x7b qwen-moe-a2.7b}"
STAGES="${STAGES:-0 1 2}"

P0=experiments/20260622-moe-ep-fp8-roofline-pilot
P1=experiments/20260622-grouped-gemm-utilization-pilot
P2=experiments/20260622-precision-placement-scheduler-pilot
P0_OUT="$P0/outputs"; [[ -n "$DRY" ]] && P0_OUT=/tmp/p0_all

nexp ()   { case "$1" in qwen-moe-a2.7b) echo 60;;   *) echo 8;;     esac; }
hidden () { case "$1" in qwen-moe-a2.7b) echo 2048;; *) echo 4096;;  esac; }
inter ()  { case "$1" in qwen-moe-a2.7b) echo 1408;; *) echo 14336;; esac; }

has () { [[ " $STAGES " == *" $1 "* ]]; }

if has 0; then
  echo "########## Paper 0: measurement ##########"
  if [[ -n "$DRY" ]]; then
    rm -rf "$P0_OUT"; mkdir -p "$P0_OUT"
    for m in $MODELS; do
      python3 $P0/profile_routing.py --dry-run --model "$m" --n-layers 4 --n-experts "$(nexp "$m")" --out "$P0_OUT/routing_${m}.json"
      python3 $P0/sensitivity.py     --dry-run --quant int8 --model "$m" --n-layers 4 --n-experts "$(nexp "$m")" --out "$P0_OUT/sensitivity_${m}.json"
    done
    python3 $P0/analyze.py --results-dir "$P0_OUT" || true
  else
    bash $P0/run_paper0.sh
    python3 $P0/analyze.py --results-dir "$P0/outputs"
  fi
fi

if has 1; then
  echo "########## Paper 1: grouped-GEMM utilization ##########"
  mkdir -p "$P1/outputs"
  for m in $MODELS; do
    if [[ -n "$DRY" ]]; then
      python3 $P1/bench_grouped_gemm.py --dry-run --n-experts "$(nexp "$m")" \
        --hidden "$(hidden "$m")" --intermediate "$(inter "$m")" --total-tokens 512 \
        --gpu "$GPU" --methods loop
    else
      for T in 64 512 4096 32768; do
        python3 $P1/bench_grouped_gemm.py --n-experts "$(nexp "$m")" \
          --hidden "$(hidden "$m")" --intermediate "$(inter "$m")" --total-tokens "$T" \
          --gpu "$GPU" --dtypes bf16,int8 --dists uniform,skew \
          --out "$P1/outputs/gg_${m}_T${T}.json"
      done
    fi
  done
fi

if has 2; then
  echo "########## Paper 2: precision x placement scheduling ##########"
  mkdir -p "$P2/outputs"
  for m in $MODELS; do
    out=""; [[ -z "$DRY" ]] && out="--out $P2/outputs/paper2_${m}.json"
    # reads P0 routing+sensitivity from $P0_OUT (real file path, even in dry-run)
    python3 $P2/eval_paper2.py --results-dir "$P0_OUT" --model "$m" \
      --gpu "$GPU" --ep-size "$EP_SIZE" $out
  done
fi

echo "########## done (stages: $STAGES; dry-run: ${DRY:-no}) ##########"
[[ -z "$DRY" ]] && echo "Outputs under each experiment's outputs/ (git-ignored)."
