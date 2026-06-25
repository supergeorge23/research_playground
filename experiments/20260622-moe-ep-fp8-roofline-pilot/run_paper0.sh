#!/usr/bin/env bash
# Paper 0 sweep orchestrator. Runs UNATTENDED on the 8xA100 box and writes
# results into outputs/ (git-ignored). Single boot: edit the grid, run once.
#
#   bash experiments/20260622-moe-ep-fp8-roofline-pilot/run_paper0.sh
#
# PRECISION on A100 (Ampere):
#   bf16  -> baseline (312 TFLOPS tensor core, 2 bytes).
#   int8  -> the REAL low-precision lever on A100 (624 TOPS tensor core + 1-byte
#            weights = ~2x compute AND ~2x weight traffic). Needs a pre-quantized
#            W8A8 / AWQ checkpoint -- set INT8_<MODEL> to its HF id (see int8_id).
#   fp8   -> NOT used on A100 (no FP8 tensor cores; would be emulated/weight-only).
#            Kept only for a future H100 run. The roofline reports the fp8@H100
#            ceiling as a projection regardless.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${RESULTS:-$HERE/outputs}"; mkdir -p "$RESULTS"

ENGINE="${ENGINE:-vllm}"          # vllm | sglang
GPU="${GPU:-a100-80-sxm}"
PORT="${PORT:-8000}"
NUM_PROMPTS="${NUM_PROMPTS:-256}"
MODELS="${MODELS:-mixtral-8x7b qwen-moe-a2.7b}"
PARALLELS="${PARALLELS:-ep8 tp8}"  # EP vs TP at world size 8 -- the core A100 result
DTYPES="${DTYPES:-bf16}"           # add int8 once you have a pre-quantized checkpoint
INLENS="${INLENS:-1024 8192}"
OUTLENS="${OUTLENS:-128}"
CONCS="${CONCS:-1 16 64}"
# sensitivity pass (model-intrinsic fake-quant; int8 is the A100-native target)
RUN_SENSITIVITY="${RUN_SENSITIVITY:-1}"
SENS_METHOD="${SENS_METHOD:-direct}"     # direct | proxy
SENS_MAX_LAYERS="${SENS_MAX_LAYERS:-8}"
SENS_QUANT="${SENS_QUANT:-int8}"

hf_id () {  # base bf16 checkpoint
  case "$1" in
    mixtral-8x7b)   echo "mistralai/Mixtral-8x7B-Instruct-v0.1";;
    qwen-moe-a2.7b) echo "Qwen/Qwen1.5-MoE-A2.7B-Chat";;
    *)              echo "$1";;
  esac
}

int8_id () {  # pre-quantized W8A8 ckpt (real compute low-prec; vLLM W8A8 = Hopper/Ada, NOT A100)
  case "$1" in
    mixtral-8x7b)   echo "${INT8_MIXTRAL:-}";;
    qwen-moe-a2.7b) echo "${INT8_QWEN:-}";;
    *)              echo "${INT8_MODEL:-}";;
  esac
}

w4a16_id () {  # weight-only 4-bit (GPTQ/AWQ) -- the runnable A100 low-prec serving point.
  case "$1" in  # NOTE: quantized-MoE serving on A100 (Marlin) is best-effort (vLLM #35922).
    qwen-moe-a2.7b) echo "${W4A16_QWEN:-Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4}";;
    mixtral-8x7b)   echo "${W4A16_MIXTRAL:-}";;
    *)              echo "${W4A16_MODEL:-}";;
  esac
}

MID=""; QFLAGS=()
resolve () {  # short dtype -> sets MID + QFLAGS; returns 1 to skip this point
  local short="$1" dtype="$2"; QFLAGS=()
  case "$dtype" in
    bf16) MID="$(hf_id "$short")";;
    int8) MID="$(int8_id "$short")"
          [[ -z "$MID" ]] && { echo "SKIP int8/$short: W8A8 needs a pre-quantized ckpt and vLLM W8A8 is Hopper/Ada (not A100); set INT8_${short^^}"; return 1; };;
    w4a16) MID="$(w4a16_id "$short")"
          [[ -z "$MID" ]] && { echo "SKIP w4a16/$short: set W4A16_${short^^} to a GPTQ/AWQ checkpoint"; return 1; }
          [[ "$GPU" == a100* ]] && echo "NOTE: w4a16 MoE serving on $GPU uses Marlin (best-effort; may hit vLLM #35922).";;
    fp8)  [[ "$GPU" == a100* ]] && { echo "SKIP fp8/$short on $GPU (no FP8 tensor cores)"; return 1; }
          MID="$(hf_id "$short")"; QFLAGS=(--quantization fp8);;
    *)    MID="$(hf_id "$short")";;
  esac
  return 0
}

CUR_PID=""
cleanup () { [[ -n "$CUR_PID" ]] && kill "$CUR_PID" 2>/dev/null || true; }
trap cleanup EXIT

wait_health () {
  for _ in $(seq 1 120); do
    curl -sf "$1/health"    >/dev/null 2>&1 && return 0
    curl -sf "$1/v1/models" >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}

launch () {  # tp ep   (uses global MID + QFLAGS)
  local tp="$1" ep="$2"
  if [[ "$ENGINE" == "vllm" ]]; then
    local e=(); [[ "$ep" == "1" ]] && e=(--enable-expert-parallel)
    vllm serve "$MID" --port "$PORT" --tensor-parallel-size "$tp" \
      "${e[@]}" "${QFLAGS[@]}" --gpu-memory-utilization 0.90 \
      >"$RESULTS/server.log" 2>&1 &
  else
    local e=(); [[ "$ep" == "1" ]] && e=(--ep-size "$tp")
    python3 -m sglang.launch_server --model-path "$MID" --port "$PORT" --tp "$tp" \
      "${e[@]}" "${QFLAGS[@]}" --mem-fraction-static 0.90 \
      >"$RESULTS/server.log" 2>&1 &
  fi
  CUR_PID=$!
}

for model in $MODELS; do
  base="$(hf_id "$model")"

  echo ">>> routing pass: $model"
  python3 "$HERE/profile_routing.py" --model "$base" --num-seqs 256 --seq-len 512 \
    --ep-size 8 --out "$RESULTS/routing_${model}.json" || echo "WARN routing $model"

  if [[ "$RUN_SENSITIVITY" == "1" ]]; then
    echo ">>> sensitivity pass: $model ($SENS_QUANT/$SENS_METHOD, max-layers $SENS_MAX_LAYERS)"
    python3 "$HERE/sensitivity.py" --model "$base" --quant "$SENS_QUANT" \
      --method "$SENS_METHOD" --max-layers "$SENS_MAX_LAYERS" --num-calib-seqs 16 \
      ${CALIB_FILE:+--calib-file "$CALIB_FILE"} \
      --out "$RESULTS/sensitivity_${model}.json" || echo "WARN sensitivity $model"
  fi

  for par in $PARALLELS; do
    tp=8; ep=0; [[ "$par" == ep* ]] && ep=1
    for dtype in $DTYPES; do
      if ! resolve "$model" "$dtype"; then continue; fi
      url="http://localhost:$PORT"
      echo ">>> launch $model $par $dtype ($MID)"
      launch "$tp" "$ep"
      if ! wait_health "$url"; then
        echo "ERROR: server unhealthy for $model/$par/$dtype (see server.log)"
        cleanup; CUR_PID=""; continue
      fi
      for inl in $INLENS; do for outl in $OUTLENS; do for c in $CONCS; do
        tag="${model}_${par}_${dtype}_in${inl}_out${outl}_c${c}"
        python3 "$HERE/bench_serving.py" --base-url "$url" --model "$MID" \
          --engine "$ENGINE" --parallel "$par" --dtype "$dtype" --gpu "$GPU" \
          --input-len "$inl" --output-len "$outl" --concurrency "$c" \
          --num-prompts "$NUM_PROMPTS" --out "$RESULTS/bench_${tag}.json" \
          || echo "WARN bench $tag"
      done; done; done
      cleanup; wait "$CUR_PID" 2>/dev/null || true; CUR_PID=""; sleep 3
    done
  done
done

echo "Done. Analyze with:"
echo "  python3 $HERE/analyze.py --results-dir $RESULTS"
