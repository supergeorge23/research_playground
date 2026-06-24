#!/usr/bin/env bash
# One-time environment setup for the Paper 0 box (run on the 8xA100 server).
# Pin versions to match the box CUDA driver; adjust as needed.
set -euo pipefail

PYBIN="${PYBIN:-python3}"
VENV="${VENV:-.venv-paper0}"

"$PYBIN" -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
pip install --upgrade pip

# Pick ONE serving engine (vLLM is the default in run_paper0.sh).
# Match the torch/CUDA build to the box; see the engine's install docs.
pip install "vllm>=0.6.3"
# Alternative: pip install "sglang[all]>=0.4"

# For the model-intrinsic routing pass (HF forward hooks):
pip install "transformers>=4.44" "datasets>=2.20" "numpy>=1.26"

echo
echo "Env ready. Next:"
echo "  source $VENV/bin/activate"
echo "  python3 experiments/20260622-moe-ep-fp8-roofline-pilot/selftest.py   # no-GPU sanity"
echo "  bash    experiments/20260622-moe-ep-fp8-roofline-pilot/run_paper0.sh # full sweep"
