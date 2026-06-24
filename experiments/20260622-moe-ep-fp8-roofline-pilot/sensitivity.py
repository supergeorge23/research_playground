#!/usr/bin/env python3
"""Per-expert quantization sensitivity probe -- closes the hotness-vs-sensitivity loop.

For each MoE expert, fake-quantize its FFN weights (fp8 E4M3 or int8) and measure
the increase in NLL on a small calibration set vs the bf16 baseline. Larger Δnll =
more sensitive. Output keys by (moe_layer_index, expert) so it pairs exactly with
profile_routing.py for the correlation in analyze.py.

methods:
  direct  -- fake-quant the expert, re-run calib, measure Δnll (accurate, ~1 fwd/expert)
  proxy   -- per-expert weight quantization MSE (instant, no forward; coarse proxy)

--dry-run synthesizes Δnll with no GPU/weights so analyze.py's correlation path is
testable locally. fp8 fake-quant needs torch>=2.1 (float8_e4m3fn); it is a numeric
cast for SIMULATION and runs fine on A100 (we measure accuracy, not speed).

Real run (subsample layers to keep it cheap):
  python3 sensitivity.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
      --quant fp8 --method direct --calib-file calib.txt --max-layers 8 \
      --out outputs/sensitivity_mixtral.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_LAYER_RE = re.compile(r"layers?\.(\d+)")
_EXPERT_RE = re.compile(r"\.experts\.(\d+)\b")

_BUILTIN_CALIB = [
    "The mixture-of-experts layer routes each token to a small subset of experts.",
    "Expert parallelism distributes expert weights across multiple GPUs.",
    "Quantization reduces memory traffic but may degrade rare experts.",
    "A roofline model separates compute-bound from memory-bound kernels.",
]


def _layer_num(name: str) -> int:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else 0


# ----------------------------- real (GPU) path ------------------------------ #

def _fake_quant(w, quant: str):
    import torch
    if quant == "fp8":
        scale = w.abs().amax().clamp(min=1e-8) / 448.0          # E4M3 max ~448
        q = torch.clamp(w / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
        return q.to(w.dtype) * scale
    # int8 symmetric per-tensor
    scale = w.abs().amax().clamp(min=1e-8) / 127.0
    return (torch.clamp((w / scale).round(), -127, 127)) * scale


def _calib_batches(tok, model, args):
    import torch
    lines = _BUILTIN_CALIB
    if args.calib_file:
        lines = [ln.strip() for ln in Path(args.calib_file).read_text().splitlines() if ln.strip()]
    lines = lines[: args.num_calib_seqs] or _BUILTIN_CALIB
    batches = []
    for ln in lines:
        ids = tok(ln, return_tensors="pt", truncation=True, max_length=args.seq_len).input_ids
        if ids.shape[1] >= 2:
            batches.append(ids.to(model.device))
    return batches


def _nll(model, batches) -> float:
    import torch
    tot, n = 0.0, 0
    with torch.no_grad():
        for ids in batches:
            out = model(ids, labels=ids)
            tot += float(out.loss.item())
            n += 1
    return tot / max(1, n)


def _experts_index(model):
    """Return {(layer_num, expert_num): [linear_modules...]} for all MoE experts."""
    import torch.nn as nn
    groups = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        me = _EXPERT_RE.search(name)
        if not me:
            continue
        key = (_layer_num(name), int(me.group(1)))
        groups.setdefault(key, []).append(mod)
    return groups


def _real_sensitivity(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="bfloat16", device_map="auto", trust_remote_code=True)
    model.eval()

    groups = _experts_index(model)
    # 0-based moe_layer_index in numeric layer order (matches profile_routing.py)
    layer_nums = sorted({L for (L, _) in groups})
    moe_idx = {L: i for i, L in enumerate(layer_nums)}
    if args.max_layers:
        keep = set(layer_nums[: args.max_layers])
        groups = {k: v for k, v in groups.items() if k[0] in keep}

    out = []
    if args.method == "proxy":
        for (L, E), mods in sorted(groups.items()):
            err = 0.0
            for lin in mods:
                w = lin.weight.data
                err += float(((w - _fake_quant(w, args.quant)) ** 2).mean().item())
            out.append((moe_idx[L], E, err / len(mods)))
        return out, None

    batches = _calib_batches(tok, model, args)
    base = _nll(model, batches)
    for (L, E), mods in sorted(groups.items()):
        saved = [lin.weight.data.clone() for lin in mods]
        for lin in mods:
            lin.weight.data.copy_(_fake_quant(lin.weight.data, args.quant))
        delta = _nll(model, batches) - base
        for lin, w0 in zip(mods, saved):
            lin.weight.data.copy_(w0)
        out.append((moe_idx[L], E, delta))
    return out, base


# ------------------------------- dry-run path ------------------------------- #

def _synth_sensitivity(args):
    import random
    rng = random.Random(1)
    out = []
    for L in range(args.n_layers):
        for E in range(args.n_experts):
            # cold experts (higher index) more fragile -> larger delta
            out.append((L, E, round((E + 1) / args.n_experts + rng.uniform(0, 0.1), 4)))
    return out, 2.345


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    ap.add_argument("--quant", choices=["fp8", "int8"], default="fp8")
    ap.add_argument("--method", choices=["direct", "proxy"], default="direct")
    ap.add_argument("--calib-file", default=None)
    ap.add_argument("--num-calib-seqs", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--max-layers", type=int, default=0, help="0 = all MoE layers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-layers", type=int, default=32)   # dry-run shape
    ap.add_argument("--n-experts", type=int, default=8)
    args = ap.parse_args()

    if args.dry_run:
        rows, base = _synth_sensitivity(args)
    else:
        rows, base = _real_sensitivity(args)

    per_expert = [{"layer": L, "expert": E, "sensitivity": round(d, 6)} for (L, E, d) in rows]
    out = {
        "config": {"model": args.model, "quant": args.quant, "method": args.method,
                   "dry_run": args.dry_run},
        "baseline_nll": base,
        "per_expert": per_expert,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    deltas = [r["sensitivity"] for r in per_expert]
    rng = (min(deltas), max(deltas)) if deltas else (0, 0)
    print(f"[sensitivity] {args.model} {args.quant}/{args.method}: "
          f"{len(per_expert)} experts, delta range [{rng[0]:.4f}, {rng[1]:.4f}] -> {args.out}")


if __name__ == "__main__":
    main()
