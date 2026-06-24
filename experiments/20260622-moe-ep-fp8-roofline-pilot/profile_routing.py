#!/usr/bin/env python3
"""Collect per-expert routing counts via HF forward hooks (one list per MoE layer).

This answers the "is sensitivity correlated with hotness?" and "how skewed is the
load?" questions WITHOUT a serving engine -- it is model-intrinsic. Works for
Mixtral- and Qwen-MoE-style gates out of the box; extend GATE_SUFFIXES for other
architectures. Writes results/routing_<tag>.json.

Layers are emitted as a 0-based moe_layer_index in NUMERIC layer order, matching
sensitivity.py so analyze.py can pair (layer, expert) exactly.

--dry-run synthesizes a skewed routing distribution (no GPU / no weights).

Real run (needs GPU + weights):
  python3 profile_routing.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
      --num-seqs 256 --seq-len 512 --out results/routing_mixtral.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.profiling import router_stats as rs  # noqa: E402

# Module-name suffixes that identify the MoE router/gate linear layer.
GATE_SUFFIXES = ("block_sparse_moe.gate", "mlp.gate", "moe.gate", "gate")
_LAYER_RE = re.compile(r"layers?\.(\d+)")


def _layer_num(name: str) -> int:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else 0


def _synth_counts(n_layers, n_experts, top_k, n_tokens, seed=0):
    import random
    rng = random.Random(seed)
    layers = []
    for _ in range(n_layers):
        weights = [1.0 / (i + 1) ** 1.3 for i in range(n_experts)]
        rng.shuffle(weights)
        counts = [0] * n_experts
        for _ in range(n_tokens * top_k):
            r = rng.random() * sum(weights)
            acc = 0.0
            for e, w in enumerate(weights):
                acc += w
                if r <= acc:
                    counts[e] += 1
                    break
        layers.append(counts)
    return layers


def _real_counts(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    n_experts = getattr(cfg, "num_local_experts", None) or getattr(cfg, "num_experts")
    top_k = getattr(cfg, "num_experts_per_tok", 2)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="bfloat16", device_map="auto", trust_remote_code=True)
    model.eval()

    counts = {}          # module name -> list[n_experts]
    handles = []

    def make_hook(name):
        def hook(_module, _inp, out):
            logits = out[0] if isinstance(out, tuple) else out
            idx = logits.reshape(-1, logits.shape[-1]).topk(top_k, dim=-1).indices
            binc = torch.bincount(idx.reshape(-1), minlength=n_experts)
            c = counts.setdefault(name, [0] * n_experts)
            for e in range(n_experts):
                c[e] += int(binc[e].item())
        return hook

    for name, mod in model.named_modules():
        if name.endswith(GATE_SUFFIXES):
            handles.append(mod.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for _ in range(args.num_seqs):
            ids = tok("hello " * args.seq_len, return_tensors="pt").input_ids
            ids = ids[:, : args.seq_len].to(model.device)
            model(ids)
    for h in handles:
        h.remove()
    # 0-based moe_layer_index in numeric layer order (matches sensitivity.py)
    return [counts[k] for k in sorted(counts, key=_layer_num)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    ap.add_argument("--num-seqs", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--ep-size", type=int, default=8,
                    help="EP degree to evaluate the straggler factor at")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-layers", type=int, default=32)
    ap.add_argument("--n-experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    args = ap.parse_args()

    if args.dry_run:
        layers = _synth_counts(args.n_layers, args.n_experts, args.top_k,
                               args.num_seqs * args.seq_len // 32)
    else:
        layers = _real_counts(args)

    per_layer = []
    for li, counts in enumerate(layers):
        m = rs.skew_metrics(counts)
        per_layer.append({
            "layer": li, "counts": counts,
            "gini": round(m.gini, 4), "norm_entropy": round(m.norm_entropy, 4),
            "max_over_mean": round(m.max_over_mean, 4),
            "frac_active": round(m.frac_active, 4),
            "ep_straggler_factor": round(rs.ep_straggler_factor(counts, args.ep_size), 4),
        })
    summary = {
        "n_layers": len(per_layer),
        "mean_gini": round(sum(p["gini"] for p in per_layer) / len(per_layer), 4),
        "mean_straggler": round(
            sum(p["ep_straggler_factor"] for p in per_layer) / len(per_layer), 4),
        "ep_size": args.ep_size,
    }
    out = {"config": {"model": args.model, "dry_run": args.dry_run, "ep_size": args.ep_size},
           "summary": summary, "per_layer": per_layer}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[routing] {args.model}: {summary['n_layers']} layers, "
          f"mean gini={summary['mean_gini']}, "
          f"mean EP({args.ep_size}) straggler={summary['mean_straggler']} -> {args.out}")


if __name__ == "__main__":
    main()
