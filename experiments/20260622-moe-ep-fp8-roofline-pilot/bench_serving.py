#!/usr/bin/env python3
"""Load client for MoE serving profiling (engine-agnostic, OpenAI-compatible).

Drives an already-running vLLM or SGLang server (launched by run_paper0.sh) and
records throughput + latency percentiles for a fixed workload. The engine,
parallelism and dtype are passed in only as metadata (echoed into the JSON) so
analyze.py can compare EP vs TP, bf16 vs fp8, etc.

Use --dry-run to exercise the aggregation / IO path with no server, for local
validation before touching the GPU box.

Example (against a running server on :8000):
  python3 bench_serving.py --base-url http://localhost:8000 --model mixtral \
      --engine vllm --parallel ep8 --dtype bf16 --gpu a100-80-sxm \
      --input-len 1024 --output-len 128 --concurrency 16 --num-prompts 128 \
      --out results/bench_mixtral_ep8_bf16_in1024_out128_c16.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def pctl(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def make_prompt(input_len):
    # ~1 word ≈ 1 token; good enough to size prompts for load generation.
    return " ".join(["hello"] * max(1, input_len))


def one_request(base_url, model, prompt, output_len, api_key, timeout):
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": output_len,
        "temperature": 0.0, "stream": True,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + api_key},
    )
    t0 = time.perf_counter()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            txt = (chunk.get("choices") or [{}])[0].get("text", "")
            if txt:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
    e2e = time.perf_counter() - t0
    return {"ttft": ttft if ttft is not None else e2e, "e2e": e2e, "out_tokens": n}


def synth_request(output_len):
    ttft = random.uniform(0.02, 0.08)
    tpot = random.uniform(0.008, 0.02)
    n = output_len
    return {"ttft": ttft, "e2e": ttft + tpot * max(0, n - 1), "out_tokens": n}


def aggregate(records, wall):
    ttfts = [r["ttft"] for r in records]
    e2es = [r["e2e"] for r in records]
    tpots = [(r["e2e"] - r["ttft"]) / (r["out_tokens"] - 1)
             for r in records if r["out_tokens"] > 1]
    out_tokens = sum(r["out_tokens"] for r in records)
    return {
        "requests": len(records),
        "wall_s": wall,
        "throughput_tokens_per_s": out_tokens / wall if wall else 0.0,
        "throughput_req_per_s": len(records) / wall if wall else 0.0,
        "ttft_ms": {"p50": _ms(pctl(ttfts, 50)), "p95": _ms(pctl(ttfts, 95)),
                    "p99": _ms(pctl(ttfts, 99))},
        "tpot_ms": {"p50": _ms(pctl(tpots, 50)), "p95": _ms(pctl(tpots, 95)),
                    "p99": _ms(pctl(tpots, 99))},
        "e2e_ms": {"p50": _ms(pctl(e2es, 50)), "p95": _ms(pctl(e2es, 95)),
                   "p99": _ms(pctl(e2es, 99))},
    }


def _ms(x):
    return None if x is None else round(x * 1000.0, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=600.0)
    # metadata (recorded, not used for the request itself)
    ap.add_argument("--engine", default="vllm")
    ap.add_argument("--parallel", default="tp8", help="e.g. tp8 / ep8 / tp2ep4")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--gpu", default="a100-80-sxm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    prompt = make_prompt(args.input_len)
    t0 = time.perf_counter()
    records = []
    if args.dry_run:
        records = [synth_request(args.output_len) for _ in range(args.num_prompts)]
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(one_request, args.base_url, args.model, prompt,
                              args.output_len, args.api_key, args.timeout)
                    for _ in range(args.num_prompts)]
            for f in as_completed(futs):
                records.append(f.result())
    wall = time.perf_counter() - t0

    out = {
        "config": {
            "engine": args.engine, "model": args.model, "parallel": args.parallel,
            "dtype": args.dtype, "gpu": args.gpu, "input_len": args.input_len,
            "output_len": args.output_len, "concurrency": args.concurrency,
            "num_prompts": args.num_prompts, "dry_run": args.dry_run,
        },
        "metrics": aggregate(records, wall),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    m = out["metrics"]
    print(f"[bench] {args.model} {args.parallel} {args.dtype} "
          f"in{args.input_len}/out{args.output_len} c{args.concurrency}: "
          f"{m['throughput_tokens_per_s']:.0f} tok/s, "
          f"TPOT p50/p95={m['tpot_ms']['p50']}/{m['tpot_ms']['p95']} ms -> {args.out}")


if __name__ == "__main__":
    main()
