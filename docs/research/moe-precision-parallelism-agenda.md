# Research Agenda: Utilization-Aware Precision × Parallelism Co-Design for MoE Inference

## Starting Observation

Two independent lines optimize MoE inference but leave the boundary between them
fixed: quantization work assumes expert GEMMs are balanced and deployable, while
expert-parallel (EP) scheduling work ignores numeric precision. In a real EP
deployment the expert computation *is* a skewed, low-precision grouped GEMM, so
the EP load distribution determines whether low precision actually delivers
speedup, and precision is an unused knob for balancing the synchronous-layer
straggler.

## Central Research Question

On real multi-GPU MoE inference, how should we *jointly* choose per-expert
precision and placement to maximize realized (roofline-bound) throughput under an
accuracy and latency budget — and what kernel makes heterogeneous-precision
grouped GEMM hit tensor-core peak in the small/skewed-token regime?

## Candidate Claims

1. Whether EP inference improves throughput/latency is regime-dependent (batch,
   sequence length, #experts, top-k, interconnect); a roofline cost model can
   predict it.
2. Low precision's realized benefit is governed by GEMM regime: compute-bound
   (hot) experts gain from reduced compute; memory-bound (cold) experts gain only
   from reduced weight traffic. Allocating precision by error alone (ignoring
   regime) wastes the low-precision budget. On the available A100 the on-device
   lever is INT8 (W8A8); FP8 is the H100 target.
3. Per-expert quantization sensitivity is not monotone in activation frequency
   (cf. MoPEQ), so precision and load are two independent axes that must be
   optimized jointly, not collapsed into a "hot -> high precision" rule.
4. Precision can serve as a load-balancing dimension: downcasting the overloaded,
   non-sensitive experts equalizes per-GPU finish times and cuts the synchronous
   EP straggler without moving tokens or experts.

## Experiment Axes

| Axis | Examples |
| --- | --- |
| Model | Mixtral-8x7B, Qwen-MoE, DeepSeek-MoE (coarse vs fine-grained experts) |
| Workload | prefill vs decode, context length, batch size, routing skew |
| Parallelism | EP degree, TP/DP/EP mix, hot-expert replication |
| Precision | BF16 (baseline) / INT8 W8A8 (A100 lever) / FP8 E4M3 (H100) / FP4 (Blackwell), scaling granularity |
| Hardware | single-GPU, NVLink, PCIe, cross-node RDMA |
| Per-expert signals | token count, GEMM roofline regime, quantization sensitivity |
| Metrics | decode tok/s, p50/p95, tensor-core utilization, straggler gap, accuracy delta |

> Hardware note: "low precision" = **INT8 (W8A8)** on the available A100 (no FP8
> tensor cores; real ~2x via INT8 tensor cores + 1-byte weights) and **FP8** on
> H100 (reported by the roofline as a projection). BF16/FP16 are the baseline, not
> a precision lever. Software gap (and Paper 1 motivation): current serving stacks
> do NOT expose W8A8-INT8 for MoE on A100 (vLLM W8A8 is Hopper/Ada-only; Marlin-MoE
> has A100 bugs, vLLM #35922). So on A100, low-precision *serving* is best-effort
> via W4A16 GPTQ, while the precision signal comes from the (fake-quant)
> sensitivity pass — closing that serving gap is exactly Paper 1.

## Paper Sequence

### Paper 0 — Measurement (low risk, de-risks the rest)

Characterize when EP inference wins and the roofline regimes of MoE GEMMs across
interconnects and model shapes. Decompose EP's effect into memory-enabled batch,
all-to-all cost, and load skew. Empirically test whether quantization sensitivity
correlates with expert hotness. Output: a reproducible methodology + open
benchmark + the claim that utilization is the missing axis. Cost: 1-2 on-demand
GPUs. First experiment: `20260622-moe-ep-fp8-roofline-pilot`.

### Paper 1 — Kernel mechanism

Utilization-aware low precision (INT8 W8A8 on A100, FP8 on H100): a
heterogeneous-precision grouped GEMM that realizes speedup when per-expert token
counts are small and skewed (padding elimination, scaling-granularity vs
tensor-core-efficiency trade-off). Builds directly on the Paper 0 roofline
characterization.

### Paper 2 — System

Roofline-driven joint precision-and-placement scheduler for EP serving: precision
as a load-balancing dimension to cut the synchronous straggler, sensitivity-aware
and utilization-aware, co-optimized with hot-expert replication/placement.

## Baselines / Comparison

- EP scheduling / load balancing: MoETuner, EPS-MoE, PROBE, AEP, ReaLB.
- MoE mixed-precision: MoPEQ, MxMoE, MC-MoE, HOBBIT, ScaleBITS, Dynamic Expert
  Quantization.
- Kernels: INT8 / W4A16 Marlin grouped GEMM (A100); DeepGEMM / CUTLASS / TMA-Adaptive FP8 grouped GEMM (H100).
- Serving stacks: vLLM / SGLang FP8 MoE paths.

## Risks

- Hot, fast-moving area with strong industrial players; must claim the precise
  niche (utilization/roofline axis + distributed EP realization + a real kernel),
  move fast, and not chase the broad "mixed-precision experts" headline.
- Heterogeneous-precision grouped GEMM may need a custom kernel; multi-precision
  weight copies cost memory — design around this.
- Industrial low-precision recipes carry IP: reproduce only on open models; commit
  no internal data, weights, or hostnames.

## Open Questions

- Is quantization sensitivity correlated, uncorrelated, or anti-correlated with
  expert hotness?
- What is the cheapest online policy for re-allocating precision under routing
  drift at serving latency?
- Where does precision-based balancing beat token/expert movement, and where do
  they compose?
- How much of the win survives on PCIe / cross-node vs NVLink?
