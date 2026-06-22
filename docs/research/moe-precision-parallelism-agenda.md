# Research Agenda: Utilization-Aware Precision × Parallelism Co-Design for MoE Inference

## Starting Observation

Two independent lines optimize MoE inference but leave the boundary between them
fixed: quantization work assumes expert GEMMs are balanced and deployable, while
expert-parallel (EP) scheduling work ignores numeric precision. In a real EP
deployment the expert computation *is* a skewed, low-precision grouped GEMM, so
the EP load distribution determines whether FP8/FP4 actually delivers speedup,
and precision is an unused knob for balancing the synchronous-layer straggler.

## Central Research Question

On real multi-GPU MoE inference, how should we *jointly* choose per-expert
precision and placement to maximize realized (roofline-bound) throughput under an
accuracy and latency budget — and what kernel makes heterogeneous-precision
grouped GEMM hit tensor-core peak in the small/skewed-token regime?

## Candidate Claims

1. Whether EP inference improves throughput/latency is regime-dependent (batch,
   sequence length, #experts, top-k, interconnect); a roofline cost model can
   predict it.
2. FP8's realized benefit is governed by GEMM regime: compute-bound (hot) experts
   gain from reduced compute; memory-bound (cold) experts gain only from reduced
   weight traffic. Allocating precision by error alone (ignoring regime) wastes
   the FP8 budget.
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
| Precision | BF16 / FP8 (E4M3) / FP4, scaling granularity (per-tensor/tile/group) |
| Hardware | single-GPU, NVLink, PCIe, cross-node RDMA |
| Per-expert signals | token count, GEMM roofline regime, quantization sensitivity |
| Metrics | decode tok/s, p50/p95, tensor-core utilization, straggler gap, accuracy delta |

## Paper Sequence

### Paper 0 — Measurement (low risk, de-risks the rest)

Characterize when EP inference wins and the roofline regimes of MoE GEMMs across
interconnects and model shapes. Decompose EP's effect into memory-enabled batch,
all-to-all cost, and load skew. Empirically test whether quantization sensitivity
correlates with expert hotness. Output: a reproducible methodology + open
benchmark + the claim that utilization is the missing axis. Cost: 1-2 on-demand
GPUs. First experiment: `20260622-moe-ep-fp8-roofline-pilot`.

### Paper 1 — Kernel mechanism

Utilization-aware FP8: a heterogeneous-precision grouped GEMM that realizes
speedup when per-expert token counts are small and skewed (padding elimination,
scaling-granularity vs tensor-core-efficiency trade-off). Builds directly on the
Paper 0 roofline characterization.

### Paper 2 — System

Roofline-driven joint precision-and-placement scheduler for EP serving: precision
as a load-balancing dimension to cut the synchronous straggler, sensitivity-aware
and utilization-aware, co-optimized with hot-expert replication/placement.

## Baselines / Comparison

- EP scheduling / load balancing: MoETuner, EPS-MoE, PROBE, AEP, ReaLB.
- MoE mixed-precision: MoPEQ, MxMoE, MC-MoE, HOBBIT, ScaleBITS, Dynamic Expert
  Quantization.
- Kernels: DeepGEMM / CUTLASS FP8 grouped GEMM, TMA-Adaptive grouped GEMM.
- Serving stacks: vLLM / SGLang FP8 MoE paths.

## Risks

- Hot, fast-moving area with strong industrial players; must claim the precise
  niche (utilization/roofline axis + distributed EP realization + a real kernel),
  move fast, and not chase the broad "mixed-precision experts" headline.
- Heterogeneous-precision grouped GEMM may need a custom kernel; multi-precision
  weight copies cost memory — design around this.
- Industrial FP8 recipes carry IP: reproduce only on open models; commit no
  internal data, weights, or hostnames.

## Open Questions

- Is quantization sensitivity correlated, uncorrelated, or anti-correlated with
  expert hotness?
- What is the cheapest online policy for re-allocating precision under routing
  drift at serving latency?
- Where does precision-based balancing beat token/expert movement, and where do
  they compose?
- How much of the win survives on PCIe / cross-node vs NVLink?
