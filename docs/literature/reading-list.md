# Literature Reading List

This file tracks papers that are directly relevant to active research tracks.
The initial seed focuses on MoE recommendation scaling laws and the systems work
needed to make them practical.

## High-Priority Papers

| Paper | Why it matters | Link |
| --- | --- | --- |
| Understanding Scaling Laws for Recommendation Models | Establishes empirical scaling laws for DLRM-style CTR recommendation models across data, parameters, and compute. Useful dense-rec baseline. | https://arxiv.org/abs/2208.08489 |
| Scaling Law of Large Sequential Recommendation Models | Shows scaling-law behavior for large sequential recommenders up to 0.8B parameters and discusses sparse/data-constrained recommendation settings. | https://arxiv.org/abs/2311.11351 |
| Realizing Scaling Laws in Recommender Systems: A Foundation-Expert Paradigm for Hyperscale Model Deployment | Production-scale Meta system paper connecting scaling laws, expert/foundation paradigm, deployment, and infra constraints. | https://arxiv.org/abs/2508.02929 |
| Towards a Comprehensive Scaling Law of Mixture-of-Experts | MoE-specific scaling-law work that decomposes data size, total params, activated params, active experts, and shared expert ratio. Useful methodology reference. | https://arxiv.org/abs/2509.23678 |
| FoldMoE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining | Systems reference for MoE training bottlenecks, attention-MoE pipelining, and all-to-all communication overlap. | https://aclanthology.org/2025.acl-long.186/ |
| Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism | Systems reference for MoE expert imbalance and sparse collectives. | https://arxiv.org/abs/2502.02581 |

## MoE Precision × Parallelism Track (LLM Inference)

These papers ground the second active track
(`docs/research/moe-precision-parallelism-agenda.md`).

| Paper | Why it matters | Link |
| --- | --- | --- |
| MoPEQ: Mixture of Mixed Precision Quantized Experts | Per-expert bit-width by Hessian sensitivity; finds sensitivity matters more than activation frequency — refutes a naive "hot expert -> high precision" rule. | https://arxiv.org/abs/2509.02512 |
| MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design | Closest prior art: jointly navigates parameter sensitivity, expert activation dynamics, and hardware resources. Defines the boundary our track must beat. | https://arxiv.org/abs/2505.05799 |
| ReaLB: Real-Time Load Balancing for Multimodal MoE Inference | Uses low-precision execution on overloaded paths to mitigate the straggler — the "precision as a load-balancing knob" idea, in the multimodal setting. | https://arxiv.org/abs/2604.19503 |
| Capacity-Aware Inference: Mitigating the Straggler Effect in MoE | Frames the synchronous-layer straggler in EP inference; baseline for the straggler objective. | https://arxiv.org/abs/2503.05066 |
| EPS-MoE / PROBE / Asynchronous Expert Parallelism | EP-inference scheduling and compute-communication co-balancing baselines. | https://arxiv.org/abs/2410.12247 |
| HOBBIT: Mixed Precision Expert Offloading for Fast MoE Inference | Precision-by-importance for offloading; reference for the memory-bound expert regime. | https://arxiv.org/abs/2411.01433 |
| TMA-Adaptive FP8 Grouped GEMM | Eliminates padding in low-precision grouped GEMM on Hopper; starting point for the Paper 1 kernel. | https://arxiv.org/abs/2508.16584 |

## Reading Note Template

Create one file per paper under `docs/literature/notes/`.

```markdown
# Paper Title

## Citation

## One-line Relevance

## Key Ideas

## What We Can Reuse

## Gaps / Weaknesses

## Questions for Our Work
```
