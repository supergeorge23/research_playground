# Literature Reading List

This file tracks papers that are directly relevant to MoE recommendation
scaling laws and the systems work needed to make them practical.

## High-Priority Papers

| Paper | Why it matters | Link |
| --- | --- | --- |
| Understanding Scaling Laws for Recommendation Models | Establishes empirical scaling laws for DLRM-style CTR recommendation models across data, parameters, and compute. Useful dense-rec baseline. | https://arxiv.org/abs/2208.08489 |
| Scaling Law of Large Sequential Recommendation Models | Shows scaling-law behavior for large sequential recommenders up to 0.8B parameters and discusses sparse/data-constrained recommendation settings. | https://arxiv.org/abs/2311.11351 |
| Realizing Scaling Laws in Recommender Systems: A Foundation-Expert Paradigm for Hyperscale Model Deployment | Production-scale Meta system paper connecting scaling laws, expert/foundation paradigm, deployment, and infra constraints. | https://arxiv.org/abs/2508.02929 |
| Towards a Comprehensive Scaling Law of Mixture-of-Experts | MoE-specific scaling-law work that decomposes data size, total params, activated params, active experts, and shared expert ratio. Useful methodology reference. | https://arxiv.org/abs/2509.23678 |
| FoldMoE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining | Systems reference for MoE training bottlenecks, attention-MoE pipelining, and all-to-all communication overlap. | https://aclanthology.org/2025.acl-long.186/ |
| Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism | Systems reference for MoE expert imbalance and sparse collectives. | https://arxiv.org/abs/2502.02581 |

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
