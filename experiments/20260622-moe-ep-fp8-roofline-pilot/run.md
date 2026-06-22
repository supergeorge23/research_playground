# Run Log

Experiment ID: `20260622-moe-ep-fp8-roofline-pilot`

## Git Commit

TBD

## Environment

TBD (container image / conda env; CUDA / NCCL / PyTorch / vLLM or SGLang versions).

## Commands

```bash
# TBD: profiling harness invocation (per model x parallelism x interconnect x precision)
```

## Notes

Inference-only profiling. Record the interconnect (single-GPU / NVLink / PCIe /
RDMA) and the GPU type per run. Keep raw traces outside Git; reference them from
`artifacts.yaml`.
