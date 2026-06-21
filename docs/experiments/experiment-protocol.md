# Experiment Protocol

## Experiment ID

Use:

```text
YYYYMMDD-short-topic-scale-owner
```

Example:

```text
20260621-moe-rec-scaling-pilot-gs
```

## Required Files

Each experiment folder must include:

- `README.md`: hypothesis, owner, status, and result summary.
- `config.yaml`: model/data/training/infra config or a pointer to the exact
  config in `configs/`.
- `run.md`: exact commands, git commit, environment, and remote machine alias.
- `artifacts.yaml`: artifact locations and checksums where available.
- `results.md`: final metrics, plots, interpretation, and follow-up decisions.

## Required Metadata

- owner
- date started
- git commit
- data snapshot
- model family
- dense or MoE
- total parameters
- activated parameters
- number of experts
- active experts
- train FLOPs or GPU-hours
- offline metrics
- serving metrics if available
- known caveats

## Review Checklist

- Is the hypothesis falsifiable?
- Are data and feature snapshots fixed?
- Is the baseline strong enough?
- Can another teammate rerun the experiment from the record?
- Are large artifacts referenced without being committed?
- Are negative or inconclusive results documented?
