# Decision 0001: Use a Single Research Mega Repo

Date: 2026-06-21

Owner: Zhengqi Wang

## Context

The project is moving from an observed MoE recommendation scaling-law signal to
a multi-stream research effort covering OKRs, literature, code, experiment
records, artifacts, and future remote GPU execution.

## Decision

Use `supergeorge23/research_playground` as the single source of truth for the
research program. Keep human-readable plans under `docs/`, reusable code under
`src/`, experiment instances under `experiments/`, lightweight artifact pointers
under `artifacts/`, and machine-readable indexes under `registry/`.

## Consequences

- Research claims, code versions, and experiment evidence can be reviewed
  together.
- Large private outputs stay outside Git and are referenced by manifests.
- Future collaborators can find work by OKR, experiment ID, paper topic, or
  remote-run alias.
