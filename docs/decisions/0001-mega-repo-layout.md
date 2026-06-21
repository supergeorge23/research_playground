# Decision 0001: Use a Single Research Mega Repo

Date: 2026-06-21

Owner: Zhengqi Wang

## Context

The project needs a shared workspace for multiple research directions, including
but not limited to MoE recommendation scaling laws. The repo must cover OKRs,
literature, code, experiment records, artifacts, and future remote GPU
execution.

## Decision

Use `supergeorge23/research_playground` as the single source of truth for the
multi-person research program. Keep human-readable plans under `docs/`,
reusable code under `src/`, experiment instances under `experiments/`,
lightweight artifact pointers under `artifacts/`, and machine-readable indexes
under `registry/`. Treat MoE recommendation scaling laws as the first active
research track, not as the repository identity.

## Consequences

- Research claims, code versions, and experiment evidence can be reviewed
  together.
- Large private outputs stay outside Git and are referenced by manifests.
- Future collaborators can find work by OKR, experiment ID, paper topic, or
  remote-run alias.
