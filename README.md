# MoE Rec Scaling Law Mega Repo

This repository is the single source of truth for research, code, experiments,
artifacts, literature, OKRs, and infrastructure notes around scaling laws for
Mixture-of-Experts (MoE) in recommender systems.

## Core Thesis

We have observed that MoE models in recommender systems exhibit scaling-law
behavior. The next research phase is to turn this observation into a rigorous,
reproducible, and publishable body of work:

- quantify how recommendation quality scales with data, total parameters,
  activated parameters, experts, compute, and serving constraints;
- explain where MoE changes the scaling frontier compared with dense
  recommendation models;
- identify the training and serving infrastructure bottlenecks that appear when
  this scaling law is pushed toward production workloads;
- maintain all code, experiment records, literature, and infra state in one
  versioned mega repo.

## Repository Map

| Path | Purpose |
| --- | --- |
| `docs/okr/` | Team OKRs, weekly execution state, milestones, and decision records. |
| `docs/research/` | Research agenda, hypotheses, paper outlines, and technical notes. |
| `docs/literature/` | Reading list, paper summaries, BibTeX, and open questions. |
| `docs/papers/` | Internal reports, paper drafts, rebuttal notes, and sanitized release plans. |
| `docs/experiments/` | Experiment protocol, naming rules, templates, and review checklist. |
| `docs/infra/` | GPU cloud, SSH, remote machines, security, and environment notes. |
| `docs/decisions/` | Lightweight decision records for repo, research, and infra choices. |
| `src/` | Production-quality research code and reusable libraries. |
| `configs/` | Versioned experiment, model, data, and infra configs. |
| `experiments/` | One folder per tracked experiment run or experiment family. |
| `artifacts/` | Lightweight artifact manifests. Large binaries stay outside Git. |
| `registry/` | Machine-readable registries for experiments, papers, models, datasets, and remotes. |
| `scripts/` | Repo automation scripts. |
| `.github/` | Pull request, issue, and validation workflow templates for collaboration. |

## Operating Rules

1. Every research direction must map to an OKR objective or a documented
   exploratory decision.
2. Every experiment must have a stable ID, config, owner, hypothesis, metrics,
   artifact manifest, and final result note.
3. Large files do not go into Git directly. Store manifests and pointers in
   `artifacts/`; use Git LFS, object storage, or DVC only after the storage
   policy is decided.
4. SSH hostnames, cloud accounts, secrets, private data paths, and tokens must
   not be committed. Use redacted aliases in `docs/infra/` and `registry/`.
5. Literature notes should capture why a paper matters to our scaling-law
   thesis, not just summarize the abstract.

## Current Priority

See `docs/okr/2026-h2-okr.md` and
`docs/research/moe-rec-scaling-law-agenda.md`.

## Quick Commands

Create a tracked experiment scaffold:

```bash
scripts/new_experiment.sh YYYYMMDD-short-topic-scale-owner
```

Validate the repository structure before pushing:

```bash
python3 scripts/validate_repo.py
```
