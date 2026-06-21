# Research Playground

This repository is the single source of truth for research, code, experiments,
artifacts, literature, OKRs, and infrastructure notes for multi-person AI
research collaboration.

## Scope

This is a general research workspace. Individual research directions live as
tracks under `docs/research/`, `experiments/`, `registry/`, and `docs/papers/`.

Current seeded tracks:

- MoE recommendation scaling laws;
- LLM4Rec training and recommendation infrastructure;
- MoE model infrastructure;
- FlashAttention and attention-kernel industrialization;
- future AI infra and applied research tracks added through OKR or decision
  records.

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
5. Literature notes should capture why a paper matters to a concrete research
   track, not just summarize the abstract.

## Current Priority

See `docs/okr/2026-h2-okr.md` for repo-level execution. The first active
research track is documented in `docs/research/moe-rec-scaling-law-agenda.md`.

## Quick Commands

Create a tracked experiment scaffold:

```bash
scripts/new_experiment.sh YYYYMMDD-short-topic-scale-owner
```

Validate the repository structure before pushing:

```bash
python3 scripts/validate_repo.py
```
