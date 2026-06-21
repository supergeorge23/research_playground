# Contributing

## Before Starting Work

1. Check `docs/okr/2026-h2-okr.md` for the relevant objective.
2. Check `registry/experiments.yaml` before creating a new experiment ID.
3. Create a focused branch for code or research-document changes.
4. Run `python3 scripts/validate_repo.py` before opening a pull request.

## Commit Expectations

- Keep commits scoped: one experiment record, one doc update, or one code change
  family at a time.
- Include experiment IDs in commit messages when applicable.
- Never commit secrets, raw private data, or large checkpoints.
- Put substantial repo, research, or infra decisions in `docs/decisions/`.
- Keep public-paper drafts and internal reports in `docs/papers/`; keep raw
  experiment evidence linked from experiment folders and artifact manifests.

## Experiment Review

An experiment is reviewable only when it has:

- config;
- run log;
- artifact manifest;
- final or interim result note;
- known caveats.
