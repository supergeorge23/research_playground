# GitHub Sync Plan

## Recommended Remote

Suggested repository:

```text
supergeorge23/moe-rec-scaling-law
```

Suggested visibility: private until papers, code, and internal experiment notes
are sanitized.

## Current Local State

- Local path: `/Users/supergeorge/Documents/AI_Research`
- Branch: `main`
- Initial commit exists.

## Create Remote Repository

Current Codex environment can access the GitHub connector for existing
repositories, but it does not expose a create-repository operation. The local
machine also does not currently have the `gh` CLI available.

Create the empty GitHub repository through the GitHub UI, then run:

```bash
cd /Users/supergeorge/Documents/AI_Research
git remote add origin https://github.com/supergeorge23/moe-rec-scaling-law.git
git push -u origin main
```

If `gh` is installed later:

```bash
cd /Users/supergeorge/Documents/AI_Research
gh repo create supergeorge23/moe-rec-scaling-law --private --source=. --remote=origin --push
```

## Branch and Protection Policy

After the remote exists:

- keep `main` protected;
- require pull requests for substantial code and experiment changes;
- allow direct commits only for initial scaffolding and urgent doc fixes;
- tag major internal reports and paper snapshots.

## Suggested Labels

- `okr`
- `experiment`
- `literature`
- `infra`
- `paper`
- `artifact`
- `security`
