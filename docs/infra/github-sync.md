# GitHub Sync Plan

## Recommended Remote

Repository:

```text
supergeorge23/research_playground
```

Suggested visibility: private until papers, code, and internal experiment notes
are sanitized.

## Current Local State

- Local path: `/Users/supergeorge/Documents/AI_Research`
- Branch: `main`
- Initial commit exists.
- GitHub remote target: `https://github.com/supergeorge23/research_playground.git`

## Sync Remote Repository

The GitHub repository already exists. Connect and push the local repo with:

```bash
cd /Users/supergeorge/Documents/AI_Research
git remote add origin https://github.com/supergeorge23/research_playground.git
git push -u origin main
```

If `origin` already exists, verify it points to the same repository:

```bash
cd /Users/supergeorge/Documents/AI_Research
git remote -v
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
