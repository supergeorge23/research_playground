#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: scripts/new_experiment.sh YYYYMMDD-short-topic-owner" >&2
  exit 1
fi

experiment_id="$1"
experiment_dir="experiments/${experiment_id}"
manifest_path="artifacts/manifests/${experiment_id}.yaml"

if [[ -e "${experiment_dir}" ]]; then
  echo "experiment already exists: ${experiment_dir}" >&2
  exit 1
fi

mkdir -p "${experiment_dir}" artifacts/manifests
cp docs/experiments/templates/README.md "${experiment_dir}/README.md"
cp docs/experiments/templates/config.yaml "${experiment_dir}/config.yaml"

cat > "${experiment_dir}/run.md" <<RUN
# Run Log

Experiment ID: \`${experiment_id}\`

## Git Commit

TBD

## Environment

TBD

## Commands

\`\`\`bash
# TBD
\`\`\`

## Notes

TBD
RUN

cat > "${experiment_dir}/results.md" <<RESULTS
# Results

Experiment ID: \`${experiment_id}\`

## Metrics

TBD

## Interpretation

TBD

## Decision

TBD
RESULTS

cat > "${experiment_dir}/artifacts.yaml" <<ARTIFACTS
artifact_manifest:
  experiment_id: ${experiment_id}
  status: planned
  owner: TBD
  git_commit: TBD
  external_artifacts: []
ARTIFACTS

cat > "${manifest_path}" <<MANIFEST
artifact_manifest:
  experiment_id: ${experiment_id}
  status: planned
  owner: TBD
  git_commit: TBD
  external_artifacts: []
  notes: Created by scripts/new_experiment.sh.
MANIFEST

echo "created ${experiment_dir}"
