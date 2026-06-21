#!/usr/bin/env python3
"""Validate the lightweight research mega-repo contract."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "README.md",
    "CONTRIBUTING.md",
    "docs/okr/2026-h2-okr.md",
    "docs/okr/weekly/README.md",
    "docs/research/moe-rec-scaling-law-agenda.md",
    "docs/literature/reading-list.md",
    "docs/experiments/experiment-protocol.md",
    "docs/infra/github-sync.md",
    "docs/infra/gpu-cloud-ssh.md",
    "docs/versioning/artifact-policy.md",
    "docs/decisions/README.md",
    "docs/papers/README.md",
    "registry/experiments.yaml",
    "registry/papers.yaml",
    "registry/models.yaml",
    "registry/datasets.yaml",
    "registry/remotes.yaml",
    "scripts/new_experiment.sh",
]

EXPERIMENT_REQUIRED_FILES = {
    "README.md",
    "config.yaml",
    "run.md",
    "results.md",
    "artifacts.yaml",
}

SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"][^'\"]{12,}['\"]"),
]

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__"}
TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".json", ".bib", ".toml", ".txt", ".sh"}


def fail(message: str) -> None:
    print(f"ERROR: {message}")
    raise SystemExit(1)


def check_required_paths() -> None:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    if missing:
        fail("missing required paths: " + ", ".join(missing))


def check_experiments() -> None:
    experiments_dir = ROOT / "experiments"
    if not experiments_dir.exists():
        fail("missing experiments directory")

    for experiment_dir in sorted(path for path in experiments_dir.iterdir() if path.is_dir()):
        missing = [
            name
            for name in sorted(EXPERIMENT_REQUIRED_FILES)
            if not (experiment_dir / name).exists()
        ]
        if missing:
            fail(f"{experiment_dir.relative_to(ROOT)} missing files: {', '.join(missing)}")

        manifest = ROOT / "artifacts" / "manifests" / f"{experiment_dir.name}.yaml"
        if not manifest.exists():
            fail(f"missing artifact manifest for {experiment_dir.name}: {manifest.relative_to(ROOT)}")


def iter_text_files():
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            yield path


def check_secret_patterns() -> None:
    for path in iter_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                fail(f"possible secret in {path.relative_to(ROOT)}")


def main() -> int:
    check_required_paths()
    check_experiments()
    check_secret_patterns()
    print("repo validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
