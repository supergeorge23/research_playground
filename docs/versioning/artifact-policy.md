# Versioning and Artifact Policy

## What Goes in Git

- source code;
- small configs;
- experiment manifests;
- paper notes;
- plots small enough for review;
- redacted logs and summarized metrics;
- scripts needed to reproduce results.

## What Does Not Go in Git

- raw datasets;
- checkpoints;
- large tensorboard or wandb directories;
- unredacted production logs;
- SSH keys or cloud credentials;
- private user data;
- vendor or internal hostnames.

## Large Artifact Options

Choose one before large-scale experiments begin:

1. Git LFS for small-to-medium binary artifacts that must be reviewed with code.
2. Object storage for checkpoints, datasets, and large logs.
3. DVC if data lineage becomes complex enough to justify an extra tool.

Until then, store only artifact manifests in this repository.
