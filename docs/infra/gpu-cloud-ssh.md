# GPU Cloud and SSH Notes

This document defines how remote compute should be recorded once GPU cloud or
online servers become available.

## Security Rules

- Do not commit SSH private keys, cloud tokens, API keys, internal hostnames, or
  private dataset paths.
- Use redacted aliases such as `prod-gpu-a`, `cloud-h100-01`, or
  `research-a10-pool`.
- Store actual credentials in the approved secret manager or local encrypted
  keychain, not in this repository.
- Record enough environment metadata for reproducibility without exposing
  infrastructure details.

## Remote Run Record

Each remote run should record:

- remote alias from `registry/remotes.yaml`;
- git commit;
- command;
- container image or conda environment;
- GPU type and count;
- driver/CUDA/NCCL/PyTorch versions;
- data snapshot alias;
- artifact location;
- failure notes if the run did not complete.

## Future Bootstrap Plan

After the first remote target is available, add:

- `scripts/bootstrap_remote.sh`;
- base Dockerfile or environment file;
- remote smoke test;
- artifact upload/download policy.
