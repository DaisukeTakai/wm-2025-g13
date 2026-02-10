# src/ AGENTS

Core library layer: models, datasets, utilities.

## Overview
`src/` should stay importable without pulling in `app/` or `evals/`.

## Where to look
| Task | Location | Notes |
|------|----------|-------|
| Logging / metrics | `utils/logging.py` | high fan-in |
| YAML + env vars | `utils/yaml_utils.py` | expands `${VAR}` |
| Distributed init | `utils/distributed.py` | SLURM-aware |
| Dataset path registry | `utils/cluster.py` | uses `JEPAWM_DSET` |
| Video dataset + loaders | `datasets/` | generic dataset plumbing |
| ViT architectures | `models/` | core model blocks |

## Conventions
- Prefer small, composable utilities; `src` is the foundation layer.

## Anti-patterns
- Don’t import `app.*` or `evals.*` from `src.*`.
