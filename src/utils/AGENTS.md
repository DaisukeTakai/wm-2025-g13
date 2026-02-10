# src/utils AGENTS

Shared utilities used across the entire repo.

## Overview
High fan-in modules; changes here have wide blast radius.

## Where to look
| Task | Location | Notes |
|------|----------|-------|
| Logging | `logging.py` | `get_logger`, CSV logging, git info |
| Config env vars | `yaml_utils.py` | `${VAR}` expansion with warnings |
| Distributed | `distributed.py` | sets TMPDIR on SLURM; nccl backend |
| Cluster integration | `cluster.py` | dataset paths + SLURM qos helpers |
| Optimizer | `adamw.py` | custom AdamW variant |

## Conventions
- Logging format is centralized in `logging.py`.
- YAML is loaded with ruamel.yaml (order/comments preserved).

## Anti-patterns
- Avoid `print()` in shared utilities; use `get_logger`.
