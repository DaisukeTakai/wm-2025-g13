# app/ AGENTS

Training entrypoints + app implementations.

## Overview
- CLI entrypoints: `app/main.py` (local) and `app/main_distributed.py` (submitit/SLURM).
- Dynamic dispatch: `app/scaffold.py` imports `app.<app>.train`.

## Where to look
| Task | Location | Notes |
|------|----------|-------|
| Local run | `main.py` | expands `folder` env vars early |
| SLURM run | `main_distributed.py` | copies repo to `<folder>/code/` and `chdir`s |
| App plugin loader | `scaffold.py` | `importlib.import_module(f"app.{app}.train")` |
| World-model app | `vjepa_wm/` | see `vjepa_wm/AGENTS.md` |
| Shared planning code | `plan_common/` | used by both training and evals |

## Conventions
- Configs are plain YAML loaded via `yaml` + `src/utils/yaml_utils.expand_env_vars`.
- Run folders contain `params-pretrain.yaml` and (distributed) `git-info.txt`.

## Anti-patterns
- Don’t touch `app/*/local/` unless user asked (ignored by git).
