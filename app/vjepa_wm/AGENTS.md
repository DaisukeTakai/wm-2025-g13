# app/vjepa_wm AGENTS

Main training application for the JEPA world model.

## Overview
Owns the training loop, model wrapper, and initialization utilities.

## Where to look
| Task | Location | Notes |
|------|----------|-------|
| Training loop | `train.py` | large; handles data/model/opt/logging/evals |
| Model wrapper | `video_wm.py` | encode/predict/unroll/rollout |
| Init helpers | `utils.py` | build model/opt, checkpoint I/O, eval-args builders |
| Planning wrapper module | `modelcustom/simu_env_planning/vit_enc_preds.py` | used by planning/unroll evals via `module_name` |

## Running
- Training is started via `python -m app.main --fname <configs/vjepa_wm/...yaml> --debug`.
- SLURM via `python -m app.main_distributed ...`.

## Conventions / gotchas
- Training may submit eval jobs via `evals.main_distributed.launch_evals_with_parsed_args`.
- `submitit.helpers.clean_env()` is used when launching child jobs (avoid leaking SLURM vars).
- `meta.plan_only_eval_mode` + `evals.dump_eval_configs` dumps runnable eval configs to `configs/dump_online_evals/<env>/`.

## Anti-patterns
- Avoid importing `evals/*` except for orchestration/launch points.
- Keep `src/` imports pure (don’t move app logic into `src`).
