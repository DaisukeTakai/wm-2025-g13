# evals/simu_env_planning AGENTS

Goal-conditioned planning evaluation suite.

## Overview
Optimizes action sequences to minimize representation distance to goal (plus optional decoding/metrics).

## Where to look
| Task | Location | Notes |
|------|----------|-------|
| Suite entrypoint | `eval.py` | distributed episodes + model init |
| Environments | `envs/` | wrappers + per-env adapters |
| Planners | `planning/planning/planner.py` | CEM/NeverGrad/GD/Adam variants |
| Objectives | `planning/planning/objectives.py` | L1/L2 etc |
| Episode definition | `planning/plan_evaluator.py` | goal sources: dset/expert/random_state |
| Grid config generator | `run_eval_grid.py` | writes `configs/cwtemp/` + `batch.yaml` |

## Commands
```bash
# single GPU
python -m evals.main --fname <eval_config.yaml> --debug

# SLURM
python -m evals.main_distributed --fname <eval_config.yaml> --account <account> --qos lowest --time 120

# generate a sweep
python -m evals.simu_env_planning.run_eval_grid --env <env> --config <eval_config.yaml>
```

## Conventions / gotchas
- Default model wrapper for planning: `app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds.EncPredWM`.
- Generated configs:
  - `configs/cwtemp/**` (scratch)
  - `configs/dump_online_evals/**` (from training)

## Anti-patterns
- Avoid adding heavy dependencies into planner core; env adapters are the right place.
