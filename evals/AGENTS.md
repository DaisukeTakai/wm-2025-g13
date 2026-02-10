# evals/ AGENTS

Evaluation entrypoints + suites.

## Overview
- CLI entrypoints: `evals/main.py` (local) and `evals/main_distributed.py` (submitit/SLURM).
- Dynamic dispatch: `evals/scaffold.py` imports `evals.<name>.eval` (or `app.<name>.eval` if prefixed).

## Where to look
| Task | Location | Notes |
|------|----------|-------|
| Local eval runner | `main.py` | argparse; multiprocess |
| SLURM eval runner | `main_distributed.py` | submitit; optional `--copy_code` |
| Eval plugin loader | `scaffold.py` | `eval_name` → module import |
| Shared eval utils | `utils.py` | dataset init + media logging |
| Planning eval suite | `simu_env_planning/` | see `simu_env_planning/AGENTS.md` |
| Counterfactual unroll/decode | `unroll_decode/` | DROID/franka-only assumptions |

## Conventions / gotchas
- Eval configs are YAML; often point at a trained run via `folder: ${JEPAWM_LOGS}/...`.
- `evals.utils.make_datasets()` builds transforms + uses `get_dataset_paths()`.

## Anti-patterns
- Don’t hand-edit generated configs under `configs/dump_online_evals/**`.
