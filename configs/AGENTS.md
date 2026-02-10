# configs/ AGENTS

YAML configs: training runs, eval templates, example eval runs.

## Overview
Configs are organized by role. Some directories are generated and ignored by git.

## Where to look
| Task | Location | Notes |
|------|----------|-------|
| Training configs | `configs/vjepa_wm/` | `folder:` usually `${JEPAWM_LOGS}/...` |
| Eval templates | `configs/online_plan_evals/` | inputs merged during training |
| Example eval configs | `configs/evals/simu_env_planning/` | runnable examples |
| Generated eval configs | `configs/dump_online_evals/` | **generated; ignored by git** |
| Sweep scratch | `configs/cwtemp/` | **generated; ignored by git** |

## Conventions
- `${JEPAWM_LOGS}`: run output root (`folder:`)
- `${JEPAWM_CKPT}`: checkpoint root (`checkpoint_folder:`)
- `${JEPAWM_DSET}`: dataset root (indirect via code path registry)
- `${JEPAWM_OSSCKPT}`: pretrained encoder weights root

## Anti-patterns
- Don’t commit or hand-edit `configs/cwtemp/**` or `configs/dump_online_evals/**`.
