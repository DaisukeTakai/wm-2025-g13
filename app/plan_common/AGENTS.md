# app/plan_common AGENTS

Shared domain library used by both training and evaluation.

## Overview
Provides dataset wrappers + preprocessing/transforms + model components (heads/decoders) + plotting utilities.

## Where to look
| Task | Location | Notes |
|------|----------|-------|
| Dataset init glue | `datasets/utils.py` | bridges into `src.datasets.*` |
| Normalization | `datasets/preprocessor.py` | action/state/proprio stats |
| Transforms | `datasets/transforms.py` | builds (inverse) transforms |
| Model heads | `models/wm_heads.py` | image/state/reward heads |
| DINO encoder integration | `models/dino.py` | may require external weights/repos |
| Planning plots | `plot/` | scripts; heavy filesystem assumptions |

## Conventions
- Observations commonly use keys: `visual` and optionally `proprio`.
- `plan_common` is “shared”; keep it importable from evals.

## Anti-patterns
- Don’t hardcode absolute paths; use `${JEPAWM_*}` + `expand_env_vars` where possible.
