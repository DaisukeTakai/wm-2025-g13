#!/usr/bin/env bash
set -euo pipefail

# Run evals in headless MuJoCo environments.
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

python -u -m evals.main --fname "$1" --debug
