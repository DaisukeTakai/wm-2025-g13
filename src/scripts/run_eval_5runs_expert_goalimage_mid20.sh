#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jepa-wms
source ../init.sh

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cfgs=(
  "configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_expert_goalimage_eval20_mid_run0.yaml"
  "configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_expert_goalimage_eval20_mid_run1.yaml"
  "configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_expert_goalimage_eval20_mid_run2.yaml"
  "configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_expert_goalimage_eval20_mid_run3.yaml"
  "configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_expert_goalimage_eval20_mid_run4.yaml"
)

failed=0
for cfg in "${cfgs[@]}"; do
  name=$(basename "$cfg" .yaml)
  out_dir="$JEPAWM_LOGS/goalhead_eval_expert_goalimage/runs/$name"
  mkdir -p "$out_dir"
  echo "=== $name: eval ==="
  set +e
  python -u -m evals.main --fname "$cfg" --debug | tee "$out_dir/eval.log"
  rc=${PIPESTATUS[0]}
  set -e
  echo "$rc" > "$out_dir/eval_exit_code.txt"
  if [ "$rc" -ne 0 ]; then
    echo "=== $name: eval failed (rc=$rc) ===" >&2
    failed=1
  fi
done

exit "$failed"
