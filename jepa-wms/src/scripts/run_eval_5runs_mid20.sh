#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jepa-wms
source ../init.sh

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Azure env for instruction generation
set -a
source ../.env
set +a
export AZURE_OPENAI_DEPLOYMENT=${AZURE_OPENAI_DEPLOYMENT:-gpt-5.2}
export AZURE_OPENAI_API_VERSION=${AZURE_OPENAI_API_VERSION:-2024-06-01}

runs=(
  "run0_seed1000003|configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_language_instructionfile_eval20_mid_run0.yaml"
  "run1_seed1010003|configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_language_instructionfile_eval20_mid_run1.yaml"
  "run2_seed1020003|configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_language_instructionfile_eval20_mid_run2.yaml"
  "run3_seed1030003|configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_language_instructionfile_eval20_mid_run3.yaml"
  "run4_seed1040003|configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_language_instructionfile_eval20_mid_run4.yaml"
)

failed=0
for r in "${runs[@]}"; do
  name=${r%%|*}
  cfg=${r#*|}
  out_dir="$JEPAWM_LOGS/goalhead_eval_instruction/runs/$name"
  mkdir -p "$out_dir"

  if [ ! -f "$out_dir/manifest.jsonl" ] || [ ! -d "$out_dir/init_images" ]; then
    echo "=== $name: dump init frames ==="
    python -u src/scripts/dump_eval_init_frames.py \
      --eval-yaml "$cfg" \
      --out "$out_dir" \
      --overwrite
  else
    echo "=== $name: dump init frames (skip; already present) ==="
  fi

  if [ ! -f "$out_dir/instructions.jsonl" ]; then
    echo "=== $name: generate instructions ==="
    python -u src/scripts/generate_eval_instructions_azure.py \
      --manifest "$out_dir/manifest.jsonl" \
      --out "$out_dir/instructions.jsonl" \
      --cache "$out_dir/instructions_cache.jsonl" \
      --concurrency 8 \
      --log-every 10
  else
    echo "=== $name: generate instructions (skip; already present) ==="
  fi

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
