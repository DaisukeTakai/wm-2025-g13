GoalHead (Instruction) Training

This folder contains configs for training a GoalHead that conditions on free-form
English instructions (not task-name tokens).

Quick start:

1) Copy and edit `metaworld_instruction_goalhead.yaml`.
2) Set `wm_checkpoint` to an existing EncPredWM checkpoint.
3) Run:

   python -m app.main --fname configs/goalhead/metaworld_instruction_goalhead.yaml --debug

Notes:
- The dataset is loaded from Hugging Face: `hf_repo: wm-2025-g13/metaworld`.
- Train expands each base sample into `k_variants_train` samples by selecting one
  of the instruction variants (virtual 10x expansion).
- Validation uses a fixed variant index (`val_fixed_variant_idx`) for stability.
