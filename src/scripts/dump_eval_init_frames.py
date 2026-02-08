#!/usr/bin/env python3

"""Dump init frames for planning eval episodes (Metaworld) + write a manifest.

We use the same ep_seed formula as PlanEvaluator.eval():
  ep_seed = (local_seed^2 + ep * local_seed) % (2^32 - 2)

This is intended to support offline language generation from init images, without
leaking goal images.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.planning.utils import make_td


def _save_jpg(path: Path, img_hwc_uint8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(img_hwc_uint8).save(
            path, format="JPEG", quality=95, subsampling=0
        )
    except Exception as e:
        raise RuntimeError("Pillow required to save JPG") from e


def _to_hwc_uint8(visual: torch.Tensor, obs_concat_channels: bool) -> np.ndarray:
    if obs_concat_channels:
        frame = visual[-3:]
    else:
        frame = visual[-1]
    frame = frame.detach().cpu()
    if frame.dtype != torch.uint8:
        frame = frame.clamp(0, 255).to(torch.uint8)
    return frame.permute(1, 2, 0).numpy()


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump init frames for eval episodes")
    ap.add_argument("--eval-yaml", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Headless defaults
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    cfg_path = Path(args.eval_yaml)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "init_images"
    manifest_path = out_dir / "manifest.jsonl"
    img_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        if manifest_path.exists():
            manifest_path.unlink()

    params = yaml.load(cfg_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    # Expand a few env vars commonly used in configs.
    for k in ("folder", "checkpoint_folder"):
        if k in params and isinstance(params[k], str):
            params[k] = os.path.expandvars(params[k])
    cfg = type("Cfg", (), {})()
    # Use OmegaConf-like dict access for make_env by passing raw dict.
    # The env maker expects an OmegaConf, but in practice it uses attribute access.
    # We'll re-load via omegaconf if present.
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(params)
    except Exception:
        raise RuntimeError("omegaconf is required to run dump_eval_init_frames")

    local_seed = int(cfg.meta.seed)
    eval_episodes = int(cfg.meta.eval_episodes)
    tasks = (
        list(cfg.tasks) if hasattr(cfg, "tasks") else [str(cfg.task_specification.task)]
    )
    obs_concat_channels = bool(cfg.task_specification.obs_concat_channels)

    env = make_env(cfg)

    for task_idx, task_name in enumerate(tasks):
        for ep in range(eval_episodes):
            ep_seed = (local_seed * local_seed + ep * local_seed) % (2**32 - 2)
            # Ensure correct task
            obs, info = env.reset(seed=int(ep_seed), task_idx=int(task_idx))
            init_obs, info = env.reset_warmup(seed=int(ep_seed))
            td = make_td(init_obs, info)
            img = _to_hwc_uint8(td["visual"], obs_concat_channels=obs_concat_channels)
            fname = f"{task_name.replace('-', '_')}_{int(ep_seed)}.jpg"
            out_path = img_dir / fname
            if out_path.exists() and not args.overwrite:
                continue
            _save_jpg(out_path, img)
            _append_jsonl(
                manifest_path,
                {
                    "task": str(task_name),
                    "ep": int(ep),
                    "ep_seed": int(ep_seed),
                    "init_image_path": str(out_path.resolve()),
                },
            )


if __name__ == "__main__":
    main()
