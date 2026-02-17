#!/usr/bin/env python3

"""Collect (init, goal) image pairs for Metaworld tasks using expert rollouts.

This script generates dataset samples for tasks like:
  - mw-reach
  - mw-reach-wall

For each episode:
  1) reset_warmup(seed=...) to get an initial observation
  2) run the Metaworld expert policy to episode termination
  3) save initial and final RGB frames (and minimal metadata)

Outputs:
  <out_dir>/images/<id>_init.jpg
  <out_dir>/images/<id>_goal.jpg
  <out_dir>/pairs.jsonl

The resulting JSONL is compatible with:
  src/scripts/annotate_pairs_with_azure_vlm.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _to_hwc_uint8(visual: torch.Tensor, obs_concat_channels: bool) -> np.ndarray:
    """Convert PixelWrapper visual tensor to HWC uint8."""
    if obs_concat_channels:
        # [3*T, H, W] -> last frame [3,H,W]
        if visual.dim() != 3 or visual.shape[0] < 3:
            raise ValueError(f"Unexpected concat visual shape: {tuple(visual.shape)}")
        frame = visual[-3:]
    else:
        # [T, 3, H, W] -> last frame [3,H,W]
        if visual.dim() != 4 or visual.shape[1] != 3:
            raise ValueError(f"Unexpected visual shape: {tuple(visual.shape)}")
        frame = visual[-1]
    frame = frame.detach().cpu()
    if frame.dtype != torch.uint8:
        frame = frame.clamp(0, 255).to(torch.uint8)
    return frame.permute(1, 2, 0).numpy()


def _save_image(path: Path, img_hwc_uint8: np.ndarray) -> None:
    """Save HWC uint8 image as JPEG/PNG (lazy imports)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"Unsupported image extension: {ext}")

    # Prefer PIL if present.
    try:
        from PIL import Image  # type: ignore

        im = Image.fromarray(img_hwc_uint8)
        if ext in {".jpg", ".jpeg"}:
            im.save(path, format="JPEG", quality=95, subsampling=0)
        else:
            im.save(path, format="PNG")
        return
    except Exception:
        pass

    # Fallback to imageio.v3.
    try:
        import imageio.v3 as iio  # type: ignore

        iio.imwrite(path, img_hwc_uint8)
        return
    except Exception:
        pass

    # Fallback to cv2.
    try:
        import cv2  # type: ignore

        bgr = img_hwc_uint8[:, :, ::-1]
        ok = cv2.imwrite(str(path), bgr)
        if not ok:
            raise RuntimeError("cv2.imwrite returned False")
        return
    except Exception as e:
        raise RuntimeError(
            "No image writer available. Install pillow or imageio/opencv-python. "
            f"Failed to save {path}: {e}"
        ) from e


@dataclass(frozen=True)
class CollectConfig:
    out_dir: Path
    tasks: list[str]
    episodes_per_task: int
    base_seed: int
    frameskip: int
    img_size: int
    num_frames: int
    num_proprios: int
    max_episode_steps: int
    obs_concat_channels: bool
    image_ext: str
    camera_fovy: float | None
    start_ep: int
    end_ep: int
    log_every: int


def _build_eval_cfg(task: str, tasks: list[str], cfg: CollectConfig) -> Any:
    # Minimal config needed by env + PlanEvaluator.
    return OmegaConf.create(
        {
            "frameskip": int(cfg.frameskip),
            "meta": {"seed": int(cfg.base_seed)},
            "tasks": tasks,
            "logging": {"tqdm_silent": True, "optional_plots": False},
            "task_specification": {
                "task": task,
                "multitask": False,
                "obs": "rgb_state",
                "obs_concat_channels": bool(cfg.obs_concat_channels),
                "num_frames": int(cfg.num_frames),
                "num_proprios": int(cfg.num_proprios),
                "img_size": int(cfg.img_size),
                "goal_source": "expert",
                "max_episode_steps": int(cfg.max_episode_steps),
                "env": {
                    "with_target": True,
                    "with_velocity": True,
                    "freeze_rand_vec": False,
                },
            },
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect Metaworld init/goal image pairs")
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=["mw-reach", "mw-reach-wall"],
        help="Metaworld task names (e.g. mw-reach mw-reach-wall)",
    )
    ap.add_argument(
        "--episodes-per-task",
        type=int,
        default=100,
        help="Episodes to collect per task",
    )
    ap.add_argument("--base-seed", type=int, default=1)
    ap.add_argument(
        "--frameskip",
        type=int,
        default=5,
        help="Frameskip used by planning/eval code (affects expert steps_left bookkeeping)",
    )
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--num-frames", type=int, default=1)
    ap.add_argument("--num-proprios", type=int, default=1)
    ap.add_argument("--max-episode-steps", type=int, default=100)
    ap.add_argument("--obs-concat-channels", action="store_true")
    ap.add_argument(
        "--image-ext", type=str, default=".jpg", choices=[".jpg", ".jpeg", ".png"]
    )
    ap.add_argument(
        "--camera-fovy",
        type=float,
        default=None,
        help="Optional camera FOV (lower=more zoom) for Metaworld corner2 camera",
    )
    ap.add_argument(
        "--resume", action="store_true", help="Skip already-written ids in pairs.jsonl"
    )
    ap.add_argument(
        "--start-ep",
        type=int,
        default=0,
        help="Start episode index (inclusive) within each task",
    )
    ap.add_argument(
        "--end-ep",
        type=int,
        default=-1,
        help="End episode index (exclusive) within each task; -1 means episodes-per-task",
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print progress every N newly written samples",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    pairs_path = out_dir / "pairs.jsonl"
    meta_path = out_dir / "meta.json"

    end_ep = int(args.end_ep)
    if end_ep < 0:
        end_ep = int(args.episodes_per_task)
    start_ep = int(args.start_ep)
    if start_ep < 0 or end_ep <= start_ep:
        raise ValueError(f"Invalid episode range: start_ep={start_ep} end_ep={end_ep}")

    cfg = CollectConfig(
        out_dir=out_dir,
        tasks=list(args.tasks),
        episodes_per_task=int(args.episodes_per_task),
        base_seed=int(args.base_seed),
        frameskip=int(args.frameskip),
        img_size=int(args.img_size),
        num_frames=int(args.num_frames),
        num_proprios=int(args.num_proprios),
        max_episode_steps=int(args.max_episode_steps),
        obs_concat_channels=bool(args.obs_concat_channels),
        image_ext=str(args.image_ext),
        camera_fovy=float(args.camera_fovy) if args.camera_fovy is not None else None,
        start_ep=start_ep,
        end_ep=end_ep,
        log_every=int(args.log_every),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Save a small metadata file for provenance.
    meta = {
        "created_unix": time.time(),
        "tasks": cfg.tasks,
        "episodes_per_task": cfg.episodes_per_task,
        "base_seed": cfg.base_seed,
        "frameskip": cfg.frameskip,
        "start_ep": cfg.start_ep,
        "end_ep": cfg.end_ep,
        "log_every": cfg.log_every,
        "img_size": cfg.img_size,
        "num_frames": cfg.num_frames,
        "num_proprios": cfg.num_proprios,
        "max_episode_steps": cfg.max_episode_steps,
        "obs_concat_channels": cfg.obs_concat_channels,
        "image_ext": cfg.image_ext,
        "camera_fovy": cfg.camera_fovy,
        "format": "pairs_jsonl_v1",
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )

    done: set[str] = set()
    if args.resume and pairs_path.exists():
        with pairs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    sid = obj.get("id")
                    if isinstance(sid, str):
                        done.add(sid)
                except Exception:
                    continue

    total = (cfg.end_ep - cfg.start_ep) * len(cfg.tasks)
    idx = 0
    wrote = 0
    for task_i, task in enumerate(cfg.tasks):
        # Build cfg/env/evaluator once per task for speed.
        eval_cfg = _build_eval_cfg(task=task, tasks=cfg.tasks, cfg=cfg)
        if cfg.camera_fovy is not None:
            eval_cfg.task_specification.env.camera_fovy = float(cfg.camera_fovy)
        env = make_env(eval_cfg)
        evaluator = PlanEvaluator(eval_cfg, agent=None)

        for ep_i in range(cfg.start_ep, cfg.end_ep):
            sid = f"{task.replace('-', '_')}_{cfg.base_seed}_{ep_i:06d}"
            if sid in done:
                idx += 1
                continue

            # Derive a deterministic but diverse per-episode seed.
            ep_seed = (
                cfg.base_seed * cfg.base_seed + (task_i + 1) * 1000003 + ep_i * 9176
            ) % (2**32 - 2)

            init_obs, goal_obs, _, expert_success = evaluator.set_episode(
                eval_cfg,
                None,
                env,
                ep_seed,
                task_idx=task_i,
            )

            init_img = _to_hwc_uint8(
                init_obs["visual"], obs_concat_channels=cfg.obs_concat_channels
            )
            goal_img = _to_hwc_uint8(
                goal_obs["visual"], obs_concat_channels=cfg.obs_concat_channels
            )

            init_path = images_dir / f"{sid}_init{cfg.image_ext}"
            goal_path = images_dir / f"{sid}_goal{cfg.image_ext}"
            _save_image(init_path, init_img)
            _save_image(goal_path, goal_img)

            row = {
                "id": sid,
                "task": task,
                "seed": int(ep_seed),
                "init_path": str(init_path.resolve()),
                "goal_path": str(goal_path.resolve()),
                "expert_success": float(expert_success),
            }
            # Also store minimal proprio/state if present.
            try:
                p0 = init_obs["proprio"]
                pg = goal_obs["proprio"]
                if isinstance(p0, torch.Tensor):
                    row["init_proprio"] = p0.detach().cpu().reshape(-1).tolist()
                if isinstance(pg, torch.Tensor):
                    row["goal_proprio"] = pg.detach().cpu().reshape(-1).tolist()
            except Exception:
                pass

            _write_jsonl(pairs_path, row)
            done.add(sid)
            idx += 1
            wrote += 1
            if wrote % cfg.log_every == 0:
                print(f"[{wrote}/{total}] wrote {sid}")


if __name__ == "__main__":
    # Keep runs deterministic across processes unless user controls PYTHONHASHSEED.
    os.environ.setdefault("PYTHONHASHSEED", "0")
    # Headless MuJoCo defaults (safe no-ops if already set).
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    main()
