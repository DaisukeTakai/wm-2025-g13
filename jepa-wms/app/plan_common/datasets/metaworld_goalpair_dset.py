from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from datasets import Dataset, load_dataset

from src.utils.logging import get_logger

log = get_logger(__name__)


def _decode_selected_frames(video_obj, frame_indices: Sequence[int]) -> np.ndarray:
    """Decode a small set of frames from a HF 'video' column.

    video_obj can be either:
    - dict with key 'bytes' (mp4 bytes)
    - torchcodec VideoDecoder-like object supporting __len__ and __getitem__
    """
    frame_indices = sorted(set(int(i) for i in frame_indices))
    if not frame_indices:
        raise ValueError("frame_indices must be non-empty")

    if isinstance(video_obj, dict) and "bytes" in video_obj:
        import io

        import imageio

        wanted = set(frame_indices)
        max_i = frame_indices[-1]
        frames = {}
        reader = imageio.get_reader(io.BytesIO(video_obj["bytes"]), format="mp4")
        try:
            for i, fr in enumerate(reader):
                if i in wanted:
                    frames[i] = fr
                if i >= max_i:
                    break
        finally:
            reader.close()
        missing = [i for i in frame_indices if i not in frames]
        if missing:
            raise ValueError(f"Missing frames after decode: {missing}")
        return np.stack([frames[i] for i in frame_indices], axis=0)

    # torchcodec path: index frames directly (C,H,W) tensors
    out = []
    for i in frame_indices:
        fr = video_obj[i]
        # VideoDecoder[i] returns tensor-like with .data in (C,H,W)
        fr_t = fr.data if hasattr(fr, "data") else fr
        fr_np = fr_t.permute(1, 2, 0).cpu().numpy()
        out.append(fr_np)
    return np.stack(out, axis=0)


@dataclass
class MetaworldGoalPairConfig:
    data_path: str
    task_set: str = "mwgreedy"
    split_ratio: float = 0.9
    seed: int = 234
    n_rollouts: Optional[int] = None
    img_size: int = 224
    frameskip: int = 5
    goal_H: int = 6
    max_text_len: int = 4
    pairs_per_rollout: int = 1


class MetaworldGoalPairDataset(torch.utils.data.Dataset):
    """Pairs (I0, Ig, task_name) from Metaworld HF parquet.

    Returns raw uint8 RGB frames (C,H,W) and raw proprio (first 4 dims).
    """

    def __init__(
        self,
        cfg: MetaworldGoalPairConfig,
        split: str,
        indices: Optional[Sequence[int]] = None,
    ):
        self.cfg = cfg
        self.split = split
        self.epoch = 0

        data_path = Path(cfg.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Metaworld HF data_path not found: {data_path}")

        log.info(f"📂 Loading Metaworld HF dataset from {data_path}...")
        ds = load_dataset("parquet", data_dir=str(data_path), split="train")

        from evals.simu_env_planning.planning.common import TASK_SET

        if cfg.task_set not in TASK_SET:
            raise KeyError(
                f"Unknown task_set: {cfg.task_set}. Available: {list(TASK_SET.keys())}"
            )
        allowed_tasks = set(TASK_SET[cfg.task_set])
        ds = ds.filter(lambda x: x["task"] in allowed_tasks)

        # Ensure rollouts are not ordered by task when subsetting.
        # Without shuffling, `select(range(n))` can pick mostly a single task.
        ds = ds.shuffle(seed=cfg.seed)

        if cfg.n_rollouts is not None:
            ds = ds.select(range(min(cfg.n_rollouts, len(ds))))

        if indices is None:
            # deterministic split using HF's train_test_split
            split_ds = ds.train_test_split(
                test_size=1.0 - cfg.split_ratio, seed=cfg.seed, shuffle=True
            )
            ds = split_ds["train"] if split == "train" else split_ds["test"]
        else:
            ds = ds.select(list(indices))

        self.ds: Dataset = ds
        log.info(f"✅ GoalPair split={split} rollouts={len(self.ds)}")

        # Assuming HF conversion matches MetaworldHFDataset convention (99 aligned frames)
        self.seq_len = 99

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.ds) * int(self.cfg.pairs_per_rollout)

    def _sample_t0(self, rollout_idx: int, pair_idx: int) -> int:
        max_t0 = self.seq_len - 1 - self.cfg.frameskip * self.cfg.goal_H
        if max_t0 < 0:
            raise ValueError(
                f"Invalid (frameskip={self.cfg.frameskip}, goal_H={self.cfg.goal_H}) for seq_len={self.seq_len}"
            )
        # random.Random expects a hashable scalar seed (not a tuple).
        seed = (
            self.cfg.seed
            + 1000003 * int(self.epoch)
            + 10007 * int(rollout_idx)
            + 97 * int(pair_idx)
        ) & 0xFFFFFFFF
        rng = random.Random(seed)
        return rng.randint(0, max_t0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int]:
        pairs_per_rollout = int(self.cfg.pairs_per_rollout)
        if pairs_per_rollout <= 0:
            raise ValueError("pairs_per_rollout must be >= 1")
        rollout_idx = int(idx) // pairs_per_rollout
        pair_idx = int(idx) % pairs_per_rollout

        row = self.ds[rollout_idx]
        task = row["task"]
        t0 = self._sample_t0(rollout_idx=rollout_idx, pair_idx=pair_idx)
        tg = t0 + self.cfg.frameskip * self.cfg.goal_H

        frames = _decode_selected_frames(row["video"], [t0, tg])  # [2,H,W,3] uint8
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)

        # Convert to CHW
        i0 = torch.from_numpy(frames[0]).permute(2, 0, 1).contiguous()
        ig = torch.from_numpy(frames[1]).permute(2, 0, 1).contiguous()

        states = np.asarray(row["states"], dtype=np.float32)
        # states length is 100 in HF, but frames aligned to first 99 steps
        p0 = torch.from_numpy(states[t0, :4].copy())
        pg = torch.from_numpy(states[tg, :4].copy())

        return {
            "i0_rgb": i0,
            "ig_rgb": ig,
            "p0": p0,
            "pg": pg,
            "task": task,
            "t0": int(t0),
            "tg": int(tg),
            "rollout_idx": int(rollout_idx),
            "pair_idx": int(pair_idx),
        }
