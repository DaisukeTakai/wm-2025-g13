from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from datasets import load_dataset

from src.utils.logging import get_logger


log = get_logger(__name__)


@dataclass(frozen=True)
class MetaworldInstructionPairHFConfig:
    hf_repo: str
    split: str = "train"  # "train" or "validation"
    k_variants: int = 10
    fixed_variant_idx: int | None = None  # for validation stability
    img_size: int = 224
    limit: int | None = None


class MetaworldInstructionPairHFDataset(torch.utils.data.Dataset):
    """(init_image, goal_image, instruction) pairs from HF dataset.

    Expands each base sample into K samples by selecting one instruction variant.
    """

    def __init__(self, cfg: MetaworldInstructionPairHFConfig):
        self.cfg = cfg
        if cfg.k_variants <= 0:
            raise ValueError("k_variants must be >= 1")

        log.info(f"📂 Loading HF dataset repo={cfg.hf_repo} split={cfg.split}...")
        ds = load_dataset(cfg.hf_repo, split=cfg.split)
        if cfg.limit is not None:
            ds = ds.select(range(min(int(cfg.limit), len(ds))))
        self.ds = ds
        log.info(f"✅ Loaded rows={len(self.ds)}")

    def __len__(self) -> int:
        return len(self.ds) * int(self.cfg.k_variants)

    def _variant_idx(self, idx: int) -> int:
        if self.cfg.fixed_variant_idx is not None:
            return int(self.cfg.fixed_variant_idx)
        return int(idx) % int(self.cfg.k_variants)

    def _to_chw_uint8(self, img: Any) -> torch.Tensor:
        # HF Image returns PIL.Image.Image or numpy array.
        try:
            import PIL.Image

            if isinstance(img, PIL.Image.Image):
                if self.cfg.img_size and (
                    img.size[0] != self.cfg.img_size or img.size[1] != self.cfg.img_size
                ):
                    img = img.resize(
                        (self.cfg.img_size, self.cfg.img_size),
                        resample=PIL.Image.BILINEAR,
                    )
                arr = np.array(img)
            else:
                arr = np.array(img)
        except Exception:
            arr = np.array(img)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape={arr.shape}")
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return t

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int]:
        k = int(self.cfg.k_variants)
        base_idx = int(idx) // k
        var_idx = self._variant_idx(idx)

        row = self.ds[base_idx]
        variants = row["instruction_variants"]
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"Missing instruction_variants for base_idx={base_idx}")
        if var_idx >= len(variants):
            # Be tolerant if some rows have fewer variants.
            var_idx = 0
        instruction = str(variants[var_idx])

        i0 = self._to_chw_uint8(row["init_image"])
        ig = self._to_chw_uint8(row["goal_image"])

        p0 = torch.tensor(
            row.get("init_proprio", [0.0, 0.0, 0.0, 0.0]), dtype=torch.float32
        )
        pg = torch.tensor(
            row.get("goal_proprio", [0.0, 0.0, 0.0, 0.0]), dtype=torch.float32
        )
        if p0.numel() >= 4:
            p0 = p0[:4]
        if pg.numel() >= 4:
            pg = pg[:4]

        return {
            "i0_rgb": i0,
            "ig_rgb": ig,
            "p0": p0,
            "pg": pg,
            "instruction": instruction,
            "task": str(row.get("task", "")),
            "base_idx": int(base_idx),
            "variant_idx": int(var_idx),
        }
