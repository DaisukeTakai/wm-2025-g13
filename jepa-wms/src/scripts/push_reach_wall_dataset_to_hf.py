#!/usr/bin/env python3

"""Build and push the reach/reach-wall dataset to Hugging Face Hub.

This script joins:
  - pairs.jsonl (init/goal image paths + metadata)
  - annotations.jsonl (instruction_variants)

and pushes a DatasetDict with Image columns to the Hub.

Expected inputs:
  pairs.jsonl line example:
    {"id": "mw_reach_1_000123", "task": "mw-reach", "seed": 123, "init_path": "/abs/..._init.jpg", "goal_path": "/abs/..._goal.jpg", ...}

  annotations.jsonl line example:
    {"id": "mw_reach_1_000123", "task": "mw-reach", "instruction_variants": ["...", ...]}
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _split_of(s: str) -> str:
    """Deterministic 90/10 split by id (no test split).

    We keep validation as a stable holdout for model selection; true testing should
    happen in simulation/real rollouts.
    """
    h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % 10
    return "validation" if h == 0 else "train"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Push reach/reach-wall VLM dataset to Hugging Face"
    )
    ap.add_argument(
        "--repo", type=str, required=True, help="Hub repo id, e.g. org/name"
    )
    ap.add_argument("--private", action="store_true", help="Create as private repo")
    ap.add_argument(
        "--pairs",
        type=str,
        default="/workspace/wm_exp/datasets/reach_wall_vlm/raw/pairs.jsonl",
    )
    ap.add_argument(
        "--annotations",
        type=str,
        default="/workspace/wm_exp/datasets/reach_wall_vlm/annotations.jsonl",
    )
    ap.add_argument(
        "--require-k",
        type=int,
        default=10,
        help="Require exactly K instruction variants per sample (0 disables)",
    )
    args = ap.parse_args()

    pairs_path = Path(args.pairs)
    ann_path = Path(args.annotations)
    if not pairs_path.exists():
        raise FileNotFoundError(f"pairs.jsonl not found: {pairs_path}")
    if not ann_path.exists():
        raise FileNotFoundError(f"annotations.jsonl not found: {ann_path}")

    # Lazy import so the script can still run basic validations without deps.
    import datasets

    # Load annotations into a dict for fast join.
    ann_by_id: dict[str, dict[str, Any]] = {}
    for rec in _read_jsonl(ann_path):
        sid = rec.get("id")
        if isinstance(sid, str):
            ann_by_id[sid] = rec

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}

    missing_ann = 0
    missing_imgs = 0
    dropped_k = 0

    for p in _read_jsonl(pairs_path):
        sid = str(p.get("id"))
        a = ann_by_id.get(sid)
        if a is None:
            missing_ann += 1
            continue

        init_path = Path(str(p.get("init_path")))
        goal_path = Path(str(p.get("goal_path")))
        if not init_path.exists() or not goal_path.exists():
            missing_imgs += 1
            continue

        variants = a.get("instruction_variants", [])
        if args.require_k and (
            not isinstance(variants, list) or len(variants) != int(args.require_k)
        ):
            dropped_k += 1
            continue

        row: dict[str, Any] = {
            "id": sid,
            "task": str(p.get("task")),
            "seed": int(p.get("seed", -1)),
            "expert_success": float(p.get("expert_success", -1.0)),
            "instruction_variants": variants,
            # datasets.Image() accepts local file paths.
            "init_image": str(init_path),
            "goal_image": str(goal_path),
        }
        # Optional proprio (if present in pairs.jsonl)
        if "init_proprio" in p:
            row["init_proprio"] = p.get("init_proprio")
        if "goal_proprio" in p:
            row["goal_proprio"] = p.get("goal_proprio")

        rows_by_split[_split_of(sid)].append(row)

    if missing_ann or missing_imgs or dropped_k:
        print(
            f"join stats: missing_ann={missing_ann} missing_imgs={missing_imgs} dropped_k={dropped_k}",
            flush=True,
        )

    # Define schema.
    feats: dict[str, Any] = {
        "id": datasets.Value("string"),
        "task": datasets.Value("string"),
        "seed": datasets.Value("int64"),
        "expert_success": datasets.Value("float32"),
        "instruction_variants": datasets.Sequence(datasets.Value("string")),
        "init_image": datasets.Image(),
        "goal_image": datasets.Image(),
    }
    if any(
        "init_proprio" in r
        for r in rows_by_split["train"] + rows_by_split["validation"]
    ):
        feats["init_proprio"] = datasets.Sequence(datasets.Value("float32"))
        feats["goal_proprio"] = datasets.Sequence(datasets.Value("float32"))

    features = datasets.Features(feats)

    dsd = datasets.DatasetDict(
        {
            split: datasets.Dataset.from_list(rows, features=features)
            for split, rows in rows_by_split.items()
        }
    )

    print({k: len(v) for k, v in dsd.items()}, flush=True)
    dsd.push_to_hub(args.repo, private=bool(args.private))
    print(f"pushed: {args.repo}", flush=True)


if __name__ == "__main__":
    main()
