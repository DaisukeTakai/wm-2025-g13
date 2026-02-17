#!/usr/bin/env python3

"""Analyze correlation between instruction text features and episode success.

Assumes each eval run directory contains:
  - episode_metrics.csv (added by eval.py)
  - run instructions.jsonl (keyed by task, ep_seed)
  - a GoalHead checkpoint with tokenizer_vocab for unk-rate estimation
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


NON_ALNUM = re.compile(r"[^a-z0-9]+")
WALL_KW = re.compile(r"\b(wall|obstacle|avoid|gap|opening|around)\b", re.I)


def toks(s: str) -> list[str]:
    return [t for t in NON_ALNUM.sub(" ", (s or "").lower()).split() if t]


def load_vocab_from_goalhead_pt(path: Path) -> set[str]:
    import torch

    ckpt = torch.load(path, map_location="cpu")
    vocab = ckpt.get("tokenizer_vocab") or []
    v = set(vocab)
    v.discard("<pad>")
    v.discard("<unk>")
    return v


def read_instructions(path: Path) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            out[(str(j["task"]), int(j["ep_seed"]))] = str(j["instruction"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run work dir containing episode_metrics.csv",
    )
    ap.add_argument(
        "--instructions", type=str, required=True, help="instructions.jsonl for the run"
    )
    ap.add_argument(
        "--goalhead-ckpt", type=str, required=True, help="goalhead_instruction.pt path"
    )
    ap.add_argument("--out", type=str, required=True, help="Output CSV path")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    epi_path = run_dir / "episode_metrics.csv"
    if not epi_path.exists():
        raise FileNotFoundError(f"Missing {epi_path}")

    ins_map = read_instructions(Path(args.instructions))
    vocab = load_vocab_from_goalhead_pt(Path(args.goalhead_ckpt))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        epi_path.open("r", encoding="utf-8") as f_in,
        out_path.open("w", newline="", encoding="utf-8") as f_out,
    ):
        r = csv.DictReader(f_in)
        fieldnames = list(r.fieldnames or []) + [
            "instr_len",
            "has_wall_kw",
            "unk_rate",
            "instruction",
        ]
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()

        for row in r:
            task = str(row["task"])
            ep_seed = int(float(row["ep_seed"]))
            instr = ins_map.get((task, ep_seed), "")
            ts = toks(instr)
            unk = sum(1 for t in ts if t not in vocab)
            unk_rate = unk / max(len(ts), 1)
            row_out = dict(row)
            row_out["instr_len"] = str(len(ts))
            row_out["has_wall_kw"] = "1" if WALL_KW.search(instr or "") else "0"
            row_out["unk_rate"] = f"{unk_rate:.6f}"
            row_out["instruction"] = instr
            w.writerow(row_out)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
