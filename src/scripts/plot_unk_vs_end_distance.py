#!/usr/bin/env python3

"""Plot UNK rate vs end_distance for instruction-conditioned eval runs.

Reads:
  - episode_metrics.csv (per-run eval output)
  - instructions.jsonl (per-run offline GPT instructions)
  - goalhead_instruction.pt (tokenizer_vocab for UNK rate estimation)

Writes:
  - logs/goalhead_eval_instruction/analysis_unk_end_distance.png
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np


RUNS = [
    ("run0_seed1000003", "reach-wall_instructionfile_eval20_mid_run0_seed1000003_r2"),
    ("run1_seed1010003", "reach-wall_instructionfile_eval20_mid_run1_seed1010003_r2"),
    ("run2_seed1020003", "reach-wall_instructionfile_eval20_mid_run2_seed1020003_r2"),
    ("run3_seed1030003", "reach-wall_instructionfile_eval20_mid_run3_seed1030003_r2"),
    ("run4_seed1040003", "reach-wall_instructionfile_eval20_mid_run4_seed1040003_r2"),
]

BASE_EVAL = Path(
    "/workspace/wm_exp/logs/goalhead_eval_instruction/simu_env_planning/local_language"
)
BASE_RUNS = Path("/workspace/wm_exp/logs/goalhead_eval_instruction/runs")
GOALHEAD = Path(
    "/workspace/wm_exp/logs/goalhead_instruction/metaworld_v2/goalhead_instruction.pt"
)

NON_ALNUM = re.compile(r"[^a-z0-9]+", re.I)


def toks(s: str) -> list[str]:
    return [t for t in NON_ALNUM.sub(" ", (s or "").lower()).split() if t]


def load_vocab() -> set[str]:
    import torch

    ckpt = torch.load(GOALHEAD, map_location="cpu")
    vocab = set(ckpt.get("tokenizer_vocab", []))
    vocab.discard("<pad>")
    vocab.discard("<unk>")
    return vocab


def load_instructions(run_dir: str) -> dict[tuple[str, int], str]:
    ins_path = BASE_RUNS / run_dir / "instructions.jsonl"
    ins: dict[tuple[str, int], str] = {}
    for line in ins_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        ins[(str(j["task"]), int(j["ep_seed"]))] = str(j["instruction"])
    return ins


def load_episode_metrics(tag: str) -> list[dict[str, str]]:
    ep_path = BASE_EVAL / tag / "episode_metrics.csv"
    rows: list[dict[str, str]] = []
    with ep_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def main() -> None:
    # Lazy import so this script can run headlessly.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vocab = load_vocab()

    data = []
    missing = 0
    for run_dir, tag in RUNS:
        ins = load_instructions(run_dir)
        eps = load_episode_metrics(tag)
        for row in eps:
            task = str(row["task"])
            ep_seed = int(float(row["ep_seed"]))
            instr = ins.get((task, ep_seed), "")
            if not instr:
                missing += 1
            ts = toks(instr)
            unk = sum(1 for t in ts if t not in vocab)
            unk_rate = unk / max(len(ts), 1)
            data.append(
                {
                    "task": task,
                    "success": 1 if float(row["success"]) > 0.5 else 0,
                    "unk_rate": float(unk_rate),
                    "end_distance": float(row["end_distance"]),
                }
            )

    print("episodes:", len(data), "missing_instruction:", missing)

    tasks = ["mw-reach", "mw-reach-wall"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, task in zip(axes, tasks):
        dd = [d for d in data if d["task"] == task]
        x = np.array([d["unk_rate"] for d in dd], dtype=np.float32)
        y = np.array([d["end_distance"] for d in dd], dtype=np.float32)
        s = np.array([d["success"] for d in dd], dtype=np.int32)

        ax.scatter(x[s == 0], y[s == 0], s=22, alpha=0.55, c="#d62728", label="fail")
        ax.scatter(x[s == 1], y[s == 1], s=22, alpha=0.75, c="#2ca02c", label="success")

        ax.set_title(task)
        ax.set_xlabel("UNK rate (OOV token ratio)")
        ax.grid(True, alpha=0.25)
        ax.text(
            0.02,
            0.98,
            f"n={len(dd)}, succ={s.mean():.3f}\nunk_mean={x.mean():.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    axes[0].set_ylabel("End distance (lower is better)")
    axes[1].legend(loc="upper right", frameon=True)

    fig.suptitle("Instruction UNK rate vs End distance (5 runs, 200 episodes)", y=1.02)
    fig.tight_layout()

    out = Path(
        "/workspace/wm_exp/logs/goalhead_eval_instruction/analysis_unk_end_distance.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote:", out)


if __name__ == "__main__":
    main()
