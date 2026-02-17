from __future__ import annotations

import os
import csv as csv_module
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from app.plan_common.datasets import get_data_stats
from app.plan_common.datasets.preprocessor import Preprocessor
from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
from app.plan_common.datasets.metaworld_goalpair_dset import (
    MetaworldGoalPairConfig,
    MetaworldGoalPairDataset,
)
from app.plan_common.models.goal_head import GoalHead, GoalHeadConfig, goalhead_loss
from app.plan_common.text.task_tokenizer import TaskTokenizer
from evals.simu_env_planning.eval import init_module as init_encpred_module
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    folder: str
    wm_checkpoint: str
    wm_module_name: str = "app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds"
    wm_pretrain_kwargs: dict | None = None
    wm_data: dict | None = None
    wm_data_aug: dict | None = None
    wm_wrapper_kwargs: dict | None = None
    # dataset
    metaworld_data_path: str = "${JEPAWM_DSET}/Metaworld/data"
    task_set: str = "mwgreedy"
    split_ratio: float = 0.9
    seed: int = 234
    n_rollouts: int | None = None
    frameskip: int = 5
    goal_H: int = 6
    img_size: int = 224
    max_text_len: int = 4
    pairs_per_rollout: int = 1
    # training
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 5
    lambda_cos: float = 1.0
    alpha_proprio: float = 1.0
    goalhead: dict | None = None
    device: str = "cuda:0"
    resume: bool = False


def _expand_env_vars(s: str) -> str:
    # Reuse same semantics as yaml_utils.expand_env_vars for common simple cases.
    return os.path.expandvars(s)


def _build_preprocessor(img_size: int, data_aug: dict | None) -> Preprocessor:
    stats = get_data_stats("metaworld")
    normalize = None
    if data_aug is not None:
        normalize = data_aug.get("normalize")
    if normalize is None:
        normalize = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    transform = make_transforms(
        img_size=img_size,
        normalize=normalize,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    inverse_transform = make_inverse_transforms(img_size=img_size, normalize=normalize)
    return Preprocessor(
        action_mean=torch.tensor(stats["action_mean"]),
        action_std=torch.tensor(stats["action_std"]),
        state_mean=torch.tensor(stats["state_mean"]),
        state_std=torch.tensor(stats["state_std"]),
        proprio_mean=torch.tensor(stats["proprio_mean"]),
        proprio_std=torch.tensor(stats["proprio_std"]),
        transform=transform,
        inverse_transform=inverse_transform,
    )


def _collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if isinstance(batch[0][k], str):
            out[k] = [b[k] for b in batch]
        elif isinstance(batch[0][k], int):
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def main(args, resume_preempt: bool = False):
    # app.main passes the full YAML dict including the `app` key.
    args = dict(args)
    args.pop("app", None)
    cfg = TrainConfig(**args)
    if cfg.wm_data_aug is None:
        cfg.wm_data_aug = {}
    work_dir = Path(_expand_env_vars(cfg.folder))
    work_dir.mkdir(parents=True, exist_ok=True)

    # Logger (append-safe; avoid duplicate headers on resume)
    csv_path = work_dir / "train.csv"
    train_fields = [
        "epoch",
        "iteration",
        "loss",
        "l2_visual",
        "cos_visual",
        "l2_proprio",
        "cos_proprio",
        "lr",
    ]
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with open(csv_path, "w", newline="") as f:
            w = csv_module.writer(f)
            w.writerow(train_fields)

    def _log_train_row(epoch: int, iteration: int, metrics: Dict[str, float]):
        row = [
            int(epoch),
            int(iteration),
            float(metrics["loss"]),
            float(metrics["l2_visual"]),
            float(metrics["cos_visual"]),
            float(metrics["l2_proprio"]),
            float(metrics["cos_proprio"]),
            float(metrics["lr"]),
        ]
        with open(csv_path, "a", newline="") as f:
            w = csv_module.writer(f)
            w.writerow(row)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Preprocessor matches the WM encode path
    preprocessor = _build_preprocessor(cfg.img_size, cfg.wm_data_aug)

    # Load frozen WM (EncPredWM)
    wm_data = cfg.wm_data or {
        "dataset_type": "custom",
        "datasets": ["METAWORLD_HF"],
        "img_size": cfg.img_size,
        "custom": {"frameskip": cfg.frameskip, "action_skip": 1, "state_skip": 1},
    }
    wm_pretrain_kwargs = cfg.wm_pretrain_kwargs or {}
    wm_wrapper_kwargs = cfg.wm_wrapper_kwargs or {
        "ctxt_window": 2,
        "proprio_mode": "predict_proprio",
    }
    wm = init_encpred_module(
        folder=str(work_dir),
        checkpoint=cfg.wm_checkpoint,
        module_name=cfg.wm_module_name,
        model_kwargs=wm_pretrain_kwargs,
        device=device,
        cfgs_data=wm_data,
        wrapper_kwargs=wm_wrapper_kwargs,
        action_dim=4,
        proprio_dim=4,
        preprocessor=preprocessor,
    )
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    # Tokenizer from task set
    tok = TaskTokenizer.from_task_set(cfg.task_set)

    # Datasets
    dcfg = MetaworldGoalPairConfig(
        data_path=_expand_env_vars(cfg.metaworld_data_path),
        task_set=cfg.task_set,
        split_ratio=cfg.split_ratio,
        seed=cfg.seed,
        n_rollouts=cfg.n_rollouts,
        img_size=cfg.img_size,
        frameskip=cfg.frameskip,
        goal_H=cfg.goal_H,
        max_text_len=cfg.max_text_len,
        pairs_per_rollout=cfg.pairs_per_rollout,
    )
    train_ds = MetaworldGoalPairDataset(dcfg, split="train")
    val_ds = MetaworldGoalPairDataset(dcfg, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=_collate,
        drop_last=False,
    )

    # Infer visual/proprio dims from WM
    with torch.no_grad():
        b0 = next(iter(train_loader))
        text_ids0 = torch.tensor(
            tok.batch_encode(b0["task"], max_len=cfg.max_text_len), dtype=torch.long
        )
        obs0 = {
            "visual": b0["i0_rgb"].unsqueeze(1),  # [B,1,C,H,W]
            "proprio": b0["p0"].unsqueeze(1),
        }
        init_enc0 = wm.encode(obs0, act=False)
        visual_dim = int(init_enc0["visual"].shape[-1])
        # wm.encode returns proprio emb as [B,T,HW,P] when use_proprio is on
        proprio_dim = int(init_enc0["proprio"].shape[-1])

    gh_cfg_in = cfg.goalhead or {}
    gh_cfg = GoalHeadConfig(
        visual_dim=visual_dim,
        proprio_dim=proprio_dim,
        num_heads=int(gh_cfg_in.get("num_heads", 8)),
        depth=int(gh_cfg_in.get("depth", 2)),
        text_embed_dim=int(gh_cfg_in.get("text_embed_dim", 128)),
        mlp_ratio=int(gh_cfg_in.get("mlp_ratio", 4)),
        dropout=float(gh_cfg_in.get("dropout", 0.0)),
    )
    goal_head = GoalHead(gh_cfg, vocab_size=len(tok.vocab)).to(device)
    opt = torch.optim.AdamW(
        goal_head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Resume (epoch-level): load latest checkpoint if requested.
    ckpt_path = work_dir / "goalhead_latest.pt"
    resume = bool(resume_preempt) or bool(cfg.resume)
    start_epoch = 0
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        goal_head.load_state_dict(ckpt["goal_head"], strict=True)
        ckpt_vocab = ckpt.get("tokenizer_vocab")
        if ckpt_vocab is not None and ckpt_vocab != tok.vocab:
            raise ValueError(
                "Tokenizer vocab mismatch when resuming. "
                "This usually means task_set changed. "
                f"ckpt_vocab_size={len(ckpt_vocab)} current_vocab_size={len(tok.vocab)}"
            )
        if ckpt.get("opt") is not None:
            opt.load_state_dict(ckpt["opt"])
            logger.info(f"Resumed optimizer state from {ckpt_path}")
        else:
            logger.info(
                f"No optimizer state in {ckpt_path}; resuming model weights only"
            )
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        logger.info(f"Resuming GoalHead from epoch {start_epoch}")

    logger.info(
        f"GoalHead params={sum(p.numel() for p in goal_head.parameters()):,} visual_dim={visual_dim} proprio_dim={proprio_dim}"
    )

    def _run_epoch(loader, train: bool, epoch: int):
        if train:
            goal_head.train()
            train_ds.set_epoch(epoch)
        else:
            goal_head.eval()
            val_ds.set_epoch(epoch)

        task_sums = None
        if not train:
            task_sums = {}

        for it, batch in enumerate(loader):
            i0 = batch["i0_rgb"].to(device, non_blocking=True).unsqueeze(1)
            ig = batch["ig_rgb"].to(device, non_blocking=True).unsqueeze(1)
            p0 = batch["p0"].to(device, non_blocking=True).unsqueeze(1)
            pg = batch["pg"].to(device, non_blocking=True).unsqueeze(1)
            text_ids = torch.tensor(
                tok.batch_encode(batch["task"], max_len=cfg.max_text_len),
                dtype=torch.long,
            )
            text_ids = text_ids.to(device, non_blocking=True)

            with torch.no_grad():
                init_enc = wm.encode({"visual": i0, "proprio": p0}, act=False)
                goal_enc = wm.encode({"visual": ig, "proprio": pg}, act=False)

            pred = goal_head(init_enc, text_ids=text_ids)
            losses = goalhead_loss(
                pred,
                goal_enc,
                lambda_cos=cfg.lambda_cos,
                alpha_proprio=cfg.alpha_proprio,
            )

            if task_sums is not None:
                # Per-task aggregation (simple: use batch-mean for every task occurrence).
                b_l2v = float(losses["l2_visual"].detach().cpu())
                b_cosv = float(losses["cos_visual"].detach().cpu())
                b_l2p = float(losses["l2_proprio"].detach().cpu())
                b_cosp = float(losses["cos_proprio"].detach().cpu())
                for task_name in batch["task"]:
                    if task_name not in task_sums:
                        task_sums[task_name] = {
                            "count": 0,
                            "l2_visual": 0.0,
                            "cos_visual": 0.0,
                            "l2_proprio": 0.0,
                            "cos_proprio": 0.0,
                        }
                    task_sums[task_name]["count"] += 1
                    task_sums[task_name]["l2_visual"] += b_l2v
                    task_sums[task_name]["cos_visual"] += b_cosv
                    task_sums[task_name]["l2_proprio"] += b_l2p
                    task_sums[task_name]["cos_proprio"] += b_cosp

            if train:
                opt.zero_grad(set_to_none=True)
                losses["loss"].backward()
                opt.step()

            if train:
                _log_train_row(
                    epoch=epoch,
                    iteration=it,
                    metrics={
                        "loss": float(losses["loss"].detach().cpu()),
                        "l2_visual": float(losses["l2_visual"].detach().cpu()),
                        "cos_visual": float(losses["cos_visual"].detach().cpu()),
                        "l2_proprio": float(losses["l2_proprio"].detach().cpu()),
                        "cos_proprio": float(losses["cos_proprio"].detach().cpu()),
                        "lr": float(opt.param_groups[0]["lr"]),
                    },
                )

        if task_sums is not None and len(task_sums) > 0:
            out_path = work_dir / f"val_task_metrics_epoch{epoch}.csv"
            with open(out_path, "w", newline="") as f:
                writer = csv_module.DictWriter(
                    f,
                    fieldnames=[
                        "task",
                        "count",
                        "l2_visual",
                        "cos_visual",
                        "l2_proprio",
                        "cos_proprio",
                    ],
                )
                writer.writeheader()
                for task_name in sorted(task_sums.keys()):
                    row = task_sums[task_name]
                    c = max(int(row["count"]), 1)
                    writer.writerow(
                        {
                            "task": task_name,
                            "count": c,
                            "l2_visual": row["l2_visual"] / c,
                            "cos_visual": row["cos_visual"] / c,
                            "l2_proprio": row["l2_proprio"] / c,
                            "cos_proprio": row["cos_proprio"] / c,
                        }
                    )
            logger.info(f"Wrote per-task val metrics to {out_path}")

    for epoch in range(start_epoch, cfg.epochs):
        _run_epoch(train_loader, train=True, epoch=epoch)
        _run_epoch(val_loader, train=False, epoch=epoch)
        ckpt = {
            "epoch": epoch,
            "goal_head": goal_head.state_dict(),
            "opt": opt.state_dict(),
            "tokenizer_vocab": tok.vocab,
            "cfg": args,
        }
        tmp_path = work_dir / "goalhead_latest.pt.tmp"
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")

    return {
        "work_dir": str(work_dir),
        "checkpoint": str(work_dir / "goalhead_latest.pt"),
    }
