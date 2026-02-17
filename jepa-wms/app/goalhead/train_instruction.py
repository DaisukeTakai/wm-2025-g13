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
from app.plan_common.datasets.metaworld_instruction_pair_hf_dset import (
    MetaworldInstructionPairHFConfig,
    MetaworldInstructionPairHFDataset,
)
from app.plan_common.models.goal_head import GoalHead, GoalHeadConfig, goalhead_loss
from app.plan_common.models.goal_head_v2 import GoalHeadV2, GoalHeadV2Config
from app.plan_common.models.goal_head_mixture import (
    GoalHeadMixture,
    GoalHeadMixtureConfig,
    goalhead_mixture_loss,
)
from app.plan_common.text.instruction_tokenizer import InstructionTokenizer
from evals.simu_env_planning.eval import init_module as init_encpred_module
from src.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class TrainInstructionConfig:
    # output
    folder: str
    # frozen WM
    wm_checkpoint: str
    wm_module_name: str = "app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds"
    wm_pretrain_kwargs: dict | None = None
    wm_data: dict | None = None
    wm_data_aug: dict | None = None
    wm_wrapper_kwargs: dict | None = None
    # HF dataset
    hf_repo: str = "wm-2025-g13/metaworld"
    img_size: int = 224
    train_limit_base: int | None = None
    val_limit_base: int | None = None
    # instruction expansion
    k_variants_train: int = 10
    k_variants_val: int = 1
    val_fixed_variant_idx: int = 0
    # tokenizer
    max_vocab: int = 20000
    min_freq: int = 1
    max_text_len: int = 32
    # training
    batch_size: int = 32
    num_workers: int = 0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 5
    lambda_cos: float = 1.0
    alpha_proprio: float = 1.0
    device: str = "cuda:0"
    resume: bool = False
    # model kind
    goalhead_kind: str = "v2"  # v1|v2|mixture_v1
    goalhead: dict | None = None


def _expand_env_vars(s: str) -> str:
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
    args = dict(args)
    args.pop("app", None)
    cfg = TrainInstructionConfig(**args)
    if cfg.wm_data_aug is None:
        cfg.wm_data_aug = {}

    work_dir = Path(_expand_env_vars(cfg.folder))
    work_dir.mkdir(parents=True, exist_ok=True)

    # Logger (append-safe)
    csv_path = work_dir / "train.csv"
    fields = [
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
            csv_module.writer(f).writerow(fields)

    def _log_row(epoch: int, iteration: int, metrics: Dict[str, float]):
        with open(csv_path, "a", newline="") as f:
            csv_module.writer(f).writerow(
                [
                    int(epoch),
                    int(iteration),
                    float(metrics["loss"]),
                    float(metrics["l2_visual"]),
                    float(metrics["cos_visual"]),
                    float(metrics["l2_proprio"]),
                    float(metrics["cos_proprio"]),
                    float(metrics["lr"]),
                ]
            )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    preprocessor = _build_preprocessor(cfg.img_size, cfg.wm_data_aug)

    wm_data = cfg.wm_data or {
        "dataset_type": "custom",
        "datasets": ["METAWORLD_HF"],
        "img_size": cfg.img_size,
        "custom": {"frameskip": 5, "action_skip": 1, "state_skip": 1},
    }
    wm_pretrain_kwargs = cfg.wm_pretrain_kwargs or {}
    # The default public Metaworld JEPA-WM uses proprio_encoding='feature' and requires a non-zero proprio_emb_dim.
    # If the user didn't provide full pretrain kwargs, supply a safe default.
    try:
        if wm_pretrain_kwargs.get("proprio_encoding") == "feature":
            wm_pretrain_kwargs.setdefault("proprio_encoder", {})
            wm_pretrain_kwargs["proprio_encoder"].setdefault("proprio_emb_dim", 16)
    except Exception:
        pass
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

    # Datasets
    train_ds = MetaworldInstructionPairHFDataset(
        MetaworldInstructionPairHFConfig(
            hf_repo=cfg.hf_repo,
            split="train",
            k_variants=cfg.k_variants_train,
            fixed_variant_idx=None,
            img_size=cfg.img_size,
            limit=cfg.train_limit_base,
        )
    )
    val_ds = MetaworldInstructionPairHFDataset(
        MetaworldInstructionPairHFConfig(
            hf_repo=cfg.hf_repo,
            split="validation",
            k_variants=cfg.k_variants_val,
            fixed_variant_idx=cfg.val_fixed_variant_idx,
            img_size=cfg.img_size,
            limit=cfg.val_limit_base,
        )
    )

    # Tokenizer vocab built from TRAIN instructions (base samples), not expanded.
    logger.info("Building instruction vocab from training split...")
    # Sample one variant per base sample to build vocab.
    base_texts = []
    for i in range(len(train_ds.ds)):
        row = train_ds.ds[i]
        vs = row.get("instruction_variants", [])
        if isinstance(vs, list) and vs:
            base_texts.append(str(vs[0]))
    tok = InstructionTokenizer.build_from_texts(
        base_texts, max_vocab=cfg.max_vocab, min_freq=cfg.min_freq
    )
    logger.info(f"Tokenizer vocab_size={len(tok.vocab)}")

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

    # Infer dims
    with torch.no_grad():
        b0 = next(iter(train_loader))
        text_ids0 = torch.tensor(
            tok.batch_encode(b0["instruction"], max_len=cfg.max_text_len),
            dtype=torch.long,
        )
        obs0 = {
            "visual": b0["i0_rgb"].to(device).unsqueeze(1),
            "proprio": b0["p0"].to(device).unsqueeze(1),
        }
        init_enc0 = wm.encode(obs0, act=False)
        visual_dim = int(init_enc0["visual"].shape[-1])
        proprio_dim = int(init_enc0["proprio"].shape[-1])

    kind = str(cfg.goalhead_kind)
    gh_cfg_in = cfg.goalhead or {}
    if kind == "v2":
        gh_cfg = GoalHeadV2Config(
            kind="v2",
            visual_dim=visual_dim,
            proprio_dim=proprio_dim,
            visual_depth=int(gh_cfg_in.get("visual_depth", 6)),
            visual_num_heads=int(
                gh_cfg_in.get("visual_num_heads", gh_cfg_in.get("num_heads", 8))
            ),
            text_embed_dim=int(gh_cfg_in.get("text_embed_dim", 256)),
            text_depth=int(gh_cfg_in.get("text_depth", 2)),
            text_num_heads=int(gh_cfg_in.get("text_num_heads", 4)),
            max_text_len=int(cfg.max_text_len),
            mlp_ratio=int(gh_cfg_in.get("mlp_ratio", 4)),
            dropout=float(gh_cfg_in.get("dropout", 0.0)),
        )
        goal_head = GoalHeadV2(gh_cfg, vocab_size=len(tok.vocab)).to(device)
        loss_fn = goalhead_loss
    elif kind == "mixture_v1":
        gh_cfg = GoalHeadMixtureConfig(
            kind="mixture_v1",
            mixture_k=int(gh_cfg_in.get("mixture_k", 4)),
            prompt_len=int(gh_cfg_in.get("prompt_len", 0)),
            component_scale=float(gh_cfg_in.get("component_scale", 1.0)),
            prompt_scale=float(gh_cfg_in.get("prompt_scale", 1.0)),
            visual_dim=visual_dim,
            proprio_dim=proprio_dim,
            num_heads=int(gh_cfg_in.get("num_heads", 8)),
            depth=int(gh_cfg_in.get("depth", 4)),
            text_embed_dim=int(gh_cfg_in.get("text_embed_dim", 256)),
            mlp_ratio=int(gh_cfg_in.get("mlp_ratio", 4)),
            dropout=float(gh_cfg_in.get("dropout", 0.0)),
            lambda_div=float(gh_cfg_in.get("lambda_div", 0.01)),
        )
        goal_head = GoalHeadMixture(gh_cfg, vocab_size=len(tok.vocab)).to(device)
        loss_fn = goalhead_mixture_loss
    else:
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
        loss_fn = goalhead_loss

    opt = torch.optim.AdamW(
        goal_head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    ckpt_path = work_dir / "goalhead_latest.pt"
    resume = bool(resume_preempt) or bool(cfg.resume)
    start_epoch = 0
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        goal_head.load_state_dict(ckpt["goal_head"], strict=False)
        if ckpt.get("opt") is not None:
            opt.load_state_dict(ckpt["opt"])
        ckpt_vocab = ckpt.get("tokenizer_vocab")
        if ckpt_vocab is not None and ckpt_vocab != tok.vocab:
            raise ValueError(
                "Tokenizer vocab mismatch when resuming. "
                f"ckpt_vocab_size={len(ckpt_vocab)} current_vocab_size={len(tok.vocab)}"
            )
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        logger.info(f"Resuming from epoch {start_epoch}")

    def _run_epoch(loader, train: bool, epoch: int):
        if train:
            goal_head.train()
        else:
            goal_head.eval()

        losses_out = []
        for it, batch in enumerate(loader):
            i0 = batch["i0_rgb"].to(device, non_blocking=True).unsqueeze(1)
            ig = batch["ig_rgb"].to(device, non_blocking=True).unsqueeze(1)
            p0 = batch["p0"].to(device, non_blocking=True).unsqueeze(1)
            pg = batch["pg"].to(device, non_blocking=True).unsqueeze(1)
            text_ids = torch.tensor(
                tok.batch_encode(batch["instruction"], max_len=cfg.max_text_len),
                dtype=torch.long,
                device=device,
            )

            with torch.no_grad():
                init_enc = wm.encode({"visual": i0, "proprio": p0}, act=False)
                goal_enc = wm.encode({"visual": ig, "proprio": pg}, act=False)

            pred = goal_head(init_enc, text_ids=text_ids)
            if loss_fn is goalhead_mixture_loss:
                losses = loss_fn(
                    pred,
                    goal_enc,
                    lambda_cos=cfg.lambda_cos,
                    alpha_proprio=cfg.alpha_proprio,
                )
            else:
                losses = loss_fn(
                    pred,
                    goal_enc,
                    lambda_cos=cfg.lambda_cos,
                    alpha_proprio=cfg.alpha_proprio,
                )

            if train:
                opt.zero_grad(set_to_none=True)
                losses["loss"].backward()
                opt.step()
                _log_row(
                    epoch,
                    it,
                    {
                        **{k: float(v.detach().cpu()) for k, v in losses.items()},
                        "lr": cfg.lr,
                    },
                )

            losses_out.append({k: float(v.detach().cpu()) for k, v in losses.items()})

        # simple mean
        mean = {
            k: sum(d[k] for d in losses_out) / max(len(losses_out), 1)
            for k in losses_out[0].keys()
        }
        return mean

    for epoch in range(start_epoch, int(cfg.epochs)):
        train_metrics = _run_epoch(train_loader, train=True, epoch=epoch)
        val_metrics = _run_epoch(val_loader, train=False, epoch=epoch)
        logger.info(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f}"
        )

        torch.save(
            {
                "goal_head": goal_head.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch,
                "tokenizer_vocab": tok.vocab,
                "cfg": {
                    "goalhead_kind": kind,
                    "max_text_len": cfg.max_text_len,
                    "max_vocab": cfg.max_vocab,
                },
            },
            ckpt_path,
        )

    final_path = work_dir / "goalhead_instruction.pt"
    torch.save(
        {
            "goal_head": goal_head.state_dict(),
            "tokenizer_vocab": tok.vocab,
            "cfg": {
                "goalhead_kind": kind,
                "max_text_len": cfg.max_text_len,
                "max_vocab": cfg.max_vocab,
            },
        },
        final_path,
    )
    logger.info(f"Saved GoalHead to {final_path}")
