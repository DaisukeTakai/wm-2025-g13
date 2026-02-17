from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.plan_common.datasets import get_data_stats
from app.plan_common.datasets.preprocessor import Preprocessor
from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
from app.plan_common.models.goal_head import GoalHead, GoalHeadConfig, goalhead_loss
from app.plan_common.text.task_tokenizer import TaskTokenizer
from evals.simu_env_planning.eval import init_module as init_encpred_module
from evals.simu_env_planning.planning.common import TASK_SET
from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FineTuneConfig:
    folder: str
    wm_checkpoint: str
    goalhead_checkpoint: str | None = None
    # What to collect / train on
    task: str = "mw-reach-wall"
    n_episodes: int = 100
    max_episode_steps: int = 100
    # Data generation
    img_size: int = 224
    frameskip: int = 5
    goal_H: int = 6
    # GoalHead
    max_text_len: int = 4
    # Optimization
    batch_size: int = 32
    num_workers: int = 0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 5
    lambda_cos: float = 1.0
    alpha_proprio: float = 0.1
    device: str = "cuda:0"
    # World model module config
    wm_module_name: str = "app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds"
    wm_pretrain_kwargs: dict | None = None
    wm_data: dict | None = None
    wm_data_aug: dict | None = None
    wm_wrapper_kwargs: dict | None = None

    # Optional GoalHead config override (and for v2 from scratch)
    goalhead: dict | None = None

    # Resume
    resume: bool = True


class _Pairs(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def _collate(batch):
    out = {}
    for k in batch[0].keys():
        if isinstance(batch[0][k], str):
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


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


def main(args, resume_preempt: bool = False):
    args = dict(args)
    args.pop("app", None)
    cfg = FineTuneConfig(**args)
    if cfg.wm_data_aug is None:
        cfg.wm_data_aug = {}

    work_dir = Path(os.path.expandvars(cfg.folder))
    work_dir.mkdir(parents=True, exist_ok=True)

    resume = bool(resume_preempt) or bool(cfg.resume)
    pairs_cache = work_dir / "pairs_cache.pt"

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    preprocessor = _build_preprocessor(cfg.img_size, cfg.wm_data_aug)

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

    # Task list
    tasks = TASK_SET.get(cfg.task, [cfg.task])
    multitask = cfg.task in TASK_SET

    # Optional: load GoalHead checkpoint + tokenizer
    ckpt = None
    tok = None
    if (
        cfg.goalhead_checkpoint is not None
        and str(cfg.goalhead_checkpoint).strip() != ""
    ):
        ckpt_path = os.path.expandvars(str(cfg.goalhead_checkpoint))
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            tok = TaskTokenizer.from_vocab(ckpt["tokenizer_vocab"])

    if tok is None:
        tok = TaskTokenizer.from_tasks(tasks)
    # Infer dims
    dummy_obs = {
        "visual": torch.zeros(1, 1, 3, cfg.img_size, cfg.img_size, dtype=torch.uint8),
        "proprio": torch.zeros(1, 1, 4, dtype=torch.float32),
    }
    with torch.no_grad():
        init_enc0 = wm.encode(dummy_obs, act=False)
    visual_dim = int(init_enc0["visual"].shape[-1])
    proprio_dim = int(init_enc0["proprio"].shape[-1])
    gh_cfg_dict = {}
    if ckpt is not None and isinstance(ckpt.get("cfg"), dict):
        gh_cfg_dict = ckpt.get("cfg", {}).get("goalhead", {})
    if cfg.goalhead is not None:
        gh_cfg_dict = {**gh_cfg_dict, **cfg.goalhead}

    kind = str(gh_cfg_dict.get("kind", "v1"))
    if kind == "v2":
        from app.plan_common.models.goal_head_v2 import GoalHeadV2, GoalHeadV2Config

        gh_cfg = GoalHeadV2Config(
            kind="v2",
            visual_dim=visual_dim,
            proprio_dim=proprio_dim,
            visual_depth=int(gh_cfg_dict.get("visual_depth", 6)),
            visual_num_heads=int(
                gh_cfg_dict.get("visual_num_heads", gh_cfg_dict.get("num_heads", 8))
            ),
            text_embed_dim=int(gh_cfg_dict.get("text_embed_dim", 256)),
            text_depth=int(gh_cfg_dict.get("text_depth", 2)),
            text_num_heads=int(gh_cfg_dict.get("text_num_heads", 4)),
            max_text_len=int(cfg.max_text_len),
            mlp_ratio=int(gh_cfg_dict.get("mlp_ratio", 4)),
            dropout=float(gh_cfg_dict.get("dropout", 0.0)),
        )
        goal_head = GoalHeadV2(gh_cfg, vocab_size=len(tok.vocab)).to(device)
    elif kind == "mixture_v1":
        from app.plan_common.models.goal_head_mixture import (
            GoalHeadMixture,
            GoalHeadMixtureConfig,
        )

        gh_cfg = GoalHeadMixtureConfig(
            kind="mixture_v1",
            mixture_k=int(gh_cfg_dict.get("mixture_k", 4)),
            prompt_len=int(gh_cfg_dict.get("prompt_len", 0)),
            component_scale=float(gh_cfg_dict.get("component_scale", 1.0)),
            prompt_scale=float(gh_cfg_dict.get("prompt_scale", 1.0)),
            visual_dim=visual_dim,
            proprio_dim=proprio_dim,
            num_heads=int(gh_cfg_dict.get("num_heads", 8)),
            depth=int(gh_cfg_dict.get("depth", 4)),
            text_embed_dim=int(gh_cfg_dict.get("text_embed_dim", 256)),
            mlp_ratio=int(gh_cfg_dict.get("mlp_ratio", 4)),
            dropout=float(gh_cfg_dict.get("dropout", 0.0)),
            lambda_div=float(gh_cfg_dict.get("lambda_div", 0.01)),
        )
        goal_head = GoalHeadMixture(gh_cfg, vocab_size=len(tok.vocab)).to(device)
    else:
        gh_cfg = GoalHeadConfig(
            visual_dim=visual_dim,
            proprio_dim=proprio_dim,
            num_heads=int(gh_cfg_dict.get("num_heads", 8)),
            depth=int(gh_cfg_dict.get("depth", 2)),
            text_embed_dim=int(gh_cfg_dict.get("text_embed_dim", 128)),
            mlp_ratio=int(gh_cfg_dict.get("mlp_ratio", 4)),
            dropout=float(gh_cfg_dict.get("dropout", 0.0)),
        )
        goal_head = GoalHead(gh_cfg, vocab_size=len(tok.vocab)).to(device)

    if ckpt is not None:
        # Allow architecture upgrade (e.g., v1 -> mixture_v1) with partial load.
        goal_head.load_state_dict(ckpt["goal_head"], strict=False)

    # Collect env-generated pairs using PlanEvaluator expert rollout.
    from omegaconf import OmegaConf

    eval_cfg = OmegaConf.create(
        {
            "frameskip": cfg.frameskip,
            "meta": {"seed": 1, "quick_debug": False},
            "logging": {"tqdm_silent": True, "optional_plots": False},
            "planner": {"planning_objective": {"alpha": cfg.alpha_proprio}},
            "task_specification": {
                "task": cfg.task,
                "obs": "rgb_state",
                "obs_concat_channels": False,
                "goal_source": "expert",
                "succ_def": "simu",
                "done_at_succ": False,
                "max_episode_steps": cfg.max_episode_steps,
                "goal_H": cfg.goal_H,
                "num_frames": 1,
                "num_proprios": 1,
                "img_size": cfg.img_size,
                "env": {
                    "with_target": True,
                    "with_velocity": True,
                    "freeze_rand_vec": False,
                },
                "multitask": multitask,
            },
            "tasks": tasks,
            "device": str(device),
            "work_dir": work_dir,
        }
    )

    # Minimal agent stub for PlanEvaluator needs: cfg, model, preprocessor
    class _Agent:
        def __init__(self, cfg, model, preprocessor):
            self.cfg = cfg
            self.model = model
            self.preprocessor = preprocessor
            self.device = model.device

    from evals.simu_env_planning.envs.init import make_env

    # NOTE: PlanEvaluator.get_goal_state_mw internally creates env_expert via make_env(env.cfg)
    # and does so for every episode. For large-scale collection this becomes very slow.
    # We reduce overhead by letting evaluator reuse a single expert env across episodes.
    env = make_env(eval_cfg)
    agent = _Agent(eval_cfg, wm, preprocessor)
    evaluator = PlanEvaluator(eval_cfg, agent)

    # Create a single persistent expert env (multitask-capable) and reuse it.
    # Creating a fresh multitask env per episode is very expensive.
    env_expert = make_env(env.cfg)
    evaluator._env_expert = env_expert

    pairs = None
    if resume and pairs_cache.exists():
        cached = torch.load(pairs_cache, map_location="cpu")
        if isinstance(cached, dict) and "pairs" in cached and "n_episodes" in cached:
            if int(cached.get("n_episodes", 0)) == int(cfg.n_episodes):
                pairs = cached["pairs"]
        elif isinstance(cached, list):
            # Backward compatible cache format (no metadata)
            pairs = cached

        if not isinstance(pairs, list) or len(pairs) == 0:
            pairs = None
        else:
            logger.info(f"Loaded cached pairs from {pairs_cache} (n={len(pairs)})")

    if pairs is None:
        pairs = []
        for ep in range(cfg.n_episodes):
            ep_seed = (12345 * 12345 + ep * 12345) % (2**32 - 2)
            task_idx = ep % len(tasks)
            env.reset(seed=ep_seed, task_idx=task_idx)
            init_obs, goal_obs, _, _ = evaluator.set_episode(
                eval_cfg, agent, env, ep_seed, task_idx=task_idx
            )
            # Ensure shape [T,C,H,W] for obs, but we only want first frame.
            i0 = init_obs["visual"][0].to(torch.uint8)
            ig = goal_obs["visual"][0].to(torch.uint8)
            p0 = init_obs["proprio"][0].to(torch.float32)
            pg = goal_obs["proprio"][0].to(torch.float32)
            pairs.append(
                {
                    "i0_rgb": i0.cpu(),
                    "ig_rgb": ig.cpu(),
                    "p0": p0.cpu(),
                    "pg": pg.cpu(),
                    "task": tasks[task_idx],
                }
            )
        torch.save(
            {"n_episodes": int(cfg.n_episodes), "tasks": tasks, "pairs": pairs},
            pairs_cache,
        )
        logger.info(f"Saved cached pairs to {pairs_cache} (n={len(pairs)})")

    loader = DataLoader(
        _Pairs(pairs),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=_collate,
        drop_last=True,
    )

    opt = torch.optim.AdamW(
        goal_head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    goal_head.train()
    last_losses = None
    for epoch in range(cfg.epochs):
        for it, batch in enumerate(loader):
            i0 = batch["i0_rgb"].to(device, non_blocking=True).unsqueeze(1)
            ig = batch["ig_rgb"].to(device, non_blocking=True).unsqueeze(1)
            p0 = batch["p0"].to(device, non_blocking=True).unsqueeze(1)
            pg = batch["pg"].to(device, non_blocking=True).unsqueeze(1)
            text_ids = torch.tensor(
                tok.batch_encode(batch["task"], max_len=cfg.max_text_len),
                dtype=torch.long,
                device=device,
            )
            with torch.no_grad():
                init_enc = wm.encode({"visual": i0, "proprio": p0}, act=False)
                goal_enc = wm.encode({"visual": ig, "proprio": pg}, act=False)
            pred = goal_head(init_enc, text_ids=text_ids)
            if kind == "mixture_v1":
                from app.plan_common.models.goal_head_mixture import (
                    goalhead_mixture_loss,
                )

                tau = gh_cfg_dict.get("tau")
                tau_start = gh_cfg_dict.get("tau_start")
                tau_end = gh_cfg_dict.get("tau_end")
                if tau_start is not None and tau_end is not None and cfg.epochs > 1:
                    frac = float(epoch) / float(cfg.epochs - 1)
                    tau = float(tau_start) + (float(tau_end) - float(tau_start)) * frac
                if tau is None:
                    tau = 0.5

                losses = goalhead_mixture_loss(
                    pred,
                    goal_enc,
                    lambda_cos=cfg.lambda_cos,
                    alpha_proprio=cfg.alpha_proprio,
                    lambda_div=float(gh_cfg_dict.get("lambda_div", 0.01)),
                    tau=float(tau),
                    lambda_ent=float(gh_cfg_dict.get("lambda_ent", 0.0)),
                    lambda_balance=float(gh_cfg_dict.get("lambda_balance", 0.0)),
                    lambda_orth=float(gh_cfg_dict.get("lambda_orth", 0.0)),
                    to_visual_weights=goal_head.get_to_visual_weight_tensor()
                    if hasattr(goal_head, "get_to_visual_weight_tensor")
                    else None,
                    lambda_comp=float(gh_cfg_dict.get("lambda_comp", 0.0)),
                    component_embed=getattr(goal_head, "component_embed", None),
                    lambda_prompt=float(gh_cfg_dict.get("lambda_prompt", 0.0)),
                    component_prompt=getattr(goal_head, "component_prompt", None),
                    gumbel_noise=float(gh_cfg_dict.get("gumbel_noise", 0.0)),
                    lambda_gate=float(gh_cfg_dict.get("lambda_gate", 0.0)),
                )
            else:
                losses = goalhead_loss(
                    pred,
                    goal_enc,
                    lambda_cos=cfg.lambda_cos,
                    alpha_proprio=cfg.alpha_proprio,
                )
            last_losses = losses
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            opt.step()
        if last_losses is not None:
            msg = f"epoch {epoch}: loss={float(last_losses['loss']):.4f}"
            l2v = last_losses.get("l2_visual")
            if l2v is None:
                l2v = last_losses.get("l2v")
            if l2v is not None:
                msg += f" l2v={float(l2v):.4f}"
            if "div" in last_losses:
                msg += f" div={float(last_losses['div']):.4f}"
            if "balance" in last_losses:
                msg += f" balance={float(last_losses['balance']):.4f}"
            if "entropy" in last_losses:
                msg += f" ent={float(last_losses['entropy']):.4f}"
            if "orth" in last_losses:
                msg += f" orth={float(last_losses['orth']):.4f}"
            if "w_bar_min" in last_losses and "w_bar_max" in last_losses:
                msg += (
                    f" wbar=[{float(last_losses['w_bar_min']):.3f},"
                    f"{float(last_losses['w_bar_max']):.3f}]"
                )
            if "w_max" in last_losses:
                msg += f" wmax={float(last_losses['w_max']):.3f}"
            if "best_k" in last_losses:
                msg += f" best_k_mean={float(last_losses['best_k']):.4f}"
            logger.info(msg)

    # Serialize minimal cfg for eval-time reconstruction.
    gh_cfg_out = {
        "kind": kind,
        "mlp_ratio": gh_cfg.mlp_ratio,
        "dropout": gh_cfg.dropout,
        "text_embed_dim": gh_cfg.text_embed_dim,
    }
    if kind == "v2":
        gh_cfg_out.update(
            {
                "visual_depth": getattr(gh_cfg, "visual_depth"),
                "visual_num_heads": getattr(gh_cfg, "visual_num_heads"),
                "text_depth": getattr(gh_cfg, "text_depth"),
                "text_num_heads": getattr(gh_cfg, "text_num_heads"),
            }
        )
    else:
        gh_cfg_out.update(
            {
                "num_heads": getattr(gh_cfg, "num_heads"),
                "depth": getattr(gh_cfg, "depth"),
            }
        )

    if kind == "mixture_v1":
        gh_cfg_out.update(
            {
                "mixture_k": getattr(gh_cfg, "mixture_k"),
                "lambda_div": float(getattr(gh_cfg, "lambda_div", 0.01)),
                "prompt_len": int(getattr(gh_cfg, "prompt_len", 0)),
                "component_scale": float(getattr(gh_cfg, "component_scale", 1.0)),
                "prompt_scale": float(getattr(gh_cfg, "prompt_scale", 1.0)),
            }
        )
        if cfg.goalhead is not None:
            if "tau" in cfg.goalhead:
                gh_cfg_out["tau"] = float(cfg.goalhead["tau"])
            if "tau_start" in cfg.goalhead and "tau_end" in cfg.goalhead:
                gh_cfg_out["tau_start"] = float(cfg.goalhead["tau_start"])
                gh_cfg_out["tau_end"] = float(cfg.goalhead["tau_end"])
            for k in ["lambda_ent", "lambda_balance", "lambda_orth"]:
                if k in cfg.goalhead:
                    gh_cfg_out[k] = float(cfg.goalhead[k])
            for k in ["lambda_comp", "lambda_prompt"]:
                if k in cfg.goalhead:
                    gh_cfg_out[k] = float(cfg.goalhead[k])
            if "gumbel_noise" in cfg.goalhead:
                gh_cfg_out["gumbel_noise"] = float(cfg.goalhead["gumbel_noise"])
            if "lambda_gate" in cfg.goalhead:
                gh_cfg_out["lambda_gate"] = float(cfg.goalhead["lambda_gate"])

    out = {
        "epoch": cfg.epochs - 1,
        "goal_head": goal_head.state_dict(),
        "opt": opt.state_dict(),
        "tokenizer_vocab": tok.vocab,
        # Keep a minimal nested cfg to allow eval to reconstruct the module.
        "cfg": {"goalhead": gh_cfg_out},
    }
    out_path = work_dir / "goalhead_finetuned.pt"
    torch.save(out, out_path)
    logger.info(f"Saved finetuned GoalHead to {out_path}")
    return {"checkpoint": str(out_path), "work_dir": str(work_dir)}
