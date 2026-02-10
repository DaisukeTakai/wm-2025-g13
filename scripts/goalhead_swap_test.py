from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from app.plan_common.datasets import get_data_stats
from app.plan_common.datasets.preprocessor import Preprocessor
from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
from app.plan_common.models.goal_head_mixture import (
    GoalHeadMixture,
    GoalHeadMixtureConfig,
)
from app.plan_common.text.task_tokenizer import TaskTokenizer
from evals.simu_env_planning.eval import init_module as init_encpred_module


def _build_preprocessor(img_size: int = 224) -> Preprocessor:
    stats = get_data_stats("metaworld")
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


def _wm_pretrain_kwargs() -> dict:
    # Mirror configs used in mw planning evals.
    return {
        "grid_size": 16,
        "tubelet_size_enc": 1,
        "use_activation_checkpointing": False,
        "action_conditioning": "token",
        "proprio_encoding": "feature",
        "num_frames_pred": 4,
        "visual_encoder": {
            "enc_type": "dino",
            "enc_version": "dinov2_vits14",
            "pretrain_enc_path": None,
            "pretrain_enc_ckpt_key": "target_encoder",
            "embed_dim": 384,
            "enc_use_rope": None,
            "enc_name": None,
            "use_sdpa_enc": True,
            "num_frames_enc": 16,
            "uniform_power": True,
        },
        "action_encoder": {
            "action_tokens": 1,
            "action_emb_dim": 0,
            "act_mlp": False,
            "action_encoder_inpred": True,
        },
        "proprio_encoder": {
            "proprio_tokens": 0,
            "proprio_emb_dim": 16,
            "prop_mlp": False,
            "proprio_encoder_inpred": False,
        },
        "predictor": {
            "tubelet_size": 1,
            "pred_num_heads": 16,
            "pred_depth": 6,
            "pred_embed_dim": 384,
            "pred_use_extrinsics": False,
            "pred_type": "AdaLN",
            "act_pred_projector": False,
            "use_SiLU": False,
            "use_rope": True,
        },
        "wm_encoding": {
            "batchify_video": True,
            "dup_image": False,
            "normalize_reps": False,
        },
        "attn": {"local_window_time": 3, "local_window_h": -1, "local_window_w": -1},
        "heads_cfg": {"architectures": {}},
    }


def _best_of_k_mse(pred_kf: torch.Tensor, tgt_f: torch.Tensor) -> float:
    # pred_kf: [K,F], tgt_f: [F]
    return float(((pred_kf - tgt_f[None, :]) ** 2).mean(dim=-1).min().detach().cpu())


def _set_min_dist(a_kf: torch.Tensor, b_kf: torch.Tensor) -> float:
    # a_kf,b_kf: [K,F]
    d = torch.cdist(a_kf, b_kf)
    return float(d.min().detach().cpu())


def _diversity_cos(a_kf: torch.Tensor) -> float:
    # mean off-diagonal cosine similarity (lower is more diverse)
    a = a_kf / (a_kf.norm(dim=-1, keepdim=True) + 1e-8)
    sim = a @ a.t()
    k = int(sim.shape[0])
    mask = ~torch.eye(k, dtype=torch.bool, device=sim.device)
    return float(sim[mask].mean().detach().cpu())


def _td_visual_to_kbf(td_visual: torch.Tensor) -> torch.Tensor:
    # Expected mixture visual shapes: [K,B,1,1,H,W,D] (current) or [K,B,...]
    v = td_visual
    # Squeeze singleton dims after K,B
    while v.ndim > 3 and v.shape[2] == 1:
        v = v.squeeze(2)
    return v.reshape(v.shape[0], v.shape[1], -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goalhead-ckpt", required=True)
    ap.add_argument("--pairs-cache", required=True)
    ap.add_argument("--task-a", default="mw-reach-wall")
    ap.add_argument("--task-b", default="mw-reach")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ckpt_path = os.path.expandvars(args.goalhead_ckpt)
    pairs_path = os.path.expandvars(args.pairs_cache)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    mix = torch.load(ckpt_path, map_location="cpu")
    tok = TaskTokenizer.from_vocab(mix["tokenizer_vocab"])
    gh_cfg_dict = mix.get("cfg", {}).get("goalhead", {})
    if str(gh_cfg_dict.get("kind")) != "mixture_v1":
        raise ValueError(
            f"Expected mixture_v1 checkpoint, got kind={gh_cfg_dict.get('kind')}"
        )

    pairs_obj = torch.load(pairs_path, map_location="cpu")
    pairs = (
        pairs_obj["pairs"]
        if isinstance(pairs_obj, dict) and "pairs" in pairs_obj
        else pairs_obj
    )
    if not isinstance(pairs, list) or len(pairs) == 0:
        raise ValueError(f"Invalid pairs cache: {pairs_path}")

    pre = _build_preprocessor(img_size=224)

    wm = init_encpred_module(
        folder="/tmp",
        checkpoint="https://dl.fbaipublicfiles.com/jepa-wms/mw_jepa-wm.pth.tar",
        module_name="app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds",
        model_kwargs=_wm_pretrain_kwargs(),
        device=device,
        cfgs_data={
            "dataset_type": "custom",
            "datasets": ["METAWORLD_HF"],
            "img_size": 224,
            "custom": {"frameskip": 5, "action_skip": 1, "state_skip": 1},
        },
        wrapper_kwargs={"ctxt_window": 2},
        action_dim=4,
        proprio_dim=4,
        preprocessor=pre,
    )
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    gh = GoalHeadMixture(
        GoalHeadMixtureConfig(
            kind="mixture_v1",
            mixture_k=int(gh_cfg_dict.get("mixture_k", 4)),
            prompt_len=int(gh_cfg_dict.get("prompt_len", 0)),
            component_scale=float(gh_cfg_dict.get("component_scale", 1.0)),
            prompt_scale=float(gh_cfg_dict.get("prompt_scale", 1.0)),
            visual_dim=384,
            proprio_dim=16,
            num_heads=int(gh_cfg_dict.get("num_heads", 8)),
            depth=int(gh_cfg_dict.get("depth", 4)),
            text_embed_dim=int(gh_cfg_dict.get("text_embed_dim", 256)),
            mlp_ratio=int(gh_cfg_dict.get("mlp_ratio", 4)),
            dropout=float(gh_cfg_dict.get("dropout", 0.0)),
            lambda_div=float(gh_cfg_dict.get("lambda_div", 0.01)),
        ),
        vocab_size=len(tok.vocab),
    ).to(device)
    # Allow running against older checkpoints that might not have gating params.
    gh.load_state_dict(mix["goal_head"], strict=False)
    gh.eval()

    def _text_ids(task: str, max_len: int = 4) -> torch.Tensor:
        return torch.tensor(
            [tok.encode(task, max_len=max_len)], dtype=torch.long, device=device
        )

    ids_a = _text_ids(args.task_a)
    ids_b = _text_ids(args.task_b)

    swap_dists = []
    div_a = []
    div_b = []
    better_a = 0
    n = 0

    with torch.no_grad():
        for it in pairs:
            if it.get("task") != args.task_a:
                continue

            i0 = it["i0_rgb"].unsqueeze(0).unsqueeze(0).to(device)
            p0 = it["p0"].unsqueeze(0).unsqueeze(0).to(device)
            ig = it["ig_rgb"].unsqueeze(0).unsqueeze(0).to(device)
            pg = it["pg"].unsqueeze(0).unsqueeze(0).to(device)

            init_enc = wm.encode({"visual": i0, "proprio": p0}, act=False)
            goal_enc = wm.encode({"visual": ig, "proprio": pg}, act=False)

            out_a = gh(init_enc, text_ids=ids_a)
            out_b = gh(init_enc, text_ids=ids_b)

            a_kbf = _td_visual_to_kbf(out_a["visual"])  # [K,B,F]
            b_kbf = _td_visual_to_kbf(out_b["visual"])

            a_kf = a_kbf[:, 0]
            b_kf = b_kbf[:, 0]
            tgt_f = goal_enc["visual"][0].reshape(-1)

            swap_dists.append(_set_min_dist(a_kf, b_kf))
            div_a.append(_diversity_cos(a_kf))
            div_b.append(_diversity_cos(b_kf))
            if _best_of_k_mse(a_kf, tgt_f) < _best_of_k_mse(b_kf, tgt_f):
                better_a += 1

            n += 1
            if n >= args.n:
                break

    print("N", n)
    print(
        "swap_minset_l2_mean",
        float(np.mean(swap_dists)) if swap_dists else float("nan"),
    )
    print(
        "swap_minset_l2_std", float(np.std(swap_dists)) if swap_dists else float("nan")
    )
    print("div_cos_a_mean", float(np.mean(div_a)) if div_a else float("nan"))
    print("div_cos_b_mean", float(np.mean(div_b)) if div_b else float("nan"))
    print("a_beats_b_rate", better_a / max(n, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
