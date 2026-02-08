from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict


@dataclass
class GoalHeadConfig:
    visual_dim: int
    proprio_dim: int
    num_heads: int = 8
    depth: int = 2
    text_embed_dim: int = 128
    mlp_ratio: int = 4
    dropout: float = 0.0


class _CrossAttnBlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0
    ):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # q: [B, N, D], kv: [B, L, D]
        q2 = self.ln_q(q)
        kv2 = self.ln_kv(kv)
        attn_out, _ = self.attn(
            q2, kv2, kv2, key_padding_mask=key_padding_mask, need_weights=False
        )
        q = q + attn_out
        q = q + self.mlp(self.ln_mlp(q))
        return q


class GoalHead(nn.Module):
    """Predict goal encoding from (init encoding, task tokens).

    Query = visual tokens, Key/Value = text tokens (as requested in TASK.md).
    """

    def __init__(self, cfg: GoalHeadConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.text_embed = nn.Embedding(vocab_size, cfg.text_embed_dim)
        self.text_proj = nn.Linear(cfg.text_embed_dim, cfg.visual_dim)
        self.blocks = nn.ModuleList(
            [
                _CrossAttnBlock(
                    cfg.visual_dim,
                    cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.to_proprio = nn.Linear(cfg.visual_dim, cfg.proprio_dim)

    def forward(self, init_enc: TensorDict, text_ids: torch.Tensor) -> TensorDict:
        """Args:
        init_enc: TensorDict with keys:
            - visual: [B, T, V, H, W, D]
            - proprio: [B, T, HW, P] (optional)
        text_ids: [B, L] token ids
        """
        visual = init_enc["visual"]
        if visual.ndim != 6:
            raise ValueError(f"Expected init_enc['visual'] ndim=6, got {visual.shape}")
        b, t, v, h, w, d = visual.shape
        if t != 1 or v != 1:
            # Training/eval here always uses 1-frame encodings.
            visual = visual[:, :1, :1]
            b, t, v, h, w, d = visual.shape
        q = visual[:, 0, 0].reshape(b, h * w, d)

        txt = self.text_embed(text_ids)
        kv = self.text_proj(txt)
        key_padding_mask = text_ids.eq(0)  # <pad>
        for blk in self.blocks:
            q = blk(q, kv, key_padding_mask=key_padding_mask)

        pred_visual = q.reshape(b, 1, 1, h, w, d)
        pred_proprio = self.to_proprio(q).reshape(b, 1, h * w, self.cfg.proprio_dim)
        return TensorDict(
            {"visual": pred_visual, "proprio": pred_proprio}, batch_size=[b]
        )


def goalhead_loss(
    pred: TensorDict,
    target: TensorDict,
    lambda_cos: float = 1.0,
    alpha_proprio: float = 1.0,
) -> dict:
    """Compute combined L2 + cosine losses (visual + proprio)."""

    def _flat(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)

    pred_v = _flat(pred["visual"])
    tgt_v = _flat(target["visual"])
    l2_v = F.mse_loss(pred_v, tgt_v)
    cos_v = 1.0 - F.cosine_similarity(pred_v, tgt_v, dim=-1).mean()

    pred_p = pred.get("proprio")
    tgt_p = target.get("proprio")
    if pred_p is None or tgt_p is None:
        l2_p = pred_v.new_tensor(0.0)
        cos_p = pred_v.new_tensor(0.0)
        alpha_proprio = 0.0
    else:
        pred_p = _flat(pred_p)
        tgt_p = _flat(tgt_p)
        l2_p = F.mse_loss(pred_p, tgt_p)
        cos_p = 1.0 - F.cosine_similarity(pred_p, tgt_p, dim=-1).mean()

    loss = l2_v + alpha_proprio * l2_p + lambda_cos * (cos_v + alpha_proprio * cos_p)
    return {
        "loss": loss,
        "l2_visual": l2_v,
        "cos_visual": cos_v,
        "l2_proprio": l2_p,
        "cos_proprio": cos_p,
    }
