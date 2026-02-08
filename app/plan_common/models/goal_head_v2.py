from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from tensordict import TensorDict


def _build_2d_sincos_pos_embed(h: int, w: int, dim: int, device) -> torch.Tensor:
    """2D sin-cos positional embedding.

    Returns: [1, h*w, dim]
    """
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4 for 2D sincos, got dim={dim}")
    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    pos = torch.stack([yy, xx], dim=-1).reshape(-1, 2)  # [N,2]

    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (dim // 4)))

    out = []
    for i in range(2):
        v = pos[:, i : i + 1] * omega.unsqueeze(0)  # [N, dim/4]
        out.append(torch.sin(v))
        out.append(torch.cos(v))
    emb = torch.cat(out, dim=-1)  # [N, dim]
    return emb.unsqueeze(0)


@dataclass
class GoalHeadV2Config:
    kind: str = "v2"

    visual_dim: int = 384
    proprio_dim: int = 16

    # Visual stream
    visual_depth: int = 6
    visual_num_heads: int = 8

    # Text stream
    text_embed_dim: int = 256
    text_depth: int = 2
    text_num_heads: int = 4
    max_text_len: int = 4

    mlp_ratio: int = 4
    dropout: float = 0.0


class _MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SelfAttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.ln1(x)
        out, _ = self.attn(x2, x2, x2, need_weights=False)
        x = x + out
        x = x + self.mlp(self.ln2(x))
        return x


class _CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q2 = self.ln_q(q)
        kv2 = self.ln_kv(kv)
        out, _ = self.attn(
            q2, kv2, kv2, key_padding_mask=key_padding_mask, need_weights=False
        )
        q = q + out
        q = q + self.mlp(self.ln_mlp(q))
        return q


class _VisualBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.self_attn = _SelfAttnBlock(
            dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )
        self.cross_attn = _CrossAttnBlock(
            dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q = self.self_attn(q)
        q = self.cross_attn(q, kv, key_padding_mask=key_padding_mask)
        return q


class GoalHeadV2(nn.Module):
    """2-stream GoalHead with text encoder + visual self-attn + cross-attn.

    Output stays compatible with planner objectives (TensorDict visual/proprio).
    """

    def __init__(self, cfg: GoalHeadV2Config, vocab_size: int):
        super().__init__()
        self.cfg = cfg

        # Text
        self.text_embed = nn.Embedding(vocab_size, cfg.text_embed_dim)
        self.text_pos = nn.Embedding(cfg.max_text_len, cfg.text_embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.text_embed_dim,
            nhead=cfg.text_num_heads,
            dim_feedforward=cfg.text_embed_dim * cfg.mlp_ratio,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.text_depth)
        self.text_to_visual = nn.Linear(cfg.text_embed_dim, cfg.visual_dim)

        # Visual
        self.blocks = nn.ModuleList(
            [
                _VisualBlock(
                    cfg.visual_dim,
                    cfg.visual_num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.visual_depth)
            ]
        )
        self.to_proprio = nn.Linear(cfg.visual_dim, cfg.proprio_dim)

    def forward(self, init_enc: TensorDict, text_ids: torch.Tensor) -> TensorDict:
        visual = init_enc["visual"]
        if visual.ndim != 6:
            raise ValueError(f"Expected init_enc['visual'] ndim=6, got {visual.shape}")
        b, t, v, h, w, d = visual.shape
        if t != 1 or v != 1:
            visual = visual[:, :1, :1]
            b, t, v, h, w, d = visual.shape

        # Visual tokens + 2D pos
        q = visual[:, 0, 0].reshape(b, h * w, d)
        q = q + _build_2d_sincos_pos_embed(h=h, w=w, dim=d, device=q.device)

        # Text tokens + pos + encoder
        if text_ids.shape[1] > self.cfg.max_text_len:
            text_ids = text_ids[:, : self.cfg.max_text_len]
        l = int(text_ids.shape[1])
        pos_ids = torch.arange(l, device=text_ids.device).unsqueeze(0).expand(b, l)
        txt = self.text_embed(text_ids) + self.text_pos(pos_ids)
        key_padding_mask = text_ids.eq(0)
        txt = self.text_encoder(txt, src_key_padding_mask=key_padding_mask)
        kv = self.text_to_visual(txt)

        for blk in self.blocks:
            q = blk(q, kv, key_padding_mask=key_padding_mask)

        pred_visual = q.reshape(b, 1, 1, h, w, d)
        pred_proprio = self.to_proprio(q).reshape(b, 1, h * w, self.cfg.proprio_dim)
        return TensorDict(
            {"visual": pred_visual, "proprio": pred_proprio}, batch_size=[b]
        )
