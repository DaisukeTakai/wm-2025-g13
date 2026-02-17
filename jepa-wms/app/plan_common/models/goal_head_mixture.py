from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict


@dataclass
class GoalHeadMixtureConfig:
    kind: str = "mixture_v1"
    mixture_k: int = 4

    # Optional per-component prompt tokens (helps avoid component collapse)
    prompt_len: int = 0

    visual_dim: int = 384
    proprio_dim: int = 16
    num_heads: int = 8
    depth: int = 4
    text_embed_dim: int = 256
    mlp_ratio: int = 4
    dropout: float = 0.0

    # diversity regularization (training-side)
    lambda_div: float = 0.01

    # Scaling applied to per-component conditioning to break symmetry.
    component_scale: float = 1.0
    prompt_scale: float = 1.0


class _CrossAttnBlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0
    ):
        super().__init__()
        # Self-attention over q tokens lets prompt/component tokens influence
        # the visual query token set (important for mixture separation).
        self.ln_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
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
        comp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attend among q tokens first.
        q2 = self.ln_self(q)
        if comp is not None:
            q2 = q2 + comp
        out, _ = self.self_attn(q2, q2, q2, need_weights=False)
        q = q + out

        q2 = self.ln_q(q)
        # Inject per-component embedding *after* LayerNorm so it isn't washed out.
        # comp is expected to be broadcastable to [B, N, D] (typically [B, 1, D]).
        if comp is not None:
            q2 = q2 + comp
        kv2 = self.ln_kv(kv)
        out, _ = self.attn(
            q2, kv2, kv2, key_padding_mask=key_padding_mask, need_weights=False
        )
        q = q + out
        q = q + self.mlp(self.ln_mlp(q))
        return q


class GoalHeadMixture(nn.Module):
    """Mixture GoalHead that outputs K candidate goal encodings.

    It reuses the same image query tokens but adds a learned component embedding per
    mixture element to allow multiple modes.
    """

    def __init__(self, cfg: GoalHeadMixtureConfig, vocab_size: int):
        super().__init__()
        if cfg.mixture_k <= 1:
            raise ValueError("mixture_k must be > 1")
        self.cfg = cfg

        self.text_embed = nn.Embedding(vocab_size, cfg.text_embed_dim)
        self.text_proj = nn.Linear(cfg.text_embed_dim, cfg.visual_dim)

        # Learned per-component embeddings (added to visual query tokens)
        self.component_embed = nn.Parameter(
            torch.randn(cfg.mixture_k, cfg.visual_dim) * 0.02
        )

        # Optional per-component prompt tokens prepended to visual tokens.
        self.prompt_len = int(getattr(cfg, "prompt_len", 0) or 0)
        if self.prompt_len > 0:
            self.component_prompt = nn.Parameter(
                torch.randn(cfg.mixture_k, self.prompt_len, cfg.visual_dim) * 0.02
            )

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

        # Predict mixture logits p(k | I0, text) from per-component features.
        # This enables weighted objectives (no min-over-K oracle during planning).
        self.gate = nn.Sequential(
            nn.LayerNorm(cfg.visual_dim),
            nn.Linear(cfg.visual_dim, 1),
        )

        # Multi-head outputs to force candidate diversity.
        self.to_visual = nn.ModuleList(
            [nn.Linear(cfg.visual_dim, cfg.visual_dim) for _ in range(cfg.mixture_k)]
        )
        self.to_proprio = nn.ModuleList(
            [nn.Linear(cfg.visual_dim, cfg.proprio_dim) for _ in range(cfg.mixture_k)]
        )

        # Initialize visual heads close to identity, with small per-head noise.
        for i, layer in enumerate(self.to_visual):
            nn.init.eye_(layer.weight)
            layer.weight.data += 1e-3 * torch.randn_like(layer.weight) * (i + 1)
            nn.init.zeros_(layer.bias)

    def get_to_visual_weight_tensor(self) -> torch.Tensor:
        """Returns stacked weights for optional regularization.

        Shape: [K, out_dim, in_dim]
        """
        return torch.stack([m.weight for m in self.to_visual], dim=0)

    def forward(self, init_enc: TensorDict, text_ids: torch.Tensor) -> TensorDict:
        visual = init_enc["visual"]
        if visual.ndim != 6:
            raise ValueError(f"Expected init_enc['visual'] ndim=6, got {visual.shape}")
        b, t, v, h, w, d = visual.shape
        if t != 1 or v != 1:
            visual = visual[:, :1, :1]
            b, t, v, h, w, d = visual.shape

        q0 = visual[:, 0, 0].reshape(b, h * w, d)  # [B,N,D]

        txt = self.text_embed(text_ids)
        kv = self.text_proj(txt)
        key_padding_mask = text_ids.eq(0)

        # Expand to K candidates
        k = self.cfg.mixture_k
        q = q0.unsqueeze(0).expand(k, b, h * w, d).contiguous()  # [K,B,N,D]
        comp_scale = float(getattr(self.cfg, "component_scale", 1.0))
        q = q + comp_scale * self.component_embed[:, None, None, :]

        # Optional: prepend per-component prompt tokens to strengthen separation.
        prompt_len = int(self.prompt_len)
        if prompt_len > 0:
            prompt_scale = float(getattr(self.cfg, "prompt_scale", 1.0))
            prompts = (prompt_scale * self.component_prompt[:, None, :, :]).expand(
                k, b, prompt_len, d
            )
            q = torch.cat([prompts, q], dim=2)  # [K,B,P+N,D]

        # Component injection tensor for post-LN injection inside blocks.
        comp = (
            (comp_scale * self.component_embed)[:, None, :]
            .expand(k, b, d)
            .reshape(k * b, 1, d)
            .contiguous()
        )

        # Flatten K into batch for MHA
        q = q.reshape(k * b, q.shape[2], d)
        kv = (
            kv.unsqueeze(0)
            .expand(k, b, kv.shape[1], kv.shape[2])
            .reshape(k * b, kv.shape[1], d)
        )
        if key_padding_mask is not None:
            key_padding_mask = (
                key_padding_mask.unsqueeze(0)
                .expand(k, b, key_padding_mask.shape[1])
                .reshape(k * b, -1)
            )

        for blk in self.blocks:
            q = blk(q, kv, key_padding_mask=key_padding_mask, comp=comp)

        # Reshape back to [K,B,NT,D].
        q = q.reshape(k, b, q.shape[1], d)

        # Gating logits for mixture weights. If prompts exist, use prompts only.
        # Otherwise use a global average pooling of all tokens.
        if prompt_len > 0:
            gate_feat = q[:, :, :prompt_len, :].mean(dim=2)  # [K,B,D]
        else:
            gate_feat = q.mean(dim=2)  # [K,B,D]
        mix_logits = self.gate(gate_feat).squeeze(-1)  # [K,B]

        # Apply per-component heads (ignore prompts in the output encoding).
        if prompt_len > 0:
            q = q[:, :, prompt_len:, :]
        pred_visual = []
        pred_proprio = []
        for i in range(k):
            qi = self.to_visual[i](q[i])
            pred_visual.append(qi.reshape(b, 1, 1, h, w, d))
            pred_proprio.append(
                self.to_proprio[i](qi).reshape(b, 1, h * w, self.cfg.proprio_dim)
            )
        pred_visual = torch.stack(pred_visual, dim=0)
        pred_proprio = torch.stack(pred_proprio, dim=0)
        return TensorDict(
            {"visual": pred_visual, "proprio": pred_proprio, "mix_logits": mix_logits},
            batch_size=[k, b],
        )


def goalhead_mixture_loss(
    pred: TensorDict,
    target: TensorDict,
    lambda_cos: float = 1.0,
    alpha_proprio: float = 1.0,
    lambda_div: float = 0.01,
    tau: float = 0.5,
    lambda_ent: float = 0.0,
    lambda_balance: float = 0.0,
    lambda_orth: float = 0.0,
    to_visual_weights: Optional[torch.Tensor] = None,
    lambda_comp: float = 0.0,
    component_embed: Optional[torch.Tensor] = None,
    lambda_prompt: float = 0.0,
    component_prompt: Optional[torch.Tensor] = None,
    gumbel_noise: float = 0.0,
    lambda_gate: float = 0.0,
    eps: float = 1e-8,
) -> dict:
    """Soft best-of-K loss + diversity regularization.

    Hard argmin (best-of-K) often collapses to identical components because
    only one component receives gradient. Softmin keeps gradient on all K.
    """

    pred_v = pred["visual"].reshape(
        pred["visual"].shape[0], pred["visual"].shape[1], -1
    )  # [K,B,F]
    tgt_v = target["visual"].reshape(1, target["visual"].shape[0], -1)  # [1,B,F]
    l2_v = (pred_v - tgt_v).pow(2).mean(-1)  # [K,B]
    cos_v = 1.0 - F.cosine_similarity(pred_v, tgt_v.expand_as(pred_v), dim=-1)  # [K,B]

    pred_p = pred["proprio"].reshape(
        pred["proprio"].shape[0], pred["proprio"].shape[1], -1
    )
    tgt_p = target["proprio"].reshape(1, target["proprio"].shape[0], -1)
    l2_p = (pred_p - tgt_p).pow(2).mean(-1)
    cos_p = 1.0 - F.cosine_similarity(pred_p, tgt_p.expand_as(pred_p), dim=-1)

    per_k = (
        l2_v + alpha_proprio * l2_p + lambda_cos * (cos_v + alpha_proprio * cos_p)
    )  # [K,B]
    # Responsibilities r_k from ground-truth matching (EM-style).
    # This keeps gradients on all components and avoids argmin-only updates.
    tau = float(tau)
    if not (tau > 0.0):
        tau = 1e-6
    resp_logits = -per_k / tau
    gn = float(gumbel_noise)
    if gn > 0.0:
        # Break symmetry/ties during training: Gumbel noise on responsibilities.
        u = torch.rand_like(resp_logits).clamp_(1e-6, 1.0 - 1e-6)
        g = -torch.log(-torch.log(u))
        resp_logits = resp_logits + gn * g
    r = torch.softmax(resp_logits, dim=0)  # [K,B]

    # Data term: expected per-k loss under responsibilities.
    data = (r * per_k).sum(dim=0)  # [B]

    # Gating loss: match predicted mixture weights to responsibilities.
    gate_kl = pred_v.new_tensor(0.0)
    w_pred = None
    if isinstance(pred, TensorDict) and ("mix_logits" in pred.keys()):
        mix_logits = pred["mix_logits"]  # [K,B]
        log_w = torch.log_softmax(mix_logits, dim=0)
        w_pred = torch.softmax(mix_logits, dim=0)
        log_r = (r + float(eps)).log()
        gate_kl = (r * (log_r - log_w)).sum(dim=0).mean()

    best, best_k = per_k.min(dim=0)  # [B] (monitor only)

    # Diversity: penalize similarity between candidate visual predictions (batch-mean)
    # Use cosine similarity on normalized predictions.
    pv = F.normalize(pred_v, dim=-1)
    sim = torch.einsum("kbf,jbf->kjb", pv, pv)  # [K,K,B]
    # exclude diagonal (boolean indexing does not broadcast)
    k = pred_v.shape[0]
    mask = ~torch.eye(k, dtype=torch.bool, device=pred_v.device)  # [K,K]
    div = sim[mask].mean() if mask.any() else sim.new_tensor(0.0)

    # Entropy of responsibilities (diagnostic).
    entropy = -(r * (r.clamp_min(1e-9).log())).sum(dim=0).mean()  # scalar

    # Encourage using multiple components (avoid one-component collapse).
    # Balance: batch-average weights close to uniform.
    w_bar = r.mean(dim=1)  # [K]
    balance = (w_bar - (1.0 / float(k))).pow(2).mean()

    # Encourage separation in component parameters (optional, tends to be stable).
    comp_div = sim.new_tensor(0.0)
    if component_embed is not None and float(lambda_comp) > 0.0:
        ce = F.normalize(component_embed.reshape(k, -1), dim=-1)
        sim_ce = ce @ ce.t()
        comp_div = sim_ce[mask].mean() if mask.any() else sim_ce.new_tensor(0.0)

    prompt_div = sim.new_tensor(0.0)
    if component_prompt is not None and float(lambda_prompt) > 0.0:
        cp = F.normalize(component_prompt.reshape(k, -1), dim=-1)
        sim_cp = cp @ cp.t()
        prompt_div = sim_cp[mask].mean() if mask.any() else sim_cp.new_tensor(0.0)

    # Encourage diversity between component heads (optional).
    orth = sim.new_tensor(0.0)
    if to_visual_weights is not None and float(lambda_orth) > 0.0:
        # to_visual_weights: [K, out_dim, in_dim] (or similar), flatten per head.
        ww = to_visual_weights.reshape(k, -1)
        ww = F.normalize(ww, dim=-1)
        sim_w = ww @ ww.t()  # [K,K]
        orth = sim_w[mask].mean() if mask.any() else sim_w.new_tensor(0.0)

    loss = (
        data.mean()
        + float(lambda_div) * div
        + float(lambda_balance) * balance
        + float(lambda_orth) * orth
        - float(lambda_ent) * entropy
        + float(lambda_comp) * comp_div
        + float(lambda_prompt) * prompt_div
        + float(lambda_gate) * gate_kl
    )
    return {
        "loss": loss,
        "data_loss": data.mean(),
        "best_loss": best.mean(),
        "best_k": best_k.float().mean(),
        "div": div,
        "balance": balance,
        "orth": orth,
        "comp_div": comp_div,
        "prompt_div": prompt_div,
        "gate_kl": gate_kl,
        "tau": pred_v.new_tensor(tau),
        "entropy": entropy,
        "w_max": r.max(dim=0).values.mean(),
        "w_bar_min": w_bar.min(),
        "w_bar_max": w_bar.max(),
        "gumbel_noise": pred_v.new_tensor(gn),
    }
