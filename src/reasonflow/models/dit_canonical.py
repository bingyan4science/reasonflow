# Adapted from Meta Platforms' DiT implementation:
# https://github.com/facebookresearch/DiT/blob/f4e74c1ec80e8/models.py
# This adapted file is licensed under CC BY-NC 4.0.
# SPDX-License-Identifier: CC-BY-NC-4.0

r"""Canonical DiT (Peebles & Xie 2022) ported to a 1-D vector latent and
extended with PixArt-style cross-attention to a question-conditioning KV.

Source: https://github.com/facebookresearch/DiT/blob/main/models.py (commit
f4e74c1ec80e8) -- I have kept variable names, layer structure, init schedule,
and adaLN-zero scheme as close to the original as possible. The deltas are:

  1. The latent here is a 1-D vector ``z_t \in R^D`` rather than a 2-D image
     ``(B, C, H, W)``. So:
       - ``PatchEmbed1D`` splits z_t into ``num_patches = D / patch_size``
         patches of ``patch_size`` dims each, then applies a *shared* linear
         per patch (same as DiT's Conv2d patchify, but in 1-D).
       - The "unpatchify" is just ``(B, N, patch_size) -> (B, D)`` reshape.
       - ``pos_embed`` is a learned 1-D parameter (DiT's 2-D sin-cos grid
         doesn't apply); this matches what most 1-D adaptations do (Stable
         Audio's DiT, MAR, etc.).

  2. We replace the class-label embedder ``LabelEmbedder`` with a PixArt-style
     cross-attention sub-layer in every block. After self-attn (modulated by
     adaLN), the block does ``x = x + cross_attn(LN(x), kv)``, then the MLP
     (modulated by adaLN). ``adaLN_modulation`` keeps **6 outputs** -- exactly
     as in canonical DiT -- and the cross-attn sublayer is *not* gated
     (matching PixArt-alpha's PixArtBlock).

  3. ``learn_sigma=False``: the network outputs a single tensor of shape
     ``(B, D)``, interpreted as ``v_hat`` or ``x0_hat`` by the caller (just
     like the rest of our pipeline).

  4. ``num_classes`` and ``class_dropout_prob`` are absent. Only ``t`` is
     embedded into ``c`` for adaLN; the question is fed via cross-attn.

Everything else (qkv_bias=True self-attn, GELU-tanh MLP, mlp_ratio=4,
LayerNorm with elementwise_affine=False and eps=1e-6, sinusoidal+MLP timestep
embedding, basic Xavier init for Linears, Xavier on patch_embed, 0.02 normal
on timestep MLP, all-zero init on adaLN_modulation[-1] and the final-layer
linear) follows DiT exactly.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers (kept verbatim where possible from the DiT repo)
# ---------------------------------------------------------------------------


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x * (1 + scale) + shift`` -- exactly DiT's helper. ``shift, scale``
    are (B, D); ``x`` is (B, T, D)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """Self-attention with qkv_bias=True. Behaviorally equivalent to
    ``timm.layers.Attention`` (which DiT imports). Re-implemented here to
    avoid an explicit timm dependency and to use ``F.scaled_dot_product_attention``
    for speed/memory efficiency."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # (B, num_heads, N, head_dim) each
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop_p if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class Mlp(nn.Module):
    """``timm.layers.Mlp``-equivalent: Linear -> GELU(tanh) -> Linear."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TimestepEmbedder(nn.Module):
    """Verbatim copy of DiT's ``TimestepEmbedder``."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        # Matches DiT exactly.
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            self.mlp[0].weight.dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class PatchEmbed1D(nn.Module):
    r"""1-D analog of ViT/DiT ``PatchEmbed``. Splits ``z_t \in R^{B \times D}``
    into ``D / patch_size`` patches of ``patch_size`` dims each, then applies
    a shared linear projection to ``hidden_size``. Equivalent to a Conv1d
    with kernel=stride=patch_size."""

    def __init__(self, latent_dim: int, patch_size: int, hidden_size: int):
        super().__init__()
        assert latent_dim % patch_size == 0, (
            f"latent_dim={latent_dim} must be divisible by patch_size={patch_size}"
        )
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.num_patches = latent_dim // patch_size
        self.proj = nn.Linear(patch_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.view(B, self.num_patches, self.patch_size)
        return self.proj(x)


# ---------------------------------------------------------------------------
# DiT block with PixArt-style cross-attention
# ---------------------------------------------------------------------------


class DiTBlockCA(nn.Module):
    """A DiT block (Peebles & Xie 2022, ``DiTBlock``) with a PixArt-alpha
    cross-attention sublayer inserted between self-attn and MLP. ``adaLN_modulation``
    keeps **6 outputs** as in canonical DiT (it modulates self-attn and MLP
    only, not the cross-attn sublayer -- this matches PixArt's ``PixArtBlock``).

    Block topology:
        (msa) x = x + gate_msa * attn(modulate(LN(x), shift_msa, scale_msa))
        (ca)  x = x + cross_attn(LN(x), kv)             # PixArt: no gate/AdaLN
        (mlp) x = x + gate_mlp * mlp(modulate(LN(x), shift_mlp, scale_mlp))
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        # PixArt-style cross-attn (its own LN, no AdaLN gate).
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        kv: torch.Tensor,
        kv_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        # PixArt cross-attn: no AdaLN gate, just norm + MHA + residual.
        ca_out, _ = self.cross_attn(
            self.norm_cross(x), kv, kv,
            key_padding_mask=kv_pad_mask, need_weights=False,
        )
        x = x + ca_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """Verbatim copy of DiT's ``FinalLayer`` (norm + AdaLN(2 outputs) +
    Linear), with output channels=1 since we predict a single tensor per
    patch (no learn_sigma)."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int = 1):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ---------------------------------------------------------------------------
# Top-level: DiT denoiser for our 1-D latent
# ---------------------------------------------------------------------------


class DiTDenoiser(nn.Module):
    """``DiT`` (Peebles & Xie 2022) for a 1-D latent + PixArt cross-attention
    to a question. Forward signature matches our ``models.gsm8k_flow_model.Denoiser``
    so the rest of the pipeline can call it interchangeably.

    Args:
      latent_dim:    D of z_t / x0_hat / v_hat (the AE latent dim, e.g. 256).
      hidden_size:   internal DiT hidden, e.g. 384/768/1024/1152 for DiT-S/B/L/XL.
      depth:         number of DiTBlockCA layers.
      num_heads:     attention heads. Must divide ``hidden_size``.
      kv_dim:        feature dim of question_kv (typically also latent_dim,
                     but kept separate so the q-encoder is free to project).
      n_latent_tokens:  ``D / patch_size``. Use 1 for "no patch" (single token);
                     16 splits a 256-d latent into 16 patches of 16 dims.
      mlp_ratio:     FFN expansion (DiT default 4).
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        kv_dim: int,
        n_latent_tokens: int = 1,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        assert latent_dim % n_latent_tokens == 0
        patch_size = latent_dim // n_latent_tokens
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_patches = n_latent_tokens
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed1D(latent_dim, patch_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # Project question features to hidden_size for cross-attn.
        self.kv_proj = nn.Linear(kv_dim, hidden_size, bias=True)
        # Learned 1-D position embedding (DiT uses 2-D sin-cos for images).
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_latent_tokens, hidden_size)
        )

        self.blocks = nn.ModuleList(
            [DiTBlockCA(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        # out_channels=1 (we are not learning sigma; one prediction per dim).
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels=1)

        self.initialize_weights()

    def initialize_weights(self):
        # 1) Basic init for all Linears (mirrors DiT._basic_init).
        def _basic_init(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        # 2) pos_embed: DiT uses 2-D sin-cos; for 1-D we use truncated normal
        # std=0.02 (matches standard ViT learnable pos-emb init).
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 3) x_embedder: re-Xavier (already Xavier from basic_init, but DiT
        # is explicit, so we mirror it).
        nn.init.xavier_uniform_(self.x_embedder.proj.weight)
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # 4) Timestep MLP: normal std=0.02 (verbatim from DiT).
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # 5) kv_proj: keep Xavier (no override).

        # 6) Zero-out adaLN modulation layers in each DiT block (THE adaLN-zero
        # init -- matches DiT exactly).
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # 7) Zero-out output: final adaLN + final linear (matches DiT).
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, num_patches, patch_size) -> (B, D)``."""
        B = x.shape[0]
        return x.reshape(B, self.latent_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        kv: torch.Tensor,
        kv_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """``z_t``: (B, D); ``t``: (B,); ``kv``: (B, T_q, kv_dim);
        ``kv_pad_mask``: (B, T_q) bool, True at pad positions.
        Returns (B, D), interpreted as v_hat or x0_hat by the caller."""
        x = self.x_embedder(z_t) + self.pos_embed  # (B, N, hidden)
        c = self.t_embedder(t)                       # (B, hidden)
        kv = self.kv_proj(kv)                        # (B, T_q, hidden)
        for block in self.blocks:
            x = block(x, c, kv, kv_pad_mask)
        x = self.final_layer(x, c)                   # (B, N, patch_size)
        return self.unpatchify(x)
