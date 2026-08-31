"""Shared flow-matching components for the GSM8K-formulas recipe.

This module holds the backbone-agnostic pieces of the per-token x0
denoising flow:

* `GSM8KFlowConfig` -- the base config dataclass.
* `Denoiser` -- the DiT-style denoiser on a single D-dim latent, with
  AdaLN-time conditioning and cross-attention to the question.
* Rectified-flow helpers: `sinusoidal_time_embedding`, `derive_x0_from_v`,
  `derive_v_from_x0`.

The Qwen3 model in `gsm8k_flow_qwen3.py` imports these and provides the
encoder/decoder roles on top of a Qwen3-0.6B + LoRA backbone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GSM8KFlowConfig:
    # Backbone
    pretrained_model: str = "Qwen/Qwen3-0.6B"

    # Latent / denoiser
    latent_dim: int = 256
    n_denoise_blocks: int = 8
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1

    # DiT-style denoiser knobs (override the n_denoise_blocks / n_heads defaults
    # above to scale the denoiser independently of the rest of the model).
    # All four default to 0 ("inherit"): denoiser_hidden -> latent_dim,
    # denoiser_depth -> n_denoise_blocks, denoiser_heads -> n_heads,
    # denoiser_ff_mult -> ff_mult. Use these for DiT-S/B/L/XL sweeps.
    denoiser_hidden: int = 0
    denoiser_depth: int = 0
    denoiser_heads: int = 0
    denoiser_ff_mult: int = 0

    # Number of latent tokens the denoiser uses INTERNALLY. The latent z_t/z_1
    # the AE consumes is still a single (B, latent_dim) vector; this knob only
    # controls how that vector is patchified to (B, N, denoiser_hidden) inside
    # the denoiser's blocks. N=1 keeps the old behavior (self-attn is a no-op,
    # purely cross-attn+FFN). N>1 lets the denoiser do real self-attn across
    # N "patches" of the latent. (Cf. multi-token latents in SD3 / Stable Audio.)
    n_latent_tokens: int = 1

    # Parameterization of the denoiser output:
    #   "v":  predict velocity v_hat ; loss = MSE(v_hat, z_1 - eps).
    #         x0_hat is then derived as z_t + (1-t) * v_hat (sphere-normed).
    #   "x0": predict x0_hat directly (sphere-normed) ; loss = MSE(x0_hat, z_1).
    #         v is then derived as (x0_hat - z_t) / (1-t) for the integrator.
    pred_target: str = "v"

    # Denoiser architecture:
    #   "local":         the in-repo Denoiser (cross-attn block tower with
    #                    9-output AdaLN; Linear+reshape patchify).
    #   "dit_canonical": faithful port of facebookresearch/DiT/models.py
    #                    (Peebles & Xie 2022) with PixArt-alpha-style
    #                    cross-attention added (6-output AdaLN, ungated cross-attn,
    #                    canonical patchify via shared per-patch Linear).
    denoiser_arch: str = "local"

    # CoT decoder
    n_cot_blocks: int = 6
    cot_max_len: int = 96  # >= M for Variant A and >= max L observed in pretraining

    # Answer decoder
    n_ans_blocks: int = 4
    ans_max_len: int = 32

    # Variant: "pad_to_M" or "no_pad"
    variant: str = "pad_to_M"
    M: int = 40  # p90 token-length on training reasoning; only used when variant=="pad_to_M"

    # Loss weights
    w_tok: float = 1.0
    w_v: float = 1.0     # flow-matching velocity MSE (was w_x0 in earlier runs)
    w_ae: float = 1.0    # autoencoding: answer head fed z_1 directly
    # Trajectory-shaped AE: if w_ae_cot > 0, also train the cot_decoder
    # during AE to reconstruct cot[k] from z_t = (1-t)*z_0 + t*z_1 with
    # k = floor(t * L_eff) (per-token-at-k CE, mirroring phase-2's flow
    # supervision). Forces z_1 to be the endpoint of a cot-decodable
    # linear trajectory, shaping the path the phase-2 denoiser will learn
    # to navigate. ae_z0_origin selects the start of the trajectory.
    w_ae_cot: float = 0.0
    ae_z0_origin: str = "noise"  # "noise" | "question_encoder"

    # Autoencoder-noise training: max stddev of Gaussian noise added to z_1
    # before feeding the answer head's L_ae path. After noise, sphere-norm is
    # re-applied so the encoder cannot drown the noise by inflating |z_1|.
    # sigma per sample is sampled U(0, ae_noise_sigma_max). Default 0 = clean AE.
    # Only affects L_ae; the diffusion target z_1 is kept clean.
    ae_noise_sigma_max: float = 0.0

    # Pooling mode for the target encoder over (q, cot, answer) hidden states.
    # "mean": mean over non-pad positions.
    # "last_token": index the last non-pad position. Standard for causal LMs
    #         like Qwen3 since that hidden state has attended to everything.
    pool_mode: str = "mean"

    # Sphere-normalize z_1 and x0_hat to constant norm sqrt(D). Required for
    # stability: without it, downstream losses reward growing |x0_hat| to
    # dominate cross-attention KV, and training catastrophically diverges
    # (verified empirically on the unnormalized run, 2026-05-10).
    sphere_norm: bool = True

    # Inference
    steps_per_rollout: int = 0  # 0 means use M (Variant A) or predicted L (Variant B)

    # Misc
    pad_token_id: Optional[int] = None  # filled in at construction
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    # Stochastic interpolant (set 0 for clean rectified flow)
    flow_noise_sigma: float = 0.0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) float in [0,1]. Returns (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t[:, None] * freqs[None, :] * 1000.0  # scale t for richer freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class AdaLNModulation(nn.Module):
    """Produces (shift, scale, gate) for residual blocks given a conditioning vector."""

    def __init__(self, cond_dim: int, target_dim: int, n_outputs: int = 6):
        super().__init__()
        self.proj = nn.Linear(cond_dim, n_outputs * target_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.n_outputs = n_outputs
        self.target_dim = target_dim

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        # c: (B, cond_dim) -> (B, n_outputs * target_dim)
        return self.proj(F.silu(c))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x: (B, T, D), shift/scale: (B, D).
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


def derive_x0_from_v(
    z_t: torch.Tensor,
    v_hat: torch.Tensor,
    t: torch.Tensor,
    sphere_norm: bool,
) -> torch.Tensor:
    """Algebraic identity for the rectified-flow path
    ``z_t = (1-t) eps + t z_1``: ``x0 = z_t + (1-t) v``.

    If ``sphere_norm`` is set, normalize the derived x0 to ``sqrt(D)`` so the
    downstream heads always consume a constant-norm latent (matching how the
    target encoder produces ``z_1``).
    """
    D = z_t.shape[-1]
    x0_hat = z_t + (1.0 - t)[:, None] * v_hat
    if sphere_norm:
        x0_hat = F.normalize(x0_hat, dim=-1) * math.sqrt(D)
    return x0_hat


class CrossAttentionBlock(nn.Module):
    """Self-attn + cross-attn + MLP, with optional AdaLN-time modulation.

    If `with_adaln=True`, the conditioning vector `c` (B, cond_dim) is consumed
    to produce shift/scale/gate triples for each of the three sub-layers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
        with_adaln: bool = True,
        cond_dim: Optional[int] = None,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal
        self.norm_self = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.with_adaln = with_adaln
        if with_adaln:
            assert cond_dim is not None
            self.adaln = AdaLNModulation(cond_dim, d_model, n_outputs=9)
        else:
            self.adaln = None

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        return torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        kv_pad_mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.with_adaln:
            mod = self.adaln(cond)  # (B, 9*D)
            chunks = mod.chunk(9, dim=-1)
            sh1, sc1, g1, sh2, sc2, g2, sh3, sc3, g3 = chunks
        else:
            zero = torch.zeros(x.shape[0], x.shape[-1], device=x.device, dtype=x.dtype)
            sh1 = sc1 = sh2 = sc2 = sh3 = sc3 = zero
            g1 = g2 = g3 = torch.ones_like(zero)

        # Self-attention
        h = modulate(self.norm_self(x), sh1, sc1)
        if self.causal:
            mask = self._causal_mask(h.shape[1], h.device)
            sa, _ = self.self_attn(h, h, h, attn_mask=mask, need_weights=False)
        else:
            sa, _ = self.self_attn(
                h, h, h, attn_mask=self_attn_mask, need_weights=False
            )
        x = x + g1[:, None, :] * self.dropout(sa)

        # Cross-attention
        h = modulate(self.norm_cross(x), sh2, sc2)
        ca, _ = self.cross_attn(
            h, kv, kv, key_padding_mask=kv_pad_mask, need_weights=False
        )
        x = x + g2[:, None, :] * self.dropout(ca)

        # FFN
        h = modulate(self.norm_ff(x), sh3, sc3)
        x = x + g3[:, None, :] * self.dropout(self.ff(h))
        return x


def derive_v_from_x0(
    z_t: torch.Tensor,
    x0_hat: torch.Tensor,
    t: torch.Tensor,
    eps_clamp: float = 1e-3,
) -> torch.Tensor:
    """Algebraic inverse of ``derive_x0_from_v`` for the rectified-flow path
    ``z_t = (1-t) eps + t z_1``: ``v = (z_1 - z_t) / (1-t)``. Used by the
    x0-parameterized integrator to get a velocity for the Heun step.

    Near t=1 the denominator goes to zero; we clamp to ``eps_clamp`` to keep
    finite values during the very last step of integration. Callers should
    treat v at t close to 1 as unreliable and rely on x0_hat directly there.
    """
    one_minus_t = (1.0 - t).clamp_min(eps_clamp)[:, None]
    return (x0_hat - z_t) / one_minus_t


class Denoiser(nn.Module):
    """DiT-style denoiser on a latent of total dim ``D = cfg.latent_dim``.

    The latent is patchified into ``N = cfg.n_latent_tokens`` internal tokens
    of width ``D_hid = cfg.denoiser_hidden`` (defaults to latent_dim when 0):

        z_t (B, D) --in_proj--> (B, N*D_hid) --reshape--> (B, N, D_hid)
        + per-token learned pos-emb
        |
        |  for each block: self-attn + cross-attn(question_kv) + FFN with
        |  AdaLN-zero gating by t_emb.
        v
        (B, N, D_hid) --reshape+out_proj--> (B, D)

    The output is interpreted as v_hat (when cfg.pred_target=="v") or
    x0_hat (when cfg.pred_target=="x0"). For x0, the consumer should
    sphere-norm before using it; v stays raw and is converted via
    derive_x0_from_v.

    When ``N=1`` the self-attn inside each block is a no-op on a single
    token, and this reduces to the original "cross-attn + FFN tower on one
    256-d vector" denoiser. When ``N>1`` self-attn becomes meaningful and
    the denoiser can exchange information across the N tokens.

    Size presets matching DiT (Peebles & Xie 2022):
      DiT-S:  hidden=384,  depth=12, heads=6
      DiT-B:  hidden=768,  depth=12, heads=12
      DiT-L:  hidden=1024, depth=24, heads=16
      DiT-XL: hidden=1152, depth=28, heads=16
    """

    def __init__(self, cfg: GSM8KFlowConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.latent_dim
        D_hid = cfg.denoiser_hidden or D
        depth = cfg.denoiser_depth or cfg.n_denoise_blocks
        heads = cfg.denoiser_heads or cfg.n_heads
        ff_mult = cfg.denoiser_ff_mult or cfg.ff_mult
        N = cfg.n_latent_tokens
        if D_hid % heads != 0:
            raise ValueError(
                f"denoiser_hidden={D_hid} must be divisible by heads={heads}"
            )
        self.D = D
        self.D_hid = D_hid
        self.N = N
        self.depth = depth
        self.heads = heads
        self.cond_dim = D_hid

        self.t_mlp = nn.Sequential(
            nn.Linear(D_hid, D_hid), nn.SiLU(), nn.Linear(D_hid, D_hid)
        )
        # Patchify: (B, D) -> (B, N*D_hid) -> (B, N, D_hid)
        self.in_proj = nn.Linear(D, N * D_hid)
        # Per-token learned positional embedding (only meaningful for N>1
        # but kept for N=1 too so checkpoints are compatible across N).
        self.pos_emb = nn.Parameter(torch.zeros(1, N, D_hid))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        # Project question_kv from D_lat to D_hid for cross-attn.
        self.kv_in_proj = nn.Linear(D, D_hid)

        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=D_hid,
                    n_heads=heads,
                    ff_mult=ff_mult,
                    dropout=cfg.dropout,
                    with_adaln=True,
                    cond_dim=D_hid,
                )
                for _ in range(depth)
            ]
        )
        self.out_norm = nn.LayerNorm(D_hid, elementwise_affine=False)
        self.out_adaln = AdaLNModulation(D_hid, D_hid, n_outputs=2)
        # Depatchify: (B, N, D_hid) -> (B, N*D_hid) -> (B, D)
        self.out_proj = nn.Linear(N * D_hid, D)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        z_t: torch.Tensor,  # (B, D)
        t: torch.Tensor,  # (B,)
        question_kv: torch.Tensor,  # (B, T_q, D_lat)
        question_pad_mask: torch.Tensor,  # (B, T_q)
    ) -> torch.Tensor:
        """Returns the raw model output (B, D), interpreted as v_hat or x0_hat
        depending on ``cfg.pred_target`` (the caller decides).
        """
        B = z_t.shape[0]
        D_hid = self.D_hid
        t_emb = sinusoidal_time_embedding(t, D_hid)
        cond = self.t_mlp(t_emb)  # (B, D_hid)
        x = self.in_proj(z_t).view(B, self.N, D_hid)
        x = x + self.pos_emb
        kv = self.kv_in_proj(question_kv)
        for blk in self.blocks:
            x = blk(x, kv=kv, kv_pad_mask=question_pad_mask, cond=cond)
        sh, sc = self.out_adaln(cond).chunk(2, dim=-1)
        x = modulate(self.out_norm(x), sh, sc)
        x_flat = x.reshape(B, self.N * D_hid)
        out = self.out_proj(x_flat)  # (B, D)
        return out
