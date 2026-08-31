"""Qwen3-0.6B + LoRA backbone for the per-token x0 GSM8K flow recipe.

Architecture:

* One frozen Qwen3-0.6B base shared across four roles, each owning a
  separate LoRA adapter (rank 32) that gets switched per-call:

      lora_q   -> QuestionEncoder    : per-token KV for the denoiser cross-attn
      lora_t   -> TargetEncoder e_phi: pooled latent z_1
      lora_cot -> CoTDecoder          : per-token CE @ position k
      lora_ans -> AnswerHead          : answer CE (autoencoding & inference paths)

* Denoiser is the same from-scratch DiT-on-a-1-token-latent module from
  ``models.gsm8k_flow_model`` (we import it directly to avoid duplication).

* x0_hat (D_lat=256) is injected into the Qwen3 decoders as a single
  *prefix soft token* (linear projection to D_qwen=1024). For the CoT
  decoder we also prepend two more soft tokens carrying the timestep ``t``
  and the target slot ``k`` (both via small MLPs from sinusoidal embeds).

* Sphere-normalization on z_1 and x0_hat is retained, plus the
  autoencoding loss L_ae on ``answer_head(z_1)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasonflow.models.dit_canonical import DiTDenoiser
from reasonflow.models.gsm8k_flow_model import (
    Denoiser,
    GSM8KFlowConfig,
    derive_v_from_x0,
    derive_x0_from_v,
    sinusoidal_time_embedding,
)


def _build_denoiser(cfg: "GSM8KFlowQwen3Config") -> nn.Module:
    """Construct the right denoiser module based on ``cfg.denoiser_arch``.

    Both branches expose the same forward signature
    ``(z_t, t, question_kv, question_pad_mask) -> (B, latent_dim)``.
    """
    if cfg.denoiser_arch == "local":
        return Denoiser(cfg)
    if cfg.denoiser_arch == "dit_canonical":
        D_hid = cfg.denoiser_hidden or cfg.latent_dim
        depth = cfg.denoiser_depth or cfg.n_denoise_blocks
        heads = cfg.denoiser_heads or cfg.n_heads
        ff_mult = cfg.denoiser_ff_mult or cfg.ff_mult
        return DiTDenoiser(
            latent_dim=cfg.latent_dim,
            hidden_size=D_hid,
            depth=depth,
            num_heads=heads,
            kv_dim=cfg.latent_dim,
            n_latent_tokens=cfg.n_latent_tokens,
            mlp_ratio=ff_mult,
        )
    raise ValueError(f"unknown denoiser_arch={cfg.denoiser_arch!r}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GSM8KFlowQwen3Config(GSM8KFlowConfig):
    """Subclass of GSM8KFlowConfig with Qwen3 + LoRA defaults."""

    pretrained_model: str = "Qwen/Qwen3-0.6B"
    qwen_hidden: int = 1024  # filled in from the actual config at construction

    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # If True, skip LoRA entirely and fully fine-tune the base Qwen3.
    # All four "roles" then share the same trunk (no role-specific weights).
    # Fine for AE-only diagnostics where only target-encoder + answer-head
    # paths are active.
    full_finetune: bool = False

    # If True, the question encoder owns a SEPARATE Qwen3-0.6B trunk (init
    # from raw HF Qwen3-0.6B), independent of the shared trunk used by
    # the AE roles. Required for stage-2 flow training: the AE-trained
    # shared trunk has been specialized as an "answer-from-tail"
    # extractor and produces hidden states that are not a useful
    # question representation for flow conditioning.
    q_encoder_separate: bool = False

    # Pooling for the question encoder cross-attn KV.
    #   "per_token": feed the full last-layer hidden states (T_q tokens)
    #                as cross-attn KV (the existing behavior).
    #   "mean":      mean-pool (with attention-mask) the last-layer
    #                hidden states to a single vector and use it as a
    #                1-token KV.
    q_encoder_pool: str = "per_token"

    # If True, the cot_decoder owns a SEPARATE Qwen3-0.6B trunk (init from
    # raw HF Qwen3-0.6B), independent of the shared AE trunk. Required for
    # stage-2 flow training when w_tok > 0: the AE-trained shared trunk
    # was only optimized for the (target_encoder, answer_head) paths under
    # w_tok=0, so the cot_decoder needs its own trainable trunk -- and we
    # cannot just unfreeze the shared trunk because that would make z_1
    # (the regression target) drift along with cot_decoder updates.
    cot_decoder_separate: bool = False

    gradient_checkpointing: bool = True


# ---------------------------------------------------------------------------
# Shared backbone
# ---------------------------------------------------------------------------


class SharedQwen3Backbone(nn.Module):
    """Qwen3-0.6B trunk shared across four roles.

    Two modes:
    * LoRA (default): base frozen, four LoRA adapters (one per role) switched
      via ``set_adapter``.
    * Full fine-tune (``cfg.full_finetune=True``): base unfrozen, no LoRA,
      ``set_adapter`` is a no-op. All roles share the same trunk weights;
      role-specific specialization comes only from the role wrappers
      (projections, MLPs, embeddings) outside this backbone.
    """

    ADAPTER_NAMES = ("lora_q", "lora_t", "lora_cot", "lora_ans")

    def __init__(self, cfg: GSM8KFlowQwen3Config):
        super().__init__()
        self.cfg = cfg
        base = AutoModelForCausalLM.from_pretrained(
            cfg.pretrained_model, dtype=torch.float32
        )

        if cfg.full_finetune:
            # Fully fine-tune the base; no PEFT wrapping.
            self.base = base
            self.peft = None
            trunk = self.base
        else:
            for p in base.parameters():
                p.requires_grad = False
            lora_cfg = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=list(cfg.lora_target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.peft = get_peft_model(base, lora_cfg, adapter_name="lora_q")
            for name in self.ADAPTER_NAMES[1:]:
                self.peft.add_adapter(name, lora_cfg)
            self.base = None
            trunk = self.peft

        if cfg.gradient_checkpointing:
            trunk.gradient_checkpointing_enable()
            # For LoRA-only training the base is frozen, so grad-ckpt requires
            # explicitly enabling input-grads to keep activation grads alive.
            # For full-FT this is unnecessary but harmless.
            if hasattr(trunk, "enable_input_require_grads"):
                trunk.enable_input_require_grads()

        self.qwen_hidden = trunk.config.hidden_size
        self.vocab_size = trunk.config.vocab_size

    @property
    def _trunk(self):
        return self.peft if self.peft is not None else self.base

    def set_adapter(self, name: str):
        assert name in self.ADAPTER_NAMES, name
        if self.peft is not None:
            self.peft.set_adapter(name)
        # else: full-FT, no adapters, no-op.

    def forward(
        self,
        adapter: str,
        *,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        return_hidden: bool = True,
    ):
        self.set_adapter(adapter)
        out = self._trunk(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=return_hidden,
            use_cache=False,
            return_dict=True,
        )
        result = {}
        if return_hidden:
            result["hidden"] = out.hidden_states[-1]
        if return_logits:
            result["logits"] = out.logits
        return result

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._trunk.get_input_embeddings()(input_ids)


# ---------------------------------------------------------------------------
# Role wrappers
# ---------------------------------------------------------------------------


class Qwen3QuestionEncoder(nn.Module):
    """Question encoder for the denoiser cross-attention.

    Two modes:
    * ``cfg.q_encoder_separate=False`` (default): uses the shared trunk
      (with ``lora_q`` adapter when LoRA is active, or just the trunk when
      ``full_finetune=True``). The trunk is whatever stage-1 ended up with.
    * ``cfg.q_encoder_separate=True``: owns its OWN ``Qwen3-0.6B`` trunk
      initialized from raw HF (``cfg.pretrained_model``). This is what we
      want for stage-2 flow because the AE-trained shared trunk is a
      specialized answer-from-tail extractor.

    Pooling controls the shape of the cross-attn KV:
    * ``per_token`` (default): KV = (B, T_q, D_lat), the full last-layer
      hidden state of the trunk projected to the latent dim. Cross-attn
      pad mask comes from ``attention_mask``.
    * ``mean``: mean-pool the last-layer hidden states with the attention
      mask, project, then KV = (B, 1, D_lat) and pad mask is all-False.
    """

    def __init__(self, cfg: GSM8KFlowQwen3Config, backbone: SharedQwen3Backbone):
        super().__init__()
        self.cfg = cfg

        if cfg.q_encoder_separate:
            base = AutoModelForCausalLM.from_pretrained(
                cfg.pretrained_model, dtype=torch.float32
            )
            if cfg.gradient_checkpointing:
                base.gradient_checkpointing_enable()
                if hasattr(base, "enable_input_require_grads"):
                    base.enable_input_require_grads()
            self.own_trunk = base
            self._shared_backbone = None
            qwen_hidden = base.config.hidden_size
        else:
            self.own_trunk = None
            # Don't register the shared backbone as a submodule (it's
            # already owned by the parent model); store on a private
            # attribute via object.__setattr__ to bypass nn.Module
            # registration.
            object.__setattr__(self, "_shared_backbone", backbone)
            qwen_hidden = backbone.qwen_hidden

        self.proj = nn.Linear(qwen_hidden, cfg.latent_dim)
        # Separate projection for z_0 (start of the trajectory when
        # cfg.ae_z0_origin == "question_encoder"). Pools the question
        # hidden state to a single sphere-normed latent in the same space
        # as z_1, so we can do z_t = (1-t)*z_0 + t*z_1 directly.
        self.z0_proj = nn.Linear(qwen_hidden, cfg.latent_dim)

    def _trunk_hidden(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run the trunk and return last-layer hidden state (B, T_q, D_qwen)."""
        if self.own_trunk is not None:
            out = self.own_trunk(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            return out.hidden_states[-1]
        out = self._shared_backbone(
            "lora_q", input_ids=input_ids, attention_mask=attention_mask
        )
        return out["hidden"]

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        h = self._trunk_hidden(input_ids, attention_mask)

        if self.cfg.q_encoder_pool == "mean":
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)
            kv = self.proj(pooled).unsqueeze(1)  # (B, 1, D_lat)
            B = h.shape[0]
            kv_pad_mask = torch.zeros(
                B, 1, dtype=torch.bool, device=h.device
            )
            return {"kv": kv, "kv_pad_mask": kv_pad_mask}
        else:  # per_token
            kv = self.proj(h)  # (B, T_q, D_lat)
            kv_pad_mask = attention_mask.eq(0)
            return {"kv": kv, "kv_pad_mask": kv_pad_mask}

    def encode_z_0(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool the question hidden state to a single z_0 latent.

        Used when cfg.ae_z0_origin == "question_encoder": the trajectory
        starts at this question-encoded state instead of pure noise. Pool
        mode follows cfg.pool_mode (last_token | mean) -- the same setting
        target_encoder uses for z_1 -- so z_0 and z_1 live in matched
        representations. Sphere-norms when cfg.sphere_norm.
        """
        h = self._trunk_hidden(input_ids, attention_mask)
        if self.cfg.pool_mode == "last_token":
            last_idx = attention_mask.sum(dim=1).long() - 1  # (B,)
            B = h.shape[0]
            pooled = h[torch.arange(B, device=h.device), last_idx]
        else:  # "mean"
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        z0 = self.z0_proj(pooled)
        if self.cfg.sphere_norm:
            z0 = F.normalize(z0, dim=-1) * math.sqrt(self.cfg.latent_dim)
        return z0


class Qwen3TargetEncoder(nn.Module):
    def __init__(self, cfg: GSM8KFlowQwen3Config, backbone: SharedQwen3Backbone):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.proj = nn.Linear(backbone.qwen_hidden, cfg.latent_dim)
        self.D = cfg.latent_dim

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.backbone(
            "lora_t", input_ids=input_ids, attention_mask=attention_mask
        )
        h = out["hidden"]  # (B, T, D_qwen)
        if self.cfg.pool_mode == "last_token":
            # Index the last non-pad position. For a causal LM this hidden
            # state has attended to every prior token (q + cot + answer).
            last_idx = attention_mask.sum(dim=1).long() - 1  # (B,)
            B = h.shape[0]
            pooled = h[torch.arange(B, device=h.device), last_idx]
        else:  # "mean"
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        z1 = self.proj(pooled)
        if self.cfg.sphere_norm:
            z1 = F.normalize(z1, dim=-1) * math.sqrt(self.D)
        return z1


class Qwen3CoTDecoder(nn.Module):
    """Causal Qwen3-LM that reads ``[question, t_tok, k_tok, x0_pref, BOS, y_<k]``.

    Two modes (parallels ``Qwen3QuestionEncoder``):
    * ``cfg.cot_decoder_separate=False`` (default): uses the shared trunk
      via the ``lora_cot`` adapter (LoRA mode) or directly (full-FT). The
      shared trunk is stored on a private attribute via ``object.__setattr__``
      so it is NOT a registered submodule of this decoder; that lets us
      unfreeze ``cot_decoder.parameters()`` in stage-2 (which then iterates
      only ``t_mlp`` + ``k_emb`` + ``x0_proj``) without inadvertently
      unfreezing ``target_encoder``'s and ``answer_head``'s shared trunk.
    * ``cfg.cot_decoder_separate=True``: owns its OWN ``Qwen3-0.6B`` trunk
      initialized from raw HF (``cfg.pretrained_model``). This is what we
      want for stage-2 flow with CoT supervision: the AE-trained shared
      trunk is held frozen, and we get a fresh fully-trainable Qwen3 trunk
      to specialize for CoT generation conditioned on ``x0_hat``.
    """

    NUM_SOFT_PREFIX = 3  # t_tok, k_tok, x0_pref

    def __init__(self, cfg: GSM8KFlowQwen3Config, backbone: SharedQwen3Backbone):
        super().__init__()
        self.cfg = cfg

        if cfg.cot_decoder_separate:
            base = AutoModelForCausalLM.from_pretrained(
                cfg.pretrained_model, dtype=torch.float32
            )
            if cfg.gradient_checkpointing:
                base.gradient_checkpointing_enable()
                if hasattr(base, "enable_input_require_grads"):
                    base.enable_input_require_grads()
            self.own_trunk = base
            object.__setattr__(self, "_shared_backbone", None)
            qwen_hidden = base.config.hidden_size
        else:
            self.own_trunk = None
            # Hide the shared backbone behind ``object.__setattr__`` so it
            # is NOT a registered submodule and ``cot_decoder.parameters()``
            # only iterates the small wrappers below. Mirrors the
            # Qwen3QuestionEncoder default-mode pattern.
            object.__setattr__(self, "_shared_backbone", backbone)
            qwen_hidden = backbone.qwen_hidden

        D_lat = cfg.latent_dim
        D_q = qwen_hidden
        self.t_mlp = nn.Sequential(
            nn.Linear(D_lat, D_lat), nn.SiLU(), nn.Linear(D_lat, D_q)
        )
        self.k_emb = nn.Embedding(cfg.cot_max_len + 1, D_q)
        self.x0_proj = nn.Linear(D_lat, D_q)

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.own_trunk is not None:
            return self.own_trunk.get_input_embeddings()(input_ids)
        return self._shared_backbone.embed_tokens(input_ids)

    def _trunk_logits(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.own_trunk is not None:
            out = self.own_trunk(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            return out.logits
        out = self._shared_backbone(
            "lora_cot",
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_logits=True,
            return_hidden=False,
        )
        return out["logits"]

    def forward(
        self,
        decoder_input_ids: torch.Tensor,  # (B, T_dec)
        x0_hat: torch.Tensor,  # (B, D_lat)
        t: torch.Tensor,  # (B,)
        k_idx: torch.Tensor,  # (B,)
        question_input_ids: torch.Tensor,  # (B, T_q)
        question_attn_mask: torch.Tensor,  # (B, T_q)
    ) -> torch.Tensor:
        B = decoder_input_ids.shape[0]
        D_lat = x0_hat.shape[-1]
        # Token embeds (use the trunk's input embeddings; embedding lookup
        # doesn't need an adapter under LoRA mode since LoRA only adapts
        # attention/MLP projections).
        q_emb = self._embed_tokens(question_input_ids)
        dec_emb = self._embed_tokens(decoder_input_ids)
        # Soft prefix tokens.
        t_emb = sinusoidal_time_embedding(t, D_lat)  # (B, D_lat)
        t_tok = self.t_mlp(t_emb)[:, None, :]  # (B, 1, D_q)
        k_tok = self.k_emb(k_idx)[:, None, :]  # (B, 1, D_q)
        x0_tok = self.x0_proj(x0_hat)[:, None, :]  # (B, 1, D_q)
        full = torch.cat([q_emb, t_tok, k_tok, x0_tok, dec_emb], dim=1)
        # Attention mask: 1 for real positions, 0 for pad. Soft-prefix and
        # decoder-input tokens are always attended.
        soft_mask = torch.ones(
            B, self.NUM_SOFT_PREFIX, device=full.device, dtype=question_attn_mask.dtype
        )
        dec_mask = (decoder_input_ids != self.cfg.pad_token_id).to(
            question_attn_mask.dtype
        )
        # First decoder slot is BOS, force-attend.
        if dec_mask.shape[1] > 0:
            dec_mask[:, 0] = 1
        full_mask = torch.cat(
            [question_attn_mask, soft_mask, dec_mask], dim=1
        )
        logits_full = self._trunk_logits(full, full_mask)
        T_q = question_input_ids.shape[1]
        # Slice out the decoder positions.
        logits = logits_full[:, T_q + self.NUM_SOFT_PREFIX:, :]
        return logits


class Qwen3AnswerHead(nn.Module):
    """Causal Qwen3-LM that reads ``[question, x0_pref, BOS, a_<i]``."""

    NUM_SOFT_PREFIX = 1  # x0_pref

    def __init__(self, cfg: GSM8KFlowQwen3Config, backbone: SharedQwen3Backbone):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.x0_proj = nn.Linear(cfg.latent_dim, backbone.qwen_hidden)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        x0_hat: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        B = decoder_input_ids.shape[0]
        q_emb = self.backbone.embed_tokens(question_input_ids)
        dec_emb = self.backbone.embed_tokens(decoder_input_ids)
        x0_tok = self.x0_proj(x0_hat)[:, None, :]
        full = torch.cat([q_emb, x0_tok, dec_emb], dim=1)
        soft_mask = torch.ones(
            B, self.NUM_SOFT_PREFIX, device=full.device, dtype=question_attn_mask.dtype
        )
        dec_mask = (decoder_input_ids != self.cfg.pad_token_id).to(
            question_attn_mask.dtype
        )
        if dec_mask.shape[1] > 0:
            dec_mask[:, 0] = 1
        full_mask = torch.cat([question_attn_mask, soft_mask, dec_mask], dim=1)
        out = self.backbone(
            "lora_ans",
            inputs_embeds=full,
            attention_mask=full_mask,
            return_logits=True,
            return_hidden=False,
        )
        T_q = question_input_ids.shape[1]
        logits = out["logits"][:, T_q + self.NUM_SOFT_PREFIX:, :]
        return logits


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class GSM8KFlowQwen3Model(nn.Module):
    """Qwen3-0.6B + LoRA flow-matching model (question encoder, target
    encoder, denoiser, cot decoder, and answer head)."""

    def __init__(self, cfg: GSM8KFlowQwen3Config):
        super().__init__()
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model)
        if self.tokenizer.pad_token_id is None:
            # Qwen3 tokenizer has pad_token by default; fall back to eos if not.
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        cfg.pad_token_id = self.tokenizer.pad_token_id
        cfg.eos_token_id = self.tokenizer.eos_token_id
        # Qwen3 has bos_token_id; if missing, use pad_token as decoder start.
        cfg.bos_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.pad_token_id
        )

        self.backbone = SharedQwen3Backbone(cfg)
        cfg.qwen_hidden = self.backbone.qwen_hidden

        self.question_encoder = Qwen3QuestionEncoder(cfg, self.backbone)
        self.target_encoder = Qwen3TargetEncoder(cfg, self.backbone)
        self.denoiser = _build_denoiser(cfg)
        self.cot_decoder = Qwen3CoTDecoder(cfg, self.backbone)
        self.answer_head = Qwen3AnswerHead(cfg, self.backbone)

    # ----------------------------------------------------------------- training

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """One training step."""
        device = batch["q_input_ids"].device
        B = batch["q_input_ids"].shape[0]

        # 1) Question encoding (lora_q)
        qenc = self.question_encoder(
            batch["q_input_ids"], batch["q_attn_mask"]
        )
        question_kv, question_pad_mask = qenc["kv"], qenc["kv_pad_mask"]

        # 2) Target encoding -> z_1 (lora_t)
        z_1 = self.target_encoder(batch["tgt_input_ids"], batch["tgt_attn_mask"])

        # 3) Sample t and form z_t along trajectory from z_0 to z_1.
        #    z_0 source is selected by cfg.ae_z0_origin so AE-side trajectory
        #    shaping (w_ae_cot > 0) and phase-2 flow training share the SAME
        #    trajectory definition (training-inference matching).
        t = torch.rand(B, device=device).clamp(1e-3, 1.0 - 1e-3)
        if self.cfg.ae_z0_origin == "question_encoder":
            # Deterministic start: z_0 = pooled question encoding via the
            # question_encoder's z0_proj head (sphere-normed inside
            # encode_z_0 when cfg.sphere_norm). NO additive Gaussian noise.
            # Phase-2 inference will use the SAME z_0 -- no x0_seed
            # randomness in the trajectory start.
            z_0 = self.question_encoder.encode_z_0(
                batch["q_input_ids"], batch["q_attn_mask"]
            )
        else:  # "noise" -- back-compat: raw randn (no sphere norm),
            # exactly matching the original eps-based formulation.
            z_0 = torch.randn_like(z_1)
        z_t = (1.0 - t)[:, None] * z_0 + t[:, None] * z_1
        if self.cfg.flow_noise_sigma > 0:
            sde_noise = torch.randn_like(z_1)
            z_t = z_t + self.cfg.flow_noise_sigma * (
                t * (1.0 - t)
            )[:, None] * sde_noise

        # 4) Denoise: model outputs either v_hat or x0_hat depending on
        # cfg.pred_target. Convert the other one algebraically so downstream
        # heads always see a sphere-normed x0_hat and the integrator/monitor
        # always sees v_hat.
        denoiser_out = self.denoiser(z_t, t, question_kv, question_pad_mask)
        # v_target is the trajectory slope: dz_t/dt = z_1 - z_0. With
        # ae_z0_origin="noise" this matches the original eps-based formula;
        # with ae_z0_origin="question_encoder", v_target is purely deterministic.
        v_target = z_1.detach() - z_0.detach()
        if self.cfg.pred_target == "v":
            v_hat = denoiser_out
            x0_hat = derive_x0_from_v(
                z_t, v_hat, t, sphere_norm=self.cfg.sphere_norm
            )
        elif self.cfg.pred_target == "x0":
            x0_hat = denoiser_out
            if self.cfg.sphere_norm:
                x0_hat = F.normalize(x0_hat, dim=-1) * math.sqrt(
                    self.cfg.latent_dim
                )
            v_hat = derive_v_from_x0(z_t, x0_hat, t)
        else:
            raise ValueError(f"unknown pred_target {self.cfg.pred_target!r}")

        # 5) k = floor(t * L_eff)
        cot_lengths = batch["cot_lengths"].to(device)
        if self.cfg.variant == "pad_to_M":
            L_eff = torch.full(
                (B,), self.cfg.M, device=device, dtype=cot_lengths.dtype
            )
        else:
            L_eff = cot_lengths.clamp_min(1)
        k = (t * L_eff.float()).floor().long()
        k = torch.clamp(k, 0, self.cfg.cot_max_len - 1)

        # 6) CoT decoder forward (lora_cot, teacher-forced).
        # Skipped when w_tok == 0 to save one Qwen3 trunk forward per step.
        # In the trajectory-shaped recipe, phase-2 sets w_tok=0 because
        # CoT supervision happens during AE training (the trajectory is
        # already shaped there); phase-2 just needs the flow + answer-head
        # losses.
        if self.cfg.w_tok > 0:
            cot_logits = self.cot_decoder(
                decoder_input_ids=batch["cot_input_ids"],
                x0_hat=x0_hat,
                t=t,
                k_idx=k,
                question_input_ids=batch["q_input_ids"],
                question_attn_mask=batch["q_attn_mask"],
            )

            # 7) Per-token CE at position k only.
            valid = k < cot_lengths
            idx = k.clamp_max(self.cfg.cot_max_len - 1)
            gold = batch["cot_target_ids"][torch.arange(B, device=device), idx]
            logits_k = cot_logits[torch.arange(B, device=device), idx]
            per_sample = F.cross_entropy(
                logits_k, gold, reduction="none", ignore_index=-100
            )
            loss_tok = (per_sample * valid.float()).sum() / valid.float().sum().clamp_min(
                1.0
            )
        else:
            loss_tok = torch.zeros((), device=device)
            valid = k < cot_lengths  # still needed for the metric below

        # 8) Flow loss. With pred_target="v" this is the standard rectified-flow
        # velocity MSE. With pred_target="x0" it is MSE(x0_hat, z_1) directly,
        # which is more stable away from the trajectory (no ill-conditioning at
        # large t). w_v controls this primary flow loss in either mode (we
        # don't introduce a separate w_x0 weight; pred_target picks which one).
        loss_v_mse = F.mse_loss(v_hat, v_target)
        loss_x0_mse = F.mse_loss(x0_hat, z_1.detach())
        if self.cfg.pred_target == "v":
            loss_v = loss_v_mse
        else:
            loss_v = loss_x0_mse

        # 9) Autoencoding (lora_ans on z_1, optionally with noise injection).
        # Trains lora_t + lora_ans + lora_q. The noise-injection variant adds
        # sigma ~ U(0, sigma_max) Gaussian noise to z_1 before the answer head,
        # then re-applies sphere-norm so the encoder cannot drown out the
        # perturbation by inflating |z_1|. This makes the answer head robust
        # to the slight off-manifold perturbations the diffusion model will
        # produce at inference.
        z_1_for_ae = z_1
        if self.cfg.ae_noise_sigma_max > 0:
            sigma = torch.rand(B, 1, device=device) * self.cfg.ae_noise_sigma_max
            ae_eps = torch.randn_like(z_1)
            z_1_for_ae = z_1 + sigma * ae_eps
            if self.cfg.sphere_norm:
                z_1_for_ae = F.normalize(z_1_for_ae, dim=-1) * math.sqrt(z_1.shape[-1])
        ae_logits = self.answer_head(
            decoder_input_ids=batch["ans_input_ids"],
            x0_hat=z_1_for_ae,
            question_input_ids=batch["q_input_ids"],
            question_attn_mask=batch["q_attn_mask"],
        )
        loss_ae = F.cross_entropy(
            ae_logits.reshape(-1, ae_logits.shape[-1]),
            batch["ans_target_ids"].reshape(-1),
            ignore_index=-100,
        )

        # Trajectory-shaped AE: train the cot_decoder to reconstruct cot[k]
        # from z_t = (1-t)*z_0 + t*z_1 with k = floor(t * L_eff), per-token-at-k
        # CE -- the SAME supervision shape phase-2 flow uses on x0_hat. Forces
        # z_1 to be the endpoint of a cot-decodable linear trajectory and
        # forces z_0 (when ae_z0_origin="question_encoder") to be a
        # cot-coherent question-conditioned start. We sample a FRESH t_traj
        # (independent of the flow t above) so AE-only training (w_v=0) still
        # gets diverse t coverage and so the AE-shaping loss isn't biased by
        # whatever t the flow loss happened to sample.
        if self.cfg.w_ae_cot > 0:
            t_traj = torch.rand(B, device=device).clamp(1e-3, 1.0 - 1e-3)
            # Reuse the same z_0 distribution as the flow-side trajectory.
            if self.cfg.ae_z0_origin == "question_encoder":
                z_0_traj = self.question_encoder.encode_z_0(
                    batch["q_input_ids"], batch["q_attn_mask"]
                )
            else:  # "noise" -- raw randn (matches flow-side back-compat)
                z_0_traj = torch.randn_like(z_1)
            s_traj = t_traj.unsqueeze(-1)
            z_t_traj = (1.0 - s_traj) * z_0_traj + s_traj * z_1
            # AE-noise robustness: optionally add per-sample sigma noise then
            # sphere-norm. Defaults to 0.3 in our recipes (matches the
            # answer-head's noise injection).
            if self.cfg.ae_noise_sigma_max > 0:
                sig = (
                    torch.rand(B, 1, device=device)
                    * self.cfg.ae_noise_sigma_max
                )
                z_t_traj = z_t_traj + sig * torch.randn_like(z_t_traj)
            if self.cfg.sphere_norm:
                z_t_traj = F.normalize(z_t_traj, dim=-1) * math.sqrt(
                    z_1.shape[-1]
                )
            # k_traj = floor(t_traj * L_eff), exactly mirroring the flow path.
            if self.cfg.variant == "pad_to_M":
                L_eff_traj = torch.full(
                    (B,), self.cfg.M, device=device, dtype=cot_lengths.dtype
                )
            else:
                L_eff_traj = cot_lengths.clamp_min(1)
            k_traj = (t_traj * L_eff_traj.float()).floor().long()
            k_traj = torch.clamp(k_traj, 0, self.cfg.cot_max_len - 1)
            ae_cot_logits = self.cot_decoder(
                decoder_input_ids=batch["cot_input_ids"],
                x0_hat=z_t_traj,
                t=t_traj,
                k_idx=k_traj,
                question_input_ids=batch["q_input_ids"],
                question_attn_mask=batch["q_attn_mask"],
            )
            valid_traj = k_traj < cot_lengths
            idx_traj = k_traj.clamp_max(self.cfg.cot_max_len - 1)
            gold_traj = batch["cot_target_ids"][
                torch.arange(B, device=device), idx_traj
            ]
            logits_at_k = ae_cot_logits[
                torch.arange(B, device=device), idx_traj
            ]
            per_sample_traj = F.cross_entropy(
                logits_at_k, gold_traj, reduction="none", ignore_index=-100
            )
            loss_ae_cot = (
                per_sample_traj * valid_traj.float()
            ).sum() / valid_traj.float().sum().clamp_min(1.0)
        else:
            loss_ae_cot = torch.zeros((), device=device)

        loss = (
            self.cfg.w_tok * loss_tok
            + self.cfg.w_v * loss_v
            + self.cfg.w_ae * loss_ae
            + self.cfg.w_ae_cot * loss_ae_cot
        )

        diff_norm = (x0_hat.detach() - z_1.detach()).norm(dim=-1).mean()
        v_err_norm = (v_hat.detach() - v_target.detach()).norm(dim=-1).mean()
        return {
            "loss": loss,
            "loss_tok": loss_tok.detach(),
            "loss_v": loss_v.detach(),
            "loss_v_mse": loss_v_mse.detach(),
            "loss_x0_mse": loss_x0_mse.detach(),
            "loss_ae": loss_ae.detach(),
            "loss_ae_cot": loss_ae_cot.detach(),
            "v_err_norm": v_err_norm,
            "valid_frac": valid.float().mean().detach(),
            "z1_norm": z_1.detach().norm(dim=-1).mean(),
            "x0_norm": x0_hat.detach().norm(dim=-1).mean(),
            "x0_z1_dist": diff_norm,
        }

    # ---------------------------------------------------------------- inference

    @torch.no_grad()
    def generate(
        self,
        q_input_ids: torch.Tensor,
        q_attn_mask: torch.Tensor,
        K: Optional[int] = None,
        return_trajectory: bool = False,
        steps_per_rollout: Optional[int] = None,
        x0_mode: str = "predicted",
        x0_seed: int = 0,
        tgt_input_ids: Optional[torch.Tensor] = None,
        tgt_attn_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Greedy generation. ``x0_mode`` controls the x0_hat fed to the CoT and
        answer heads (test-time ablation; the diffusion integrator still uses
        the predicted x0_hat regardless):

          - "predicted": x0_hat = denoiser(z_t, t)  (default, trained behaviour)
          - "gaussian":  x0_hat = sphere-norm random vector (||x0_hat||=sqrt(D))
          - "zero":      x0_hat = 0
          - "oracle_z1": x0_hat = target_encoder(tgt_input_ids, tgt_attn_mask)
                         — uses the gold (q,reasoning,answer) sequence; upper bound.
        """
        device = q_input_ids.device
        B = q_input_ids.shape[0]
        D = self.cfg.latent_dim

        if K is None:
            K = (
                self.cfg.M
                if self.cfg.variant == "pad_to_M"
                else self.cfg.cot_max_len
            )
        K = max(1, int(K))
        steps_per_rollout = steps_per_rollout or (
            self.cfg.steps_per_rollout
            if self.cfg.steps_per_rollout > 0
            else K
        )
        steps_per_rollout = max(1, int(steps_per_rollout))
        if self.cfg.w_tok > 0:
            # Token-emitting variants need at least one integration interval
            # before every CoT-token-prefix boundary.
            steps_per_rollout = max(K, steps_per_rollout)

        qenc = self.question_encoder(q_input_ids, q_attn_mask)
        question_kv, question_pad_mask = qenc["kv"], qenc["kv_pad_mask"]

        # Pre-compute oracle z_1 once if needed.
        oracle_z1: Optional[torch.Tensor] = None
        if x0_mode == "oracle_z1":
            assert tgt_input_ids is not None and tgt_attn_mask is not None, (
                "x0_mode=oracle_z1 requires tgt_input_ids/tgt_attn_mask"
            )
            oracle_z1 = self.target_encoder(tgt_input_ids, tgt_attn_mask)

        # Per-batch RNG so flow start noise + gaussian-ablation noise are
        # both seeded by ``x0_seed``. Without this the GLOBAL torch RNG was
        # used for ``z_t = torch.randn(...)`` below, making val EM
        # non-reproducible across runs (same checkpoint + same x0_seed
        # could give 32% vs 52.6% on train_n500). One generator drives all
        # randn() calls in this generate() invocation.
        flow_gen = torch.Generator(device=device).manual_seed(int(x0_seed))

        def apply_x0_mode(x0_hat: torch.Tensor) -> torch.Tensor:
            if x0_mode == "predicted":
                return x0_hat
            if x0_mode == "gaussian":
                v = torch.randn(x0_hat.shape, device=device, generator=flow_gen)
                return F.normalize(v, dim=-1) * math.sqrt(D)
            if x0_mode == "zero":
                return torch.zeros_like(x0_hat)
            if x0_mode == "oracle_z1":
                # Broadcast the precomputed z_1 (B, D) to whatever shape x0_hat has.
                assert oracle_z1 is not None
                return oracle_z1
            raise ValueError(f"unknown x0_mode {x0_mode}")

        # Initial z_t = z_0 at t=0. Source matches AE training (cfg.ae_z0_origin)
        # so inference and training use identical trajectory geometry.
        if self.cfg.ae_z0_origin == "question_encoder":
            # Deterministic start; question_encoder is frozen here so this is
            # consistent across runs. x0_seed has no effect on z_0 in this
            # mode -- only on x0_mode='gaussian' ablations.
            z_t = self.question_encoder.encode_z_0(q_input_ids, q_attn_mask)
        else:  # "noise" -- back-compat with original training (raw randn)
            z_t = torch.randn(B, D, device=device, generator=flow_gen)
        tokens = torch.full(
            (B, K + 1), self.cfg.pad_token_id, device=device, dtype=torch.long
        )
        tokens[:, 0] = self.cfg.bos_token_id

        trajectory: List[torch.Tensor] = []
        cur_t = torch.zeros(B, device=device)
        cot_finished = torch.zeros(B, dtype=torch.bool, device=device)

        def _v_and_x0(z, t_):
            """Run the denoiser at (z, t_) and return (v_hat, x0_hat)
            consistently with cfg.pred_target. Both forms are needed:
            v for the integrator, x0 (sphere-normed) for the heads."""
            out = self.denoiser(z, t_, question_kv, question_pad_mask)
            if self.cfg.pred_target == "v":
                v_ = out
                x0_ = derive_x0_from_v(z, v_, t_, sphere_norm=self.cfg.sphere_norm)
            else:  # "x0"
                x0_ = out
                if self.cfg.sphere_norm:
                    x0_ = F.normalize(x0_, dim=-1) * math.sqrt(D)
                v_ = derive_v_from_x0(z, x0_, t_)
            return v_, x0_

        def _heun_step(t_next: torch.Tensor) -> None:
            """Advance ``z_t`` from ``cur_t`` to ``t_next`` with Heun's rule."""
            nonlocal z_t, cur_t
            v, _ = _v_and_x0(z_t, cur_t)
            dt = (t_next - cur_t)[:, None]
            z_mid = z_t + dt * v
            v_mid, _ = _v_and_x0(z_mid, t_next)
            z_t = z_t + 0.5 * dt * (v + v_mid)
            cur_t = t_next

        if self.cfg.w_tok <= 0:
            # Strict answer-only inference: integrate on an exact global
            # N-step grid, never consult the (untrained) CoT decoder, and
            # always reach t=1.
            for solver_step in range(steps_per_rollout):
                t_next = torch.full(
                    (B,), (solver_step + 1) / steps_per_rollout, device=device
                )
                _heun_step(t_next)
                if return_trajectory:
                    _, x0_hat = _v_and_x0(z_t, cur_t)
                    trajectory.append(apply_x0_mode(x0_hat).detach().cpu())
        else:
            # Allocate all requested solver intervals across the K token
            # prefixes. The floor-difference construction supports N values
            # that are not divisible by K while landing exactly at every
            # token-prefix boundary t=(k+1)/K.
            for k_step in range(K):
                n_substeps = (
                    ((k_step + 1) * steps_per_rollout) // K
                    - (k_step * steps_per_rollout) // K
                )
                t_target = torch.full((B,), (k_step + 1) / K, device=device)
                for substep in range(n_substeps):
                    remaining = n_substeps - substep
                    t_next = cur_t + (t_target - cur_t) / remaining
                    _heun_step(t_next)

                _, x0_hat = _v_and_x0(z_t, cur_t)
                x0_hat_for_decoder = apply_x0_mode(x0_hat)
                if return_trajectory:
                    trajectory.append(x0_hat_for_decoder.detach().cpu())
                if cot_finished.all():
                    # Keep integrating to t=1 after every sequence has
                    # emitted EOS, but avoid unnecessary decoder forwards.
                    continue
                decoder_input = tokens[:, : k_step + 1]
                k_idx = torch.full((B,), k_step, device=device, dtype=torch.long)
                logits = self.cot_decoder(
                    decoder_input_ids=decoder_input,
                    x0_hat=x0_hat_for_decoder,
                    t=cur_t,
                    k_idx=k_idx,
                    question_input_ids=q_input_ids,
                    question_attn_mask=q_attn_mask,
                )
                tok = logits[:, k_step].argmax(dim=-1)
                # Once a sample emits EOS in CoT, stop appending — leave pad.
                tokens[:, k_step + 1] = torch.where(
                    cot_finished, torch.full_like(tok, self.cfg.pad_token_id), tok
                )
                cot_finished = cot_finished | (tok == self.cfg.eos_token_id)

        # Final answer at t=1.
        # - For pred_target="v": algebraically x0(z_K, t=1) = z_K + 0*v = z_K,
        #   so we just sphere-norm the integrator's z_K and skip the denoiser.
        # - For pred_target="x0": call the denoiser at t=1 once to get its
        #   direct prediction, which is generally more accurate than z_K when
        #   the integrator has accumulated error (and avoids the v=(x0-z_t)/0
        #   blow-up that would happen if we tried to derive v at t=1).
        if self.cfg.pred_target == "v":
            x0_hat_final = z_t
            if self.cfg.sphere_norm:
                x0_hat_final = F.normalize(x0_hat_final, dim=-1) * math.sqrt(D)
        else:  # "x0"
            x0_raw = self.denoiser(
                z_t, torch.ones(B, device=device), question_kv, question_pad_mask
            )
            if self.cfg.sphere_norm:
                x0_hat_final = F.normalize(x0_raw, dim=-1) * math.sqrt(D)
            else:
                x0_hat_final = x0_raw
        x0_hat_final_for_ans = apply_x0_mode(x0_hat_final)
        ans_tokens = self._answer_greedy(
            x0_hat_final_for_ans,
            q_input_ids,
            q_attn_mask,
            max_len=self.cfg.ans_max_len,
        )

        out = {
            "cot_tokens": (
                tokens[:, 1:] if self.cfg.w_tok > 0 else tokens[:, :0]
            ),
            "ans_tokens": ans_tokens,
            "x0_hat_final": x0_hat_final_for_ans.detach(),
        }
        if return_trajectory:
            out["x0_hat_trajectory"] = torch.stack(trajectory, dim=1)
        return out

    @torch.no_grad()
    def _answer_greedy(
        self,
        x0_hat: torch.Tensor,
        q_input_ids: torch.Tensor,
        q_attn_mask: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        B = x0_hat.shape[0]
        device = x0_hat.device
        tokens = torch.full(
            (B, max_len + 1),
            self.cfg.pad_token_id,
            device=device,
            dtype=torch.long,
        )
        tokens[:, 0] = self.cfg.bos_token_id
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for i in range(max_len):
            logits = self.answer_head(
                decoder_input_ids=tokens[:, : i + 1],
                x0_hat=x0_hat,
                question_input_ids=q_input_ids,
                question_attn_mask=q_attn_mask,
            )
            tok = logits[:, i].argmax(dim=-1)
            tokens[:, i + 1] = torch.where(
                finished, torch.full_like(tok, self.cfg.pad_token_id), tok
            )
            finished = finished | (tok == self.cfg.eos_token_id)
            if finished.all():
                break
        return tokens[:, 1:]
