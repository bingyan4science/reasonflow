"""Phase-2 anchored diffusion training.

Trains a DiT denoiser to follow the time-varying clean target
``z_clean(t) = lerp(z^a_k, z^a_{k+1}, K*t-k)`` produced by a frozen
anchor encoder. At inference, integrates noise -> z_at_t=1 ~ z^a_K
and decodes the answer via the frozen suffix decoder at k=K.

Frozen modules:
    * Anchor LM    -- finetuned Qwen3-0.6B AR LM (anchor source).
                      Anchors extracted via extract_anchors (mean or
                      last-token pool of middle-layer hidden states at
                      each cot-step boundary).
    * Suffix decoder -- trained on top of the frozen anchors
                      (see scripts/train_suffix_decoder.py).
                      Used at inference as the answer head at k=K.

Trainable modules:
    * Question encoder (fresh Qwen3-0.6B for cross-attn KV).
    * DiT denoiser (latent_dim = D_qwen = 1024; sphere-normed on
      input and output).

Usage:
    python scripts/train_anchored_phase2.py \\
        --v1_pool_mode mean \\
        --v1_lm_ckpt      <CHECKPOINT_ROOT>/outputs/anchor_lm/last_checkpoint \\
        --v1_decoder_ckpt <CHECKPOINT_ROOT>/outputs/suffix_decoder_mean/last_checkpoint.pt \\
        --data_dir        <REPO_ROOT>/data/gsm8k_formulas \\
        --output_dir      <CHECKPOINT_ROOT>/outputs/anchored_phase2_v1_mean \\
        --M 48  --max_q_len 192 --cot_max_len 80 --ans_max_len 24 \\
        --dit_arch ditXL --batch_size 32 --max_steps 200000 ...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reasonflow.models.anchored.anchor_extraction import extract_anchors  # noqa: E402
from reasonflow.models.anchored.dataset_utils import (  # noqa: E402
    build_anchored_target,
)
from reasonflow.models.anchored.suffix_decoder import SuffixDecoder  # noqa: E402
from reasonflow.models.dit_canonical import DiTDenoiser  # noqa: E402
from reasonflow.utils.answer_normalizers import get_normalizer  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset / collator  (M-padded variant-A; emits cot_step_positions)
# ---------------------------------------------------------------------------


class AnchoredPhase2Dataset(Dataset):
    """Wraps GSM8K-formulas. Per item we emit:
        q_ids, ans_ids, cot_ids, cot_length,
        anchor_input_ids,           = q_ids + cot_ids[:cot_length]
        cot_step_positions          length M+1; positions in anchor_input_ids
                                    where each anchor lives.

    The anchor_input_ids match the convention the frozen encoder used when the
    suffix decoder was trained: input = [q_ids, cot_prefix_ids], no sep1
    between q and cot, no sep2/answer on the end. extract_anchors over
    anchor_input_ids with cot_step_positions reproduces the frozen encoder's
    z_a_k for k = 0..K in a single forward pass.

    cot_step_positions[k]:
        k = 0:    Lq - 1                            (last token of q only)
        k = 1..K: Lq + min(k, cot_length) - 1       (last cot token seen)
        For k > cot_length, position is replicated (variant-A convention).
    """

    def __init__(self, json_path: Path, tokenizer, *, M: int, max_q_len: int,
                 cot_max_len: int, ans_max_len: int,
                 max_examples: Optional[int] = None):
        with json_path.open("r") as fh:
            self.examples = json.load(fh)
        self.examples = [
            e for e in self.examples
            if len(e.get("reasoning_text", "")) > 0
            and len(e.get("answer", "")) > 0
        ]
        if max_examples is not None and max_examples > 0:
            self.examples = self.examples[:max_examples]
        self.tok = tokenizer
        self.M = M
        self.max_q_len = max_q_len
        self.cot_max_len = cot_max_len
        self.ans_max_len = ans_max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        # We deliberately call build_anchored_target with M=None to get the
        # raw q_ids / cot_ids / ans_ids; we don't reuse its tgt_ids /
        # cot_step_positions because those include sep1/sep2 which the
        # suffix decoder was NOT trained to see in the encoder input.
        built = build_anchored_target(
            self.tok,
            e["question"], e["reasoning_text"], e["answer"],
            M=None,
            cot_max_len=self.cot_max_len,
            ans_max_len=self.ans_max_len,
            max_q_len=self.max_q_len,
        )
        q_ids = built["q_ids"]
        cot_ids = built["cot_ids"]
        ans_ids = built["ans_ids"]
        cot_length = built["cot_length"]

        Lq = len(q_ids)
        Lcot = cot_length
        Lans = len(ans_ids)
        # Cot anchors: pos in (q + cot) = q + cot[:k] last position.
        # k=0 -> last token of q (Lq-1). k=k -> Lq + k - 1.
        anchor_input_ids = q_ids + cot_ids
        cot_step_positions = [Lq - 1]
        for k in range(1, self.M + 1):
            if k <= Lcot:
                cot_step_positions.append(Lq + k - 1)
            else:
                cot_step_positions.append(cot_step_positions[-1])

        # For z_1 = "v1_endofans" anchor, we additionally need a forward
        # over (q + cot + ans) with the last position being the LAST
        # ANSWER TOKEN (just before EOS). Position in this longer
        # sequence: Lq + Lcot + Lans - 1 (0-indexed last real ans token).
        ans_endpoint_input_ids = q_ids + cot_ids + ans_ids
        ans_endpoint_pos = Lq + Lcot + Lans - 1

        return {
            "q_ids": q_ids,
            "cot_ids": cot_ids,
            "ans_ids": ans_ids,
            "anchor_input_ids": anchor_input_ids,
            "cot_step_positions": cot_step_positions,
            "cot_length": cot_length,
            "cot_start_position": Lq,         # = first cot token in anchor_input_ids
            "ans_endpoint_input_ids": ans_endpoint_input_ids,
            "ans_endpoint_pos": ans_endpoint_pos,
            "answer": e["answer"],
            "question": e["question"],
        }


def make_collator(pad_id: int, M: int):
    def _pad(seqs, length, pad_value):
        out = torch.full((len(seqs), length), pad_value, dtype=torch.long)
        for i, s in enumerate(seqs):
            if len(s) > 0:
                out[i, : min(len(s), length)] = torch.tensor(
                    s[:length], dtype=torch.long
                )
        return out

    def collate(items):
        max_q = max(len(it["q_ids"]) for it in items)
        max_a = max(len(it["ans_ids"]) for it in items)
        max_c = max(max(len(it["cot_ids"]) for it in items), 1)
        max_anchor = max(len(it["anchor_input_ids"]) for it in items)
        max_endpt = max(len(it["ans_endpoint_input_ids"]) for it in items)

        q_ids = _pad([it["q_ids"] for it in items], max_q, pad_id)
        q_attn = (q_ids != pad_id).long()
        ans_ids = _pad([it["ans_ids"] for it in items], max_a, pad_id)
        ans_attn = (ans_ids != pad_id).long()
        cot_ids = _pad([it["cot_ids"] for it in items], max_c, pad_id)

        anchor_input_ids = _pad(
            [it["anchor_input_ids"] for it in items], max_anchor, pad_id
        )
        anchor_attn = torch.zeros((len(items), max_anchor), dtype=torch.long)
        for i, it in enumerate(items):
            anchor_attn[i, : len(it["anchor_input_ids"])] = 1

        # ans_endpoint_input_ids = q + cot + ans, used for z_1 = end-of-answer
        # hidden state extraction (variant include_z1=v1_endofans).
        ans_endpoint_input_ids = _pad(
            [it["ans_endpoint_input_ids"] for it in items], max_endpt, pad_id
        )
        ans_endpoint_attn = torch.zeros((len(items), max_endpt), dtype=torch.long)
        for i, it in enumerate(items):
            ans_endpoint_attn[i, : len(it["ans_endpoint_input_ids"])] = 1
        ans_endpoint_pos = torch.tensor(
            [it["ans_endpoint_pos"] for it in items], dtype=torch.long
        )

        cot_step_positions = torch.stack([
            torch.tensor(it["cot_step_positions"], dtype=torch.long)
            for it in items
        ])  # (B, M+1)
        assert cot_step_positions.shape[1] == M + 1, (
            f"cot_step_positions has shape {tuple(cot_step_positions.shape)}, "
            f"expected (..., {M + 1})"
        )

        cot_lengths = torch.tensor(
            [it["cot_length"] for it in items], dtype=torch.long
        )
        cot_start_positions = torch.tensor(
            [it["cot_start_position"] for it in items], dtype=torch.long
        )

        return {
            "q_input_ids": q_ids,
            "q_attn_mask": q_attn,
            "ans_ids": ans_ids,
            "ans_attn_mask": ans_attn,
            "cot_ids": cot_ids,
            "anchor_input_ids": anchor_input_ids,
            "anchor_attn_mask": anchor_attn,
            "ans_endpoint_input_ids": ans_endpoint_input_ids,
            "ans_endpoint_attn_mask": ans_endpoint_attn,
            "ans_endpoint_pos": ans_endpoint_pos,
            "cot_step_positions": cot_step_positions,
            "cot_lengths": cot_lengths,
            "cot_start_positions": cot_start_positions,
            "answers": [it["answer"] for it in items],
            "questions": [it["question"] for it in items],
        }

    return collate


# ---------------------------------------------------------------------------
# Model  (frozen anchor LM + suffix decoder + trainable q-encoder + DiT)
# ---------------------------------------------------------------------------


_DIT_PRESETS = {
    "ditS":  dict(hidden_size=384,  depth=12, num_heads=6),
    "ditB":  dict(hidden_size=768,  depth=12, num_heads=12),
    "ditL":  dict(hidden_size=1024, depth=24, num_heads=16),
    "ditXL": dict(hidden_size=1152, depth=28, num_heads=16),
    "ditH":  dict(hidden_size=1536, depth=32, num_heads=24),  # bigger than XL
}


class AnchoredPhase2Model(nn.Module):
    def __init__(
        self,
        *,
        v1_lm_ckpt: str,
        v1_decoder_ckpt: str,
        v1_pool_mode: str,
        decoder_base_model: str = "Qwen/Qwen3-0.6B",
        q_encoder_base_model: str = "Qwen/Qwen3-0.6B",
        dit_arch: str = "ditXL",
        n_latent_tokens: int = 1,
        K_total: int = 48,
        gradient_checkpointing: bool = True,
        # Trajectory-shape ablation flags (one change at a time):
        #   include_z1: which (if any) z_1 control point to APPEND after
        #     the K cot anchors. "none" (default) = K+1 anchors; the
        #     other options append a (K+1)-th control point of various
        #     flavors:
        #       "v1_endofans" = anchor LM hidden state at the LAST ANSWER
        #          token (just before EOS) over [q, cot, ans]. Cheapest;
        #          requires no extra training.
        #       "ae_new"      = a separately trained AE's z_1. Requires a
        #          pre-trained AE checkpoint that produces vectors in the
        #          same dim as the anchor LM's d_model (1024).
        #       "ae_new_smooth" = same as ae_new but trained with an MSE
        #          smoothness penalty pulling z_1 toward the last cot
        #          anchor. (TBD: not yet wired.)
        #   z0_source: where the FIRST control point comes from.
        #     "none" (default) = z^a_0 from anchor LM (encoding of q only).
        #     "noise" = sphere-normed Gaussian noise sample per example.
        include_z1: str = "none",
        z0_source: str = "none",
        ae_z1_ckpt: Optional[str] = None,
        # Anchor-dimension projection: keep anchors at their native
        # 1024-d for compatibility with the frozen probe-decoder, but
        # have the DiT operate in a smaller D_proj-d subspace. Projects
        # via P (D_proj, D_native); back-projects via P^T at inference.
        anchor_proj_method: str = "none",        # "none" / "truncate" / "svd"
        anchor_proj_dim: Optional[int] = None,
        anchor_proj_path: Optional[str] = None,  # required for "svd"
    ):
        super().__init__()
        self.K_total = K_total
        self.include_z1 = include_z1
        self.z0_source = z0_source
        self.ae_z1_ckpt = ae_z1_ckpt
        self.anchor_proj_method = anchor_proj_method
        self.anchor_proj_dim = anchor_proj_dim
        self.anchor_proj_path = anchor_proj_path
        if include_z1 not in ("none", "v1_endofans", "ae_new", "ae_new_smooth", "ans_ae"):
            raise ValueError(f"unknown include_z1={include_z1!r}")
        if z0_source not in ("none", "noise"):
            raise ValueError(f"unknown z0_source={z0_source!r}")
        if include_z1 in ("ae_new", "ae_new_smooth") and ae_z1_ckpt is None:
            raise ValueError(
                f"include_z1={include_z1} requires --ae_z1_ckpt"
            )

        assert v1_pool_mode in ("mean", "last_token", "mean_cot_only")
        self._build_encoder(v1_lm_ckpt, v1_decoder_ckpt, v1_pool_mode,
                            decoder_base_model)

        # k value to feed the decoder at inference. The probe decoder was
        # trained with k uniform on {0..k_max}. At phase-2 inference, we run
        # the decoder at k=K_total -- the pad_to_M "answer head" position
        # where the encoder has seen all available cot.
        self.k_for_decode = K_total

        # === Trainable: question encoder (separate Qwen3 trunk for cross-attn) ===
        self.q_encoder = AutoModelForCausalLM.from_pretrained(
            q_encoder_base_model, dtype=torch.float32
        )
        if gradient_checkpointing:
            self.q_encoder.gradient_checkpointing_enable()
            if hasattr(self.q_encoder, "enable_input_require_grads"):
                self.q_encoder.enable_input_require_grads()
        self.q_kv_dim = self.q_encoder.config.hidden_size

        # === Anchor projection (optional dim reduction for the DiT) ===
        # The probe-decoder was trained at the NATIVE anchor dim; we keep
        # that for back-projection at inference. The DiT operates in
        # D_proj. proj_matrix has shape (D_proj, D_native): forward
        # x_proj = x @ P^T, back x_back = x_proj @ P.
        self.D_lat_native = self.D_lat
        if anchor_proj_method == "none":
            self.proj_matrix = None
        else:
            assert anchor_proj_dim is not None and anchor_proj_dim < self.D_lat_native, (
                f"anchor_proj_dim={anchor_proj_dim} must be < native {self.D_lat_native}"
            )
            if anchor_proj_method == "truncate":
                P = torch.zeros(anchor_proj_dim, self.D_lat_native)
                for i in range(anchor_proj_dim):
                    P[i, i] = 1.0
            elif anchor_proj_method == "svd":
                if anchor_proj_path is None or not Path(anchor_proj_path).exists():
                    raise ValueError(
                        f"anchor_proj_method=svd requires --anchor_proj_path "
                        f"pointing to a precomputed projection tensor of shape "
                        f"({anchor_proj_dim}, {self.D_lat_native}); got {anchor_proj_path}"
                    )
                P = torch.load(anchor_proj_path, map_location="cpu",
                               weights_only=False).to(torch.float32)
                assert P.shape == (anchor_proj_dim, self.D_lat_native), (
                    f"loaded proj has shape {tuple(P.shape)}; expected "
                    f"({anchor_proj_dim}, {self.D_lat_native})"
                )
            else:
                raise ValueError(f"unknown anchor_proj_method={anchor_proj_method!r}")
            self.register_buffer("proj_matrix", P)
            self.D_lat = anchor_proj_dim
            print(f"[anchored-p2] anchor_proj: {anchor_proj_method} "
                  f"D_native={self.D_lat_native} -> D_proj={self.D_lat}")

        # === Trainable: DiT denoiser (latent_dim = self.D_lat) ===
        if dit_arch not in _DIT_PRESETS:
            raise ValueError(
                f"Unknown dit_arch={dit_arch}; choose from {list(_DIT_PRESETS)}"
            )
        ad = _DIT_PRESETS[dit_arch]
        self.denoiser = DiTDenoiser(
            latent_dim=self.D_lat,
            hidden_size=ad["hidden_size"],
            depth=ad["depth"],
            num_heads=ad["num_heads"],
            kv_dim=self.q_kv_dim,
            n_latent_tokens=n_latent_tokens,
            mlp_ratio=4.0,
        )

        self._sphere_scale = float(self.D_lat ** 0.5)
        self._sphere_scale_native = float(self.D_lat_native ** 0.5)

        # Optional: load a separately trained AE-z_1 model (Qwen3-based
        # target encoder reading [q, sep1, cot, sep2, ans]) when
        # include_z1 in {ae_new, ae_new_smooth}. We keep its target encoder
        # frozen and use its forward to compute z_1 per example at training
        # / eval time.
        self.ae_z1_encoder = None
        if include_z1 in ("ae_new", "ae_new_smooth", "ans_ae"):
            self._build_ae_z1(ae_z1_ckpt)

    def _build_ae_z1(self, ae_ckpt: str):
        """Load the frozen target encoder of a Qwen3-based AE trained by
        scripts/train_z1_ae.py. The AE produces z_1 in d=1024 (matches
        the anchor LM's d_model). The target encoder is loaded as a stand-alone
        Qwen3 LM + sphere-norm logic in forward().
        """
        ck = torch.load(ae_ckpt, map_location="cpu", weights_only=False)
        ae_args = ck.get("args", {}) or {}
        target_model_name = ae_args.get("target_model_name", "Qwen/Qwen3-0.6B")
        ae_lm = AutoModelForCausalLM.from_pretrained(
            target_model_name, dtype=torch.float32
        )
        # The target encoder ckpt key prefixes are ``lm.<name>``; restore.
        target_state = {
            k.replace("lm.", "", 1): v
            for k, v in ck["target_encoder_state"].items()
            if k.startswith("lm.")
        }
        missing, unexpected = ae_lm.load_state_dict(target_state, strict=False)
        if missing or unexpected:
            print(f"[anchored-p2] AE-z_1 target_encoder load: "
                  f"missing={len(missing)} unexpected={len(unexpected)}")
        for p in ae_lm.parameters():
            p.requires_grad = False
        ae_lm.eval()
        self.ae_z1_encoder = ae_lm
        self.ae_z1_d = ae_lm.config.hidden_size  # 1024 for Qwen3-0.6B
        if self.ae_z1_d != self.D_lat:
            print(f"[anchored-p2] WARNING: ae_z1_d={self.ae_z1_d} != "
                  f"D_lat={self.D_lat}; trajectory will mix dims if used.")

    def _build_encoder(self, v1_lm_ckpt, v1_decoder_ckpt, v1_pool_mode,
                       decoder_base_model):
        # Frozen anchor LM (anchor source)
        self.v1_lm = AutoModelForCausalLM.from_pretrained(
            v1_lm_ckpt, dtype=torch.float32
        )
        for p in self.v1_lm.parameters():
            p.requires_grad = False
        self.v1_lm.eval()
        self.D_lat = self.v1_lm.config.hidden_size
        self.v1_pool_mode = v1_pool_mode

        # Frozen probe-decoder (answer head at inference)
        ck = torch.load(v1_decoder_ckpt, map_location="cpu", weights_only=False)
        ck_args = ck.get("args", {}) or {}
        decoder_K = (ck_args.get("k_max") or ck_args.get("M") or self.K_total)
        d_model_anchor = ck_args.get("d_model_anchor", self.D_lat)
        self.anchor_decoder = SuffixDecoder(
            decoder_model_name=decoder_base_model,
            d_model_anchor=d_model_anchor,
            K=decoder_K,
            gradient_checkpointing=False,
        )
        self.anchor_decoder.load_state_dict(ck["decoder_state"])
        for p in self.anchor_decoder.parameters():
            p.requires_grad = False
        self.anchor_decoder.eval()
        self.decoder_K = decoder_K

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ---- helpers ----
    def _sphere(self, z):
        return F.normalize(z, dim=-1) * self._sphere_scale

    def _sphere_native(self, z):
        return F.normalize(z, dim=-1) * self._sphere_scale_native

    def _project_anchor(self, anchors):
        """Project (..., D_native) -> (..., D_proj) and re-sphere-norm."""
        if self.proj_matrix is None:
            return anchors
        z = torch.matmul(anchors, self.proj_matrix.t())
        return self._sphere(z)

    def _backproject(self, z_proj):
        """Back-project (..., D_proj) -> (..., D_native), then sphere-norm
        in native space. With orthonormal SVD basis, x_back = x_proj @ P
        is the optimal least-squares reconstruction in the rank-D_proj
        subspace; the sphere-norm in the original D_native space then
        gives the probe-decoder a vector with the norm it was trained on."""
        if self.proj_matrix is None:
            return z_proj
        z_back = torch.matmul(z_proj, self.proj_matrix)
        return self._sphere_native(z_back)

    def _question_kv(self, q_ids, q_attn):
        out = self.q_encoder(
            input_ids=q_ids,
            attention_mask=q_attn,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        kv = out.hidden_states[-1]  # (B, T_q, D_qwen)
        return kv

    def _extract_v1_endofans_z1(self, batch):
        """z_1 = anchor LM hidden state at LAST ANSWER token over [q, cot, ans].

        Returns (B, D_lat) sphere-normed (in projected space if proj on).
        """
        with torch.no_grad():
            anchors = extract_anchors(
                self.v1_lm,
                batch["ans_endpoint_input_ids"],
                batch["ans_endpoint_attn_mask"],
                batch["ans_endpoint_pos"].unsqueeze(-1),  # (B, 1)
                pool_mode=self.v1_pool_mode,
                sphere_norm=True,
            )
        z = anchors.squeeze(1)  # (B, D_native)
        if self.proj_matrix is not None:
            z = self._project_anchor(z)
        return z

    def _extract_ae_new_z1(self, batch):
        """z_1 = AE target encoder's last-token sphere-normed output.

        For include_z1 in {ae_new, ae_new_smooth}: encoder forwards over
        the full sequence (we approximate via ans_endpoint_input_ids =
        q+cot+ans).
        For include_z1=ans_ae: encoder forwards over ANS TOKENS ONLY,
        matching its training input. This is the principled "z_1 = AE(ans)"
        trajectory endpoint.

        Returns (B, ae_z1_d) sphere-normed.
        """
        if self.include_z1 == "ans_ae":
            input_ids = batch["ans_ids"]
            attn = batch["ans_attn_mask"]
        else:
            # Sep-aware matching would need tgt_ids; we use q+cot+ans as
            # a close approximation since the trunk is the same Qwen3.
            input_ids = batch["ans_endpoint_input_ids"]
            attn = batch["ans_endpoint_attn_mask"]
        with torch.no_grad():
            out = self.ae_z1_encoder(
                input_ids=input_ids, attention_mask=attn,
                output_hidden_states=True, use_cache=False, return_dict=True,
            )
            h = out.hidden_states[-1]
            last_idx = (attn.sum(dim=1) - 1).clamp_min(0)
            B = h.shape[0]
            z = h[torch.arange(B, device=h.device), last_idx]
            z = F.normalize(z, dim=-1) * (self.ae_z1_d ** 0.5)
        return z.detach()

    def _extract_anchors(self, batch):
        """Extract (B, M+1, D_lat) anchors from the frozen anchor LM.

        The encoder input is [q, cot] (no sep1/sep2), matching how the
        probe decoder was trained. We forward over [q, cot], gather hidden
        states at the K+1 cot_step_positions, then pool and project.
        """
        with torch.no_grad():
            kw = {}
            if self.v1_pool_mode == "mean_cot_only":
                kw["cot_start_positions"] = batch["cot_start_positions"]
            anchors = extract_anchors(
                self.v1_lm,
                batch["anchor_input_ids"],
                batch["anchor_attn_mask"],
                batch["cot_step_positions"],
                pool_mode=self.v1_pool_mode,
                sphere_norm=True,
                **kw,
            )
        return self._project_anchor(anchors)

    def _build_trajectory_anchors(self, batch):
        """Build the FULL trajectory control-point tensor according to the
        configured z_0 / z_1 options.

        Default (no flags): K+1 anchors [z^a_0, z^a_1, ..., z^a_K] from anchor LM.

        With include_z1 != "none": appends z_1 as the (K+1)-th control point,
        giving K+2 anchors [z^a_0, ..., z^a_K, z_1].

        With z0_source == "noise": REPLACES z^a_0 (the first control point)
        with sphere-normed Gaussian noise per example, giving K+1 or K+2
        anchors with z_0 = noise as the first.

        Returns (B, n_ctrl, D) sphere-normed anchors.
        """
        anchors = self._extract_anchors(batch)               # (B, K+1, D)
        B, K_plus_1, D = anchors.shape

        if self.z0_source == "noise":
            # Replace anchors[:, 0, :] with sampled noise.
            eps0 = torch.randn(B, D, device=anchors.device)
            anchors = torch.cat(
                [self._sphere(eps0).unsqueeze(1), anchors[:, 1:, :]],
                dim=1,
            )

        if self.include_z1 == "v1_endofans":
            z1 = self._extract_v1_endofans_z1(batch)         # (B, D)
            anchors = torch.cat([anchors, z1.unsqueeze(1)], dim=1)
        elif self.include_z1 in ("ae_new", "ae_new_smooth", "ans_ae"):
            z1 = self._extract_ae_new_z1(batch)              # (B, D_ae)
            # We require ae_z1_d == D_lat for direct concatenation. The
            # train_z1_ae.py default is Qwen3-0.6B with D=1024, matching
            # the anchor LM's d_model. If a future config breaks this, we'd need a
            # projection here.
            assert z1.shape[-1] == anchors.shape[-1], (
                f"AE-z_1 dim {z1.shape[-1]} != anchor dim {anchors.shape[-1]}"
            )
            anchors = torch.cat([anchors, z1.unsqueeze(1)], dim=1)

        return anchors

    # ---- training ----
    def forward(self, batch):
        device = batch["q_input_ids"].device
        ctrl = self._build_trajectory_anchors(batch)         # (B, n_ctrl, D)
        B, n_ctrl, D = ctrl.shape
        n_seg = n_ctrl - 1                                    # number of segments

        t = torch.rand(B, device=device).clamp(1e-3, 1.0 - 1e-3)
        seg = (t * n_seg).floor().long().clamp_max(n_seg - 1)
        s = (t * n_seg - seg.float()).unsqueeze(-1)           # (B, 1)

        z_a_k   = torch.gather(
            ctrl, 1, seg[:, None, None].expand(B, 1, D)
        ).squeeze(1)
        z_a_kp1 = torch.gather(
            ctrl, 1, (seg + 1)[:, None, None].expand(B, 1, D)
        ).squeeze(1)
        z_clean = (1 - s) * z_a_k + s * z_a_kp1                # (B, D)

        eps = self._sphere(torch.randn_like(z_clean))
        z_t = (1 - t.unsqueeze(-1)) * eps + t.unsqueeze(-1) * z_clean

        kv = self._question_kv(batch["q_input_ids"], batch["q_attn_mask"])
        kv_pad_mask = batch["q_attn_mask"] == 0
        x0_hat_raw = self.denoiser(z_t, t, kv, kv_pad_mask)
        x0_hat = self._sphere(x0_hat_raw)

        loss = F.mse_loss(x0_hat, z_clean)

        # Diagnostics
        with torch.no_grad():
            x0_z_clean_dist = (x0_hat - z_clean).norm(dim=-1).mean().item()
            adj_norm = (ctrl[:, 1:, :] - ctrl[:, :-1, :]).norm(dim=-1).mean().item()
        return {
            "loss": loss,
            "loss_x0_mse": loss.detach(),
            "x0_z_clean_dist": x0_z_clean_dist,
            "anchor_adj_norm_mean": adj_norm,
            "z_clean_norm": z_clean.norm(dim=-1).mean().item(),
            "n_ctrl": n_ctrl,
        }

    # ---- inference: integrate noise -> z, decode answer ----
    @torch.no_grad()
    def generate_answers(self, batch, n_steps: int = 64, max_new_tokens: int = 64):
        device = batch["q_input_ids"].device
        B = batch["q_input_ids"].shape[0]

        kv = self._question_kv(batch["q_input_ids"], batch["q_attn_mask"])
        kv_pad_mask = batch["q_attn_mask"] == 0

        # Start from sphere-normed Gaussian noise
        eps = torch.randn(B, self.D_lat, device=device)
        z = self._sphere(eps)

        # Heun integrator
        ts = torch.linspace(0, 1, n_steps + 1, device=device)
        for i in range(n_steps):
            t_curr = ts[i].expand(B)
            t_next = ts[i + 1].expand(B)
            dt = (t_next - t_curr).unsqueeze(-1)

            x0_hat_1 = self._sphere(self.denoiser(z, t_curr, kv, kv_pad_mask))
            v1 = (x0_hat_1 - z) / (1 - t_curr.unsqueeze(-1)).clamp(min=1e-3)
            # Heun second-order correction
            z_pred = z + dt * v1
            t_next_b = t_next
            x0_hat_2 = self._sphere(self.denoiser(z_pred, t_next_b, kv, kv_pad_mask))
            v2 = (x0_hat_2 - z_pred) / (1 - t_next_b.unsqueeze(-1)).clamp(min=1e-3)
            z = z + 0.5 * dt * (v1 + v2)

        # z_at_t=1 ~ z^a_K. Decode answer via the frozen variant decoder
        # at k=K_total ("answer head" position). If we're operating in
        # a projected subspace, back-project to D_native and re-sphere
        # before feeding the probe.
        z_for_probe = self._backproject(z) if self.proj_matrix is not None else z
        k_tensor = torch.full((B,), self.k_for_decode, dtype=torch.long, device=device)
        gen = self.anchor_decoder.generate_suffix(
            q_ids=batch["q_input_ids"],
            q_attn=batch["q_attn_mask"],
            z_a_k=z_for_probe,
            k=k_tensor,
            max_new_tokens=max_new_tokens,
            bos_id=None,
            eos_id=None,
        )
        return gen, z


# ---------------------------------------------------------------------------
# Eval: greedy answer EM on val
# ---------------------------------------------------------------------------


def _extract_answer_after_sep2(text: str) -> str:
    """Extract the answer that the suffix decoder emitted after the
    first ' #### ' SEP2 marker. We then also stop at the NEXT '####'
    (a known failure mode where the decoder repeats '#### ans #### ans
    #### ...') or at a newline. Final whitespace is stripped.

    Generic across all 5 tasks; the per-task answer_normalizer is
    applied separately by the caller.
    """
    if "####" in text:
        tail = text.split("####", 1)[1]
    else:
        tail = text
    if "####" in tail:
        tail = tail.split("####", 1)[0]
    if "\n" in tail:
        tail = tail.split("\n", 1)[0]
    return tail.strip()


@torch.no_grad()
def eval_val_em(model: AnchoredPhase2Model, val_loader, tokenizer, device,
                max_examples=200, answer_normalizer=None,
                max_new_tokens: int = 64, n_steps: int = 64):
    """Greedy answer-EM eval. answer_normalizer is a callable str -> str
    applied to BOTH the predicted answer (extracted after '####') and
    the gold answer. Defaults to a stripped numeric normalizer for
    backward-compat with the original GSM8K-only behavior; pass an
    explicit normalizer (e.g. from reasonflow.utils.answer_normalizers.get_normalizer)
    for non-numeric tasks.

    max_new_tokens controls the suffix-decoder generation budget. For
    GSM8K-formulas 64 is plenty; for tasks with long SMILES answers
    (mech, retro), bump to 256-384.
    """
    if answer_normalizer is None:
        def answer_normalizer(s):
            s = s.strip().replace(",", "")
            while s.endswith("."):
                s = s[:-1]
            return s

    model.eval()
    n_correct = 0
    n_total = 0
    for batch in val_loader:
        batch_dev = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        gen, _ = model.generate_answers(
            batch_dev, n_steps=n_steps, max_new_tokens=max_new_tokens,
        )
        for b in range(gen.shape[0]):
            text = tokenizer.decode(gen[b].tolist(), skip_special_tokens=True)
            pred_raw = _extract_answer_after_sep2(text)
            pred = answer_normalizer(pred_raw)
            gold = answer_normalizer(batch_dev["answers"][b])
            if pred == gold and pred != "":
                n_correct += 1
            n_total += 1
            if n_total >= max_examples:
                break
        if n_total >= max_examples:
            break
    return n_correct / max(1, n_total), n_total


def cosine_lr(step, warmup, total, base, min_lr):
    if step < warmup:
        return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * p))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1_lm_ckpt", default=None,
                        help="Path to the finetuned anchor LM (HF format dir).")
    parser.add_argument("--v1_decoder_ckpt", default=None,
                        help="Path to the SuffixDecoder ckpt (last_checkpoint.pt).")
    parser.add_argument("--v1_pool_mode",
                        choices=["mean", "last_token", "mean_cot_only"], default=None)
    parser.add_argument("--decoder_base_model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--q_encoder_base_model", default="Qwen/Qwen3-0.6B")

    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--M", type=int, default=48)
    parser.add_argument("--max_q_len", type=int, default=192)
    parser.add_argument("--cot_max_len", type=int, default=80)
    parser.add_argument("--ans_max_len", type=int, default=24)
    parser.add_argument("--dit_arch", choices=list(_DIT_PRESETS), default="ditXL")
    parser.add_argument("--n_latent_tokens", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--val_every", type=int, default=2000)
    parser.add_argument("--val_em_every", type=int, default=5000)
    parser.add_argument("--val_em_n", type=int, default=200)
    parser.add_argument("--val_em_max_new_tokens", type=int, default=64,
                        help="Suffix-decoder generation budget during the "
                             "training-time val_em check. Default 64 is "
                             "fine for gsm8k/prontoqa; bump to 256-384 "
                             "for mech/retro/molpuzzle so the answer "
                             "isn't truncated mid-SMILES.")
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--max_train_examples", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", type=lambda s: s.lower() == "true",
                        default=True)

    # Trajectory-shape ablation flags. Default values reproduce the
    # K+1-anchor baseline (no z_1 added, z_0 = z^a_0).
    parser.add_argument(
        "--include_z1",
        choices=["none", "v1_endofans", "ae_new", "ae_new_smooth", "ans_ae"],
        default="none",
        help="If non-'none', append a z_1 control point after the K cot "
             "anchors, giving K+2 total. v1_endofans extracts the anchor LM's "
             "hidden state at the LAST ANSWER TOKEN over [q, cot, ans]. "
             "ae_new / ae_new_smooth load a pre-trained AE-z_1 (--ae_z1_ckpt).",
    )
    parser.add_argument(
        "--z0_source",
        choices=["none", "noise"],
        default="none",
        help="If 'noise', REPLACE the first control point (z^a_0 = anchor LM "
             "encoding of q only) with sphere-normed Gaussian noise per "
             "example. The trajectory then starts at noise and traces "
             "through cot anchors (and optionally z_1).",
    )
    parser.add_argument(
        "--ae_z1_ckpt",
        default=None,
        help="(z_1 = ae_new[_smooth]) Path to a separately trained AE "
             "checkpoint that produces 1024-dim sphere-normed z_1 over "
             "[q, cot, ans]. Wired up in a follow-up commit.",
    )
    parser.add_argument(
        "--answer_normalizer",
        default="numeric",
        help="Per-task answer normalizer name (utils/answer_normalizers.py): "
             "numeric / smiles / set_smiles / classify. Used by val EM "
             "eval AND best-val-em ckpt selection. Default 'numeric' "
             "preserves the GSM8K-formulas behavior.",
    )
    # Anchor-dim projection for the DiT (probe-decoder still sees native D)
    parser.add_argument(
        "--anchor_proj_method",
        choices=["none", "truncate", "svd"], default="none",
        help="If not 'none', the DiT operates in a projected subspace of "
             "the anchor space. truncate = take first --anchor_proj_dim "
             "dims. svd = load a precomputed orthonormal projection from "
             "--anchor_proj_path (rows are top right-singular-vectors of "
             "the train anchor matrix).",
    )
    parser.add_argument("--anchor_proj_dim", type=int, default=None,
                        help="Projected DiT latent dim (e.g. 256).")
    parser.add_argument("--anchor_proj_path", type=str, default=None,
                        help="Path to precomputed SVD projection tensor "
                             "of shape (--anchor_proj_dim, native_dim).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    val_em_path  = args.output_dir / "val_em_curve.json"

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[anchored-p2] device={device}, output={args.output_dir}")

    tok = AutoTokenizer.from_pretrained(args.decoder_base_model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    print(f"[anchored-p2] tok pad={tok.pad_token_id} eos={tok.eos_token_id} "
          f"bos={tok.bos_token_id}")

    assert args.v1_lm_ckpt and args.v1_decoder_ckpt and args.v1_pool_mode, (
        "--v1_lm_ckpt, --v1_decoder_ckpt, --v1_pool_mode are required"
    )
    print(f"[anchored-p2] Building model: LM={args.v1_lm_ckpt}, "
          f"decoder={args.v1_decoder_ckpt}, pool={args.v1_pool_mode}, "
          f"dit={args.dit_arch}, n_lat={args.n_latent_tokens}")

    model = AnchoredPhase2Model(
        v1_lm_ckpt=args.v1_lm_ckpt,
        v1_decoder_ckpt=args.v1_decoder_ckpt,
        v1_pool_mode=args.v1_pool_mode,
        decoder_base_model=args.decoder_base_model,
        q_encoder_base_model=args.q_encoder_base_model,
        dit_arch=args.dit_arch,
        n_latent_tokens=args.n_latent_tokens,
        K_total=args.M,
        gradient_checkpointing=args.gradient_checkpointing,
        include_z1=args.include_z1,
        z0_source=args.z0_source,
        ae_z1_ckpt=args.ae_z1_ckpt,
        anchor_proj_method=args.anchor_proj_method,
        anchor_proj_dim=args.anchor_proj_dim,
        anchor_proj_path=args.anchor_proj_path,
    ).to(device)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"[anchored-p2] D_lat={model.D_lat}, kv_dim={model.q_kv_dim}, "
          f"trainable params: {n_train:,}")

    print(f"[anchored-p2] Loading datasets from {args.data_dir}")
    train_ds = AnchoredPhase2Dataset(
        args.data_dir / "train.json", tok,
        M=args.M, max_q_len=args.max_q_len,
        cot_max_len=args.cot_max_len, ans_max_len=args.ans_max_len,
        max_examples=args.max_train_examples or None,
    )
    val_ds = AnchoredPhase2Dataset(
        args.data_dir / "val.json", tok,
        M=args.M, max_q_len=args.max_q_len,
        cot_max_len=args.cot_max_len, ans_max_len=args.ans_max_len,
    )
    print(f"[anchored-p2] train={len(train_ds)}, val={len(val_ds)}")

    coll = make_collator(tok.pad_token_id, M=args.M)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=coll, num_workers=args.num_workers, drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=coll, num_workers=2,
    )

    optim = torch.optim.AdamW(
        model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    answer_norm = get_normalizer(args.answer_normalizer)
    print(f"[anchored-p2] answer_normalizer={args.answer_normalizer}")

    best_val_em = -1.0
    val_em_curve = []
    step = 0
    t0 = time.time()
    train_iter = iter(train_loader)

    print(f"[anchored-p2] Starting training for {args.max_steps} steps")
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        lr = cosine_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for g in optim.param_groups:
            g["lr"] = lr

        model.train()
        out = model(batch)
        loss = out["loss"]
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.max_grad_norm)
        optim.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"[step {step:6d}] loss_x0_mse={out['loss_x0_mse'].item():.4f} "
                  f"|x0-z_clean|={out['x0_z_clean_dist']:.3f} "
                  f"adj_norm={out['anchor_adj_norm_mean']:.3f} "
                  f"|z_clean|={out['z_clean_norm']:.3f} "
                  f"lr={lr:.2e} elapsed={elapsed:.0f}s")
            with metrics_path.open("a") as fh:
                fh.write(json.dumps({
                    "step": step,
                    "loss_x0_mse": float(out["loss_x0_mse"].item()),
                    "x0_z_clean_dist": out["x0_z_clean_dist"],
                    "anchor_adj_norm_mean": out["anchor_adj_norm_mean"],
                    "z_clean_norm": out["z_clean_norm"],
                    "lr": lr,
                    "elapsed_s": elapsed,
                }) + "\n")

        if args.val_em_every > 0 and step > 0 and step % args.val_em_every == 0:
            print(f"[step {step:6d}] running val EM eval (n={args.val_em_n})...")
            em, n_eval = eval_val_em(model, val_loader, tok, device,
                                     max_examples=args.val_em_n,
                                     answer_normalizer=answer_norm,
                                     max_new_tokens=args.val_em_max_new_tokens)
            val_em_curve.append({"step": step, "val_em": em, "n": n_eval})
            print(f"[step {step:6d}] val_em={em:.4f} (n={n_eval})")
            with val_em_path.open("w") as fh:
                json.dump(val_em_curve, fh, indent=2)
            with metrics_path.open("a") as fh:
                fh.write(json.dumps({"step": step, "val_em": em, "val_em_n": n_eval}) + "\n")
            if em > best_val_em:
                best_val_em = em
                ckpt_path = args.output_dir / "best_val_em.pt"
                state = {
                    "step": step,
                    "best_val_em": best_val_em,
                    "denoiser_state": model.denoiser.state_dict(),
                    "q_encoder_state": model.q_encoder.state_dict(),
                    "args": vars(args),
                }
                torch.save(state, ckpt_path)
                print(f"[step {step:6d}] saved best_val_em ({best_val_em:.4f})")

        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            ckpt_path = args.output_dir / "last_checkpoint.pt"
            state = {
                "step": step,
                "denoiser_state": model.denoiser.state_dict(),
                "q_encoder_state": model.q_encoder.state_dict(),
                "args": vars(args),
            }
            torch.save(state, ckpt_path)

        step += 1

    # Final eval + save (after loop, regardless of step alignment)
    print(f"[anchored-p2] Final val EM eval (n={args.val_em_n})...")
    final_em, final_n = eval_val_em(model, val_loader, tok, device,
                                    max_examples=args.val_em_n,
                                    answer_normalizer=answer_norm,
                                    max_new_tokens=args.val_em_max_new_tokens)
    val_em_curve.append({"step": step, "val_em": final_em, "n": final_n,
                         "stage": "final"})
    print(f"[anchored-p2] FINAL val_em={final_em:.4f} (n={final_n})")
    with val_em_path.open("w") as fh:
        json.dump(val_em_curve, fh, indent=2)
    with metrics_path.open("a") as fh:
        fh.write(json.dumps({"step": step, "val_em": final_em,
                             "val_em_n": final_n, "stage": "final"}) + "\n")
    if final_em > best_val_em:
        best_val_em = final_em
        torch.save({
            "step": step,
            "best_val_em": best_val_em,
            "denoiser_state": model.denoiser.state_dict(),
            "q_encoder_state": model.q_encoder.state_dict(),
            "args": vars(args),
        }, args.output_dir / "best_val_em.pt")

    state = {
        "step": step,
        "denoiser_state": model.denoiser.state_dict(),
        "q_encoder_state": model.q_encoder.state_dict(),
        "args": vars(args),
    }
    torch.save(state, args.output_dir / "last_checkpoint.pt")
    print(f"[anchored-p2] Training complete. Best val EM = {best_val_em:.4f}, "
          f"final val EM = {final_em:.4f}")


if __name__ == "__main__":
    main()
