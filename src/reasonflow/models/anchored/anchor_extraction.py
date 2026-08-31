"""Anchor extraction.

Anchors are read off the middle-layer hidden states of a finetuned causal
LM at each cot-step boundary. There is no trainable projection: the anchors
live in the LM's d_model space, and downstream consumers (the suffix decoder
and the diffusion DiT) absorb the dimensionality mismatch via their own
trainable input projections.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Middle-layer pooling (post-hoc, no trainable params)
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_anchors(
    lm: AutoModelForCausalLM,
    tgt_input_ids: torch.Tensor,        # (B, T)
    tgt_attn_mask: torch.Tensor,        # (B, T)
    cot_step_positions: torch.Tensor,   # (B, K+1) long
    *,
    middle_layer: Optional[int] = None, # default: len(hidden_states) // 2
    pool_mode: str = "mean",            # "mean" / "last_token" / "mean_cot_only"
    sphere_norm: bool = True,
    cot_start_positions: Optional[torch.Tensor] = None,   # (B,) long; required
                                                          # for pool_mode=
                                                          # "mean_cot_only"
) -> torch.Tensor:
    """Extract per-step anchor latents from a finetuned causal LM.

    Three pool modes:
        * "mean": z_a_k = mean over j=0..pos_k of h_middle[j]
                  (covers question + cot prefix; long shared question
                  dominates the average and reduces anchor distinguishability)
        * "last_token": z_a_k = h_middle[pos_k]
                        (just the hidden state at the boundary position)
        * "mean_cot_only": z_a_k = mean over j=cot_start..pos_k of h_middle[j]
                          (cot tokens only; question excluded from the mean.
                          For pos_k < cot_start (i.e. k=0 where the boundary
                          is the last q token), falls back to last_token.)

    All pool from the same middle layer, then sphere-norm.

    Args:
        lm: a Qwen3-0.6B (or any HF causal LM) instance, in eval mode.
        tgt_input_ids: (B, T) the joined (q, sep1, cot, sep2, ans) token ids.
        tgt_attn_mask: (B, T) the attention mask for tgt_input_ids.
        cot_step_positions: (B, K+1) the K+1 boundary positions.
        middle_layer: which transformer layer's hidden state to extract.
            Default = len(hidden_states) // 2.
        pool_mode: "mean", "last_token", or "mean_cot_only".
        sphere_norm: if True, sphere-norm each anchor to ||z|| = sqrt(D).
        cot_start_positions: (B,) the index of the first cot token per
            example (= len(q_ids)). Required for "mean_cot_only".

    Returns:
        anchors: (B, K+1, D_qwen)
    """
    out = lm(
        input_ids=tgt_input_ids,
        attention_mask=tgt_attn_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    if middle_layer is None:
        middle_layer = len(out.hidden_states) // 2

    h = out.hidden_states[middle_layer]              # (B, T, D)
    B, T, D = h.shape
    K_plus_1 = cot_step_positions.shape[1]

    if pool_mode == "mean":
        mask = tgt_attn_mask.to(h.dtype).unsqueeze(-1)   # (B, T, 1)
        h_masked = h * mask
        cum = h_masked.cumsum(dim=1)                     # (B, T, D)
        cnt = mask.cumsum(dim=1)                         # (B, T, 1)
        idx = cot_step_positions.clamp(min=0, max=T - 1).unsqueeze(-1)
        idx_d = idx.expand(B, K_plus_1, D)
        cum_at = torch.gather(cum, dim=1, index=idx_d)
        cnt_at = torch.gather(cnt, dim=1, index=idx).clamp_min(1.0)
        anchors = cum_at / cnt_at                         # (B, K+1, D)
    elif pool_mode == "last_token":
        idx = cot_step_positions.clamp(min=0, max=T - 1).unsqueeze(-1)   # (B, K+1, 1)
        idx_d = idx.expand(B, K_plus_1, D)                                # (B, K+1, D)
        anchors = torch.gather(h, dim=1, index=idx_d)                     # (B, K+1, D)
    elif pool_mode == "mean_cot_only":
        if cot_start_positions is None:
            raise ValueError(
                "pool_mode='mean_cot_only' requires cot_start_positions (B,)"
            )
        mask = tgt_attn_mask.to(h.dtype).unsqueeze(-1)   # (B, T, 1)
        h_masked = h * mask
        cum = h_masked.cumsum(dim=1)                     # (B, T, D)
        cnt = mask.cumsum(dim=1)                         # (B, T, 1)
        # Mean over [cot_start..pos_k] inclusive =
        #   (cum_at_pos_k - cum_at_cot_start_minus_one)
        #   / (cnt_at_pos_k - cnt_at_cot_start_minus_one)
        # We compute cum_at index `pos` for inclusive sum up to pos.
        end = cot_step_positions.clamp(min=0, max=T - 1)         # (B, K+1)
        start = cot_start_positions.unsqueeze(-1).expand(B, K_plus_1)
        start_minus_1 = (start - 1).clamp_min(0)
        # Need cum_at_start_minus_1 only when start_minus_1 >= 0 (always true)
        # and we need to ZERO it when end < start (k=0 case).
        end_d = end.unsqueeze(-1).expand(B, K_plus_1, D)
        sm1_d = start_minus_1.unsqueeze(-1).expand(B, K_plus_1, D)
        cum_end = torch.gather(cum, dim=1, index=end_d)
        cum_sm1 = torch.gather(cum, dim=1, index=sm1_d)
        cnt_end = torch.gather(cnt, dim=1, index=end.unsqueeze(-1))
        cnt_sm1 = torch.gather(cnt, dim=1, index=start_minus_1.unsqueeze(-1))
        # If start == 0 (no q?), cum_sm1/cnt_sm1 represent cum at index 0
        # which includes h[0]. Normally q is non-empty so start >= 1.
        # For k=0 the boundary is at start-1 (last q token), so end < start;
        # we fall back to last_token in those positions.
        sum_cot = cum_end - cum_sm1                                  # (B, K+1, D)
        cnt_cot = (cnt_end - cnt_sm1).clamp_min(1.0)                 # (B, K+1, 1)
        mean_cot = sum_cot / cnt_cot                                 # (B, K+1, D)

        # Fallback to last_token when end < start (no cot tokens covered):
        idx_lt = end.unsqueeze(-1).expand(B, K_plus_1, D)
        last_tok = torch.gather(h, dim=1, index=idx_lt)              # (B, K+1, D)
        use_lt = (end < start).unsqueeze(-1).to(h.dtype)             # (B, K+1, 1)
        anchors = use_lt * last_tok + (1.0 - use_lt) * mean_cot
    else:
        raise ValueError(
            f"pool_mode must be 'mean'/'last_token'/'mean_cot_only', got {pool_mode!r}"
        )

    if sphere_norm:
        anchors = F.normalize(anchors, dim=-1) * (D ** 0.5)

    return anchors


# ---------------------------------------------------------------------------
# Sanity check helpers
# ---------------------------------------------------------------------------


def anchor_diagnostics(anchors: torch.Tensor) -> dict:
    """Return a dict of sanity-check stats for anchors.

    Args:
        anchors: (B, K+1, D)

    Returns:
        dict with keys:
            shape:                 tuple(B, K+1, D)
            norm_mean:             float -- mean ||z_a_k||
            norm_std:              float -- std of ||z_a_k|| across (B, K)
            adj_diff_mean:         float -- mean ||z_a_{k+1} - z_a_k||
            adj_diff_relative:     float -- adj_diff_mean / sqrt(D), should
                                            be > 0.1 for non-degenerate
            min_adj_diff:          float -- min ||z_a_{k+1} - z_a_k||
            n_degenerate_segments: int -- # (b, k) with ||z_a_{k+1} - z_a_k||
                                          < 0.01 * sqrt(D)  (zero-velocity)
    """
    B, K_plus_1, D = anchors.shape
    sqrt_D = D ** 0.5

    norms = anchors.norm(dim=-1)                                 # (B, K+1)
    diffs = (anchors[:, 1:, :] - anchors[:, :-1, :]).norm(dim=-1)  # (B, K)
    adj_diff_mean = diffs.mean().item()
    adj_diff_relative = adj_diff_mean / sqrt_D
    n_degenerate = (diffs < 0.01 * sqrt_D).sum().item()

    return {
        "shape": (B, K_plus_1, D),
        "norm_mean": norms.mean().item(),
        "norm_std": norms.std().item(),
        "adj_diff_mean": adj_diff_mean,
        "adj_diff_relative": adj_diff_relative,
        "min_adj_diff": diffs.min().item(),
        "n_degenerate_segments": n_degenerate,
        "n_total_segments": int(B * (K_plus_1 - 1)),
    }
