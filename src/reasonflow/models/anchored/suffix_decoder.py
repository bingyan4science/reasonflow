"""SuffixDecoder: reads out cot[k:] + ans from an anchor latent.

Architecture:

    inputs_embeds = [
        q_embeds (Lq, D_model),       # decoder.embed_tokens(q_ids)
        proj_in(z_a_k) (1, D_model),  # trainable Linear from D_lat to D_model
        k_emb(k) (1, D_model),        # learned embedding table size K+1
        suffix_embeds (Lsuf, D_model) # decoder.embed_tokens(suffix_ids)
    ]
    logits = decoder(inputs_embeds=...).logits   # (B, Lq+2+Lsuf, V)

    # CE only on suffix positions:
    # Logits at positions [Lq+1 .. Lq+Lsuf] predict suffix_ids[0 .. Lsuf-1]
    # via standard next-token shift.
    suffix_logits = logits[:, Lq+1 : Lq+1+Lsuf, :]
    loss = CE(suffix_logits, suffix_ids)

The decoder predicts the FULL suffix = cot[:, k:] + ans (everything not
seen by the encoder). At k=0 this is full cot + ans (longest, hardest).
At k=K this is just ans (shortest -- this is the answer-head case used
at Phase 3 inference).

Key design decisions:
* q_embeds use the decoder's OWN embedding layer (= passing q_ids and
  letting the decoder embed internally). No separate question encoder.
* Two soft prefix tokens (separate z and k_emb) for ablation cleanness:
  zero out z (or q, or k) at inference to test which signals the
  decoder relies on.
* proj_in(z_a_k) is trainable -- it's part of the decoder, so the
  d_model_anchor != d_model_decoder dim mismatch is absorbed via this
  layer where the gradient signal exists (the frozen encoder's anchors
  stay at d_model_anchor).
* k_embedding_table is a learned nn.Embedding of size K+1.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


class SuffixDecoder(nn.Module):
    """Qwen3-based suffix decoder with question + (z_a_k, k) soft-prefix
    conditioning. Trained on top of a frozen anchor encoder.
    """

    def __init__(
        self,
        *,
        decoder_model_name: str = "Qwen/Qwen3-0.6B",
        d_model_anchor: int,
        K: int,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.K = K
        self.d_model_anchor = d_model_anchor

        self.decoder = AutoModelForCausalLM.from_pretrained(
            decoder_model_name, dtype=torch.float32
        )
        self.d_model_decoder = self.decoder.config.hidden_size
        self.vocab_size = self.decoder.config.vocab_size

        if gradient_checkpointing:
            self.decoder.gradient_checkpointing_enable()
            if hasattr(self.decoder, "enable_input_require_grads"):
                self.decoder.enable_input_require_grads()

        # Trainable input projection from d_model_anchor -> d_model_decoder.
        self.proj_in = nn.Linear(d_model_anchor, self.d_model_decoder)

        # Learned k-embedding (table of size K+1; k in {0, 1, ..., K}).
        self.k_embedding = nn.Embedding(K + 1, self.d_model_decoder)
        # Initialise k_embedding small so it doesn't dominate.
        nn.init.normal_(self.k_embedding.weight, mean=0.0, std=0.02)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Use the decoder's own embedding table."""
        return self.decoder.get_input_embeddings()(input_ids)

    def _build_inputs_embeds(
        self,
        q_ids: torch.Tensor,         # (B, Lq)
        z_a_k: torch.Tensor,         # (B, d_model_anchor)
        k: torch.Tensor,             # (B,) long
        suffix_ids: torch.Tensor,    # (B, Lsuf)
        *,
        zero_q: bool = False,        # ablation: zero out q_embeds
        zero_z: bool = False,        # ablation: zero out z prefix
        zero_k: bool = False,        # ablation: zero out k prefix
    ):
        B, Lq = q_ids.shape
        _, Lsuf = suffix_ids.shape

        q_embeds = self.embed_tokens(q_ids)               # (B, Lq, D)
        if zero_q:
            q_embeds = torch.zeros_like(q_embeds)

        prefix_z = self.proj_in(z_a_k).unsqueeze(1)        # (B, 1, D)
        if zero_z:
            prefix_z = torch.zeros_like(prefix_z)

        prefix_k = self.k_embedding(k).unsqueeze(1)        # (B, 1, D)
        if zero_k:
            prefix_k = torch.zeros_like(prefix_k)

        suffix_embeds = self.embed_tokens(suffix_ids)      # (B, Lsuf, D)

        inputs_embeds = torch.cat(
            [q_embeds, prefix_z, prefix_k, suffix_embeds], dim=1
        )
        return inputs_embeds, Lq, Lsuf

    def forward_with_prefix(
        self,
        q_ids: torch.Tensor,                 # (B, Lq)
        q_attn: Optional[torch.Tensor],      # (B, Lq) or None (then assume all-1s)
        z_a_k: torch.Tensor,                 # (B, d_model_anchor)
        k: torch.Tensor,                     # (B,) long
        suffix_ids: torch.Tensor,            # (B, Lsuf)
        suffix_attn: Optional[torch.Tensor] = None,   # (B, Lsuf) or None
        *,
        zero_q: bool = False,
        zero_z: bool = False,
        zero_k: bool = False,
    ) -> torch.Tensor:
        """Run the decoder over [q | z | k | suffix] and return raw logits.

        Returns:
            logits: (B, Lq + 2 + Lsuf, V)
        """
        device = q_ids.device
        B, Lq = q_ids.shape
        _, Lsuf = suffix_ids.shape

        inputs_embeds, _, _ = self._build_inputs_embeds(
            q_ids, z_a_k, k, suffix_ids,
            zero_q=zero_q, zero_z=zero_z, zero_k=zero_k,
        )

        # Attention mask: all-1s for the soft prefix and suffix; copy
        # provided masks for q and suffix.
        if q_attn is None:
            q_mask = torch.ones((B, Lq), dtype=torch.long, device=device)
        else:
            q_mask = q_attn
        soft_mask = torch.ones((B, 2), dtype=torch.long, device=device)
        if suffix_attn is None:
            suf_mask = torch.ones((B, Lsuf), dtype=torch.long, device=device)
        else:
            suf_mask = suffix_attn
        attention_mask = torch.cat([q_mask, soft_mask, suf_mask], dim=1)

        out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.logits

    def compute_suffix_loss(
        self,
        logits: torch.Tensor,          # (B, Lq+2+Lsuf, V)
        suffix_ids: torch.Tensor,      # (B, Lsuf)
        suffix_attn: Optional[torch.Tensor] = None,   # (B, Lsuf) -- 1 if real token
        *,
        Lq: int,
        ignore_index: int = -100,
    ):
        """CE only on the suffix positions of the logits.

        Logits at positions [Lq+1, ..., Lq+Lsuf] predict suffix_ids[0..Lsuf-1]
        via next-token shift (position Lq+1 = k_emb's position predicts
        the FIRST suffix token).

        Returns:
            loss: scalar
            tok_acc: scalar (fraction of suffix-positions correctly predicted,
                            masked by suffix_attn)
        """
        Lsuf = suffix_ids.shape[1]
        suffix_logits = logits[:, Lq + 1 : Lq + 1 + Lsuf, :]
        flat_logits = suffix_logits.reshape(-1, suffix_logits.size(-1))
        flat_targets = suffix_ids.reshape(-1)

        if suffix_attn is None:
            mask = torch.ones_like(flat_targets, dtype=torch.float)
        else:
            mask = suffix_attn.reshape(-1).float()

        ce = F.cross_entropy(flat_logits, flat_targets, reduction="none",
                             ignore_index=ignore_index)
        # ignore_index already gives 0 on padding, but we also use mask for clarity.
        masked_ce = (ce * mask).sum() / mask.sum().clamp_min(1.0)
        with torch.no_grad():
            preds = flat_logits.argmax(dim=-1)
            correct = ((preds == flat_targets).float() * mask).sum()
            tok_acc = correct / mask.sum().clamp_min(1.0)
        return masked_ce, tok_acc

    @torch.no_grad()
    def generate_suffix(
        self,
        q_ids: torch.Tensor,                 # (B, Lq)
        q_attn: Optional[torch.Tensor],      # (B, Lq)
        z_a_k: torch.Tensor,                 # (B, d_model_anchor)
        k: torch.Tensor,                     # (B,) long
        *,
        max_new_tokens: int,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        zero_q: bool = False,
        zero_z: bool = False,
        zero_k: bool = False,
    ) -> torch.Tensor:
        """Greedy autoregressive generation of the suffix.

        Returns:
            generated suffix_ids: (B, max_new_tokens) (may include EOS;
                                  caller can truncate at first EOS per row)
        """
        B = q_ids.shape[0]
        device = q_ids.device

        # Start with an empty suffix; we'll grow it one token at a time
        # by re-running the decoder (no KV cache for simplicity; OK for
        # ~50-token suffixes during eval).
        # Use BOS as the seed token if provided, otherwise start with an
        # empty suffix and let the model produce a token from the k_emb
        # position.
        if bos_id is not None:
            cur = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        else:
            cur = torch.empty((B, 0), dtype=torch.long, device=device)

        # If suffix is empty, the first prediction comes from logits at
        # the k_emb position (Lq+1 in the prefix). After that, suffix has
        # >= 1 token and predictions come from the last suffix position.
        Lq = q_ids.shape[1]

        for step in range(max_new_tokens):
            # Build inputs_embeds with current suffix.
            inputs_embeds, _, Lsuf = self._build_inputs_embeds(
                q_ids, z_a_k, k, cur if cur.numel() else cur.new_zeros((B, 1)),
                zero_q=zero_q, zero_z=zero_z, zero_k=zero_k,
            )
            if cur.numel() == 0:
                # cur was empty; we passed a dummy 1-token suffix above only
                # to satisfy the shape, but we need to predict from the
                # k_emb position (Lq+1). So drop the dummy suffix embed.
                inputs_embeds = inputs_embeds[:, : Lq + 2, :]
                Lsuf_eff = 0
            else:
                Lsuf_eff = cur.shape[1]

            attn_mask = torch.ones(
                (B, Lq + 2 + Lsuf_eff), dtype=torch.long, device=device
            )
            if q_attn is not None:
                attn_mask[:, :Lq] = q_attn

            out = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits[:, -1, :]   # (B, V) -- next-token logits
            next_tok = logits.argmax(dim=-1, keepdim=True)   # (B, 1)
            cur = torch.cat([cur, next_tok], dim=1) if cur.numel() else next_tok

            # Early stop if all sequences emitted EOS.
            if eos_id is not None:
                done = (cur == eos_id).any(dim=1).all().item()
                if done:
                    break

        return cur
