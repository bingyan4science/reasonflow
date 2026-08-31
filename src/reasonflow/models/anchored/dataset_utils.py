"""Helpers for anchored CoT trajectory experiments.

The key data structure is `cot_step_positions`: per-example, a length-(K+1)
LongTensor giving the positions in `tgt_input_ids` where each anchor lives.

Anchor convention (matches plan, indexing convention section):
    z_a_k = enc(q, c_1, ..., c_k)  for k in {0, 1, ..., K}

For variant-A (pad_to_M, K = M = 48), every example has exactly K+1 = 49
anchor positions. If the example's actual cot length < M, the trailing
anchors are filled by replicating the position of the last real cot token
(so anchors[cot_length..K] all collapse to anchors[cot_length]). This
matches the existing variant-A convention where padding contributes a
zero-length segment to the trajectory.

Position semantics (causal-LM hidden state at position p depends on
tokens 0..p inclusive):
    pos_0  = position of the LAST sep1 token (after q + " => ")
             -- causal hidden state here represents "encoder has seen q,
                no cot tokens yet" -> z_a_0 = enc(q)
    pos_k  = position of the k-th cot token in tgt_ids (1-indexed
             k = 1..K)
             -- causal hidden state here represents "encoder has seen q
                + sep1 + c_1..c_k" -> z_a_k = enc(q, c_<=k)

For tokenization stability we tokenize each segment separately and
concatenate, rather than tokenizing the joined string. This avoids
context-dependent BPE merges across segment boundaries (which would
make the cot's position in tgt_ids depend on neighboring text).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

SEP1_TEXT = " => "      # separator between question and reasoning
SEP2_TEXT = " #### "    # separator between reasoning and answer (cot/ans boundary).
                         # Matches the anchor-LM convention so the SuffixDecoder
                         # learns the same boundary signal the underlying LM has
                         # been pretrained-and-finetuned on.


def build_anchored_target(
    tokenizer,
    question: str,
    reasoning: str,
    answer: str,
    *,
    cot_max_len: int,
    ans_max_len: int,
    max_q_len: int,
    M: Optional[int] = None,
) -> Dict[str, List[int]]:
    """Tokenize each segment separately and produce tgt_ids + (optional)
    cot_step_positions.

    Two distinct phases use this:
        * Data-shaping phase (anchor LM + suffix decoder training):
              - Pass M=None (default). cot_step_positions is NOT computed.
              - cot_ids has its NATURAL length (truncated to cot_max_len).
              - The training samples k from {0..cot_max_len} per batch step.
              - For example i with cot_length_i < k, the encoder via
                attention mask effectively returns z_a_{cot_length_i} (=
                "encoder saw all real cot for this example").
        * Phase 3 diffusion training (variant-A pad_to_M):
              - Pass M=48 (or whatever pace). cot_step_positions is
                computed: K+1 = M+1 positions in tgt_ids, with positions
                for k > cot_length replicating the last real position
                (matches the variant-A no-op convention).

    Returns a dict with:
        tgt_ids:             List[int] (q + sep1 + cot + sep2 + ans)
        cot_step_positions:  List[int] of length M+1 if M is given, else []
        cot_length:          int -- actual number of cot tokens in tgt_ids
        q_ids:               List[int] -- question only (with BOS if any)
        ans_ids:             List[int] -- answer tokens only (no BOS, with EOS)
        cot_ids:             List[int] -- cot tokens only (no BOS, no EOS),
                                          truncated to cot_max_len

    Design notes:
        * cot is always truncated to cot_max_len (not min(cot_max_len, M)).
          For data-shaping phase, M is irrelevant; for diffusion phase,
          examples with cot_length > M just get extra anchors clipped.
        * We DO truncate ans and q to their max lengths to bound the
          tgt_ids length.
    """
    # 1) Question.
    q_ids = tokenizer.encode(question, add_special_tokens=True)[:max_q_len]
    Lq = len(q_ids)

    # 2) Separator 1: " => "
    sep1_ids = tokenizer.encode(SEP1_TEXT, add_special_tokens=False)
    Lsep1 = len(sep1_ids)

    # 3) Cot. Always truncate to cot_max_len (not min(cot_max_len, M)).
    cot_ids = tokenizer.encode(reasoning, add_special_tokens=False)
    cot_ids = cot_ids[:cot_max_len]
    Lcot = len(cot_ids)

    # 4) Separator 2: " #### "
    sep2_ids = tokenizer.encode(SEP2_TEXT, add_special_tokens=False)

    # 5) Answer (no BOS; manual EOS).
    eos_id = tokenizer.eos_token_id
    raw_ans = tokenizer.encode(answer, add_special_tokens=False)
    if eos_id is not None and len(raw_ans) > 0 and raw_ans[-1] == eos_id:
        raw_ans = raw_ans[:-1]
    raw_ans = raw_ans[: ans_max_len - 1]
    if eos_id is not None:
        raw_ans = raw_ans + [eos_id]
    ans_ids = raw_ans

    # 6) Concatenate segments.
    tgt_ids = q_ids + sep1_ids + cot_ids + sep2_ids + ans_ids

    # 7) Compute cot_step_positions ONLY if M is given (Phase 3 diffusion).
    cot_step_positions: List[int] = []
    if M is not None:
        cot_start = Lq + Lsep1
        pos_0 = cot_start - 1
        cot_step_positions = [pos_0]
        for k in range(1, M + 1):
            if k <= Lcot:
                cot_step_positions.append(cot_start + k - 1)
            else:
                cot_step_positions.append(cot_step_positions[-1])

    return {
        "tgt_ids": tgt_ids,
        "cot_step_positions": cot_step_positions,
        "cot_length": Lcot,
        "q_ids": q_ids,
        "ans_ids": ans_ids,
        "cot_ids": cot_ids,
    }


def make_anchored_collator(
    pad_id: int,
    *,
    K: int,
):
    """Collate function that pads tgt_ids and stacks cot_step_positions.

    K = number of cot tokens (= M for variant-A). All examples have exactly
    K+1 anchor positions; collator just stacks them.

    Returns a callable that takes a list of dicts (each from
    `build_anchored_target` plus any other fields) and returns padded
    tensors.
    """

    def _pad_to(seqs: List[List[int]], length: int, pad_value: int) -> torch.Tensor:
        out = torch.full((len(seqs), length), pad_value, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    def collate(items: List[Dict]) -> Dict[str, torch.Tensor]:
        B = len(items)

        # tgt_ids: variable length; pad to max in batch.
        max_tgt = max(len(it["tgt_ids"]) for it in items)
        tgt_ids = _pad_to([it["tgt_ids"] for it in items], max_tgt, pad_id)
        tgt_attn = (tgt_ids != pad_id).long()

        # cot_step_positions: always K+1 long.
        cot_step_positions = torch.stack(
            [torch.tensor(it["cot_step_positions"], dtype=torch.long) for it in items]
        )
        # Sanity: should be (B, K+1)
        assert cot_step_positions.shape == (B, K + 1), (
            f"cot_step_positions shape {tuple(cot_step_positions.shape)} != ({B}, {K + 1})"
        )

        cot_lengths = torch.tensor(
            [it["cot_length"] for it in items], dtype=torch.long
        )

        # q_ids and ans_ids (separate, padded individually).
        max_q = max(len(it["q_ids"]) for it in items)
        q_ids = _pad_to([it["q_ids"] for it in items], max_q, pad_id)
        q_attn = (q_ids != pad_id).long()

        max_a = max(len(it["ans_ids"]) for it in items)
        ans_ids = _pad_to([it["ans_ids"] for it in items], max_a, pad_id)
        ans_attn = (ans_ids != pad_id).long()

        max_c = max(len(it["cot_ids"]) for it in items)
        cot_ids = _pad_to([it["cot_ids"] for it in items], max_c, pad_id)
        cot_attn = (cot_ids != pad_id).long()

        return {
            "tgt_input_ids": tgt_ids,
            "tgt_attn_mask": tgt_attn,
            "cot_step_positions": cot_step_positions,
            "cot_lengths": cot_lengths,
            "q_input_ids": q_ids,
            "q_attn_mask": q_attn,
            "ans_input_ids": ans_ids,
            "ans_attn_mask": ans_attn,
            "cot_input_ids_only": cot_ids,
            "cot_attn_mask_only": cot_attn,
        }

    return collate
