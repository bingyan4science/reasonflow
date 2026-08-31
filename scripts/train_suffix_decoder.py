"""Training driver for the suffix decoder.

Loads a frozen Qwen3 LM checkpoint (from train_anchor_lm.py), extracts
anchors via middle-layer pooling, and trains a fresh SuffixDecoder on
top. The anchor LM is FROZEN.

Loss: standard suffix CE.
    For each batch step, sample one k uniformly from {0, 1, ..., K}.
    Encoder sees (q, cot[:, :k]); decoder predicts cot[:, k:] + ans.

Eval (periodic):
    * Per-k stratified suffix EM (k in {0, K/4, K/2, 3K/4, K})
    * Per-k stratified answer EM (extract last segment of decoded suffix)
    * Ablation triple at k=0 and k=K-1: full / z=0 / q=0 / z=0+q=0

Output dir layout:
    last_checkpoint.pt          (full state dict + cfg + step)
    best_val_suffix_em.pt       (saved when mean-suffix-EM across k improves)
    metrics.jsonl
    eval_curves.json            (periodic eval results)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reasonflow.models.anchored.dataset_utils import (  # noqa: E402
    SEP2_TEXT,
    build_anchored_target,
)
from reasonflow.models.anchored.suffix_decoder import SuffixDecoder  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset (variable-K-aware, per-batch shared k)
# ---------------------------------------------------------------------------


class AnchorTrainDataset(Dataset):
    """Per-example: (q_ids, cot_ids, ans_ids).

    No padding to M -- cot has its NATURAL length per example (truncated
    to cot_max_len). Per-batch shared k samples uniformly from
    {0, ..., k_max}; the encoder's attention mask handles examples where
    k > cot_length naturally.
    """

    def __init__(
        self,
        json_path: Path,
        tokenizer,
        *,
        max_q_len: int,
        cot_max_len: int,
        ans_max_len: int,
        max_examples: Optional[int] = None,
    ):
        with json_path.open("r") as fh:
            self.examples = json.load(fh)
        self.examples = [
            e
            for e in self.examples
            if len(e.get("reasoning_text", "")) > 0
            and len(e.get("answer", "")) > 0
        ]
        if max_examples is not None and max_examples > 0:
            self.examples = self.examples[:max_examples]
        self.tok = tokenizer
        self.max_q_len = max_q_len
        self.cot_max_len = cot_max_len
        self.ans_max_len = ans_max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        built = build_anchored_target(
            self.tok,
            e["question"],
            e["reasoning_text"],
            e["answer"],
            M=None,                                 # data-shaping: don't compute step positions
            cot_max_len=self.cot_max_len,
            ans_max_len=self.ans_max_len,
            max_q_len=self.max_q_len,
        )
        # cot has its natural length up to cot_max_len. No padding here;
        # collator pads to max-in-batch.
        return {
            "q_ids": built["q_ids"],
            "cot_ids": built["cot_ids"],            # natural length (<= cot_max_len)
            "ans_ids": built["ans_ids"],
            "cot_length": built["cot_length"],
            "answer": e["answer"],
            "question": e["question"],
        }


def make_anchor_collator(pad_id: int):
    """Pads to max-in-batch for q, cot, ans. cot can be empty (length 0)
    for examples where reasoning_text was filtered short; we ensure at
    least length 1 to keep tensor shape valid.
    """

    def _pad(seqs, length, pad_value):
        out = torch.full((len(seqs), length), pad_value, dtype=torch.long)
        for i, s in enumerate(seqs):
            if len(s) > 0:
                out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    def collate(items):
        max_q = max(len(it["q_ids"]) for it in items)
        max_c = max(max(len(it["cot_ids"]) for it in items), 1)
        max_a = max(len(it["ans_ids"]) for it in items)

        q_ids = _pad([it["q_ids"] for it in items], max_q, pad_id)
        q_attn = (q_ids != pad_id).long()

        cot_ids = _pad([it["cot_ids"] for it in items], max_c, pad_id)
        # cot_attn marks real cot positions (1) vs padding (0).
        # Compute from explicit lengths to be robust to pad_id collisions.
        cot_attn = torch.zeros((len(items), max_c), dtype=torch.long)
        for i, it in enumerate(items):
            n = len(it["cot_ids"])
            if n > 0:
                cot_attn[i, :n] = 1

        ans_ids = _pad([it["ans_ids"] for it in items], max_a, pad_id)
        ans_attn = (ans_ids != pad_id).long()

        cot_lengths = torch.tensor([it["cot_length"] for it in items], dtype=torch.long)

        return {
            "q_ids": q_ids,
            "q_attn": q_attn,
            "cot_ids": cot_ids,
            "cot_attn": cot_attn,
            "ans_ids": ans_ids,
            "ans_attn": ans_attn,
            "cot_lengths": cot_lengths,
            "answers": [it["answer"] for it in items],
            "questions": [it["question"] for it in items],
        }

    return collate


# ---------------------------------------------------------------------------
# Frozen anchor encoder: wraps the frozen LM + middle-layer pooling.
# ---------------------------------------------------------------------------


class FrozenEncoder(nn.Module):
    """Wraps a frozen Qwen3 LM + middle-layer pooling.

    Two pool modes (matches extract_anchors):
        * "mean": cumulative mean over [0..last_valid] (covers everything
                  the encoder has seen so far)
        * "last_token": just h[middle_layer, last_valid] (only the last
                        token's hidden state; expected to give more
                        distinguishable anchors since mean is dominated
                        by the long shared prefix)

    Returns z_a_k at d_model_anchor = D_qwen (no projection).
    """

    def __init__(
        self,
        lm_ckpt: str,
        *,
        sphere_norm: bool,
        pool_mode: str = "mean",
        middle_layer: Optional[int] = None,
    ):
        super().__init__()
        assert pool_mode in ("mean", "last_token", "mean_cot_only"), pool_mode
        self.lm = AutoModelForCausalLM.from_pretrained(lm_ckpt, dtype=torch.float32)
        for p in self.lm.parameters():
            p.requires_grad = False
        self.lm.eval()
        self.sphere_norm = sphere_norm
        self.pool_mode = pool_mode
        self.middle_layer = middle_layer
        self.D_qwen = self.lm.config.hidden_size

    def forward(
        self,
        q_ids: torch.Tensor,
        q_attn: Optional[torch.Tensor],
        cot_prefix_ids: torch.Tensor,
        cot_prefix_attn: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute z_a_k from a frozen LM forward over (q, cot_prefix)."""
        device = q_ids.device
        B, Lq = q_ids.shape

        if q_attn is None:
            q_attn = torch.ones((B, Lq), dtype=torch.long, device=device)

        if cot_prefix_ids is None or cot_prefix_ids.numel() == 0 or cot_prefix_ids.shape[1] == 0:
            input_ids = q_ids
            attn = q_attn
        else:
            input_ids = torch.cat([q_ids, cot_prefix_ids], dim=1)
            cot_attn = (cot_prefix_attn if cot_prefix_attn is not None
                        else torch.ones_like(cot_prefix_ids))
            attn = torch.cat([q_attn, cot_attn], dim=1)

        with torch.no_grad():
            out = self.lm(
                input_ids=input_ids,
                attention_mask=attn,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = out.hidden_states
        ml = self.middle_layer if self.middle_layer is not None else len(hidden_states) // 2
        h = hidden_states[ml]                               # (B, T, D)
        last_idx = (attn.sum(dim=1) - 1).clamp_min(0)        # (B,)

        if self.pool_mode == "mean":
            mask = attn.to(h.dtype).unsqueeze(-1)            # (B, T, 1)
            cum = (h * mask).cumsum(dim=1)
            cnt = mask.cumsum(dim=1).clamp_min(1.0)
            idx_d = last_idx.view(B, 1, 1).expand(B, 1, self.D_qwen)
            cum_at = torch.gather(cum, dim=1, index=idx_d).squeeze(1)        # (B, D)
            cnt_at = torch.gather(cnt, dim=1, index=idx_d[:, :, :1]).squeeze(1)
            z = cum_at / cnt_at.clamp_min(1.0)
        elif self.pool_mode == "mean_cot_only":
            # Mean over the cot positions only ([Lq..last_idx]). When the
            # encoder sees ZERO cot tokens (cot_prefix is empty / k=0), we
            # fall back to last_token at last_idx (= last q token).
            mask = attn.to(h.dtype).unsqueeze(-1)            # (B, T, 1)
            cum = (h * mask).cumsum(dim=1)
            cnt = mask.cumsum(dim=1)
            cot_start = torch.full((B,), Lq, dtype=torch.long, device=device)
            sm1 = (cot_start - 1).clamp_min(0)
            end_d = last_idx.view(B, 1, 1).expand(B, 1, self.D_qwen)
            sm1_d = sm1.view(B, 1, 1).expand(B, 1, self.D_qwen)
            cum_end = torch.gather(cum, dim=1, index=end_d).squeeze(1)
            cum_sm1 = torch.gather(cum, dim=1, index=sm1_d).squeeze(1)
            cnt_end = torch.gather(cnt, dim=1, index=end_d[:, :, :1]).squeeze(1)
            cnt_sm1 = torch.gather(cnt, dim=1, index=sm1_d[:, :, :1]).squeeze(1)
            sum_cot = cum_end - cum_sm1
            cnt_cot = (cnt_end - cnt_sm1).clamp_min(1.0)
            mean_cot = sum_cot / cnt_cot                                      # (B, D)
            last_tok = h[torch.arange(B, device=device), last_idx]            # (B, D)
            use_lt = (last_idx < cot_start).unsqueeze(-1).to(h.dtype)
            z = use_lt * last_tok + (1.0 - use_lt) * mean_cot
        else:  # last_token
            z = h[torch.arange(B, device=device), last_idx]                  # (B, D)

        if self.sphere_norm:
            z = F.normalize(z, dim=-1) * (self.D_qwen ** 0.5)
        return z.detach()


# ---------------------------------------------------------------------------
# Eval: per-k suffix EM, per-k answer EM, ablation triple
# ---------------------------------------------------------------------------


_ANSWER_RE = re.compile(r"#### *(-?\d[\d,\.]*)")


def _normalize_numeric(s: str) -> str:
    s = s.strip().replace(",", "")
    while s.endswith("."):
        s = s[:-1]
    return s


def _build_clean_suffix_external(
    batch: Dict[str, torch.Tensor],
    k: int,
    pad_id: int,
    sep2_tensor: torch.Tensor,         # (Lsep2,) the " #### " separator token ids
    answer_only: bool = False,
):
    """Build per-batch suffix targets.

    Default (full): real_cot[k:cot_length] + sep2_tensor + real_ans.

    answer_only=True: just real_ans (no cot, no sep2). Used to train a
    tiny answer-head for phase-2 inference -- avoids generating long
    cot strings + '####' separators at readout time, which is fragile
    for tasks like mech with 200+ token SMILES answers.

    Returns (suffix_ids, suffix_attn): (B, Lsuf) each, padded at the END.
    """
    cot_ids_b = batch["cot_ids"]
    cot_lengths = batch["cot_lengths"]
    ans_ids_b = batch["ans_ids"]
    ans_attn_b = batch["ans_attn"]
    B, _ = cot_ids_b.shape
    sep2_on_dev = sep2_tensor.to(cot_ids_b.device)
    suffixes = []
    for i in range(B):
        L_cot = int(cot_lengths[i].item())
        L_ans = int(ans_attn_b[i].sum().item())
        ans_part = ans_ids_b[i, :L_ans]
        if answer_only:
            suffixes.append(ans_part)
            continue
        if k < L_cot:
            cot_part = cot_ids_b[i, k:L_cot]
        else:
            cot_part = cot_ids_b.new_zeros((0,), dtype=torch.long)
        suffixes.append(torch.cat([cot_part, sep2_on_dev, ans_part]))
    max_len = max(s.shape[0] for s in suffixes) if suffixes else 0
    if max_len == 0:
        max_len = 1
    suffix_ids_out = cot_ids_b.new_full((B, max_len), pad_id)
    suffix_attn_out = torch.zeros(
        (B, max_len), dtype=torch.long, device=cot_ids_b.device
    )
    for i, s in enumerate(suffixes):
        n = s.shape[0]
        if n > 0:
            suffix_ids_out[i, :n] = s
            suffix_attn_out[i, :n] = 1
    return suffix_ids_out, suffix_attn_out


@torch.no_grad()
def eval_per_k_em(
    encoder: nn.Module,
    decoder: SuffixDecoder,
    val_loader: DataLoader,
    *,
    K: int,
    device: torch.device,
    tokenizer,
    sep2_tensor: torch.Tensor,
    max_eval_examples: int = 200,
    k_buckets: Optional[List[int]] = None,
    ablation: str = "full",   # "full" | "z=0" | "q=0" | "z=0,q=0"
) -> Dict[str, float]:
    """Compute per-k suffix EM and per-k answer EM on a val subset.

    Args:
        encoder: FrozenEncoder over (q, cot[:k]) -> anchor latent.
        decoder: SuffixDecoder.
        val_loader: DataLoader over the anchor-training dataset.
        K: cot pace.

    Returns dict with per-k metrics under keys like "k=0/answer_em",
    "k=K/answer_em", and overall mean.
    """
    if k_buckets is None:
        k_buckets = sorted({0, max(1, K // 4), K // 2, 3 * K // 4, K})
    encoder.eval()
    decoder.eval()
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    zero_q = "q=0" in ablation
    zero_z = "z=0" in ablation
    zero_k = False

    per_k_results = {}
    for kb in k_buckets:
        n_eval = 0
        n_correct_suffix = 0
        n_correct_answer = 0
        for batch in val_loader:
            B = batch["q_ids"].shape[0]
            for k_field in [
                "q_ids", "q_attn", "cot_ids", "cot_attn", "ans_ids", "ans_attn",
                "cot_lengths",
            ]:
                if k_field in batch:
                    batch[k_field] = batch[k_field].to(device)
            cot_ids = batch["cot_ids"]
            cot_attn = batch["cot_attn"]
            ans_attn = batch["ans_attn"]
            answers = batch["answers"]

            # Encoder forward.
            cot_prefix = cot_ids[:, :kb]
            cot_prefix_attn = cot_attn[:, :kb] if kb > 0 else None
            z_a_k = encoder(batch["q_ids"], batch["q_attn"],
                            cot_prefix, cot_prefix_attn)

            # Build CLEAN suffix: real_cot[k:cot_length] + sep2 + real_ans,
            # padded at the END only. Matches the training-side construction.
            suffix_gold, suffix_gold_attn = _build_clean_suffix_external(
                batch, kb, pad_id, sep2_tensor
            )

            # Greedy decode up to max_suffix_len_in_batch + small slack.
            max_new = suffix_gold.shape[1] + 4
            k_tensor = torch.full((B,), kb, dtype=torch.long, device=device)
            gen = decoder.generate_suffix(
                q_ids=batch["q_ids"], q_attn=batch["q_attn"],
                z_a_k=z_a_k, k=k_tensor,
                max_new_tokens=max_new,
                bos_id=None, eos_id=eos_id,
                zero_q=zero_q, zero_z=zero_z, zero_k=zero_k,
            )

            for b in range(B):
                Lsuf_b = int(suffix_gold_attn[b].sum().item())
                gold_seq = suffix_gold[b, :Lsuf_b].tolist()
                gen_seq = gen[b, :Lsuf_b].tolist() if gen.shape[1] >= Lsuf_b else gen[b].tolist()
                if gold_seq == gen_seq:
                    n_correct_suffix += 1

                # Answer-EM via the " #### " marker. The suffix is now
                # `real_cot + " #### " + real_ans`, so the decoder is
                # supervised to emit "####" between cot and ans. At
                # inference we decode the generation as text, find the
                # marker, and extract the trailing number.
                gen_text = tokenizer.decode(gen[b].tolist(), skip_special_tokens=True)
                m = _ANSWER_RE.search(gen_text)
                if m:
                    pred_ans = _normalize_numeric(m.group(1))
                else:
                    # Fallback (rare): no marker in gen. Use the last La
                    # tokens of the generation as the answer (works when
                    # gen exactly matches gold length, e.g., overfit-100).
                    La = int(ans_attn[b].sum().item())
                    if La > 0 and gen.shape[1] >= La:
                        pred_tail_ids = gen[b, -La:].tolist()
                        pred_ans = _normalize_numeric(
                            tokenizer.decode(pred_tail_ids, skip_special_tokens=True)
                        )
                    else:
                        pred_ans = ""
                gold_ans = _normalize_numeric(answers[b])
                if pred_ans == gold_ans and pred_ans != "":
                    n_correct_answer += 1
                n_eval += 1
                if n_eval >= max_eval_examples:
                    break
            if n_eval >= max_eval_examples:
                break
        per_k_results[f"k={kb}/suffix_em"] = n_correct_suffix / max(1, n_eval)
        per_k_results[f"k={kb}/answer_em"] = n_correct_answer / max(1, n_eval)
        per_k_results[f"k={kb}/n"] = n_eval

    suffix_ems = [per_k_results[f"k={kb}/suffix_em"] for kb in k_buckets]
    answer_ems = [per_k_results[f"k={kb}/answer_em"] for kb in k_buckets]
    per_k_results["mean_suffix_em"] = sum(suffix_ems) / len(suffix_ems)
    per_k_results["mean_answer_em"] = sum(answer_ems) / len(answer_ems)
    per_k_results["k_buckets"] = k_buckets
    per_k_results["ablation"] = ablation
    return per_k_results


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def cosine_lr(step: int, warmup: int, total: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * p))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--encoder_ckpt", type=str, default="",
                        help="Path to the frozen Qwen3 LM ckpt directory "
                             "(HF format) from train_anchor_lm.py.")

    parser.add_argument("--decoder_model_name", default="Qwen/Qwen3-0.6B",
                        help="Base model for the SuffixDecoder.")

    parser.add_argument("--k_max", type=int, default=80,
                        help="Max k value sampled during data-shaping training "
                             "(per-batch shared k uniform on {0..k_max}). "
                             "Should equal cot_max_len for full coverage. "
                             "Sets the size of the k_embedding table to k_max+1.")
    parser.add_argument("--v1_pool_mode", choices=["last_token", "mean", "mean_cot_only"],
                        default="mean",
                        help="Frozen-LM pooling. 'mean' = cumulative mean over "
                             "[0..last_valid]; 'last_token' = just middle-layer "
                             "h at the last valid position; 'mean_cot_only' = "
                             "mean over the cot tokens only.")

    parser.add_argument("--max_q_len", type=int, default=192)
    parser.add_argument("--cot_max_len", type=int, default=80)
    parser.add_argument("--ans_max_len", type=int, default=24)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=8000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--val_every", type=int, default=500,
                        help="Run val CE every N steps (cheap; teacher-forced).")
    parser.add_argument("--val_em_every", type=int, default=2000,
                        help="Run per-k suffix-EM eval every N steps. 0 = disable.")
    parser.add_argument("--val_em_max_examples", type=int, default=200,
                        help="How many val examples for per-k EM eval.")
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--max_train_examples", type=int, default=0)

    parser.add_argument("--gradient_checkpointing",
                        type=lambda s: s.lower() == "true", default=True)
    parser.add_argument("--answer_only_suffix",
                        type=lambda s: s.lower() == "true", default=False,
                        help="If true, train the decoder to output the ANSWER "
                             "only (no cot prefix, no '####' separator) at a "
                             "fixed k=k_max. Use this to train a tiny "
                             "answer-only head for phase-2 inference.")
    parser.add_argument("--sphere_norm", type=lambda s: s.lower() == "true",
                        default=True)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    eval_curve_path = args.output_dir / "eval_curves.json"

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[suffix-decoder] Output dir: {args.output_dir}")
    print(f"[suffix-decoder] Device: {device}")

    # Tokenizer must match the decoder's (and the frozen encoder ckpt's,
    # which is Qwen3's).
    print(f"[suffix-decoder] Loading tokenizer from {args.decoder_model_name}")
    tok = AutoTokenizer.from_pretrained(args.decoder_model_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Build model. k_max determines the k_embedding table size (k=0..k_max).
    k_max = args.k_max
    if not args.encoder_ckpt:
        print("ERROR: --encoder_ckpt required.")
        return 1
    print(f"[suffix-decoder] Loading frozen LM encoder from {args.encoder_ckpt} "
          f"(pool_mode={args.v1_pool_mode})")
    encoder = FrozenEncoder(
        args.encoder_ckpt,
        sphere_norm=args.sphere_norm,
        pool_mode=args.v1_pool_mode,
    ).to(device)
    decoder = SuffixDecoder(
        decoder_model_name=args.decoder_model_name,
        d_model_anchor=encoder.D_qwen,
        K=k_max,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)

    # Datasets
    print(f"[suffix-decoder] Loading datasets from {args.data_dir}")
    train_ds = AnchorTrainDataset(
        args.data_dir / "train.json", tok,
        max_q_len=args.max_q_len, cot_max_len=args.cot_max_len,
        ans_max_len=args.ans_max_len,
        max_examples=args.max_train_examples or None,
    )
    val_ds = AnchorTrainDataset(
        args.data_dir / "val.json", tok,
        max_q_len=args.max_q_len, cot_max_len=args.cot_max_len,
        ans_max_len=args.ans_max_len,
    )
    print(f"[suffix-decoder] train={len(train_ds)}, val={len(val_ds)}")

    collator = make_anchor_collator(tok.pad_token_id)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=args.num_workers, drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=2,
    )

    # Optimizer: only the decoder is trainable (encoder is frozen).
    trainable = list(decoder.parameters())
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    n_trainable = sum(p.numel() for p in trainable)
    print(f"[suffix-decoder] Trainable params: {n_trainable:,}")

    pad_id = tok.pad_token_id

    # Tokenize the " #### " separator once. Inserted between real_cot and
    # real_ans in every suffix (both training and eval). This gives the
    # decoder an explicit "cot done; answer starts here" signal and lets
    # us extract the answer at inference via the standard regex.
    sep2_ids_list = tok.encode(SEP2_TEXT, add_special_tokens=False)
    sep2_tensor = torch.tensor(sep2_ids_list, dtype=torch.long, device=device)
    print(f"[suffix-decoder] sep2 ' #### ' tokenizes to {len(sep2_ids_list)} tokens: {sep2_ids_list}")

    def _build_clean_suffix(batch, k: int):
        """Construct suffix.

        Default: real_cot[k:cot_length] + sep2 + real_ans (full suffix).
        --answer_only_suffix=true: just real_ans (no cot, no sep2).
        """
        return _build_clean_suffix_external(
            batch, k, pad_id, sep2_tensor,
            answer_only=args.answer_only_suffix,
        )

    # Helper: one training-step forward + backward.
    def train_step(batch, k: int):
        batch = {k_: (v.to(device) if torch.is_tensor(v) else v) for k_, v in batch.items()}
        B = batch["q_ids"].shape[0]
        Lq = batch["q_ids"].shape[1]
        k_tensor = torch.full((B,), k, dtype=torch.long, device=device)

        cot_prefix = batch["cot_ids"][:, :k]
        cot_prefix_attn = batch["cot_attn"][:, :k] if k > 0 else None

        z_a_k = encoder(batch["q_ids"], batch["q_attn"],
                        cot_prefix, cot_prefix_attn)

        suffix_ids, suffix_attn = _build_clean_suffix(batch, k)

        logits = decoder.forward_with_prefix(
            q_ids=batch["q_ids"], q_attn=batch["q_attn"],
            z_a_k=z_a_k, k=k_tensor,
            suffix_ids=suffix_ids, suffix_attn=suffix_attn,
        )
        loss, tok_acc = decoder.compute_suffix_loss(
            logits, suffix_ids, suffix_attn=suffix_attn, Lq=Lq,
        )
        return loss, tok_acc, {"k": k, "Lsuf": suffix_ids.shape[1]}

    # Training loop.
    K_total = k_max
    step = 0
    t0 = time.time()
    train_iter = iter(train_loader)
    best_mean_answer_em = -1.0
    eval_curves = []

    print(f"[suffix-decoder] Starting training for {args.max_steps} steps")
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Sample shared k for this step. For --answer_only_suffix we
        # fix k=K_total (the head is only ever used at the answer
        # position in phase-2 inference, so train at that single k).
        if args.answer_only_suffix:
            k = K_total
        else:
            k = int(torch.randint(0, K_total + 1, (1,)).item())

        lr = cosine_lr(step, args.warmup_steps, args.max_steps,
                       args.lr, args.min_lr)
        for g in optim.param_groups:
            g["lr"] = lr

        decoder.train()
        loss, tok_acc, diag = train_step(batch, k)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optim.step()

        if step % args.log_every == 0:
            line = {
                "step": step, "k": k, "loss": loss.item(),
                "tok_acc": tok_acc.item(), "lr": lr,
                "elapsed_s": time.time() - t0,
            }
            print(f"[suffix-decoder step {step}] k={k} loss={loss.item():.4f} "
                  f"tok_acc={tok_acc.item():.3f} lr={lr:.2e}")
            with metrics_path.open("a") as fh:
                fh.write(json.dumps(line) + "\n")

        if args.val_every > 0 and step > 0 and step % args.val_every == 0:
            decoder.eval()
            with torch.no_grad():
                val_losses = []
                for vbatch in val_loader:
                    # For val CE, use a fixed grid of k values for stable comparison.
                    vk = K_total // 2  # mid-k for val CE diagnostic
                    vl, va, _ = train_step(vbatch, vk)
                    val_losses.append(vl.item())
                    if len(val_losses) >= 8:
                        break
            val_ce = sum(val_losses) / max(1, len(val_losses))
            print(f"[suffix-decoder step {step}] val_ce(k=K/2)={val_ce:.4f}")
            with metrics_path.open("a") as fh:
                fh.write(json.dumps({"step": step, "val_ce_mid": val_ce}) + "\n")

        if (args.val_em_every > 0 and step > 0 and step % args.val_em_every == 0
                and not args.answer_only_suffix):
            # answer_only_suffix mode: skip per-k eval (it expects
            # cot+sep2+ans suffix structure with the GSM8K numeric
            # regex). We re-evaluate this head separately via
            # scripts/eval_anchored.py once trained.
            print(f"[suffix-decoder step {step}] running per-k EM eval...")
            res_full = eval_per_k_em(
                encoder, decoder, val_loader,
                K=K_total, device=device, tokenizer=tok,
                sep2_tensor=sep2_tensor,
                max_eval_examples=args.val_em_max_examples,
                ablation="full",
            )
            res_full["step"] = step
            eval_curves.append(res_full)
            print(f"  full: mean_ans_em={res_full['mean_answer_em']:.4f} "
                  f"mean_suf_em={res_full['mean_suffix_em']:.4f}")
            for kb in res_full["k_buckets"]:
                print(f"    k={kb}: ans={res_full[f'k={kb}/answer_em']:.3f} "
                      f"suf={res_full[f'k={kb}/suffix_em']:.3f}")
            with eval_curve_path.open("w") as fh:
                json.dump(eval_curves, fh, indent=2)
            with metrics_path.open("a") as fh:
                fh.write(json.dumps({"step": step, "val_em_full": res_full}) + "\n")

            if res_full["mean_answer_em"] > best_mean_answer_em:
                best_mean_answer_em = res_full["mean_answer_em"]
                ckpt_path = args.output_dir / "best_val_answer_em.pt"
                state = {
                    "step": step,
                    "decoder_state": decoder.state_dict(),
                    "best_mean_answer_em": best_mean_answer_em,
                    "args": vars(args),
                }
                torch.save(state, ckpt_path)
                print(f"[suffix-decoder] Saved best_val_answer_em ckpt @ step {step} "
                      f"(mean_ans_em={best_mean_answer_em:.4f})")

        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            ckpt_path = args.output_dir / "last_checkpoint.pt"
            state = {
                "step": step,
                "decoder_state": decoder.state_dict(),
                "args": vars(args),
            }
            torch.save(state, ckpt_path)
            print(f"[suffix-decoder] Saved last_checkpoint @ step {step}")

        step += 1

    # Final save.
    last_path = args.output_dir / "last_checkpoint.pt"
    state = {
        "step": step,
        "decoder_state": decoder.state_dict(),
        "args": vars(args),
    }
    torch.save(state, last_path)
    print(f"[suffix-decoder] Training complete. Final ckpt @ {last_path}")


if __name__ == "__main__":
    main()
