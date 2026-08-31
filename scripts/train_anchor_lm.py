"""Anchor LM finetune for the anchored CoT trajectory pipeline.

Standard autoregressive LM finetune of Qwen3-0.6B on GSM8K-formulas:
    input  : q + " => " + cot + " #### " + ans + EOS  (single sequence)
    loss   : next-token CE; positions inside the question are EXCLUDED
             from the loss (we only want the LM to learn to produce
             cot+ans given q, not to "predict the question")
    save   : best-val-LM-CE checkpoint; periodic standard-GSM8K-eval
             answer EM (greedy decode given question alone) -- the
             eventual LM upper bound for the anchors.

Usage (overfit-100 gate):
    python scripts/train_anchor_lm.py \\
        --data_dir <REPO_ROOT>/data/gsm8k_formulas \\
        --output_dir <CHECKPOINT_ROOT>/outputs/anchor_lm \\
        --max_train_examples 100 \\
        --max_steps 8000 \\
        --batch_size 16 --lr 2e-5 \\
        --val_every 500

Usage (full 385k):
    python scripts/train_anchor_lm.py \\
        --data_dir <REPO_ROOT>/data/gsm8k_formulas \\
        --output_dir <CHECKPOINT_ROOT>/outputs/anchor_lm \\
        --max_steps 8000 \\
        --batch_size 256 --lr 2e-5 \\
        --val_every 500 --val_em_every 1000

Outputs to --output_dir:
    last_checkpoint.pt        -- HF model.save_pretrained format
    best_val_loss/            -- HF model.save_pretrained format
    metrics.jsonl             -- per-step log
    val_em_curve.json         -- standard-GSM8K-eval answer EM at each
                                 val_em_every step (LM upper bound)
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reasonflow.models.anchored.dataset_utils import (  # noqa: E402
    SEP1_TEXT,
    build_anchored_target,
)

# ---------------------------------------------------------------------------
# Dataset / collator (LM finetune, separate from the flow Dataset)
# ---------------------------------------------------------------------------


class LMDataset(Dataset):
    """Tokenize each example into a single sequence + a loss mask.

    The loss mask is 1 for positions in (cot + sep2 + ans) and 0 for
    positions in (q + sep1). Standard "instruction-tuning" LM finetune
    where we only score the model on producing the answer-cot suffix
    given the question prompt.
    """

    def __init__(
        self,
        json_path: Path,
        tokenizer,
        *,
        max_q_len: int,
        cot_max_len: int,
        ans_max_len: int,
        M: int = 48,                       # unused now (kept for arg-compat); LM doesn't need M
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
        # Cache the sep1 token count (constant per tokenizer; we don't need
        # to re-tokenize per example).
        self._Lsep1 = len(tokenizer.encode(SEP1_TEXT, add_special_tokens=False))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        built = build_anchored_target(
            self.tok,
            e["question"],
            e["reasoning_text"],
            e["answer"],
            cot_max_len=self.cot_max_len,
            ans_max_len=self.ans_max_len,
            max_q_len=self.max_q_len,
            # M=None: anchor LM finetune doesn't need cot_step_positions.
        )
        tgt_ids = built["tgt_ids"]
        Lq = len(built["q_ids"])
        # First cot token is at position Lq + Lsep1 in tgt_ids.
        cot_start = Lq + self._Lsep1
        # Loss mask: 1 for positions whose target token is in (cot+sep2+ans),
        # 0 for question + sep1. In next-token-prediction loss, we predict
        # tokens at positions 1..N from inputs at positions 0..N-1.
        # We want to score predictions for cot + sep2 + ans tokens, which
        # are at positions cot_start..len(tgt_ids)-1. So the loss should be
        # active for inputs[i] where i+1 in [cot_start, len(tgt_ids)).
        N = len(tgt_ids)
        loss_mask = [0] * N
        for i in range(cot_start, N):
            loss_mask[i] = 1
        return {
            "tgt_ids": tgt_ids,
            "loss_mask": loss_mask,
            "Lq": Lq,
            "cot_start": cot_start,
            "answer": e["answer"],
            "question": e["question"],
        }


def make_lm_collator(pad_id: int):
    def _pad(seqs, length, pad_value):
        out = torch.full((len(seqs), length), pad_value, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    def collate(items):
        max_len = max(len(it["tgt_ids"]) for it in items)
        tgt_ids = _pad([it["tgt_ids"] for it in items], max_len, pad_id)
        loss_mask = _pad([it["loss_mask"] for it in items], max_len, 0)
        attn = (tgt_ids != pad_id).long()
        return {
            "input_ids": tgt_ids,
            "attention_mask": attn,
            "loss_mask": loss_mask,
            "answers": [it["answer"] for it in items],
            "questions": [it["question"] for it in items],
        }

    return collate


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def lm_loss(model, batch, device):
    """Standard next-token CE with a position-level loss mask.

    model.forward returns logits of shape (B, T, V). We shift:
        predict batch.input_ids[:, 1:] from logits[:, :-1, :]
    Loss is averaged over positions where loss_mask[:, 1:] == 1 (i.e.,
    positions whose target token is in the cot + ans region).
    """
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    loss_mask = batch["loss_mask"].to(device)

    out = model(
        input_ids=input_ids,
        attention_mask=attn,
        use_cache=False,
        return_dict=True,
    )
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = loss_mask[:, 1:].float()

    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)

    ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    masked_ce = (ce * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)
    # Also report token accuracy for diagnostics.
    with torch.no_grad():
        preds = flat_logits.argmax(dim=-1)
        correct = (preds == flat_targets).float() * flat_mask
        acc = correct.sum() / flat_mask.sum().clamp_min(1.0)
    return masked_ce, acc


# ---------------------------------------------------------------------------
# Standard GSM8K eval (LM upper bound)
# ---------------------------------------------------------------------------


_ANSWER_RE = re.compile(r"#### *(-?\d[\d,\.]*)")


def _normalize_numeric(s: str) -> str:
    """Match the gsm8k normalizer used elsewhere in the project."""
    s = s.strip()
    s = s.replace(",", "")
    # Drop trailing "." (e.g. "10." -> "10")
    while s.endswith("."):
        s = s[:-1]
    return s


@torch.no_grad()
def standard_gsm8k_eval(
    model,
    tokenizer,
    examples: List[Dict],
    *,
    device: torch.device,
    max_new_tokens: int = 128,
    max_q_len: int = 192,
) -> Dict[str, float]:
    """Greedy generate (cot, ans) given just the question prompt.

    For each example, the input prompt is `question + " => "`; the model
    generates up to `max_new_tokens` tokens (greedy). The predicted answer
    is the first numeric string after `####` in the generation, or ""
    if no `####` marker appears.

    Returns:
        {"answer_em": float, "n": int, "n_with_marker": int}
    """
    model.eval()
    n_correct = 0
    n_total = 0
    n_with_marker = 0
    for e in examples:
        prompt = e["question"] + " => "
        ids = tokenizer.encode(prompt, add_special_tokens=True)[:max_q_len]
        ids = torch.tensor([ids], dtype=torch.long, device=device)
        attn = torch.ones_like(ids)
        out = model.generate(
            input_ids=ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen = out[0, ids.shape[1] :]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        m = _ANSWER_RE.search(text)
        if m:
            n_with_marker += 1
            pred = _normalize_numeric(m.group(1))
        else:
            pred = ""
        gold = _normalize_numeric(e["answer"])
        if pred == gold and pred != "":
            n_correct += 1
        n_total += 1

    return {
        "answer_em": n_correct / max(1, n_total),
        "n": n_total,
        "n_with_marker": n_with_marker,
    }


# ---------------------------------------------------------------------------
# Cosine LR schedule (matches existing train_gsm8k_flow.py pattern)
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
    parser.add_argument("--model_name", default="Qwen/Qwen3-0.6B")

    parser.add_argument("--M", type=int, default=48,
                        help="Cot pace (variant-A pad_to_M); affects only "
                             "the dataset's cot truncation in build_anchored_target.")
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
                        help="Run val LM CE every N steps; track best.")
    parser.add_argument("--val_em_every", type=int, default=1000,
                        help="Run standard-GSM8K-eval (LM upper bound) "
                             "every N steps. 0 = disabled.")
    parser.add_argument("--val_em_n", type=int, default=200,
                        help="Number of val examples for standard-GSM8K-eval.")
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--max_train_examples", type=int, default=0,
                        help="Cap on training-set size. 0 = use all.")

    parser.add_argument("--gradient_checkpointing",
                        type=lambda s: s.lower() == "true", default=True)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    val_em_path = args.output_dir / "val_em_curve.json"

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[anchor-lm] Output dir: {args.output_dir}")
    print(f"[anchor-lm] Device: {device}")

    print(f"[anchor-lm] Loading tokenizer + model: {args.model_name}")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model = model.to(device)

    print(f"[anchor-lm] Loading datasets from {args.data_dir}")
    train_ds = LMDataset(
        args.data_dir / "train.json", tok,
        max_q_len=args.max_q_len, cot_max_len=args.cot_max_len,
        ans_max_len=args.ans_max_len, M=args.M,
        max_examples=args.max_train_examples or None,
    )
    val_ds = LMDataset(
        args.data_dir / "val.json", tok,
        max_q_len=args.max_q_len, cot_max_len=args.cot_max_len,
        ans_max_len=args.ans_max_len, M=args.M,
    )
    print(f"[anchor-lm] train={len(train_ds)}, val={len(val_ds)}")

    collator = make_lm_collator(tok.pad_token_id)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=args.num_workers, drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=2,
    )

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_ce = float("inf")
    val_em_curve = []
    step = 0
    t0 = time.time()
    train_iter = iter(train_loader)

    print(f"[anchor-lm] Starting training for {args.max_steps} steps")
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        lr = cosine_lr(step, args.warmup_steps, args.max_steps,
                       args.lr, args.min_lr)
        for g in optim.param_groups:
            g["lr"] = lr

        model.train()
        loss, acc = lm_loss(model, batch, device)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            line = {
                "step": step, "loss": loss.item(), "tok_acc": acc.item(),
                "lr": lr, "elapsed_s": elapsed,
            }
            print(f"[anchor-lm step {step}] loss={loss.item():.4f} acc={acc.item():.3f} "
                  f"lr={lr:.2e} elapsed={elapsed:.0f}s")
            with metrics_path.open("a") as fh:
                fh.write(json.dumps(line) + "\n")

        if args.val_every > 0 and step > 0 and step % args.val_every == 0:
            model.eval()
            val_losses = []
            val_accs = []
            with torch.no_grad():
                for vbatch in val_loader:
                    vl, va = lm_loss(model, vbatch, device)
                    val_losses.append(vl.item())
                    val_accs.append(va.item())
            val_ce = sum(val_losses) / max(1, len(val_losses))
            val_acc = sum(val_accs) / max(1, len(val_accs))
            print(f"[anchor-lm step {step}] val_lm_ce={val_ce:.4f} val_tok_acc={val_acc:.3f}")
            with metrics_path.open("a") as fh:
                fh.write(json.dumps({
                    "step": step, "val_lm_ce": val_ce, "val_tok_acc": val_acc
                }) + "\n")
            if val_ce < best_val_ce:
                best_val_ce = val_ce
                ckpt_dir = args.output_dir / "best_val_loss"
                model.save_pretrained(ckpt_dir)
                tok.save_pretrained(ckpt_dir)
                print(f"[anchor-lm] Saved best_val_loss ckpt @ step {step} (val_ce={val_ce:.4f})")

        if args.val_em_every > 0 and step > 0 and step % args.val_em_every == 0:
            model.eval()
            val_examples = val_ds.examples[: args.val_em_n]
            em_metrics = standard_gsm8k_eval(
                model, tok, val_examples, device=device,
                max_new_tokens=args.cot_max_len + args.ans_max_len + 8,
                max_q_len=args.max_q_len,
            )
            print(f"[anchor-lm step {step}] standard-GSM8K-eval answer_em="
                  f"{em_metrics['answer_em']:.4f} (n={em_metrics['n']}, "
                  f"with_marker={em_metrics['n_with_marker']})")
            val_em_curve.append({"step": step, **em_metrics})
            with val_em_path.open("w") as fh:
                json.dump(val_em_curve, fh, indent=2)
            with metrics_path.open("a") as fh:
                fh.write(json.dumps({
                    "step": step,
                    "val_standard_gsm8k_em": em_metrics["answer_em"],
                    "val_em_n": em_metrics["n"],
                    "val_em_n_with_marker": em_metrics["n_with_marker"],
                }) + "\n")

        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            ckpt_dir = args.output_dir / "last_checkpoint"
            model.save_pretrained(ckpt_dir)
            tok.save_pretrained(ckpt_dir)
            print(f"[anchor-lm] Saved last_checkpoint @ step {step}")

        step += 1

    # Final save.
    last_dir = args.output_dir / "last_checkpoint"
    model.save_pretrained(last_dir)
    tok.save_pretrained(last_dir)
    print(f"[anchor-lm] Training complete. Final ckpt @ {last_dir}")

    # Final standard-GSM8K-eval on a larger val subset.
    if args.val_em_every > 0:
        model.eval()
        val_examples = val_ds.examples[: args.val_em_n]
        em_metrics = standard_gsm8k_eval(
            model, tok, val_examples, device=device,
            max_new_tokens=args.cot_max_len + args.ans_max_len + 8,
            max_q_len=args.max_q_len,
        )
        print(f"[anchor-lm FINAL] standard-GSM8K-eval answer_em="
              f"{em_metrics['answer_em']:.4f} (n={em_metrics['n']})")
        val_em_curve.append({"step": "FINAL", **em_metrics})
        with val_em_path.open("w") as fh:
            json.dump(val_em_curve, fh, indent=2)


if __name__ == "__main__":
    main()
