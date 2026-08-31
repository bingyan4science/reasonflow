"""Standalone-LM upper bound evaluator for the finetuned anchor LM.

For a given anchor LM ckpt + data dir, prompt with ``q + " => "``, greedy-generate
up to (cot_max_len + ans_max_len + 8) tokens, locate ``####`` in the generated
text, take the answer string after it, and apply the per-task normalizer.

This produces a single standalone-LM val answer EM number per task, which is
the upper bound for any anchor-encoder scheme derived from this LM.

Usage:
    python scripts/eval_ar_lm_upper_bound.py \\
        --ckpt <CHECKPOINT_ROOT>/outputs/anchor_lm/last_checkpoint \\
        --data_dir <REPO_ROOT>/data/gsm8k_formulas \\
        --split val \\
        --answer_normalizer numeric \\
        --max_q_len 192 --cot_max_len 80 --ans_max_len 24 \\
        --out_path <CHECKPOINT_ROOT>/outputs/ar_lm_upper_bound/gsm8k.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reasonflow.utils.answer_normalizers import get_normalizer  # noqa: E402


def _extract_answer_after_sep2(text: str) -> str:  # noqa: F811
    """Extract answer after FIRST '####'; also stop at next '####' (decoder
    sometimes repeats `#### ans #### ans ...`) and at newline.
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
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="Anchor LM HF ckpt directory (last_checkpoint).")
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--max_examples", type=int, default=0,
                    help="0 = full split.")
    ap.add_argument("--max_q_len", type=int, default=192)
    ap.add_argument("--cot_max_len", type=int, default=80)
    ap.add_argument("--ans_max_len", type=int, default=24)
    ap.add_argument("--answer_normalizer", default="numeric",
                    choices=["numeric", "smiles", "set_smiles", "classify"])
    ap.add_argument("--out_path", type=Path, required=True)
    args = ap.parse_args()

    norm = get_normalizer(args.answer_normalizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[lm-upper-bound] device={device}, ckpt={args.ckpt}")

    tok = AutoTokenizer.from_pretrained(str(args.ckpt))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    lm = AutoModelForCausalLM.from_pretrained(
        str(args.ckpt), dtype=torch.float32
    ).to(device).eval()

    with (args.data_dir / f"{args.split}.json").open() as fh:
        examples = json.load(fh)
    examples = [
        e for e in examples
        if len(e.get("answer", "")) > 0
    ]
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    print(f"[lm-upper-bound] {args.split}: n={len(examples)}")

    n_correct = 0
    n_total = 0
    n_with_marker = 0
    max_new = args.cot_max_len + args.ans_max_len + 8
    for ei, e in enumerate(examples):
        prompt = e["question"] + " => "
        ids = tok.encode(prompt, add_special_tokens=True)[: args.max_q_len]
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        attn = torch.ones_like(inp)
        out = lm.generate(
            input_ids=inp, attention_mask=attn,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        gen = out[0, inp.shape[1]:]
        text = tok.decode(gen, skip_special_tokens=True)
        if "####" in text:
            n_with_marker += 1
        pred = norm(_extract_answer_after_sep2(text))
        gold = norm(e["answer"])
        if pred == gold and pred != "":
            n_correct += 1
        n_total += 1
        if (ei + 1) % 50 == 0:
            print(f"  [{ei+1}/{len(examples)}] ans_em={n_correct/n_total:.4f} "
                  f"with_marker={n_with_marker/n_total:.4f}")

    em = n_correct / max(1, n_total)
    print(f"[lm-upper-bound] FINAL: ans_em={em:.4f} "
          f"(n={n_total}, with_marker={n_with_marker})")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w") as fh:
        json.dump({
            "ckpt": str(args.ckpt),
            "data_dir": str(args.data_dir),
            "split": args.split,
            "n": n_total,
            "n_with_marker": n_with_marker,
            "answer_em": em,
            "answer_normalizer": args.answer_normalizer,
        }, fh, indent=2)
    print(f"[lm-upper-bound] Wrote {args.out_path}")


if __name__ == "__main__":
    main()
