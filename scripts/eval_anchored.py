"""Re-evaluate val_em on a saved AnchoredPhase2Model checkpoint with a
correct (per-task) answer normalizer.

The original train_anchored_phase2.py used a GSM8K-only `####
<numeric>` regex that returned no match on the 4 transfer tasks
(retro / mech / molpuzzle / prontoqa_hard), causing val_em=0 and
nonsensical best-val-em ckpt selection. This script lets us recompute
the actual val_em for any saved ckpt without retraining.

Usage:
    python scripts/eval_anchored.py \\
        --checkpoint /path/to/last_checkpoint.pt \\
        --data_dir   /path/to/qwen3_<task>      \\
        --answer_normalizer set_smiles          \\
        --output_path /path/to/val_em_reeval.json \\
        --val_em_n 200 --n_int_steps 64

The script reads `args` out of the checkpoint to rebuild the model
exactly. Only `--data_dir` / `--answer_normalizer` need to come from
the user (since the original args have the wrong normalizer).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train_anchored_phase2 import (  # noqa: E402
    AnchoredPhase2Dataset,
    AnchoredPhase2Model,
    eval_val_em,
    make_collator,
)

from reasonflow.utils.answer_normalizers import get_normalizer  # noqa: E402


def _bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="last_checkpoint.pt or best_val_em.pt from "
                         "train_anchored_phase2.py.")
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--answer_normalizer", required=True,
                    choices=["numeric", "smiles", "set_smiles", "classify"])
    ap.add_argument("--output_path", type=Path, required=True,
                    help="JSON file to write the re-evaluated val_em into.")
    ap.add_argument("--split", default="val", choices=["val", "test", "train"],
                    help="Which split to evaluate. 'train' computes training "
                         "EM as an overfit diagnostic.")
    ap.add_argument("--val_em_n", type=int, default=200)
    ap.add_argument("--n_int_steps", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=64,
                    help="Suffix-decoder generation budget. 64 OK for "
                         "gsm8k_formulas, prontoqa_hard. Use 256-384 "
                         "for mech / retro / molpuzzle (long SMILES).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    args_cli = ap.parse_args()

    torch.manual_seed(args_cli.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[reeval] device={device}")

    print(f"[reeval] loading {args_cli.checkpoint}")
    ck = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    ck_args = ck["args"]
    if not isinstance(ck_args, dict):
        ck_args = vars(ck_args)
    print(f"[reeval] ckpt args: M={ck_args.get('M')} "
          f"dit={ck_args.get('dit_arch')} step={ck.get('step')}")

    tok = AutoTokenizer.from_pretrained(ck_args.get(
        "decoder_base_model", "Qwen/Qwen3-0.6B"
    ))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Rebuild the model from ckpt args (no gradient checkpointing — eval only).
    model = AnchoredPhase2Model(
        v1_lm_ckpt=ck_args.get("v1_lm_ckpt"),
        v1_decoder_ckpt=ck_args.get("v1_decoder_ckpt"),
        v1_pool_mode=ck_args.get("v1_pool_mode"),
        decoder_base_model=ck_args.get("decoder_base_model", "Qwen/Qwen3-0.6B"),
        q_encoder_base_model=ck_args.get("q_encoder_base_model", "Qwen/Qwen3-0.6B"),
        dit_arch=ck_args.get("dit_arch", "ditXL"),
        n_latent_tokens=ck_args.get("n_latent_tokens", 1),
        K_total=ck_args["M"],
        gradient_checkpointing=False,
        include_z1=ck_args.get("include_z1", "none"),
        z0_source=ck_args.get("z0_source", "none"),
        ae_z1_ckpt=ck_args.get("ae_z1_ckpt"),
    ).to(device)

    if "denoiser_state" in ck:
        model.denoiser.load_state_dict(ck["denoiser_state"])
    if "q_encoder_state" in ck:
        model.q_encoder.load_state_dict(ck["q_encoder_state"])
    print("[reeval] loaded denoiser_state + q_encoder_state")

    json_name = f"{args_cli.split}.json"
    val_ds = AnchoredPhase2Dataset(
        args_cli.data_dir / json_name, tok,
        M=ck_args["M"],
        max_q_len=ck_args["max_q_len"],
        cot_max_len=ck_args["cot_max_len"],
        ans_max_len=ck_args["ans_max_len"],
        max_examples=args_cli.val_em_n if args_cli.split == "train" else None,
    )
    coll = make_collator(tok.pad_token_id, M=ck_args["M"])
    val_loader = DataLoader(
        val_ds, batch_size=args_cli.batch_size, shuffle=False,
        collate_fn=coll, num_workers=args_cli.num_workers,
    )
    print(f"[reeval] val={len(val_ds)}, batch_size={args_cli.batch_size}")

    answer_norm = get_normalizer(args_cli.answer_normalizer)
    print(f"[reeval] answer_normalizer={args_cli.answer_normalizer}, "
          f"running eval (n={args_cli.val_em_n}, "
          f"n_int_steps={args_cli.n_int_steps})...")

    em, n_eval = eval_val_em(
        model, val_loader, tok, device,
        max_examples=args_cli.val_em_n,
        answer_normalizer=answer_norm,
        max_new_tokens=args_cli.max_new_tokens,
        n_steps=args_cli.n_int_steps,
    )
    print(f"[reeval] FINAL val_em={em:.4f} (n={n_eval})")

    args_cli.output_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "checkpoint": str(args_cli.checkpoint),
        "data_dir": str(args_cli.data_dir),
        "split": args_cli.split,
        "answer_normalizer": args_cli.answer_normalizer,
        "step": ck.get("step"),
        "val_em": em,
        "n": n_eval,
        "n_int_steps": args_cli.n_int_steps,
        "seed": args_cli.seed,
    }
    with args_cli.output_path.open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[reeval] wrote {args_cli.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
