"""Train the per-token x0-parameterized denoising flow on GSM8K-formulas.

Single-GPU and DDP-aware. Loads JSON in the format produced by
`scripts/preprocess_gsm8k_formulas.py`. Two variants:
    --variant pad_to_M   : Variant A, fixed pace (k = floor(t * M))
    --variant no_pad     : Variant B, per-sample pace (k = floor(t * L))

Loss: per-token CE at position k only (pad-position-masked for Variant A),
plus x0 MSE on x0_hat vs z_1, plus answer CE at t=1.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from reasonflow.models.gsm8k_flow_qwen3 import GSM8KFlowQwen3Config, GSM8KFlowQwen3Model

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class GSM8KFormulasDataset(Dataset):
    """Tokenize on the fly with the model's tokenizer.

    Each item returns python lists / ints; the collator handles padding.
    """

    def __init__(self, json_path: Path, tokenizer, max_q_len: int, cot_max_len: int,
                 ans_max_len: int, no_cot_ae: bool = False):
        """no_cot_ae: when True, build tgt_text as ``q + " #### " + ans`` (no
        reasoning), so the AE encodes a strict no-CoT triple. The cot_input_ids
        / cot_target_ids the collator emits are still computed from the
        per-example reasoning_text and remain available for any cot-decoder
        loss; the CHANGE here is purely the AE-target sequence.
        """
        with json_path.open("r") as fh:
            self.examples = json.load(fh)
        self.tok = tokenizer
        self.max_q_len = max_q_len
        self.cot_max_len = cot_max_len
        self.ans_max_len = ans_max_len
        self.no_cot_ae = no_cot_ae
        # filter degenerate examples. We check reasoning_text directly
        # (rather than n_tokens_reasoning, which is an optional pre-computed
        # stat that synthesized data doesn't carry).
        self.examples = [
            e
            for e in self.examples
            if len(e.get("reasoning_text", "")) > 0
            and len(e.get("answer", "")) > 0
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        e = self.examples[idx]
        q_ids = self.tok.encode(e["question"], add_special_tokens=True)[: self.max_q_len]
        # Reasoning tokens (no special tokens; we add BOS/EOS manually).
        cot_ids = self.tok.encode(e["reasoning_text"], add_special_tokens=False)
        cot_ids = cot_ids[: self.cot_max_len - 1]  # leave room for the final BOS at start
        cot_len = len(cot_ids)

        # Target sequence for e_phi.
        # Default: question + ' => ' + reasoning + ' #### ' + answer
        #   (z_1 encodes the full (input, cot, answer) triple).
        # Strict no-CoT (--no_cot_ae): question + ' #### ' + answer
        #   (z_1 encodes only (input, answer); the AE never sees the cot).
        if self.no_cot_ae:
            tgt_text = e["question"] + " #### " + e["answer"]
            # In no-cot mode the tgt sequence is short; we keep the same cap
            # for consistency with the collator's max_t logic.
            tgt_max = self.max_q_len + self.ans_max_len + 8
        else:
            tgt_text = (
                e["question"] + " => " + e["reasoning_text"] + " #### " + e["answer"]
            )
            tgt_max = self.max_q_len + self.cot_max_len + self.ans_max_len + 8
        tgt_ids = self.tok.encode(tgt_text, add_special_tokens=True)
        tgt_ids = tgt_ids[:tgt_max]

        # Answer ids — always end with EOS so the answer head can learn to stop.
        # Qwen3 tokenizer does NOT append EOS via add_special_tokens=True; Flan-T5
        # does. We strip and re-append by hand to be backbone-agnostic.
        eos_id = self.tok.eos_token_id
        raw_ans = self.tok.encode(e["answer"], add_special_tokens=False)
        if eos_id is not None and len(raw_ans) > 0 and raw_ans[-1] == eos_id:
            raw_ans = raw_ans[:-1]
        raw_ans = raw_ans[: self.ans_max_len - 1]
        if eos_id is not None:
            raw_ans = raw_ans + [eos_id]
        ans_ids = raw_ans

        # CoT ids — also end with EOS so Variant B can early-stop at inference.
        # The per-token-at-k CE loss masks pad positions, but the EOS slot will
        # contribute a valid CE term that teaches the model to halt.
        cot_ids_with_eos = list(cot_ids)
        if eos_id is not None and len(cot_ids_with_eos) < self.cot_max_len:
            cot_ids_with_eos = cot_ids_with_eos + [eos_id]
        cot_ids = cot_ids_with_eos
        cot_len = len(cot_ids)

        return {
            "q_ids": q_ids,
            "cot_ids": cot_ids,
            "tgt_ids": tgt_ids,
            "ans_ids": ans_ids,
            "cot_len": cot_len,
        }


def make_collator(pad_id: int, bos_id: int, cot_max_len: int, ans_max_len: int):
    def pad_to(seqs: List[List[int]], length: int, pad_value: int) -> torch.Tensor:
        out = torch.full((len(seqs), length), pad_value, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    def collate(items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        B = len(items)
        # Question
        max_q = max(len(it["q_ids"]) for it in items)
        q_ids = pad_to([it["q_ids"] for it in items], max_q, pad_id)
        q_attn = (q_ids != pad_id).long()

        # Target (for e_phi)
        max_t = max(len(it["tgt_ids"]) for it in items)
        tgt_ids = pad_to([it["tgt_ids"] for it in items], max_t, pad_id)
        tgt_attn = (tgt_ids != pad_id).long()

        # CoT decoder inputs/targets. We need a sequence [BOS, y_1..y_{Tmax-1}],
        # with corresponding targets [y_0, y_1, ..., y_{Tmax-1}] = the gold tokens
        # at slots 0..Tmax-1. Slots beyond the true length are pad-targeted (-100
        # via ignore_index — but loss is only computed at position k, so this is
        # belt-and-suspenders).
        Tmax = cot_max_len
        cot_inputs = torch.full((B, Tmax), pad_id, dtype=torch.long)
        cot_targets = torch.full((B, Tmax), -100, dtype=torch.long)
        cot_lengths = torch.zeros(B, dtype=torch.long)
        for i, it in enumerate(items):
            ids = it["cot_ids"][: Tmax - 1]
            L = len(ids)
            cot_lengths[i] = L
            # decoder input = [bos, y_0, y_1, ..., y_{L-1}, pad...]
            cot_inputs[i, 0] = bos_id
            if L > 0:
                cot_inputs[i, 1 : 1 + L] = torch.tensor(ids, dtype=torch.long)
            # targets: position j in cot_inputs predicts y_j.
            # cot_inputs[:, 0] is BOS -> predict y_0 = ids[0]
            # cot_inputs[:, j] is y_{j-1} -> predict y_j = ids[j]
            for j in range(L):
                cot_targets[i, j] = ids[j]

        # Answer head
        Tans = ans_max_len + 1
        ans_inputs = torch.full((B, Tans), pad_id, dtype=torch.long)
        ans_targets = torch.full((B, Tans), -100, dtype=torch.long)
        for i, it in enumerate(items):
            ids = it["ans_ids"][:ans_max_len]
            La = len(ids)
            ans_inputs[i, 0] = bos_id
            if La > 0:
                ans_inputs[i, 1 : 1 + La] = torch.tensor(ids, dtype=torch.long)
            for j in range(La):
                ans_targets[i, j] = ids[j]

        return {
            "q_input_ids": q_ids,
            "q_attn_mask": q_attn,
            "tgt_input_ids": tgt_ids,
            "tgt_attn_mask": tgt_attn,
            "cot_input_ids": cot_inputs,
            "cot_target_ids": cot_targets,
            "cot_lengths": cot_lengths,
            "ans_input_ids": ans_inputs,
            "ans_target_ids": ans_targets,
        }

    return collate


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------


def setup_ddp() -> Dict[str, Any]:
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        return {"is_ddp": True, "rank": rank, "world_size": world_size,
                "local_rank": local_rank, "device": device}
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        local_rank = 0
    else:
        device = torch.device("cpu")
        local_rank = 0
    return {"is_ddp": False, "rank": 0, "world_size": 1,
            "local_rank": local_rank, "device": device}


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
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("<CHECKPOINT_ROOT>/data/gsm8k_formulas"),
        help="Canonical data dir. train/val/test are all read from here unless "
             "--extra_train_dirs is also passed (in which case the train set "
             "is concatenated across data_dir + extra_train_dirs).",
    )
    parser.add_argument(
        "--extra_train_dirs",
        type=str,
        default="",
        help="Comma-separated list of EXTRA dirs (each with a train.json) to "
             "concatenate into the training set. val/test always come from "
             "--data_dir only, never from these extras (so val stays clean).",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["pad_to_M", "no_pad"], required=True)
    parser.add_argument("--M", type=int, default=40)
    parser.add_argument(
        "--steps_per_rollout",
        type=int,
        default=0,
        help="Inference steps used by validation-EM generation. "
        "0 uses the model default.",
    )
    parser.add_argument("--cot_max_len", type=int, default=64)
    parser.add_argument("--ans_max_len", type=int, default=24)
    parser.add_argument("--max_q_len", type=int, default=192)
    parser.add_argument("--no_cot_ae", type=lambda s: s.lower() == "true",
                        default=False,
                        help="Strict no-CoT AE mode. When True, the AE input "
                             "sequence (tgt_ids) is q + ' #### ' + ans (NO "
                             "reasoning), so z_1 encodes only the (q, ans) "
                             "pair. The cot decoder loss / cot_target_ids are "
                             "still emitted by the collator for compatibility, "
                             "but should be left unused (w_tok=0) since the "
                             "AE has not seen the cot at all.")

    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--n_denoise_blocks", type=int, default=8)
    parser.add_argument("--n_cot_blocks", type=int, default=6)
    parser.add_argument("--n_ans_blocks", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--backbone", choices=["qwen3"], default="qwen3",
                        help="Encoder/decoder backbone family. qwen3 uses Qwen/Qwen3-0.6B "
                             "frozen with 4 LoRA adapters (one per role).")
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank for the qwen3 backbone.")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha for the qwen3 backbone.")
    parser.add_argument("--full_finetune", type=lambda s: s.lower() == "true",
                        default=False,
                        help="Skip LoRA and fully fine-tune the qwen3 base "
                             "(all roles share one trunk).")
    parser.add_argument("--denoiser_hidden", type=int, default=0,
                        help="DiT internal hidden dim. 0 = inherit latent_dim.")
    parser.add_argument("--denoiser_depth", type=int, default=0,
                        help="DiT depth. 0 = inherit n_denoise_blocks.")
    parser.add_argument("--denoiser_heads", type=int, default=0,
                        help="DiT heads. 0 = inherit n_heads.")
    parser.add_argument("--denoiser_ff_mult", type=int, default=0,
                        help="DiT FFN multiplier. 0 = inherit ff_mult.")
    parser.add_argument("--n_latent_tokens", type=int, default=1,
                        help="Number of latent tokens INSIDE the DiT (the AE "
                             "interface stays single-vector). 1 = old behavior; "
                             ">1 enables intra-latent self-attention.")
    parser.add_argument("--pred_target", choices=["v", "x0"], default="v",
                        help="Denoiser prediction target. 'v' = velocity "
                             "(rectified-flow MSE); 'x0' = clean latent "
                             "(direct MSE to z_1, sphere-normed).")
    parser.add_argument("--denoiser_arch",
                        choices=["local", "dit_canonical"], default="local",
                        help="Denoiser architecture. 'local' = in-repo "
                             "cross-attn-block tower (9-output AdaLN, "
                             "Linear+reshape patchify). 'dit_canonical' = "
                             "faithful port of facebookresearch/DiT with "
                             "PixArt-alpha cross-attention (6-output AdaLN, "
                             "ungated cross-attn, canonical per-patch Linear).")
    parser.add_argument("--q_encoder_separate",
                        type=lambda s: s.lower() == "true",
                        default=False,
                        help="If True, the question encoder owns its own "
                             "Qwen3-0.6B trunk init from raw HF, separate "
                             "from the AE shared trunk. Required for stage-2 "
                             "flow training. Qwen3 only.")
    parser.add_argument("--q_encoder_pool",
                        choices=["per_token", "mean"],
                        default="per_token",
                        help="Question encoder cross-attn KV pooling. "
                             "'per_token' feeds the full last-layer hidden "
                             "states (T_q tokens) as KV. 'mean' mean-pools "
                             "with attention-mask to a 1-token KV.")
    parser.add_argument("--cot_decoder_separate",
                        type=lambda s: s.lower() == "true",
                        default=False,
                        help="If True, the cot_decoder owns its own "
                             "Qwen3-0.6B trunk init from raw HF, separate "
                             "from the AE shared trunk. Required for "
                             "stage-2 flow training with w_tok > 0 because "
                             "we cannot unfreeze the shared trunk (it would "
                             "drift z_1, the regression target). Qwen3 only.")
    parser.add_argument("--save_best_val",
                        type=lambda s: s.lower() == "true",
                        default=False,
                        help="If True, also save 'best_model.pt' whenever "
                             "val loss improves (overwriting any previous "
                             "best). Disabled by default to avoid the disk "
                             "bloat we hit on 2026-05-16 at 256-cell scale. "
                             "Useful for longer-training sweeps where val "
                             "EM has not plateaued and we want to evaluate "
                             "the val-best step rather than the final step. "
                             "NOTE: val LOSS is a worse selector than val EM "
                             "for diffusion (loss continues to descend long "
                             "after EM has peaked); prefer --save_best_val_em "
                             "instead.")
    parser.add_argument("--save_best_val_em",
                        type=lambda s: s.lower() == "true",
                        default=False,
                        help="If True, periodically run model.generate() on "
                             "a small val subset (--val_em_n examples) and "
                             "save 'best_model_em.pt' whenever val AnsEM "
                             "improves. Diffusion benefits from val-EM-best "
                             "ckpt selection (vs val-loss-best, which mirrors "
                             "image-generation FID-vs-CE selection: loss "
                             "keeps dropping after generation-quality peaks).")
    parser.add_argument("--val_em_n",
                        type=int,
                        default=200,
                        help="Number of val examples to run model.generate() "
                             "on at each val_every step for val-EM tracking "
                             "(used when --save_best_val_em=true).")
    parser.add_argument("--save_best_train_em",
                        type=lambda s: s.lower() == "true",
                        default=False,
                        help="If True, also track train-EM at every "
                             "val_every step and save 'best_model_train_em.pt' "
                             "when train-EM improves. Used to debug joint-AE "
                             "finetuning where val-EM is a bad signal due to "
                             "overfitting on small data; train-EM tells us if "
                             "the architecture/loss is still capable of fitting.")
    parser.add_argument("--train_em_n",
                        type=int,
                        default=200,
                        help="Number of train examples for train-EM tracking. "
                             "Set 0 to disable.")
    parser.add_argument("--answer_normalizer",
                        type=str,
                        default="numeric",
                        choices=["numeric", "smiles", "set_smiles", "classify"],
                        help="Per-task answer-string normalizer used for "
                             "val-EM ckpt selection (run_val_em). Default "
                             "'numeric' matches gsm8k. Use 'smiles' for "
                             "mech/molpuzzle, 'set_smiles' for retro, "
                             "'classify' for prontoqa/proofwriter.")
    parser.add_argument("--gradient_checkpointing", type=lambda s: s.lower() == "true",
                        default=True,
                        help="Enable gradient checkpointing on the qwen3 base.")

    parser.add_argument("--w_tok", type=float, default=1.0)
    parser.add_argument("--w_v", type=float, default=1.0,
                        help="Flow-matching velocity MSE weight.")
    parser.add_argument("--w_x0", type=float, default=None,
                        help="DEPRECATED alias for --w_v. The model now predicts "
                             "velocity, not x0; the loss is MSE on v. Kept for "
                             "backward compat with old sbatch scripts.")
    parser.add_argument("--w_ae", type=float, default=1.0,
                        help="Autoencoding loss weight (answer head fed z_1)")
    parser.add_argument("--w_ae_cot", type=float, default=0.0,
                        help="Trajectory-shaping CoT loss weight: if > 0, "
                             "during AE training, also train cot_decoder on "
                             "intermediate latents z_t = (1-t)*z_0 + t*z_1 "
                             "with per-token-at-k CE (k = floor(t * L_eff), "
                             "matching phase-2's flow supervision). Shapes "
                             "z_1 to be the endpoint of a cot-decodable "
                             "linear trajectory. z_0 source is set by "
                             "--ae_z0_origin.")
    parser.add_argument("--ae_z0_origin", type=str, default="noise",
                        choices=["noise", "question_encoder"],
                        help="Source of z_0 (start of trajectory) for "
                             "BOTH AE-side trajectory shaping and phase-2 "
                             "flow training+inference. 'noise' = raw randn "
                             "(back-compat with original eps-based "
                             "diffusion); 'question_encoder' = "
                             "question_encoder.encode_z_0(q) -- a "
                             "deterministic question-conditioned start "
                             "(no x0_seed randomness in the trajectory). "
                             "Always match training and inference -- the "
                             "denoiser learns v* = z_1 - z_0, so the "
                             "z_0 distribution must be identical.")
    parser.add_argument("--ae_noise_sigma_max", type=float, default=0.0,
                        help="Max stddev of Gaussian noise added to z_1 before "
                             "the answer head's L_ae path. Per-sample sigma is "
                             "sampled U(0, this). Sphere-norm is re-applied "
                             "after noise. 0 = clean autoencoder.")
    parser.add_argument("--pool_mode", choices=["mean", "last_token"], default="mean",
                        help="Pooling mode for target encoder over (q,cot,a).")
    parser.add_argument("--sphere_norm", type=lambda s: s.lower() == "true",
                        default=True, help="Sphere-normalize z_1 and x0_hat")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=2000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.add_argument("--max_train_examples", type=int, default=0,
                        help="If >0, subsample training set (debugging).")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init_ae_checkpoint", type=Path, default=None,
                        help="Path to a stage-1 AE checkpoint. Loaded into the "
                             "model state_dict (strict=False) before the "
                             "optimizer is built. Use with --freeze_ae for "
                             "stage-2 flow training.")
    parser.add_argument("--freeze_ae", type=lambda s: s.lower() == "true",
                        default=False,
                        help="Freeze the AE components after init: backbone, "
                             "target_encoder, answer_head. Only the denoiser "
                             "and (by default) question_encoder + cot_decoder "
                             "are trainable. Use with --init_ae_checkpoint.")
    parser.add_argument("--freeze_question_encoder",
                        type=lambda s: s.lower() == "true",
                        default=False,
                        help="In addition to --freeze_ae, also freeze the "
                             "question_encoder. Used for the trajectory-"
                             "shaped AE recipe where question_encoder was "
                             "trained during AE phase to produce z_0 and "
                             "should be frozen for phase-2.")
    parser.add_argument("--freeze_cot_decoder",
                        type=lambda s: s.lower() == "true",
                        default=False,
                        help="In addition to --freeze_ae, also freeze the "
                             "cot_decoder. Used for the trajectory-shaped "
                             "AE recipe where cot_decoder was trained during "
                             "AE phase and phase-2 has --w_tok 0 so the "
                             "cot_decoder isn't needed (also skipped in "
                             "the forward when w_tok=0 to save compute).")
    args = parser.parse_args()

    if args.w_x0 is not None:
        # User passed deprecated --w_x0. Honor it as --w_v and tell them.
        print(
            f"[deprecation] --w_x0 is deprecated; using --w_v={args.w_x0}. "
            "The denoiser now predicts velocity (v=z_1-eps), not x0; the loss "
            "is MSE on v."
        )

    ddp_info = setup_ddp()
    is_master = ddp_info["rank"] == 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO if is_master else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger("gsm8k_flow")

    if is_master:
        log.info(f"args: {vars(args)}")
        log.info(f"ddp: {ddp_info}")

    torch.manual_seed(args.seed + ddp_info["rank"])

    # Build model + tokenizer
    common_kwargs = dict(
        latent_dim=args.latent_dim,
        n_denoise_blocks=args.n_denoise_blocks,
        n_cot_blocks=args.n_cot_blocks,
        n_ans_blocks=args.n_ans_blocks,
        n_heads=args.n_heads,
        dropout=args.dropout,
        cot_max_len=args.cot_max_len,
        ans_max_len=args.ans_max_len,
        variant=args.variant,
        M=args.M,
        steps_per_rollout=args.steps_per_rollout,
        w_tok=args.w_tok,
        w_v=args.w_v if args.w_x0 is None else args.w_x0,
        w_ae=args.w_ae,
        w_ae_cot=args.w_ae_cot,
        ae_z0_origin=args.ae_z0_origin,
        ae_noise_sigma_max=args.ae_noise_sigma_max,
        pool_mode=args.pool_mode,
        sphere_norm=args.sphere_norm,
        denoiser_hidden=args.denoiser_hidden,
        denoiser_depth=args.denoiser_depth,
        denoiser_heads=args.denoiser_heads,
        denoiser_ff_mult=args.denoiser_ff_mult,
        n_latent_tokens=args.n_latent_tokens,
        pred_target=args.pred_target,
        denoiser_arch=args.denoiser_arch,
    )
    cfg = GSM8KFlowQwen3Config(
        **common_kwargs,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        full_finetune=args.full_finetune,
        q_encoder_separate=args.q_encoder_separate,
        q_encoder_pool=args.q_encoder_pool,
        cot_decoder_separate=args.cot_decoder_separate,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model = GSM8KFlowQwen3Model(cfg)
    device = ddp_info["device"]
    model = model.to(device)

    if ddp_info["is_ddp"]:
        model = DDP(model, device_ids=[ddp_info["local_rank"]], find_unused_parameters=True)

    raw = model.module if isinstance(model, DDP) else model

    # Datasets
    train_ds = GSM8KFormulasDataset(
        args.data_dir / "train.json",
        raw.tokenizer,
        max_q_len=args.max_q_len,
        cot_max_len=args.cot_max_len,
        ans_max_len=args.ans_max_len,
        no_cot_ae=args.no_cot_ae,
    )
    # Optional: concatenate extra train.json files (e.g. synthetic data tiers).
    # val/test stay from args.data_dir so val never includes synth examples.
    if args.extra_train_dirs:
        extra_paths = [
            Path(p.strip())
            for p in args.extra_train_dirs.split(",")
            if p.strip()
        ]
        for ep in extra_paths:
            extra_ds = GSM8KFormulasDataset(
                ep / "train.json",
                raw.tokenizer,
                max_q_len=args.max_q_len,
                cot_max_len=args.cot_max_len,
                ans_max_len=args.ans_max_len,
                no_cot_ae=args.no_cot_ae,
            )
            n_before = len(train_ds.examples)
            train_ds.examples = train_ds.examples + extra_ds.examples
            if is_master:
                log.info(
                    f"extra_train_dir {ep}: appended {len(extra_ds)} -> "
                    f"train_size {n_before} -> {len(train_ds.examples)}"
                )
    val_ds = GSM8KFormulasDataset(
        args.data_dir / "val.json",
        raw.tokenizer,
        max_q_len=args.max_q_len,
        cot_max_len=args.cot_max_len,
        ans_max_len=args.ans_max_len,
        no_cot_ae=args.no_cot_ae,
    )
    if args.max_train_examples > 0:
        train_ds.examples = train_ds.examples[: args.max_train_examples]
    if is_master:
        log.info(f"train: {len(train_ds)}; val: {len(val_ds)}")

    collate = make_collator(
        pad_id=raw.cfg.pad_token_id,
        bos_id=raw.cfg.bos_token_id,
        cot_max_len=args.cot_max_len,
        ans_max_len=args.ans_max_len,
    )

    if ddp_info["is_ddp"]:
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    # Optional: load a stage-1 AE checkpoint into the model state_dict
    # *before* the optimizer is constructed and *before* freezing, so that
    # both frozen and trainable components start from the AE-trained weights
    # (the denoiser was at random init in the AE-only run because w_v=0; that
    # matches what stage-2 wants anyway).
    if args.init_ae_checkpoint is not None and args.init_ae_checkpoint.exists():
        ae_ckpt = torch.load(
            args.init_ae_checkpoint, map_location="cpu", weights_only=False
        )
        # Drop any AE-checkpoint keys whose shape does not match the current
        # model. This handles the case where stage-2 uses a different-sized
        # denoiser (e.g. DiT-S/B/L/XL) than the one stored in the stage-1 AE
        # checkpoint, OR a different denoiser_arch entirely. Stage-2 always
        # starts the denoiser fresh anyway (w_v=0 in stage-1).
        cur_state = raw.state_dict()
        filtered = {}
        skipped_shape, skipped_unknown = [], []
        for k, v in ae_ckpt["model"].items():
            if k not in cur_state:
                skipped_unknown.append(k)
                continue
            if cur_state[k].shape != v.shape:
                skipped_shape.append(
                    f"{k}: ckpt={tuple(v.shape)} cur={tuple(cur_state[k].shape)}"
                )
                continue
            filtered[k] = v
        result = raw.load_state_dict(filtered, strict=False)
        if is_master:
            log.info(
                f"init_ae_checkpoint: loaded {args.init_ae_checkpoint} "
                f"loaded={len(filtered)} "
                f"missing={len(result.missing_keys)} "
                f"unexpected={len(result.unexpected_keys)} "
                f"skipped_shape={len(skipped_shape)} "
                f"skipped_unknown={len(skipped_unknown)}"
            )
            if skipped_shape:
                log.info(
                    "init_ae_checkpoint: skipping shape-mismatched keys "
                    f"(showing up to 5/{len(skipped_shape)}): "
                    + "; ".join(skipped_shape[:5])
                )

    if args.freeze_ae:
        # Freeze everything, then unfreeze the stage-2 trainable components:
        # the denoiser (always), and conditionally question_encoder and
        # cot_decoder (per --freeze_question_encoder / --freeze_cot_decoder).
        #
        # For the trajectory-shaped AE recipe, both encoders should be
        # frozen during phase-2 (--freeze_question_encoder true,
        # --freeze_cot_decoder true) since they were already trained
        # during the AE phase to define the trajectory. For the legacy
        # frozen-AE recipe with phase-2 cot supervision (w_tok > 0), keep
        # cot_decoder trainable (the default).
        for p in raw.parameters():
            p.requires_grad = False
        for p in raw.denoiser.parameters():
            p.requires_grad = True
        if not args.freeze_question_encoder:
            for p in raw.question_encoder.parameters():
                p.requires_grad = True
        if not args.freeze_cot_decoder:
            for p in raw.cot_decoder.parameters():
                p.requires_grad = True
        if is_master:
            n_total = sum(p.numel() for p in raw.parameters())
            n_train = sum(
                p.numel() for p in raw.parameters() if p.requires_grad
            )
            log.info(
                f"freeze_ae: trainable {n_train/1e6:.2f}M / total "
                f"{n_total/1e6:.2f}M ({100*n_train/n_total:.2f}%)"
            )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    scaler = GradScaler(enabled=args.use_amp)

    step = 0
    best_val = float("inf")
    best_val_em = -1.0  # -1 sentinel so first measured EM is always saved
    best_train_em = -1.0  # -1 sentinel so first measured EM is always saved

    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        # strict=False so checkpoints predating new submodules (e.g. the
        # question_encoder.z0_proj head added for trajectory-shaped AE)
        # still load -- those keys initialize fresh from the module
        # construction. Used for resume-and-extend across schema-evolving
        # model versions; missing keys are logged for visibility.
        missing, unexpected = raw.load_state_dict(ckpt["model"], strict=False)
        if is_master and (missing or unexpected):
            log.info(
                f"resume strict=False: missing={len(missing)} "
                f"unexpected={len(unexpected)}"
            )
            if missing:
                log.info(f"  missing (showing 5/{len(missing)}): {missing[:5]}")
            if unexpected:
                log.info(f"  unexpected (showing 5/{len(unexpected)}): {unexpected[:5]}")
        # Only load optim state if the trainable-param set matches.
        # Resuming with a different freeze config (e.g., source had
        # freeze_ae=true and we now want freeze_ae=false) changes the
        # number of param groups -- pytorch's load_state_dict throws.
        # In that case start fresh optim state.
        try:
            optim.load_state_dict(ckpt["optim"])
        except (ValueError, KeyError) as e:
            if is_master:
                log.warning(
                    f"could not load optim state from {args.resume} "
                    f"({type(e).__name__}: {e}); starting fresh optimizer "
                    f"state. This is expected when --freeze_ae or related "
                    f"freeze flags differ between the source ckpt and this run."
                )
        step = ckpt.get("step", 0)
        best_val = ckpt.get("best_val", best_val)
        best_val_em = ckpt.get("best_val_em", best_val_em)
        best_train_em = ckpt.get("best_train_em", best_train_em)
        if is_master:
            log.info(f"resumed from {args.resume} at step {step}")

    def save_ckpt(path: Path, step: int, best_val: float):
        """Save then immediately verify the file is a loadable zip.

        After the disk-full crisis on 2026-05-16, several `last_checkpoint.pt`
        files were silently truncated -- the python process did not crash,
        the slurm job was marked COMPLETED, and only the downstream eval
        step caught the corruption. The fix: after every save, attempt to
        re-open the zip and list its members. If that fails we treat the
        checkpoint as corrupt and raise loudly so the training job exits
        non-zero.
        """
        if not is_master:
            return
        path = Path(path)
        torch.save(
            {
                "model": raw.state_dict(),
                "optim": optim.state_dict(),
                "step": step,
                "best_val": best_val,
                "best_val_em": best_val_em,
                "best_train_em": best_train_em,
                "args": vars(args),
                "cfg": asdict(raw.cfg),
            },
            path,
        )
        # Verify: torch.save uses a zip container; if disk filled mid-save,
        # the zip's central directory is missing/truncated.
        import zipfile  # local import to keep top-of-file imports tidy
        try:
            with zipfile.ZipFile(path, "r") as zf:
                _ = zf.namelist()
        except Exception as e:
            log.error(
                f"CHECKPOINT CORRUPT after save: {path} "
                f"(size={path.stat().st_size if path.exists() else 'missing'} bytes); "
                f"raising so the slurm job is marked FAILED"
            )
            raise RuntimeError(
                f"corrupt checkpoint {path}: {type(e).__name__}: {e}"
            ) from e

    @torch.no_grad()
    def run_val() -> Dict[str, float]:
        model.eval()
        agg = {"loss": 0.0, "loss_tok": 0.0, "loss_v": 0.0,
               "loss_ae": 0.0, "n": 0}
        for batch in val_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast(enabled=args.use_amp, dtype=torch.bfloat16):
                out = model(batch)
            B = batch["q_input_ids"].shape[0]
            agg["loss"] += float(out["loss"]) * B
            agg["loss_tok"] += float(out["loss_tok"]) * B
            agg["loss_v"] += float(out["loss_v"]) * B
            agg["loss_ae"] += float(out["loss_ae"]) * B
            agg["n"] += B
        if ddp_info["is_ddp"]:
            t = torch.tensor(
                [agg["loss"], agg["loss_tok"], agg["loss_v"],
                 agg["loss_ae"], agg["n"]],
                device=device,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            agg = {
                "loss": float(t[0]), "loss_tok": float(t[1]), "loss_v": float(t[2]),
                "loss_ae": float(t[3]), "n": float(t[4]),
            }
        n = agg["n"] or 1
        return {k: v / n for k, v in agg.items() if k != "n"}

    # Val-EM ckpt selection: optionally run model.generate() on the FIRST
    # ``args.val_em_n`` val examples and compute AnsEM. We mirror the eval
    # in test_gsm8k_flow.py but inline here to avoid importing the test
    # harness mid-train.
    val_em_loader = None
    if args.save_best_val_em:
        # Take the first val_em_n examples to keep the EM eval fast and
        # reproducible across val_every checkpoints.
        val_em_examples = val_ds.examples[: args.val_em_n]
        from copy import copy as _copy_obj
        val_em_ds = _copy_obj(val_ds)
        val_em_ds.examples = val_em_examples
        val_em_loader = DataLoader(
            val_em_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=2,
            pin_memory=True,
        )
        if is_master:
            log.info(f"val-EM ckpt selection enabled: "
                     f"{len(val_em_examples)} examples per val_every step")

    @torch.no_grad()
    def run_val_em() -> float:
        """Run model.generate() on val_em_n examples; return AnsEM in [0,1].

        Uses x0_mode=predicted, x0_seed=0 (now that the unseeded-noise bug
        is fixed in models.gsm8k_flow_qwen3.generate()) so this metric is
        deterministic across val_every steps.
        """
        if val_em_loader is None:
            return 0.0
        model.eval()
        n_correct = 0
        n_total = 0
        idx = 0
        # Per-task answer normalizer (gsm8k=numeric default, smiles for
        # chemistry tasks, classify for prontoqa/proofwriter, etc.).
        from reasonflow.utils.answer_normalizers import get_normalizer
        normalize = get_normalizer(args.answer_normalizer)
        for batch in val_em_loader:
            B = batch["q_input_ids"].shape[0]
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast(enabled=args.use_amp, dtype=torch.bfloat16):
                gen = model.generate(
                    batch["q_input_ids"], batch["q_attn_mask"],
                    K=None, return_trajectory=False,
                    x0_mode="predicted", x0_seed=0,
                )
            ans_tokens = gen["ans_tokens"]
            for i in range(B):
                pred = raw.tokenizer.decode(ans_tokens[i].tolist(), skip_special_tokens=True).strip()
                pred_n = normalize(pred)
                gold_n = normalize(val_em_examples[idx]["answer"])
                if pred_n == gold_n:
                    n_correct += 1
                n_total += 1
                idx += 1
        if ddp_info["is_ddp"]:
            t = torch.tensor([n_correct, n_total], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            n_correct, n_total = int(t[0].item()), int(t[1].item())
        return n_correct / max(1, n_total)

    # Train-EM tracking: mirrors the val-EM block but uses train examples.
    # Doesn't gate ckpt selection by default (use --save_best_train_em to
    # also save best_model_train_em.pt). Headline use case: debugging
    # joint-AE finetune where val EM is a bad signal due to overfitting.
    train_em_loader = None
    if args.save_best_train_em and args.train_em_n > 0:
        # Use the FIRST train_em_n train examples; same examples every step
        # so the trajectory is comparable across checkpoints.
        train_em_examples = train_ds.examples[: args.train_em_n]
        from copy import copy as _copy_obj_t
        train_em_ds = _copy_obj_t(train_ds)
        train_em_ds.examples = train_em_examples
        train_em_loader = DataLoader(
            train_em_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=2,
            pin_memory=True,
        )
        if is_master:
            log.info(f"train-EM tracking enabled: "
                     f"{len(train_em_examples)} examples per val_every step")

    @torch.no_grad()
    def run_train_em() -> float:
        if train_em_loader is None:
            return 0.0
        model.eval()
        n_correct = 0
        n_total = 0
        idx = 0
        from reasonflow.utils.answer_normalizers import get_normalizer
        normalize = get_normalizer(args.answer_normalizer)
        for batch in train_em_loader:
            B = batch["q_input_ids"].shape[0]
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast(enabled=args.use_amp, dtype=torch.bfloat16):
                gen = model.generate(
                    batch["q_input_ids"], batch["q_attn_mask"],
                    K=None, return_trajectory=False,
                    x0_mode="predicted", x0_seed=0,
                )
            ans_tokens = gen["ans_tokens"]
            for i in range(B):
                pred = raw.tokenizer.decode(ans_tokens[i].tolist(), skip_special_tokens=True).strip()
                pred_n = normalize(pred)
                gold_n = normalize(train_em_examples[idx]["answer"])
                if pred_n == gold_n:
                    n_correct += 1
                n_total += 1
                idx += 1
        if ddp_info["is_ddp"]:
            t = torch.tensor([n_correct, n_total], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            n_correct, n_total = int(t[0].item()), int(t[1].item())
        return n_correct / max(1, n_total)

    model.train()
    t0 = time.time()
    epoch = 0
    while step < args.max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            if step >= args.max_steps:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            lr = cosine_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
            for g in optim.param_groups:
                g["lr"] = lr

            optim.zero_grad(set_to_none=True)
            with autocast(enabled=args.use_amp, dtype=torch.bfloat16):
                out = model(batch)
                loss = out["loss"]

            if not torch.isfinite(loss):
                log.warning(f"skip non-finite loss at step {step}: {float(loss)}")
                step += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            scaler.step(optim)
            scaler.update()

            if is_master and step % args.log_every == 0:
                dt = time.time() - t0
                log.info(
                    f"step {step:6d} | lr {lr:.2e} | loss {float(loss):.4f} | "
                    f"tok {float(out['loss_tok']):.4f} | v {float(out['loss_v']):.4f} | "
                    f"ae {float(out['loss_ae']):.4f} | "
                    f"ae_cot {float(out.get('loss_ae_cot', 0.0)):.4f} | "
                    f"valid {float(out['valid_frac']):.3f} | "
                    f"|z1| {float(out['z1_norm']):.2f} | |x0| {float(out['x0_norm']):.2f} | "
                    f"|x0-z1| {float(out['x0_z1_dist']):.2f} | "
                    f"|v_err| {float(out['v_err_norm']):.2f} | {dt:.1f}s"
                )

            if step > 0 and step % args.val_every == 0:
                v = run_val()
                # Val-EM eval (only if --save_best_val_em). Doing this on
                # all ranks so DDP reductions are well-formed.
                val_em_score = run_val_em() if args.save_best_val_em else None
                # Train-EM tracking (only if --save_best_train_em). Same DDP
                # rule. Doesn't gate ckpt selection unless --save_best_train_em.
                train_em_score = run_train_em() if args.save_best_train_em else None
                if is_master:
                    log.info(
                        f"VAL  step {step:6d} | loss {v['loss']:.4f} | "
                        f"tok {v['loss_tok']:.4f} | v_loss {v['loss_v']:.4f} | "
                        f"ae {v['loss_ae']:.4f}"
                    )
                    if val_em_score is not None:
                        log.info(f"VAL-EM step {step:6d} | AnsEM "
                                 f"{val_em_score*100:.2f}% (n={args.val_em_n})")
                    if train_em_score is not None:
                        log.info(f"TRAIN-EM step {step:6d} | AnsEM "
                                 f"{train_em_score*100:.2f}% (n={args.train_em_n})")
                    if v["loss"] < best_val:
                        best_val = v["loss"]
                        # We default to NOT saving best_model.pt because at
                        # 256-cell scale on 2026-05-16 each cell wrote
                        # best + last + final = 75 GB and blew the shared
                        # filesystem. With --save_best_val=true the user
                        # opts back into the val-best save (one file per
                        # cell, overwritten in place, ~25 GB each).
                        log.info(f"new best val @ step {step} val={best_val:.4f}")
                        if args.save_best_val:
                            save_ckpt(args.output_dir / "best_model.pt", step, best_val)
                            log.info(f"saved best_model.pt @ step {step}")
                    if val_em_score is not None and val_em_score > best_val_em:
                        best_val_em = val_em_score
                        log.info(f"new best val-EM @ step {step} "
                                 f"em={best_val_em*100:.2f}%")
                        if args.save_best_val_em:
                            save_ckpt(args.output_dir / "best_model_em.pt",
                                      step, best_val)
                            log.info(f"saved best_model_em.pt @ step {step} "
                                     f"em={best_val_em*100:.2f}%")
                    if train_em_score is not None and train_em_score > best_train_em:
                        best_train_em = train_em_score
                        log.info(f"new best train-EM @ step {step} "
                                 f"em={best_train_em*100:.2f}%")
                        if args.save_best_train_em:
                            save_ckpt(args.output_dir / "best_model_train_em.pt",
                                      step, best_val)
                            log.info(f"saved best_model_train_em.pt @ step {step} "
                                     f"em={best_train_em*100:.2f}%")
                model.train()

            if step > 0 and step % args.save_every == 0:
                save_ckpt(args.output_dir / "last_checkpoint.pt", step, best_val)

            step += 1
        epoch += 1

    # Final save -> last_checkpoint.pt (was a separate final_model.pt; same
    # filesystem reason as the best_model.pt removal above).
    save_ckpt(args.output_dir / "last_checkpoint.pt", step, best_val)
    if is_master:
        log.info("training complete")

    if ddp_info["is_ddp"]:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
