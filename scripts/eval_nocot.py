"""Evaluate a GSM8K-formulas flow model.

Reports answer EM, reasoning Exact Sequence Match (ESeq), Prefix Match,
Step Accuracy, and normalized edit distance.

Also dumps:
- depth_curve.json   : answer EM and reasoning ESeq vs. number of K steps
- trajectory.json    : per-correctness trajectory diagnostics
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from train_ae import GSM8KFormulasDataset, make_collator

from reasonflow.models.gsm8k_flow_qwen3 import GSM8KFlowQwen3Config, GSM8KFlowQwen3Model

# Faithful GSM8K calculator-annotation regex. Both the upstream da03 corpus
# and the Qwen3 tokenizer preserve the `<<...>>` markers verbatim, so we
# parse them as-is. (The earlier `([^\s>]+)>>` form was a Flan-T5 workaround
# because that tokenizer mapped `<` to <unk>, dropping the leading `<<`.)
STEP_RE = re.compile(r"<<([^>]+)>>")


def edit_distance(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


# Default normalizer (numeric/gsm8k); replaced at module level by `set_normalizer`
# at startup based on the --answer_normalizer CLI flag.
def normalize_answer(s: str) -> str:
    s = s.strip()
    # strip thousand separators / dollar signs / trailing punctuation
    s = s.replace(",", "").replace("$", "")
    s = s.rstrip(".")
    return s


def set_normalizer(name: str) -> None:
    """Swap the module-level `normalize_answer` to the named one. Called
    once from main() after argparse so the rest of the script doesn't
    need plumbing."""
    global normalize_answer
    from reasonflow.utils.answer_normalizers import get_normalizer
    normalize_answer = get_normalizer(name)


def split_steps(reasoning_text: str) -> List[str]:
    return STEP_RE.findall(reasoning_text)


def cot_chain_end_answer(pred_cot: str) -> Optional[str]:
    """Pull the RHS of the LAST `<<...>>` step and return its leading integer."""
    steps_ = split_steps(pred_cot)
    if not steps_:
        return None
    last = steps_[-1]
    rhs = last.split("=")[-1].strip() if "=" in last else last.strip()
    m = re.match(r"-?\d+", rhs)
    return m.group(0) if m else None


def reasoning_metrics(pred_text: str, gold_text: str) -> Dict[str, float]:
    p_steps = split_steps(pred_text)
    g_steps = split_steps(gold_text)
    eseq = float(p_steps == g_steps)
    common_prefix = 0
    for a, b in zip(p_steps, g_steps):
        if a == b:
            common_prefix += 1
        else:
            break
    prefix_match = common_prefix / max(len(g_steps), 1)
    step_acc = sum(int(s in g_steps) for s in p_steps) / max(len(p_steps), 1)
    ed = edit_distance(p_steps, g_steps)
    norm_ed = ed / max(len(g_steps), 1)
    return {
        "ESeq": eseq,
        "PrefixMatch": prefix_match,
        "StepAcc": step_acc,
        "EditDist": float(ed),
        "NormEditDist": float(norm_ed),
        "n_pred_steps": float(len(p_steps)),
        "n_gold_steps": float(len(g_steps)),
    }


def decode_clean(tokenizer, ids: torch.Tensor) -> str:
    txt = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
    return txt


def evaluate(
    model,
    loader,
    device,
    examples: List[Dict[str, Any]],
    K: Optional[int],
    log,
    x0_mode: str = "predicted",
    x0_seed: int = 0,
    sample_dump_limit: int = 0,
    steps_per_rollout: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Generate predictions and compute metrics.

    `x0_mode` is forwarded to the model's `generate()`:
      - "predicted"  : default, x0_hat = denoiser(z_t, t).
      - "gaussian"   : substitute a fresh sphere-normalised random vector for x0_hat.
      - "zero"       : substitute a zero vector for x0_hat (degenerate control).
      - "oracle_z1"  : run the target encoder on the gold (q,reasoning,answer)
                       sequence and use that z_1 as x0_hat — upper bound.
    `sample_dump_limit=0` means dump every sample; otherwise dump first N.
    """
    model.eval()
    tokenizer = model.tokenizer
    answer_em_total = 0
    cot_end_correct_total = 0
    cc_aa = 0  # CoT correct, answer correct
    cc_aw = 0  # CoT correct, answer wrong
    cw_aa = 0  # CoT wrong,   answer correct
    cw_aw = 0  # CoT wrong,   answer wrong
    metrics_acc: Dict[str, float] = {}
    n = 0
    sample_outputs: List[Dict[str, Any]] = []

    idx_global = 0
    with torch.no_grad():
        for batch in loader:
            B = batch["q_input_ids"].shape[0]
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            gen_kwargs = dict(K=K, return_trajectory=False, x0_mode=x0_mode,
                              x0_seed=x0_seed)
            if steps_per_rollout is not None and steps_per_rollout > 0:
                gen_kwargs["steps_per_rollout"] = steps_per_rollout
            if x0_mode == "oracle_z1":
                gen_kwargs["tgt_input_ids"] = batch["tgt_input_ids"]
                gen_kwargs["tgt_attn_mask"] = batch["tgt_attn_mask"]
            gen = model.generate(
                batch["q_input_ids"], batch["q_attn_mask"], **gen_kwargs
            )
            cot_tokens = gen["cot_tokens"]
            ans_tokens = gen["ans_tokens"]

            for i in range(B):
                ex = examples[idx_global]
                idx_global += 1
                pred_cot = decode_clean(tokenizer, cot_tokens[i])
                pred_ans = decode_clean(tokenizer, ans_tokens[i])
                pred_ans_n = normalize_answer(pred_ans)
                gold_ans_n = normalize_answer(ex["answer"])
                em = int(pred_ans_n == gold_ans_n)
                answer_em_total += em
                # CoT-end-correctness: read answer off the last <<...>> rhs.
                cot_end = cot_chain_end_answer(pred_cot)
                cot_end_norm = normalize_answer(cot_end) if cot_end is not None else ""
                cot_ok = int(cot_end_norm == gold_ans_n) if cot_end is not None else 0
                cot_end_correct_total += cot_ok
                if cot_ok and em:
                    cc_aa += 1
                elif cot_ok and not em:
                    cc_aw += 1
                elif (not cot_ok) and em:
                    cw_aa += 1
                else:
                    cw_aw += 1
                m = reasoning_metrics(pred_cot, ex["reasoning_text"])
                for k_, v in m.items():
                    metrics_acc[k_] = metrics_acc.get(k_, 0.0) + v
                n += 1

                if sample_dump_limit == 0 or len(sample_outputs) < sample_dump_limit:
                    sample_outputs.append(
                        {
                            "question": ex["question"],
                            "gold_cot": ex["reasoning_text"],
                            "pred_cot": pred_cot,
                            "gold_ans": ex["answer"],
                            "pred_ans": pred_ans,
                            "cot_end_ans": cot_end_norm,
                            "answer_em": em,
                            "cot_end_em": cot_ok,
                            **m,
                        }
                    )
            if n % 200 == 0:
                log.info(
                    f"  evaluated {n} samples; running AnswerEM={answer_em_total/n:.3f}"
                    f" CoTEndsCorrect={cot_end_correct_total/n:.3f}"
                )

    out = {
        "AnswerEM": answer_em_total / max(n, 1),
        "CoTEndsCorrect": cot_end_correct_total / max(n, 1),
        "OracleEM": (cc_aa + cc_aw + cw_aa) / max(n, 1),  # right by either head
        "CoTcorrect_AnsCorrect": cc_aa / max(n, 1),
        "CoTcorrect_AnsWrong": cc_aw / max(n, 1),
        "CoTwrong_AnsCorrect": cw_aa / max(n, 1),
        "CoTwrong_AnsWrong": cw_aw / max(n, 1),
        "n": n,
    }
    for k, v in metrics_acc.items():
        out[k] = v / max(n, 1)
    # Convenience aliases for x0_mode=oracle_z1: in that mode AnswerEM is
    # the AE answer-decode oracle EM, and ESeq is the CoT-decode oracle EM
    # (full-sequence match). Surface them under explicit names so dual-head
    # AE eval reports are unambiguous.
    if x0_mode == "oracle_z1":
        out["oracle_ans_em"] = out["AnswerEM"]
        out["oracle_cot_em"] = out.get("ESeq", 0.0)
        out["oracle_cot_step_acc"] = out.get("StepAcc", 0.0)
    return out, sample_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("<CHECKPOINT_ROOT>/data/gsm8k_formulas"),
    )
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--K", type=int, default=0,
                        help="Number of token-emission steps (0 = use M for variant A, "
                             "predicted/cot_max_len for variant B)")
    parser.add_argument("--steps_per_rollout", type=int, default=0,
                        help="Override the Heun integrator step count "
                             "(0 = use the checkpoint's model default).")
    parser.add_argument("--depth_sweep", type=int, nargs="*", default=[])
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--x0_mode",
        choices=["predicted", "gaussian", "zero", "oracle_z1"],
        default="predicted",
        help="Test-time ablation on x0_hat. 'predicted' is the trained model. "
             "'gaussian' replaces x0_hat with sphere-norm random noise. "
             "'oracle_z1' substitutes the gold target-encoder z_1.",
    )
    parser.add_argument("--x0_seed", type=int, default=0)
    parser.add_argument("--sample_dump_limit", type=int, default=0,
                        help="Cap on samples.json size; 0 = dump all.")
    parser.add_argument("--answer_normalizer",
                        type=str,
                        default="numeric",
                        choices=["numeric", "smiles", "set_smiles", "classify"],
                        help="Per-task answer-string normalizer. Default "
                             "'numeric' matches gsm8k. Use 'smiles' for "
                             "mech/molpuzzle, 'set_smiles' for retro, "
                             "'classify' for prontoqa/proofwriter.")
    args = parser.parse_args()
    # Install the per-task normalizer before any evaluation reads it.
    set_normalizer(args.answer_normalizer)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.out_dir / "eval.log"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger("gsm8k_flow_eval")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["cfg"]
    # Drop any obsolete fields saved by older training runs (e.g. w_ans,
    # train_answer_at_t1). The model code may have evolved since the ckpt
    # was saved; a config dataclass cannot accept unknown kwargs.
    valid_keys = {f.name for f in fields(GSM8KFlowQwen3Config)}
    dropped = [k for k in cfg_dict if k not in valid_keys]
    if dropped:
        log.info(f"dropping obsolete cfg fields: {dropped}")
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    cfg = GSM8KFlowQwen3Config(**cfg_dict)
    model = GSM8KFlowQwen3Model(cfg)
    log.info(
        f"qwen3 checkpoint: variant={cfg.variant} M={cfg.M} latent_dim={cfg.latent_dim} lora_rank={cfg.lora_rank}"
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    log.info(f"missing keys: {len(missing)}, unexpected: {len(unexpected)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    ds = GSM8KFormulasDataset(
        args.data_dir / f"{args.split}.json",
        model.tokenizer,
        max_q_len=192,
        cot_max_len=cfg.cot_max_len,
        ans_max_len=cfg.ans_max_len,
    )
    if args.max_examples > 0:
        ds.examples = ds.examples[: args.max_examples]
    log.info(f"split={args.split} n={len(ds)}")

    collate = make_collator(
        pad_id=cfg.pad_token_id,
        bos_id=cfg.bos_token_id,
        cot_max_len=cfg.cot_max_len,
        ans_max_len=cfg.ans_max_len,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate,
    )

    K = args.K if args.K > 0 else None
    log.info(f"x0_mode={args.x0_mode} x0_seed={args.x0_seed}")
    metrics, samples = evaluate(
        model, loader, device, ds.examples, K, log,
        x0_mode=args.x0_mode, x0_seed=args.x0_seed,
        sample_dump_limit=args.sample_dump_limit,
        steps_per_rollout=args.steps_per_rollout,
    )
    log.info(f"final metrics: {json.dumps(metrics, indent=2)}")

    with (args.out_dir / "metrics.json").open("w") as fh:
        json.dump(metrics, fh, indent=2)
    with (args.out_dir / "samples.json").open("w") as fh:
        json.dump(samples, fh, indent=2)

    if args.depth_sweep:
        sweep = {}
        for k in args.depth_sweep:
            log.info(f"depth sweep K={k}")
            m, _ = evaluate(
                model, loader, device, ds.examples, k, log,
                x0_mode=args.x0_mode, x0_seed=args.x0_seed,
                sample_dump_limit=0,
                steps_per_rollout=args.steps_per_rollout,
            )
            sweep[str(k)] = m
        with (args.out_dir / "depth_curve.json").open("w") as fh:
            json.dump(sweep, fh, indent=2)


if __name__ == "__main__":
    main()
