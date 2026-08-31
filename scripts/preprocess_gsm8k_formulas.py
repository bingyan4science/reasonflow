"""Parse the da03/implicit_chain_of_thought GSM8K formulas dataset
into the JSON shape the GSM8K flow trainer expects, and compute the
token-length p90 used for Variant A's M.

Input files are ``train.txt``, ``valid.txt``, and ``test.txt``. Their format
is one example per line:
    question||<<f1>> <<f2>> ... <<fK>> #### answer

Output: data/gsm8k_formulas/{train,val,test}.json with entries
    {
        "question": str,
        "reasoning_text": str,         # space-separated <<...>> chunks
        "answer": str,
        "n_chars_reasoning": int,
        "n_tokens_reasoning": int,     # Flan-T5 BPE
        "n_steps": int,                # number of <<...>> chunks
    }

Plus data/gsm8k_formulas/stats.json with the token-length distribution
and the p90, p95, p99, max values used to pick M for Variant A.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
from transformers import AutoTokenizer

LINE_RE = re.compile(r"^(?P<q>.*?)\|\|(?P<rest>.*)$")
ANS_RE = re.compile(r"^(?P<r>.*?)#### (?P<a>.*)$")
STEP_RE = re.compile(r"<<[^>]*>>")


def parse_line(line: str) -> Dict[str, str] | None:
    line = line.rstrip("\n")
    m = LINE_RE.match(line)
    if not m:
        return None
    q = m.group("q").strip()
    rest = m.group("rest")
    m2 = ANS_RE.match(rest)
    if not m2:
        return None
    r = m2.group("r").strip()
    a = m2.group("a").strip()
    if not q or not a:
        return None
    # Keep upstream `<<...>>` markers verbatim; they are part of the GSM8K
    # calculator-annotation convention. The Qwen3 tokenizer encodes `<<` and
    # `>>` as single dedicated tokens (ids 2442 / 2452 verified), so no
    # waste. We only stripped them for the Flan-T5 backbone, which mapped
    # `<` to <unk>. Do not modify the literature's data format.
    return {"question": q, "reasoning_text": r, "answer": a}


def parse_file(path: Path) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as fh:
        first_line = fh.readline()
        if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(
                f"{path} is a Git LFS pointer, not dataset content. "
                "Hydrate the upstream files with git lfs pull first."
            )
        fh.seek(0)
        for line in fh:
            entry = parse_line(line)
            if entry is None:
                skipped += 1
                continue
            out.append(entry)
    print(f"  {path.name}: parsed {len(out)} (skipped {skipped})")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("<REPO_ROOT>/data/gsm8k_formulas"),
        help="Directory containing upstream train.txt, valid.txt, and test.txt.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(
            "<CHECKPOINT_ROOT>/data/gsm8k_formulas"
        ),
        help="Where to write parsed JSON (and a copy of stats.json).",
    )
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--stats_suffix",
        type=str,
        default="",
        help="If set, write stats to stats_<suffix>.json instead of stats.json so multiple tokenizers can coexist.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    splits = {"train": "train.txt", "val": "valid.txt", "test": "test.txt"}
    train_token_lengths: List[int] = []

    for split_name, raw_name in splits.items():
        raw_path = args.raw_dir / raw_name
        entries = parse_file(raw_path)

        out: List[Dict] = []
        for e in entries:
            steps = STEP_RE.findall(e["reasoning_text"])
            ids = tok.encode(e["reasoning_text"], add_special_tokens=False)
            out.append(
                {
                    "question": e["question"],
                    "reasoning_text": e["reasoning_text"],
                    "answer": e["answer"],
                    "n_chars_reasoning": len(e["reasoning_text"]),
                    "n_tokens_reasoning": len(ids),
                    "n_steps": len(steps),
                }
            )
            if split_name == "train":
                train_token_lengths.append(len(ids))

        out_path = args.out_dir / f"{split_name}.json"
        with out_path.open("w") as fh:
            json.dump(out, fh)
        print(f"  wrote {out_path}: {len(out)} entries")

    arr = np.array(train_token_lengths)
    stats = {
        "tokenizer": args.tokenizer,
        "split": "train",
        "n": int(arr.size),
        "min": int(arr.min()),
        "p50": int(np.percentile(arr, 50)),
        "p75": int(np.percentile(arr, 75)),
        "p90": int(np.ceil(np.percentile(arr, 90))),
        "p95": int(np.ceil(np.percentile(arr, 95))),
        "p99": int(np.ceil(np.percentile(arr, 99))),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }
    stats_name = "stats.json" if not args.stats_suffix else f"stats_{args.stats_suffix}.json"
    with (args.out_dir / stats_name).open("w") as fh:
        json.dump(stats, fh, indent=2)
    print("\ntoken-length stats (train, tokens):")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\n  -> Variant A M (p90, ceil): {stats['p90']}")


if __name__ == "__main__":
    main()
