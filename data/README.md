# Data

Dataset files are not redistributed in this repository. The released recipe
uses the GSM8K-Aug formulas-only data from Deng et al. (2023,
[arXiv:2311.01460](https://arxiv.org/abs/2311.01460)), in which each reasoning
chain is represented by calculator equations followed by a numeric answer.

Obtain the raw GSM8K files from the upstream
[`da03/implicit_chain_of_thought`](https://github.com/da03/implicit_chain_of_thought/tree/main/data/gsm8k)
repository and comply with its terms. The original GSM8K dataset is available
from [`openai/grade-school-math`](https://github.com/openai/grade-school-math)
under its own license.

## Layout

```
data/gsm8k_formulas/
    train.json         # generated training records
    val.json           # generated validation records
    test.json          # generated test records
    stats_qwen3.json   # generated token-length statistics
```

## Obtain and verify the upstream files

The upstream repository tracks `data/gsm8k/train.txt` with Git LFS. Clone it
and hydrate all three inputs before preprocessing:

```bash
git clone https://github.com/da03/implicit_chain_of_thought.git
git -C implicit_chain_of_thought lfs pull \
    --include='data/gsm8k/train.txt,data/gsm8k/valid.txt,data/gsm8k/test.txt'
```

Verify that `train.txt` contains examples rather than the three-line Git LFS
pointer. The upstream LFS object may be temporarily unavailable when its
bandwidth quota is exhausted; in that case, obtain the exact file from the
upstream authors before proceeding. The repository's historical tarball has a
different training count and is not an exact substitute for this recipe.

## Generate the JSON files

Run the preprocessor directly on the upstream `data/gsm8k/` directory:

```bash
python scripts/preprocess_gsm8k_formulas.py \
    --raw_dir /path/to/implicit_chain_of_thought/data/gsm8k \
    --out_dir data/gsm8k_formulas \
    --tokenizer Qwen/Qwen3-0.6B \
    --stats_suffix qwen3
```

The SLURM templates read the generated JSON files from
`<REPO_ROOT>/data/gsm8k_formulas/`. The upstream files contain 385,620 train,
500 validation, and 1,319 test records. The anchored loaders discard records
with empty required fields, leaving 385,531 / 494 / 1,301. This public corpus
corresponds to the 32.0% recipe in the README; it does not reproduce the
paper's separate 1.385M-example, 33.0% setup.

## Record schema

```json
{
  "question": "Out of 600 employees, 30% got promoted while 10% received bonus. How many got neither?",
  "reasoning_text": "<<600*30/100=180>> <<600*10/100=60>> <<180+60=240>> <<600-240=360>>",
  "answer": "360",
  "n_chars_reasoning": 67,
  "n_tokens_reasoning": 56,
  "n_steps": 4
}
```

## Provenance

The preprocessing script parses the raw formulas-only files distributed by
Deng et al.; it does not synthesize augmented examples from a Hugging Face
GSM8K download. The underlying questions originate from GSM8K
([Cobbe et al. 2021](https://arxiv.org/abs/2110.14168)). Dataset files remain
subject to their upstream terms and are not covered by this repository's MIT
license.
