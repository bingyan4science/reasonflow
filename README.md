# ReasonFlow: Flow Matching that Thinks Step by Step

ReasonFlow uses chain-of-thought supervision to shape continuous latent trajectories in flow matching. The paper studies arithmetic, logical reasoning, and chemistry. This repository provides the GSM8K reproduction.

[Paper](https://aclanthology.org/2026.findings-emnlp.2268/) · [Guide: flow matching for reasoning](https://bingyan.me/research/flow-matching-reasoning/) · [PDF](https://bingyan.me/reasonflow.pdf) · [BibTeX](https://bingyan.me/research/reasonflow.bib)

The GSM8K-Aug source contains 385,620 training examples, and our loaders retain 385,531 after filtering empty required fields.

## Result on GSM8K

| Method                       | Answer EM |
| ---                          | ---       |
| AR LM with CoT (upper bound) | 69.4%     |
| No-CoT flow matching         | 1.0%      |
| **ReasonFlow** (mean pool)   | **32.0%** |


## Install

```bash
git clone https://github.com/bingyan4science/reasonflow.git
cd reasonflow
conda env create -f environment.yml
conda activate reasonflow
pip install -e .
```

`pip install -e .` installs both the source-layout ReasonFlow package and its runtime dependencies. `requirements.txt` is a dependency-only reference. `environment.yml` targets Python 3.11 and includes the PyTorch CUDA 12.4 wheel index; resolve the appropriate PyTorch build for your hardware.

Before running anything, replace these three placeholders in [`slurm/_common.sh`](slurm/_common.sh) and the `slurm/*.sbatch` scripts with paths for your environment:

| Placeholder | Meaning | Example |
|---|---|---|
| `<REPO_ROOT>` | Absolute path to the repo on your machine | `/path/to/ReasonFlow` |
| `<CHECKPOINT_ROOT>` | Where training output checkpoints will be written | `/path/to/checkpoints` |
| `<CONDA_ROOT>` | Root of your conda installation | `${HOME}/miniconda3` |

Training data are not redistributed. Follow [`data/README.md`](data/README.md) to obtain the upstream GSM8K-Aug files and generate `data/gsm8k_formulas/` locally.

The tracked `train_logs/` directory exists so SLURM can open its output path before each job starts; generated logs remain ignored.


## Reproduce the GSM8K

The anchored ReasonFlow recipe has three training stages plus one inference step. Each stage has a SLURM template under `slurm/`.

1. **Stage 1 -- finetune the anchor LM.** Trains Qwen3-0.6B as an autoregressive LM on `[q, cot, ans]`. The LM's middle-layer hidden states at CoT-token-prefix positions become the anchor latents at phase 2.

   ```bash
   sbatch slurm/train_anchor_lm.sbatch
   # Output: <CHECKPOINT_ROOT>/outputs/anchor_lm/last_checkpoint
   ```

2. **Stage 2 -- train the suffix decoder.** A separate Qwen3-0.6B is trained on top of the frozen anchor LM to map `(q, z^a_k, k) -> cot[k:] + ans`. At phase-2 inference we call it with `k=K` to read the answer from the integrated `z^a_K`.

   ```bash
   sbatch slurm/train_suffix_decoder.sbatch
   # Output: <CHECKPOINT_ROOT>/outputs/suffix_decoder_mean/last_checkpoint.pt
   ```

3. **Stage 3 -- anchored phase-2 flow training.** Trains the DiT denoiser to predict the time-varying anchored clean target `z^a(t)` along the piecewise-linear path through the K+1 anchor latents. Anchor LM and suffix decoder are frozen.

   ```bash
   sbatch slurm/train_anchored_phase2.sbatch
   # Output: <CHECKPOINT_ROOT>/outputs/anchored_phase2_v1_mean/best_val_em.pt
   ```

4. **Eval.** Greedy single-seed answer EM at `n_int=64` Heun integration steps:

   ```bash
   python scripts/eval_anchored.py \
       --checkpoint <CHECKPOINT_ROOT>/outputs/anchored_phase2_v1_mean/best_val_em.pt \
       --data_dir <REPO_ROOT>/data/gsm8k_formulas \
       --answer_normalizer numeric \
       --output_path eval_results.json \
       --split val --val_em_n 200 --n_int_steps 64
   ```

## Reproduce the baselines

**No-CoT flow matching** (the same `scripts/train_ae.py` is used for both stages, with `--no_cot_ae true`):

```bash
STAGE=ae sbatch slurm/train_strict_nocot.sbatch       # Stage 1: AE on (q, ans)
STAGE=phase2 sbatch slurm/train_strict_nocot.sbatch    # Stage 2: DiT phase-2
python scripts/eval_nocot.py --checkpoint <CHECKPOINT_ROOT>/outputs/strict_nocot_phase2/best_model_em.pt \
    --data_dir <REPO_ROOT>/data/gsm8k_formulas \
    --split val --max_examples 200 \
    --steps_per_rollout 64 \
    --out_dir outputs/strict_nocot_eval
```

The same training code supports the **strict** baseline (AE never sees cot) and the **weak** ablation (AE pretrained on `(q, cot, ans)` but phase-2 has `w_tok=0`); flip the `--no_cot_ae` flag to switch between them. The paper's weak-baseline vote@13 number requires a multi-seed majority-vote wrapper, which is not included in this source snapshot.

**AR LM upper bound** (the same Qwen3-0.6B from Stage 1 of the anchored pipeline, evaluated as a standalone autoregressive LM):

```bash
sbatch slurm/eval_ar_lm_upper_bound.sbatch
# Output: <CHECKPOINT_ROOT>/outputs/ar_lm_upper_bound/gsm8k.json
```

## Repository layout

```
src/reasonflow/             # importable package
  models/
    gsm8k_flow_qwen3.py     # Qwen3 backbone + AE + answer head
    dit_canonical.py        # DiT denoiser
    anchored/               # anchor extraction + suffix decoder
  data/gsm_dataset.py
  utils/answer_normalizers.py

scripts/                    # runnable training + eval entry points
slurm/                      # SLURM templates (one per stage)
data/README.md              # how to obtain and preprocess GSM8K-Aug
```

## Paper

**ReasonFlow: Flow Matching that Thinks Step by Step**

Bing Yan, Ge Sun, Juan J. de Pablo, Stefano Martiniani, and Kyunghyun Cho.
Findings of EMNLP 2026, pages 48554–48571. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.2268/) · [DOI: 10.18653/2026.findings-emnlp.2268](https://doi.org/10.18653/2026.findings-emnlp.2268) · [BibTeX](https://bingyan.me/research/reasonflow.bib)

## License

Original ReasonFlow code is released under the [MIT License](LICENSE). [`src/reasonflow/models/dit_canonical.py`](src/reasonflow/models/dit_canonical.py) is adapted from Meta's DiT and remains under [CC BY-NC 4.0](https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt), including its noncommercial restriction. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
