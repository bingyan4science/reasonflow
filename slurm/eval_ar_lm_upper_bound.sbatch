#!/bin/bash
#SBATCH --job-name=ar-lm-upper-bound
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=<REPO_ROOT>/train_logs/ar_lm_upper_bound-%j.out

# AR LM upper bound: greedy single-seed answer EM of the finetuned
# Qwen3-0.6B autoregressive LM (= the same backbone used as the
# anchor encoder). Provides the ceiling reference for the flow-matching
# block in tab:main_results. Reproduces the GSM8K AR LM = 69.4% number.

source "<REPO_ROOT>/slurm/_common.sh"
mkdir -p train_logs

DATA_DIR="<REPO_ROOT>/data/gsm8k_formulas"
CKPT="<CHECKPOINT_ROOT>/outputs/anchor_lm/last_checkpoint"

if [ ! -d "$CKPT" ]; then
    echo "ERROR: anchor LM ckpt missing at $CKPT"; exit 1
fi

OUT_DIR="<CHECKPOINT_ROOT>/outputs/ar_lm_upper_bound"
mkdir -p "$OUT_DIR"

python scripts/eval_ar_lm_upper_bound.py \
    --ckpt "$CKPT" \
    --data_dir "$DATA_DIR" \
    --split val \
    --max_q_len 192 \
    --cot_max_len 80 \
    --ans_max_len 24 \
    --answer_normalizer numeric \
    --out_path "$OUT_DIR/gsm8k.json" \
    2>&1 | tee "$OUT_DIR/gsm8k.log"

echo "DONE: ar_lm_upper_bound"
