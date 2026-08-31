#!/bin/bash
#SBATCH --job-name=anchor-lm
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=30:00:00
#SBATCH --output=<REPO_ROOT>/train_logs/anchor_lm-%j.out

# Stage 1 of the anchored pipeline: finetune Qwen3-0.6B as an
# autoregressive LM on (q, cot, ans). Hidden states from this LM
# at cot-step boundary tokens become the anchor latents z^a_k for
# the phase-2 anchored flow training.
#
# Output: <CHECKPOINT_ROOT>/outputs/anchor_lm/last_checkpoint.

source "<REPO_ROOT>/slurm/_common.sh"
mkdir -p train_logs

DATA_DIR="<REPO_ROOT>/data/gsm8k_formulas"
OUT="<CHECKPOINT_ROOT>/outputs/anchor_lm"
mkdir -p "$OUT"

python scripts/train_anchor_lm.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUT" \
    --model_name "Qwen/Qwen3-0.6B" \
    --M 48 \
    --max_q_len 192 --cot_max_len 80 --ans_max_len 24 \
    --batch_size 64 --lr 2e-5 --warmup_steps 1000 --max_steps 24000 \
    --val_every 1000 --val_em_every 2000 --val_em_n 200 \
    --save_every 4000 --num_workers 8 \
    --gradient_checkpointing true --log_every 100 \
    2>&1 | tee -a "$OUT/train.log"

echo "DONE: anchor_lm"
