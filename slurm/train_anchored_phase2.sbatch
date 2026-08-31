#!/bin/bash
#SBATCH --job-name=anchored-phase2
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=40:00:00
#SBATCH --output=<REPO_ROOT>/train_logs/anchored_phase2-%j.out

# Stage 3 of the anchored pipeline: train the DiT denoiser to predict
# the time-varying anchored clean target z^a(t) along the piecewise-linear
# path through anchor latents. Anchor LM and suffix decoder are frozen
# throughout phase-2.
#
# Headline configuration: mean-pool anchors, DiT-XL, n_lat=1,
# 200k steps batch 32 at lr 5e-5. This public 385K-example recipe produced
# 32.0% EM; the paper's 33.0% row used a separate 1.385M-example setup.
#
# Depends on slurm/train_anchor_lm.sbatch and slurm/train_suffix_decoder.sbatch.

source "<REPO_ROOT>/slurm/_common.sh"
mkdir -p train_logs

DATA_DIR="<REPO_ROOT>/data/gsm8k_formulas"
ANCHOR_LM="<CHECKPOINT_ROOT>/outputs/anchor_lm/last_checkpoint"
SUFFIX_DEC="<CHECKPOINT_ROOT>/outputs/suffix_decoder_mean/last_checkpoint.pt"

if [ ! -d "$ANCHOR_LM" ] || [ ! -f "$SUFFIX_DEC" ]; then
    echo "ERROR: missing prerequisites:"
    echo "  ANCHOR_LM=$ANCHOR_LM ($([ -d "$ANCHOR_LM" ] && echo OK || echo MISSING))"
    echo "  SUFFIX_DEC=$SUFFIX_DEC ($([ -f "$SUFFIX_DEC" ] && echo OK || echo MISSING))"
    exit 1
fi

OUT="<CHECKPOINT_ROOT>/outputs/anchored_phase2_v1_mean"
mkdir -p "$OUT"

python scripts/train_anchored_phase2.py \
    --v1_lm_ckpt "$ANCHOR_LM" \
    --v1_decoder_ckpt "$SUFFIX_DEC" \
    --v1_pool_mode mean \
    --decoder_base_model "Qwen/Qwen3-0.6B" \
    --q_encoder_base_model "Qwen/Qwen3-0.6B" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUT" \
    --M 48 --max_q_len 192 --cot_max_len 80 --ans_max_len 24 \
    --dit_arch ditXL --n_latent_tokens 1 \
    --batch_size 32 --lr 5e-5 --min_lr 1e-6 \
    --weight_decay 0.02 --warmup_steps 2000 --max_steps 200000 \
    --max_grad_norm 1.0 --num_workers 8 --seed 42 \
    --log_every 100 --val_em_every 5000 --val_em_n 200 \
    --save_every 5000 --gradient_checkpointing true \
    --answer_normalizer numeric \
    2>&1 | tee -a "$OUT/train.log"

echo "DONE: anchored_phase2_v1_mean"
