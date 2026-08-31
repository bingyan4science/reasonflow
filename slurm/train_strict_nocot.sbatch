#!/bin/bash
#SBATCH --job-name=strict-nocot
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=40:00:00
#SBATCH --output=<REPO_ROOT>/train_logs/strict_nocot-%j.out

# Strict no-CoT flow-matching baseline (the no-CoT row of the paper's
# main results table for GSM8K). Two stages:
#
#   Stage 1: AE pretrain on (q, ans) only, never sees the cot.
#   Stage 2: phase-2 trains a DiT to predict the resulting z_1 from
#            Gaussian noise; w_tok=0 (no cot supervision).
#
# Both stages use the same script (scripts/train_ae.py) with --no_cot_ae true.
# Toggle STAGE below to choose which step to run.

source "<REPO_ROOT>/slurm/_common.sh"
mkdir -p train_logs

DATA_DIR="<REPO_ROOT>/data/gsm8k_formulas"
STAGE="${STAGE:-phase2}"   # ae | phase2

if [ "$STAGE" = "ae" ]; then
    OUT="<CHECKPOINT_ROOT>/outputs/strict_nocot_ae"
    EXTRA_ARGS=(
        --max_steps 8000
        --w_ae 1.0 --w_tok 0.0 --w_v 0.0
    )
elif [ "$STAGE" = "phase2" ]; then
    OUT="<CHECKPOINT_ROOT>/outputs/strict_nocot_phase2"
    AE_CKPT="<CHECKPOINT_ROOT>/outputs/strict_nocot_ae/last_checkpoint.pt"
    if [ ! -f "$AE_CKPT" ]; then
        echo "ERROR: strict no-CoT AE missing at $AE_CKPT (run STAGE=ae first)"; exit 1
    fi
    EXTRA_ARGS=(
        --init_ae_checkpoint "$AE_CKPT"
        --freeze_ae true
        --freeze_cot_decoder true
        --pred_target x0
        --steps_per_rollout 64
        --max_steps 200000
        --w_tok 0.0 --w_v 1.0 --w_ae 0.0
        --save_best_val_em true --val_em_n 200
    )
else
    echo "Unknown STAGE: $STAGE (expected 'ae' or 'phase2')"; exit 1
fi
mkdir -p "$OUT"

python scripts/train_ae.py \
    --backbone qwen3 \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUT" \
    --variant pad_to_M \
    --M 48 \
    --cot_max_len 80 --ans_max_len 24 --max_q_len 192 \
    --no_cot_ae true \
    --latent_dim 256 \
    --batch_size 64 --lr 5e-5 --warmup_steps 2000 \
    --max_grad_norm 1.0 \
    --val_every 5000 --save_every 5000 --log_every 200 --num_workers 4 \
    --pool_mode last_token --sphere_norm true \
    --full_finetune true --q_encoder_separate true \
    --q_encoder_pool per_token --cot_decoder_separate false \
    --denoiser_arch dit_canonical \
    --denoiser_hidden 1152 --denoiser_depth 28 --denoiser_heads 16 \
    --n_latent_tokens 1 \
    --answer_normalizer numeric \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$OUT/train.log"

echo "DONE: strict_nocot ($STAGE)"
