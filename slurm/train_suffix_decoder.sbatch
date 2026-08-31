#!/bin/bash
#SBATCH --job-name=suffix-decoder
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=20:00:00
#SBATCH --output=<REPO_ROOT>/train_logs/suffix_decoder-%j.out

# Stage 2 of the anchored pipeline: train a Qwen3-0.6B suffix decoder
# on top of the frozen anchor LM. The decoder maps (q, z^a_k, k) ->
# cot[k:] + ans for k sampled uniformly in {0..K}; at phase-2 inference
# we call it with k=K to read out the answer from the integrated z^a_K.
#
# Default uses mean-pool anchors (the headline configuration).
# Depends on slurm/train_anchor_lm.sbatch having finished.

source "<REPO_ROOT>/slurm/_common.sh"
mkdir -p train_logs

DATA_DIR="<REPO_ROOT>/data/gsm8k_formulas"
ANCHOR_LM="<CHECKPOINT_ROOT>/outputs/anchor_lm/last_checkpoint"
POOL="mean"   # mean | last_token | mean_cot_only

if [ ! -d "$ANCHOR_LM" ]; then
    echo "ERROR: anchor LM ckpt missing: $ANCHOR_LM"; exit 1
fi

OUT="<CHECKPOINT_ROOT>/outputs/suffix_decoder_${POOL}"
mkdir -p "$OUT"

python scripts/train_suffix_decoder.py \
    --encoder_ckpt "$ANCHOR_LM" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUT" \
    --decoder_model_name "Qwen/Qwen3-0.6B" \
    --k_max 80 \
    --v1_pool_mode "$POOL" \
    --max_q_len 192 --cot_max_len 80 --ans_max_len 24 \
    --batch_size 32 --lr 2e-5 --warmup_steps 500 --max_steps 24000 \
    --val_every 500 --val_em_every 2000 --val_em_max_examples 200 \
    --save_every 4000 --num_workers 8 \
    --gradient_checkpointing true --sphere_norm true \
    --log_every 100 \
    2>&1 | tee -a "$OUT/train.log"

echo "DONE: suffix_decoder_${POOL}"
