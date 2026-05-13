#!/usr/bin/env bash
# Experiment 7: On-Policy Distillation (OPD)
# DDO-RM with iterative on-policy candidate regeneration
set -euo pipefail

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET="ultrafeedback_binarized"
OUT="results/exp7_opd"

for SEED in 42 13 3407; do
    echo "========================================"
    echo "  Exp 7 | Seed $SEED"
    echo "========================================"

    SFT_PATH="results/exp1_same_rm/sft_seed${SEED}/final"
    RM_PATH="results/exp1_same_rm/rm_seed${SEED}/final"

    # Offline DDO-RM (baseline)
    python -m ddorm train ddorm \
        --model_name "$SFT_PATH" \
        --reward_model_path "$RM_PATH" \
        --dataset "$DATASET" \
        --train_split "train_prefs" \
        --seed $SEED \
        --output_dir "$OUT/offline_seed${SEED}" \
        --lambda_temp 1.0 \
        --num_candidates 16 \
        --learning_rate 5e-6 \
        --num_epochs 3

    # OPD: on-policy distillation with refresh
    python -m ddorm train opd \
        --model_name "$SFT_PATH" \
        --reward_model_path "$RM_PATH" \
        --dataset "$DATASET" \
        --train_split "train_prefs" \
        --seed $SEED \
        --output_dir "$OUT/opd_seed${SEED}" \
        --lambda_temp 1.0 \
        --num_candidates 16 \
        --refresh_interval 500 \
        --learning_rate 5e-6 \
        --num_epochs 3

    # GRPO baseline
    python -m ddorm train grpo \
        --model_name "$SFT_PATH" \
        --reward_model_path "$RM_PATH" \
        --ref_model_path "$SFT_PATH" \
        --dataset "$DATASET" \
        --train_split "train_prefs" \
        --seed $SEED \
        --output_dir "$OUT/grpo_seed${SEED}" \
        --group_size 16 \
        --learning_rate 5e-6 \
        --num_epochs 3

    # Evaluate all
    for METHOD in offline opd grpo; do
        python -m ddorm eval \
            --model_path "$OUT/${METHOD}_seed${SEED}/final" \
            --dataset "$DATASET" \
            --eval_split "test_prefs" \
            --ref_model_path "$SFT_PATH" \
            --reward_model_path "$RM_PATH" \
            --output_dir "$OUT/${METHOD}_seed${SEED}/eval"
    done
done

echo "Experiment 7 complete. Results in $OUT/"
