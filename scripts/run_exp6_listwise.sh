#!/usr/bin/env bash
# Experiment 6: Listwise Multi-Candidate Evaluation
# Uses UltraFeedback 4-candidate data for listwise metrics
set -euo pipefail

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET="ultrafeedback_4candidate"
OUT="results/exp6_listwise"

for SEED in 42 13 3407; do
    echo "========================================"
    echo "  Exp 6 | Seed $SEED"
    echo "========================================"

    SFT_PATH="results/exp1_same_rm/sft_seed${SEED}/final"
    RM_PATH="results/exp1_same_rm/rm_seed${SEED}/final"

    # DDO-RM with K=4 (matches data)
    python -m ddorm train ddorm \
        --model_name "$SFT_PATH" \
        --reward_model_path "$RM_PATH" \
        --dataset "$DATASET" \
        --train_split "train" \
        --seed $SEED \
        --output_dir "$OUT/ddorm_seed${SEED}" \
        --lambda_temp 1.0 \
        --num_candidates 4 \
        --learning_rate 5e-6 \
        --num_epochs 3

    # DPO on best-worst pair from 4-candidate data
    python -m ddorm train dpo \
        --model_name "$SFT_PATH" \
        --ref_model_path "$SFT_PATH" \
        --dataset "$DATASET" \
        --train_split "train" \
        --seed $SEED \
        --output_dir "$OUT/dpo_seed${SEED}" \
        --beta 0.1 \
        --learning_rate 5e-6 \
        --num_epochs 3

    # Evaluate
    for METHOD in ddorm dpo; do
        python -m ddorm eval \
            --model_path "$OUT/${METHOD}_seed${SEED}/final" \
            --dataset "$DATASET" \
            --eval_split "test" \
            --ref_model_path "$SFT_PATH" \
            --reward_model_path "$RM_PATH" \
            --output_dir "$OUT/${METHOD}_seed${SEED}/eval"
    done
done

echo "Experiment 6 complete. Results in $OUT/"
