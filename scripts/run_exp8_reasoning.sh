#!/usr/bin/env bash
# Experiment 8: Reasoning Domain — GSM8K
# Verifiable rewards (exact-match) instead of learned RM
set -euo pipefail

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET="reasoning_gsm8k"
OUT="results/exp8_reasoning"

for SEED in 42 13; do
    echo "========================================"
    echo "  Exp 8 | Seed $SEED"
    echo "========================================"

    SFT_PATH="results/exp1_same_rm/sft_seed${SEED}/final"

    # DDO-RM with exact-match reward
    for K in 8 16 32; do
        python -m ddorm train ddorm \
            --model_name "$SFT_PATH" \
            --dataset "$DATASET" \
            --train_split "train" \
            --seed $SEED \
            --output_dir "$OUT/ddorm_K${K}_seed${SEED}" \
            --lambda_temp 0.5 \
            --num_candidates $K \
            --candidate_generator temperature \
            --gen_temperature 0.7 \
            --reward_type exact_match \
            --learning_rate 5e-6 \
            --num_epochs 3
    done

    # GRPO baseline
    python -m ddorm train grpo \
        --model_name "$SFT_PATH" \
        --dataset "$DATASET" \
        --train_split "train" \
        --seed $SEED \
        --output_dir "$OUT/grpo_seed${SEED}" \
        --group_size 16 \
        --reward_type exact_match \
        --learning_rate 5e-6 \
        --num_epochs 3

    # Evaluate
    for DIR in ddorm_K8_seed${SEED} ddorm_K16_seed${SEED} ddorm_K32_seed${SEED} grpo_seed${SEED}; do
        python -m ddorm eval \
            --model_path "$OUT/${DIR}/final" \
            --dataset "$DATASET" \
            --eval_split "test" \
            --reward_type exact_match \
            --output_dir "$OUT/${DIR}/eval"
    done
done

echo "Experiment 8 complete. Results in $OUT/"
