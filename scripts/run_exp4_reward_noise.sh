#!/usr/bin/env bash
# Experiment 4: Reward Noise Sweep
# Inject calibrated noise into RM, measure degradation
# Tests Theorem 1 prediction: error ≤ λlog(1/ρ) + 2ε_RM
set -euo pipefail

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET="ultrafeedback_binarized"
OUT="results/exp4_reward_noise"
K=16

for SEED in 42 13; do
    echo "========================================"
    echo "  Exp 4 | Seed $SEED"
    echo "========================================"

    SFT_PATH="results/exp1_same_rm/sft_seed${SEED}/final"
    RM_PATH="results/exp1_same_rm/rm_seed${SEED}/final"

    # Sweep noise levels and types
    for NOISE_TYPE in gaussian uniform rank_flip; do
        for NOISE_SCALE in 0.0 0.1 0.25 0.5 1.0 2.0; do
            TAG="${NOISE_TYPE}_s${NOISE_SCALE}"
            echo "--- $TAG, Seed=$SEED ---"

            python -m ddorm reward_noise \
                --model_path "$SFT_PATH" \
                --reward_model_path "$RM_PATH" \
                --dataset "$DATASET" \
                --eval_split "test_prefs" \
                --num_candidates $K \
                --noise_type "$NOISE_TYPE" \
                --noise_scale $NOISE_SCALE \
                --seed $SEED \
                --output_dir "$OUT/${TAG}_seed${SEED}"
        done
    done
done

echo "Experiment 4 complete. Results in $OUT/"
