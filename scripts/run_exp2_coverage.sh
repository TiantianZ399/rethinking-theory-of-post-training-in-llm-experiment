#!/usr/bin/env bash
# Experiment 2: Coverage Curve — performance vs ρ(C_K) for varying K
# Tests the core theory: gap = λ log(1/ρ)
set -euo pipefail

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET="ultrafeedback_binarized"
OUT="results/exp2_coverage"

for SEED in 42 13 3407; do
    echo "========================================"
    echo "  Exp 2 | Seed $SEED"
    echo "========================================"

    SFT_PATH="results/exp1_same_rm/sft_seed${SEED}/final"
    RM_PATH="results/exp1_same_rm/rm_seed${SEED}/final"

    # Sweep K ∈ {2, 4, 8, 16, 32, 64}
    for K in 2 4 8 16 32 64; do
        echo "--- K=$K, Seed=$SEED ---"

        # 1. Train DDO-RM at this K
        python -m ddorm train ddorm \
            --model_name "$SFT_PATH" \
            --reward_model_path "$RM_PATH" \
            --dataset "$DATASET" \
            --train_split "train_prefs" \
            --seed $SEED \
            --output_dir "$OUT/ddorm_K${K}_seed${SEED}" \
            --lambda_temp 1.0 \
            --num_candidates $K \
            --candidate_generator temperature \
            --gen_temperature 0.8 \
            --learning_rate 5e-6 \
            --num_epochs 3

        # 2. Measure coverage
        python -m ddorm coverage \
            --model_path "$SFT_PATH" \
            --reward_model_path "$RM_PATH" \
            --dataset "$DATASET" \
            --eval_split "test_prefs" \
            --num_candidates $K \
            --candidate_generator temperature \
            --gen_temperature 0.8 \
            --output_dir "$OUT/coverage_K${K}_seed${SEED}"

        # 3. Evaluate
        python -m ddorm eval \
            --model_path "$OUT/ddorm_K${K}_seed${SEED}/final" \
            --dataset "$DATASET" \
            --eval_split "test_prefs" \
            --ref_model_path "$SFT_PATH" \
            --reward_model_path "$RM_PATH" \
            --output_dir "$OUT/ddorm_K${K}_seed${SEED}/eval"
    done
done

echo "Experiment 2 complete. Results in $OUT/"
