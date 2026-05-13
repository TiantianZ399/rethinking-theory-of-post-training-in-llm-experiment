#!/usr/bin/env bash
# Experiment 3: Candidate Generator Comparison
# Temperature vs Nucleus vs Beam vs DiverseBeam vs RewardDiverse
set -euo pipefail

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET="ultrafeedback_binarized"
OUT="results/exp3_generators"
K=16

for SEED in 42 13 3407; do
    echo "========================================"
    echo "  Exp 3 | Seed $SEED"
    echo "========================================"

    SFT_PATH="results/exp1_same_rm/sft_seed${SEED}/final"
    RM_PATH="results/exp1_same_rm/rm_seed${SEED}/final"

    # --- Temperature sampling (τ=0.8) ---
    python -m ddorm train ddorm \
        --model_name "$SFT_PATH" \
        --reward_model_path "$RM_PATH" \
        --dataset "$DATASET" \
        --train_split "train_prefs" \
        --seed $SEED \
        --output_dir "$OUT/temperature_seed${SEED}" \
        --lambda_temp 1.0 \
        --num_candidates $K \
        --candidate_generator temperature \
        --gen_temperature 0.8 \
        --learning_rate 5e-6 \
        --num_epochs 3

    # --- Nucleus sampling (p=0.9) ---
    python -m ddorm train ddorm \
        --model_name "$SFT_PATH" \
        --reward_model_path "$RM_PATH" \
        --dataset "$DATASET" \
        --train_split "train_prefs" \
        --seed $SEED \
        --output_dir "$OUT/nucleus_seed${SEED}" \
        --lambda_temp 1.0 \
        --num_candidates $K \
        --candidate_generator nucleus \
        --gen_top_p 0.9 \
        --learning_rate 5e-6 \
        --num_epochs 3

    # --- Diverse beam search ---
    python -m ddorm train ddorm \
        --model_name "$SFT_PATH" \
        --reward_model_path "$RM_PATH" \
        --dataset "$DATASET" \
        --train_split "train_prefs" \
        --seed $SEED \
        --output_dir "$OUT/diverse_beam_seed${SEED}" \
        --lambda_temp 1.0 \
        --num_candidates $K \
        --candidate_generator diverse_beam \
        --diversity_penalty 1.0 \
        --learning_rate 5e-6 \
        --num_epochs 3

    # --- Reward-diverse (submodular) ---
    python -m ddorm train ddorm \
        --model_name "$SFT_PATH" \
        --reward_model_path "$RM_PATH" \
        --dataset "$DATASET" \
        --train_split "train_prefs" \
        --seed $SEED \
        --output_dir "$OUT/reward_diverse_seed${SEED}" \
        --lambda_temp 1.0 \
        --num_candidates $K \
        --candidate_generator reward_diverse \
        --learning_rate 5e-6 \
        --num_epochs 3

    # --- Coverage diagnostics for each ---
    for GEN in temperature nucleus diverse_beam reward_diverse; do
        python -m ddorm coverage \
            --model_path "$SFT_PATH" \
            --reward_model_path "$RM_PATH" \
            --dataset "$DATASET" \
            --eval_split "test_prefs" \
            --num_candidates $K \
            --candidate_generator $GEN \
            --output_dir "$OUT/${GEN}_seed${SEED}/coverage"

        python -m ddorm eval \
            --model_path "$OUT/${GEN}_seed${SEED}/final" \
            --dataset "$DATASET" \
            --eval_split "test_prefs" \
            --ref_model_path "$SFT_PATH" \
            --reward_model_path "$RM_PATH" \
            --output_dir "$OUT/${GEN}_seed${SEED}/eval"
    done
done

echo "Experiment 3 complete. Results in $OUT/"
