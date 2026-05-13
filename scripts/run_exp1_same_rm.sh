#!/usr/bin/env bash
# Experiment 1: Same-RM PPO-RLHF vs DDO-RM
# THE MAIN EXPERIMENT — apples-to-apples comparison
# Same SFT, same RM, same prompts, same compute budget
set -euo pipefail

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET="ultrafeedback_binarized"
OUT="results/exp1_same_rm"

for SEED in 42 13 3407; do
    echo "========================================"
    echo "  Exp 1 | Seed $SEED"
    echo "========================================"

    SFT_DIR="$OUT/sft_seed${SEED}"
    SFT_PATH="$SFT_DIR/final"
    RM_DIR="$OUT/rm_seed${SEED}"
    RM_PATH="$RM_DIR/final"

    # Phase 1: SFT
    python -m ddorm train sft \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --train_split "train_prefs" \
        --seed $SEED \
        --output_dir "$SFT_DIR" \
        --learning_rate 2e-5 \
        --batch_size 8 \
        --num_epochs 3

    # Phase 2: Train shared reward model
    python -m ddorm train reward_model \
        --model_name "$SFT_PATH" \
        --dataset "$DATASET" \
        --train_split "train_prefs" \
        --seed $SEED \
        --output_dir "$RM_DIR" \
        --learning_rate 1e-5 \
        --num_epochs 1

    # Phase 3: PPO-RLHF
    python -m ddorm train ppo \
        --model_name "$SFT_PATH" \
        --reward_model_path "$RM_PATH" \
        --ref_model_path "$SFT_PATH" \
        --dataset "$DATASET" \
        --train_split "train_prefs" \
        --seed $SEED \
        --output_dir "$OUT/ppo_seed${SEED}" \
        --learning_rate 1.5e-6 \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --num_epochs 1 \
        --kl_coeff 0.1 \
        --clip_range 0.2

    # Phase 4: DDO-RM (sweep K)
    for K in 4 8 16; do
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
    done

    # Phase 5: DPO reference
    python -m ddorm train dpo \
        --model_name "$SFT_PATH" \
        --ref_model_path "$SFT_PATH" \
        --dataset "$DATASET" \
        --train_split "train_prefs" \
        --seed $SEED \
        --output_dir "$OUT/dpo_seed${SEED}" \
        --beta 0.1 \
        --learning_rate 5e-6 \
        --num_epochs 3

    # Phase 6: Evaluate all
    for METHOD_DIR in ppo_seed${SEED} ddorm_K4_seed${SEED} ddorm_K8_seed${SEED} ddorm_K16_seed${SEED} dpo_seed${SEED}; do
        python -m ddorm eval \
            --model_path "$OUT/${METHOD_DIR}/final" \
            --dataset "$DATASET" \
            --eval_split "test_prefs" \
            --ref_model_path "$SFT_PATH" \
            --reward_model_path "$RM_PATH" \
            --output_dir "$OUT/${METHOD_DIR}/eval"
    done
done

echo "Experiment 1 complete. Results in $OUT/"
