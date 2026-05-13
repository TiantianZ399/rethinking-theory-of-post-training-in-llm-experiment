#!/usr/bin/env bash
# Experiment 0: Reproduce existing DPO reference benchmark
# Pythia-410M, UltraFeedback Binarized, seeds (42, 13, 3407)
set -euo pipefail

MODEL="EleutherAI/pythia-410m"
DATASET="ultrafeedback_binarized"
OUT="results/exp0_reference"

for SEED in 42 13 3407; do
    echo "========================================"
    echo "  Exp 0 | Seed $SEED"
    echo "========================================"

    SFT_DIR="$OUT/sft_seed${SEED}"
    SFT_PATH="$SFT_DIR/final"

    # 1. SFT
    python -m ddorm train sft \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --train_split "train_prefs" \
        --seed $SEED \
        --output_dir "$SFT_DIR" \
        --learning_rate 2e-5 \
        --batch_size 8 \
        --num_epochs 3

    # 2. DPO
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

    # 3. DDO-RM (K=2 for pairwise data)
    python -m ddorm train ddorm \
        --model_name "$SFT_PATH" \
        --reward_model_path "$SFT_PATH" \
        --dataset "$DATASET" \
        --train_split "train_prefs" \
        --seed $SEED \
        --output_dir "$OUT/ddorm_seed${SEED}" \
        --lambda_temp 1.0 \
        --num_candidates 2 \
        --learning_rate 5e-6 \
        --num_epochs 3

    # 4. Evaluate all
    for METHOD in dpo ddorm; do
        python -m ddorm eval \
            --model_path "$OUT/${METHOD}_seed${SEED}/final" \
            --dataset "$DATASET" \
            --eval_split "test_prefs" \
            --ref_model_path "$SFT_PATH" \
            --output_dir "$OUT/${METHOD}_seed${SEED}/eval"
    done
done

echo "Experiment 0 complete. Results in $OUT/"
