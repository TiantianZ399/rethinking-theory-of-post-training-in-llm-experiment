#!/usr/bin/env bash
# Experiment 5: Reranking Ablation
# DDO-RM full distillation vs RM-argmax (Best-of-K) vs random baseline
set -euo pipefail

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET="ultrafeedback_binarized"
OUT="results/exp5_rerank_ablation"

for SEED in 42 13 3407; do
    echo "========================================"
    echo "  Exp 5 | Seed $SEED"
    echo "========================================"

    SFT_PATH="results/exp1_same_rm/sft_seed${SEED}/final"
    RM_PATH="results/exp1_same_rm/rm_seed${SEED}/final"

    for K in 4 8 16; do
        # 1. DDO-RM (soft Boltzmann distillation)
        python -m ddorm train ddorm \
            --model_name "$SFT_PATH" \
            --reward_model_path "$RM_PATH" \
            --dataset "$DATASET" \
            --train_split "train_prefs" \
            --seed $SEED \
            --output_dir "$OUT/ddorm_K${K}_seed${SEED}" \
            --lambda_temp 1.0 \
            --num_candidates $K \
            --learning_rate 5e-6 \
            --num_epochs 3

        # 2. Best-of-K (argmax reranking, SFT on winner)
        python -m ddorm train ddorm \
            --model_name "$SFT_PATH" \
            --reward_model_path "$RM_PATH" \
            --dataset "$DATASET" \
            --train_split "train_prefs" \
            --seed $SEED \
            --output_dir "$OUT/bestofK_K${K}_seed${SEED}" \
            --lambda_temp 0.01 \
            --num_candidates $K \
            --learning_rate 5e-6 \
            --num_epochs 3

        # Evaluate both
        for METHOD in ddorm bestofK; do
            python -m ddorm eval \
                --model_path "$OUT/${METHOD}_K${K}_seed${SEED}/final" \
                --dataset "$DATASET" \
                --eval_split "test_prefs" \
                --ref_model_path "$SFT_PATH" \
                --reward_model_path "$RM_PATH" \
                --output_dir "$OUT/${METHOD}_K${K}_seed${SEED}/eval"
        done
    done
done

echo "Experiment 5 complete. Results in $OUT/"
