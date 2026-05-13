#!/usr/bin/env bash
# Run All Oral-Level Experiments
# Runs E0–E8 sequentially. Use with SLURM or screen/tmux.
#
# Minimum packages:
#   Workshop:  E0 + E1 + E5
#   ICML main: E1–E6
#   Oral:      E1–E7 + E9 (optional E8)
#
# Usage:
#   bash scripts/run_all_oral_suite.sh          # run everything
#   bash scripts/run_all_oral_suite.sh workshop  # workshop subset
#   bash scripts/run_all_oral_suite.sh icml      # ICML main subset
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
SUITE="${1:-oral}"

run_exp() {
    local script="$1"
    local name="$2"
    echo ""
    echo "############################################################"
    echo "#  Starting: $name"
    echo "#  Script:   $script"
    echo "#  Time:     $(date)"
    echo "############################################################"
    echo ""
    bash "$SCRIPTS_DIR/$script"
    echo ""
    echo ">>> $name finished at $(date)"
    echo ""
}

case "$SUITE" in
    workshop)
        run_exp run_exp0_reference.sh    "E0: DPO Reference Benchmark"
        run_exp run_exp1_same_rm.sh      "E1: Same-RM PPO vs DDO-RM"
        run_exp run_exp5_rerank_ablation.sh "E5: Reranking Ablation"
        ;;
    icml)
        run_exp run_exp1_same_rm.sh      "E1: Same-RM PPO vs DDO-RM"
        run_exp run_exp2_coverage.sh     "E2: Coverage Curve"
        run_exp run_exp3_generators.sh   "E3: Candidate Generators"
        run_exp run_exp4_reward_noise.sh "E4: Reward Noise Sweep"
        run_exp run_exp5_rerank_ablation.sh "E5: Reranking Ablation"
        run_exp run_exp6_listwise.sh     "E6: Listwise Evaluation"
        ;;
    oral|all)
        run_exp run_exp0_reference.sh    "E0: DPO Reference Benchmark"
        run_exp run_exp1_same_rm.sh      "E1: Same-RM PPO vs DDO-RM"
        run_exp run_exp2_coverage.sh     "E2: Coverage Curve"
        run_exp run_exp3_generators.sh   "E3: Candidate Generators"
        run_exp run_exp4_reward_noise.sh "E4: Reward Noise Sweep"
        run_exp run_exp5_rerank_ablation.sh "E5: Reranking Ablation"
        run_exp run_exp6_listwise.sh     "E6: Listwise Evaluation"
        run_exp run_exp7_opd.sh          "E7: On-Policy Distillation"
        run_exp run_exp8_reasoning.sh    "E8: Reasoning (GSM8K)"
        ;;
    *)
        echo "Unknown suite: $SUITE"
        echo "Usage: $0 {workshop|icml|oral|all}"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  Suite '$SUITE' complete at $(date)"
echo "  Results in results/"
echo "============================================================"
