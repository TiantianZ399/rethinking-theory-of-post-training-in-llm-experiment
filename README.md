# DDO-RM: Support-Restricted Entropy Policy Improvement with Reward Models

**DDO-RM** (Direct Distribution Optimization with Reward Models) is a post-training framework for LLMs that distills a Boltzmann-optimal policy over a finite candidate set into the language model via supervised cross-entropy.

> **Core idea:** Given a prompt *x*, generate *K* candidates from a base policy, score them with a reward model, form the Boltzmann distribution q\*\_i ∝ p\_i · exp(r\_i / λ), and distill into the LM via cross-entropy loss.

## Key Properties

- **No RL instability** — pure supervised distillation, no policy gradients or value functions
- **Bounded error** — oracle inequality: gap ≤ λ log(1/ρ) + 2ε\_RM + ε\_proj (Theorem 1)
- **Coverage-aware** — explicit coverage diagnostic ρ(C\_K; x) connects candidate quality to theory
- **Submodular candidate construction** — greedy selection with (1-1/e) approximation guarantee
- **Drop-in baselines** — DPO, PPO-RLHF, GRPO, Best-of-K, OPD all implemented in the same framework

## Installation

```bash
git clone https://github.com/<org>/ddorm.git
cd ddorm
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.1, Transformers ≥ 4.36

## Quick Start

```bash
# 1. SFT a base model
python -m ddorm train sft \
    --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dataset ultrafeedback_binarized \
    --output_dir results/sft

# 2. Train a reward model
python -m ddorm train reward_model \
    --model_name results/sft/final \
    --dataset ultrafeedback_binarized \
    --output_dir results/rm

# 3. DDO-RM training
python -m ddorm train ddorm \
    --model_name results/sft/final \
    --reward_model_path results/rm/final \
    --dataset ultrafeedback_binarized \
    --lambda_temp 1.0 \
    --num_candidates 16 \
    --output_dir results/ddorm

# 4. Evaluate
python -m ddorm eval \
    --model_path results/ddorm/final \
    --dataset ultrafeedback_binarized \
    --ref_model_path results/sft/final \
    --reward_model_path results/rm/final \
    --output_dir results/ddorm/eval
```

## CLI Reference

All commands use `python -m ddorm <command>`:

| Command | Description |
|---------|-------------|
| `train {sft,rm,dpo,ddorm,ppo,grpo,opd}` | Train a model with the specified method |
| `eval` | Evaluate a trained model (pairwise win-rate, KL, reward) |
| `coverage` | Compute coverage diagnostics ρ(C\_K; x) |
| `reward_noise` | Run reward noise injection experiments |
| `run --config <yaml>` | Run a full experiment from a YAML config |

## Project Structure

```
ddorm-oral/
├── configs/
│   ├── model/          # Model configs (pythia-410m, tinyllama, qwen)
│   ├── data/           # Dataset configs (ultrafeedback, gsm8k)
│   ├── train/          # Training method configs
│   └── experiments/    # Full experiment configs (exp0–exp9)
├── scripts/
│   ├── run_exp0_reference.sh       # E0: DPO reference benchmark
│   ├── run_exp1_same_rm.sh         # E1: Same-RM PPO vs DDO-RM ★
│   ├── run_exp2_coverage.sh        # E2: Coverage curve
│   ├── run_exp3_generators.sh      # E3: Candidate generator comparison
│   ├── run_exp4_reward_noise.sh    # E4: Reward noise sweep
│   ├── run_exp5_rerank_ablation.sh # E5: Reranking ablation
│   ├── run_exp6_listwise.sh        # E6: Listwise evaluation
│   ├── run_exp7_opd.sh             # E7: On-policy distillation
│   ├── run_exp8_reasoning.sh       # E8: Reasoning (GSM8K)
│   └── run_all_oral_suite.sh       # Master runner (workshop/icml/oral)
└── src/ddorm/
    ├── cli.py              # Unified CLI entry point
    ├── objectives/         # Loss functions: DDO-RM, DPO, PPO, GRPO, OPD
    ├── candidates/         # Candidate generators: temp, nucleus, beam, diverse, submodular
    ├── theory/             # Coverage, reward noise, submodular selection, variance
    ├── data/               # Dataset loaders (UltraFeedback, GSM8K)
    ├── models/             # Model wrappers (policy, RM, reference, teacher)
    ├── train/              # Training loops for each method
    └── eval/               # Evaluation: pairwise, listwise, coverage, KL, bootstrap
```

## Experiment Suite

| ID | Experiment | Priority | What it tests |
|----|-----------|----------|---------------|
| E0 | DPO reference benchmark | Setup | Reproduce existing DPO results (Pythia-410M) |
| E1 | Same-RM PPO vs DDO-RM | **Must-have** | Apples-to-apples: same SFT, same RM, same prompts |
| E2 | Coverage curve | High | Performance vs ρ(C\_K) for K ∈ {2,4,8,16,32,64} |
| E3 | Candidate generators | Medium | Temperature vs nucleus vs diverse-beam vs submodular |
| E4 | Reward noise sweep | Medium | Degradation under calibrated RM noise (tests Thm 1) |
| E5 | Reranking ablation | Medium | Boltzmann distillation vs Best-of-K argmax |
| E6 | Listwise evaluation | Medium | Multi-candidate (K=4) UltraFeedback |
| E7 | On-policy distillation | Medium | OPD with iterative candidate refresh vs GRPO |
| E8 | Reasoning (GSM8K) | Optional | Verifiable rewards, no learned RM |
| E9 | Scaling | Optional | Model size sweep: 410M → 1B → 3B |

Run experiment subsets:

```bash
bash scripts/run_all_oral_suite.sh workshop  # E0 + E1 + E5
bash scripts/run_all_oral_suite.sh icml      # E1–E6
bash scripts/run_all_oral_suite.sh oral      # E0–E8
```

## Theory

The error decomposition (Theorem 1) bounds the suboptimality gap:

```
V(π*) - V(π_θ) ≤ λ log(1/ρ) + 2ε_RM + ε_proj
```

where ρ = coverage of the optimal policy mass by C\_K, ε\_RM = reward model error, and ε\_proj = distillation projection error. This cleanly separates the three sources of error and provides actionable diagnostics via the `coverage` command.

## Citation

```bibtex
@article{zhang2025ddorm,
  title={DDO-RM: Support-Restricted Entropy Policy Improvement
         with Reward Models for LLM Post-Training},
  author={Zhang, Tiantian and Zuo, Jierui and Chen, Michael},
  year={2025}
}
```

## License

MIT
