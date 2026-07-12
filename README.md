# DDO-RM Coverage Experiments

This repository skeleton supports the paper story:

> **Decision Distribution Optimization with Reward Models (DDO-RM)** studies support-restricted entropy policy improvement for reward-model-based LLM post-training.

The key point is not that coverage matters in general. The Coverage Principle studies whether a pretrained/current policy assigns mass to high-quality responses. This repo studies a more local object: whether a **generated finite support** captures enough mass of the **ideal entropy-improved policy** for a TPO/DDO-style target-matching update to be reliable.

## Core object

For prompt `x`, current policy `pi_theta`, reward score `r(x, y)`, and generated support

```text
C_K(x) = {y_1, ..., y_K},
```

DDO-RM computes

```math
q_C^*(y_i) = p_C(y_i) exp(r_i / lambda) / sum_j p_C(y_j) exp(r_j / lambda)
```

where

```math
p_C(y_i) = pi_theta(y_i | x) / sum_j pi_theta(y_j | x).
```

The coverage diagnostic is

```math
rho_lambda(C_K; x) = q_lambda^*(C_K(x) | x),
q_lambda^*(y | x) proportional to pi_theta(y | x) exp(r(x,y)/lambda).
```

## What this repo should test

1. **Real coverage-performance curve**: does DDO-specific coverage predict target-matching performance better than generic support statistics?
2. **Vanilla TPO vs coverage-aware support**: does support construction improve target matching?
3. **Same-RM PPO-RLHF comparison**: with same SFT, same reward model, same prompts, and same budget.
4. **RM-argmax / Best-of-K ablation**: is policy training more than inference-time reranking?
5. **Reward-noise sweep**: does degradation track the reward-error term?

## Repository status

This is a skeleton. The core math utilities and toy tests are implemented; GPU-heavy trainers are placeholders with explicit TODOs.

⭕️ Current workshop minimum package is from (workshop roadmap)[https://github.com/TiantianZ399/rethinking-theory-of-post-training-in-llm-experiment/blob/main/docs/DDO_RM_MINIMAL_WORKSHOP_README.md]

## Suggested run order

```bash
pip install -r requirements.txt
pytest -q
python scripts/smoke_core_math.py
```

Then assign experiment blocks in `docs/EXPERIMENT_ROADMAP.md`.
