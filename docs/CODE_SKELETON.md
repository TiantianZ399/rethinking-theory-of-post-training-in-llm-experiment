# Code Skeleton Overview

## Main abstractions

```python
CandidateGenerator.generate(prompts, policy, K) -> CandidateBatch
RewardModel.score(prompts, candidates) -> rewards[B, K]
PolicyScorer.score(prompts, candidates) -> scores[B, K]
make_ddo_target(scores, rewards, lam, tau) -> p, q
estimate_coverage(pool_scores, pool_rewards, selected_indices, lam) -> rho
```

## Result JSON schema

```json
{
  "experiment": "exp1_same_rm_ppo_vs_ddorm",
  "seed": 42,
  "model": "EleutherAI/pythia-410m",
  "dataset": "UltraFeedback",
  "candidate_generator": "nucleus",
  "K": 8,
  "lambda": 1.0,
  "metrics": {
    "pair_accuracy": 0.0,
    "auc": 0.0,
    "mean_margin": 0.0,
    "reward_score": 0.0,
    "kl_to_reference": 0.0,
    "coverage_rho": 0.0,
    "gpu_hours": 0.0
  }
}
```
