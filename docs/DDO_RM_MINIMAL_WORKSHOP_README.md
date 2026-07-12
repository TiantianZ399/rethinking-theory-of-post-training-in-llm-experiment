# DDO-RM Minimal Workshop Experiment README

This README is for the **minimal NeurIPS/ICML workshop version** of the DDO-RM project. The goal is not to claim a full post-training benchmark win. The goal is to make the theory paper empirically coherent with a small, reproducible set of diagnostics.

## 0. Core claim

**DDO-RM = Decision Distribution Optimization with Reward Models.**

The method studies **support-restricted entropy policy improvement**. For each prompt \(x\), a candidate generator produces a finite support

\[
C_K(x)=\{y_1,\dots,y_K\}.
\]

The current policy induces a restricted decision distribution

\[
p_i = \frac{\pi_\theta(y_i\mid x)}{\sum_{j=1}^K \pi_\theta(y_j\mid x)}.
\]

A reward model gives scores

\[
\hat r_i = \hat r(x,y_i).
\]

DDO-RM computes the target

\[
q_i^* = \frac{p_i\exp(\hat r_i/\lambda)}{\sum_j p_j\exp(\hat r_j/\lambda)}
\]

and distills this target into the language model using

\[
\mathcal L_{\mathrm{DDO-RM}} = -\sum_i q_i^*\log p_{\theta,i}.
\]

The key theory term is support coverage:

\[
\rho_\lambda(C_K;x)=q_\lambda^*(C_K(x)\mid x),
\]

where

\[
q_\lambda^*(y\mid x) \propto \pi_\theta(y\mid x)\exp(r(x,y)/\lambda).
\]

The support restriction gap is

\[
V_{\rm full}-V_C = \lambda\log\frac{1}{\rho_\lambda(C_K;x)}.
\]

So the empirical objective is to verify:

\[
\text{coverage} \uparrow \Rightarrow \text{DDO/TPO-style target matching improves}.
\]

## 1. Minimal workshop experiment list

The workshop version needs four experiment blocks.

| ID | Experiment | Required? | Purpose | GPU cost |
|---|---|---:|---|---:|
| E0 | Existing DDO-RM vs DPO held-out reference | Already done | Public minimal LLM sanity check | none |
| E1 | Real or semi-real coverage diagnostic | Yes | Check whether \(\rho_\lambda(C_K)\) predicts value/performance | low |
| E2 | Full-on-support vs sampled-on-support | Yes | Check whether full candidate reward vector improves stability | low |
| E3 | RM-argmax / Best-of-K ablation | Yes | Check DDO-RM is not merely inference-time reranking | low |
| E4 | Same-RM PPO-RLHF minimal run | Recommended | Address reviewer objection: compare to PPO-RLHF with same RM | medium |

For a workshop deadline, **E0--E3 are sufficient** if clearly framed as theory diagnostics. **E4** makes the submission much stronger if time/GPU allows.

## 2. Reference links to build on

### Existing project repository

- Current DDO-RM benchmark repo: https://github.com/TiantianZ399/ddorm-llm-preference-benchmark

This repo already contains the minimal DDO-RM vs DPO benchmark on Pythia-410M and UltraFeedback Binarized.

### Hugging Face / TRL infrastructure

- TRL main docs: https://huggingface.co/docs/trl/en/index
- TRL GitHub: https://github.com/huggingface/trl
- TRL PPOTrainer docs: https://huggingface.co/docs/trl/en/ppo_trainer
- TRL DPOTrainer docs: https://huggingface.co/docs/trl/en/dpo_trainer
- TRL GRPOTrainer docs: https://huggingface.co/docs/trl/en/grpo_trainer
- HuggingFace Alignment Handbook: https://github.com/huggingface/alignment-handbook

### Dataset links

- UltraFeedback Binarized dataset: https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized
- Dataset viewer for `train_prefs` / `test_prefs`: https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/viewer/default/train_prefs

Useful splits:

```text
train_prefs: preference training pairs
train_sft: SFT training responses
test_prefs: held-out pairwise preference eval
test_sft: held-out SFT eval
train_gen / test_gen: generation prompts
```

The dataset viewer reports about **61.1k `train_prefs` rows** and **2k `test_prefs` rows**.

### Closest related algorithmic primitive

- Target Policy Optimization paper: https://arxiv.org/abs/2604.06159
- TPO code: https://github.com/JeanKaddour/tpo

Use TPO as the **target-matching primitive** reference. Do **not** claim the detached exponential target itself is new.

### Coverage and OPD background

- Coverage Principle paper: https://arxiv.org/abs/2510.15020
- Revisiting OPD paper: https://arxiv.org/abs/2603.25562
- Revisiting OPD code/blog page: https://yuqianfu.notion.site/Revisiting-On-Policy-Distillation-Empirical-Failure-Modes-and-Simple-Fixes-31dd5cc40dd181f89eead3de7181df1d

Use the Coverage Principle as the upstream motivation: coverage matters for post-training. Use DDO-RM as the finite-support, reward-model policy-improvement specialization.

## 3. GPU requirements

### Minimal workshop level

Recommended hardware:

```text
1 x RTX 4090 / A10 / L4 / T4 / P100
16GB VRAM minimum, 24GB preferred
```

Expected GPU time:

```text
E1 coverage diagnostic:                 1-4 GPUh
E2 full-on-support vs sampled support:  1-4 GPUh
E3 RM-argmax / Best-of-K ablation:      1-4 GPUh
E4 minimal PPO-RLHF same-RM:             8-25 GPUh for one seed
Total without PPO:                       3-12 GPUh
Total with one PPO run:                  15-40 GPUh
```

Kaggle is enough for E1--E3 and possibly one small E4 run. Do not use Kaggle for the full ICML suite.

### Minimal model/dataset choice

Use:

```yaml
base_model: EleutherAI/pythia-410m
dataset: HuggingFaceH4/ultrafeedback_binarized
train_subset: 500-2000 prompts for quick tests
eval_subset: 500-2000 prompts
max_new_tokens: 64-128
candidate_K: [2, 4, 8]
seeds: [42] first, then [42, 13, 3407] if stable
```

## 4. Experiment E0: existing DPO reference

### Purpose

Keep the current public result as a direct-preference reference.

### Existing result

The current repo reports:

```text
Base model: EleutherAI/pythia-410m
Dataset: HuggingFaceH4/ultrafeedback_binarized
Eval split: test_prefs
Seeds: 42, 13, 3407
Methods: DPO, DDO-RM
```

Mean results:

| Metric | DPO | DDO-RM |
|---|---:|---:|
| Pair accuracy | 0.5238 | 0.5602 |
| AUC | 0.5315 | 0.5382 |
| Mean margin | 0.1377 | 0.5353 |

### Interpretation

This is **not** the main fairness comparison because DDO-RM uses a reward model and DPO does not. It is a public sanity check.

## 5. Experiment E1: coverage diagnostic

### Goal

Verify the theory term:

\[
\rho_\lambda(C_K;x) \uparrow \Rightarrow \text{DDO-RM value/performance} \uparrow.
\]

### Procedure

For each prompt:

1. Generate a large reference pool

   \[
   P_M(x)=\{y_1,\dots,y_M\}, \quad M \gg K.
   \]

2. Score all reference candidates with the reward model.

3. Compute approximate full-pool target

   \[
   \tilde q_M(y_m) \propto p_M(y_m)\exp(\hat r_m/\lambda).
   \]

4. Select smaller supports \(C_K\subset P_M\) by different methods.

5. Estimate coverage

   \[
   \hat \rho_\lambda(C_K;x)=\sum_{y_m\in C_K}\tilde q_M(y_m).
   \]

6. Run DDO-RM target matching on \(C_K\) and measure performance.

### Methods for choosing \(C_K\)

```text
random sample
nucleus sample
beam top-K
diverse sample
reward-top-K
coverage/submodular selected subset
```

### Metrics

```text
coverage rho
expected reward under q_C
held-out pair accuracy
reward score
regret to best candidate in reference pool
correlation: rho vs performance
```

### Output files

```text
results/e1_coverage/coverage_by_prompt.csv
results/e1_coverage/coverage_vs_reward.csv
figures/e1_coverage_vs_reward.png
figures/e1_coverage_vs_regret.png
```

### Success criterion

A positive correlation between \(\hat\rho_\lambda(C_K;x)\) and reward/performance. Stronger: coverage predicts performance better than average reward, max reward, candidate entropy, or diversity alone.

## 6. Experiment E2: full-on-support vs sampled-on-support

### Goal

Test the mechanism:

```text
sampled update on C_K  vs  full reward-vector update on C_K
```

This does **not** assume access to the full response space. “Full” means full information **on the generated finite support**.

### Methods

| Method | Description |
|---|---|
| sampled-on-support | sample one or few candidates from \(C_K\), update from those rewards |
| full-on-support DDO/TPO | use all \(K\) reward scores and compute \(q_i\propto p_i e^{r_i/\lambda}\) |

### Metrics

```text
mean reward improvement per step
gradient variance proxy
KL drift
final expected reward
seed variance
```

### Output files

```text
results/e2_full_vs_sampled/training_curves.csv
figures/e2_full_vs_sampled_reward.png
figures/e2_gradient_variance.png
```

### Success criterion

Full-on-support update has lower variance and/or better reward improvement at matched KL.

## 7. Experiment E3: RM-argmax / Best-of-K ablation

### Goal

Answer the reviewer objection:

```text
Is DDO-RM just reward-model reranking?
```

### Methods

| Method | Training? | Inference? |
|---|---:|---|
| RM-argmax | no | choose \(\arg\max_i \hat r_i\) on the same candidates |
| Best-of-K | no | generate K and pick best by RM |
| DDO-RM | yes | distill \(q^*\) into model, then evaluate on new prompts/candidates |

### Correct evaluation

Do **not** evaluate only on the same candidate set used to train. That favors RM-argmax.

Evaluate on:

```text
new held-out prompts
new candidate sets
same K and different K
```

### Metrics

```text
held-out pair accuracy
reward score on new candidates
best-of-K win rate
response margin
KL drift
```

### Output files

```text
results/e3_rerank_ablation/table.csv
figures/e3_rerank_vs_training.png
```

### Success criterion

DDO-RM trained policy improves on new prompts/candidate sets beyond pure inference-time RM-argmax.

## 8. Experiment E4: minimal same-RM PPO-RLHF

### Goal

Address the key workshop reviewer objection:

```text
You claim DDO-RM is relevant to reward-model-based policy improvement, but where is PPO-RLHF?
```

### Methods

```text
SFT
PPO-RLHF same RM
DDO-RM same RM
RM-argmax / Best-of-K
DPO reference
```

### Matched setting

Use the same:

```text
SFT checkpoint
reward model
prompts
candidate/generation budget
max_new_tokens
KL target if possible
training steps
```

### Metrics

```text
held-out pair accuracy
reward score
KL to SFT/reference
mean margin
reward improvement per GPU-hour
response length
seed stability
```

### Output files

```text
results/e4_same_rm_ppo/table.csv
figures/e4_reward_curve.png
figures/e4_kl_curve.png
figures/e4_pair_accuracy.png
```

### Success criterion

For workshop, it is enough if DDO-RM is competitive with PPO-RLHF and has lower KL drift, better sample efficiency, or more stable training. It does not need to dominate final score everywhere.

## 9. Minimal code skeleton

Create this directory structure:

```text
ddorm-workshop/
  README.md
  requirements.txt
  configs/
    model/pythia410m.yaml
    data/ultrafeedback_binarized.yaml
    exp/e1_coverage.yaml
    exp/e2_full_vs_sampled.yaml
    exp/e3_rerank_ablation.yaml
    exp/e4_same_rm_ppo.yaml
  src/ddorm/
    __init__.py
    core.py                 # ddorm target, loss, gradients
    coverage.py             # coverage estimator, support gap
    candidates.py           # candidate generation wrappers
    rewards.py              # reward model scoring wrapper
    metrics.py              # pair acc, AUC, margin, KL, reward
    submodular.py            # greedy coverage selection
    baselines.py             # RM-argmax, Best-of-K, sampled update
  scripts/
    prepare_data.py
    train_reward_model.py
    run_e1_coverage.py
    run_e2_full_vs_sampled.py
    run_e3_rerank_ablation.py
    run_e4_ppo_rlhf_minimal.py
    plot_results.py
  results/
    e1_coverage/
    e2_full_vs_sampled/
    e3_rerank_ablation/
    e4_same_rm_ppo/
  figures/
  tests/
    test_core.py
    test_coverage.py
```

### `core.py` minimum functions

```python
import torch
import torch.nn.functional as F


def restricted_policy(policy_scores: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    return F.softmax(policy_scores / tau, dim=-1)


def ddorm_target(policy_scores: torch.Tensor, reward_scores: torch.Tensor,
                 lam: float = 1.0, tau: float = 1.0,
                 center: bool = True) -> torch.Tensor:
    p = restricted_policy(policy_scores, tau=tau)
    r = reward_scores
    if center:
        baseline = (p * r).sum(dim=-1, keepdim=True)
        r = r - baseline
    log_q = torch.log(p.clamp_min(1e-12)) + r / lam
    return F.softmax(log_q, dim=-1)


def ddorm_loss(policy_scores_new: torch.Tensor, q_target: torch.Tensor,
               tau: float = 1.0) -> torch.Tensor:
    log_p = F.log_softmax(policy_scores_new / tau, dim=-1)
    return -(q_target.detach() * log_p).sum(dim=-1).mean()
```

### `coverage.py` minimum functions

```python
import torch
import torch.nn.functional as F


def reference_target(pool_policy_scores: torch.Tensor, pool_rewards: torch.Tensor,
                     lam: float = 1.0, tau: float = 1.0) -> torch.Tensor:
    p = F.softmax(pool_policy_scores / tau, dim=-1)
    log_q = torch.log(p.clamp_min(1e-12)) + pool_rewards / lam
    return F.softmax(log_q, dim=-1)


def support_coverage(q_ref: torch.Tensor, support_indices: torch.Tensor) -> torch.Tensor:
    # q_ref: [B, M], support_indices: [B, K]
    return q_ref.gather(dim=-1, index=support_indices).sum(dim=-1)


def support_gap(coverage: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return lam * torch.log(1.0 / coverage.clamp_min(1e-12))
```

### `baselines.py` minimum functions

```python
import torch


def rm_argmax(reward_scores: torch.Tensor) -> torch.Tensor:
    return reward_scores.argmax(dim=-1)


def best_of_k(reward_scores: torch.Tensor) -> torch.Tensor:
    return rm_argmax(reward_scores)


def sampled_support_update(policy_probs: torch.Tensor, reward_scores: torch.Tensor):
    idx = torch.multinomial(policy_probs, num_samples=1).squeeze(-1)
    sampled_reward = reward_scores.gather(dim=-1, index=idx[:, None]).squeeze(-1)
    return idx, sampled_reward
```

## 10. Collaborator workload split

### Person A: Coverage diagnostic

Owns E1.

Tasks:

```text
large candidate pool generation
q_ref approximation
rho computation
rho vs performance plots
```

Deliverables:

```text
results/e1_coverage/coverage_by_prompt.csv
figures/e1_coverage_vs_reward.png
```

### Person B: Full-on-support vs sampled-on-support

Owns E2.

Tasks:

```text
implement sampled update baseline
implement full DDORM/TPO update
compare reward improvement and variance
```

Deliverables:

```text
figures/e2_full_vs_sampled_reward.png
figures/e2_gradient_variance.png
```

### Person C: Reranking ablation

Owns E3.

Tasks:

```text
RM-argmax baseline
Best-of-K baseline
DDO-RM trained policy evaluation on new candidate sets
```

Deliverables:

```text
results/e3_rerank_ablation/table.csv
figures/e3_rerank_vs_training.png
```

### Person D: Minimal PPO-RLHF

Owns E4.

Tasks:

```text
build TRL PPOTrainer or clean PPO-RLHF run
use same reward model and same prompts as DDORM
report reward/KL/pair accuracy
```

Deliverables:

```text
results/e4_same_rm_ppo/table.csv
figures/e4_reward_curve.png
figures/e4_kl_curve.png
```

### Person E: Paper integration

Tasks:

```text
update experiment section
update limitations
insert figures
check claims remain conservative
```

## 11. Workshop claim language

Use this in the paper:

```text
We do not claim that coverage itself is new; the Coverage Principle establishes coverage as a fundamental quantity for post-training and test-time extraction. We instantiate this idea inside support-restricted target-policy optimization under learned rewards. We also do not claim novelty for detached exponential target matching, which is closely related to TPO. Our contribution is the finite-support coverage analysis and diagnostics showing when such target matching is reliable in reward-model-based LLM post-training.
```

## 12. Go / no-go criteria

Proceed to workshop submission if:

```text
E0 current DPO reference remains positive
E1 rho correlates with performance
E2 full-on-support beats sampled-on-support or is more stable
E3 DDO-RM is not completely dominated by RM-argmax on new prompts/candidates
```

Add E4 if possible.

Proceed to main-conference scaling only if:

```text
real coverage curves hold
candidate generator comparison shows coverage is actionable
same-RM PPO-RLHF is competitive or favorable to DDO-RM
```

