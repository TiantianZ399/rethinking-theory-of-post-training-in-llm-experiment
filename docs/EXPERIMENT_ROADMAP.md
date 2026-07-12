# Experiment Roadmap: Coverage-Guided DDO-RM

This roadmap is designed for collaborators. Each experiment has a clearly defined purpose and deliverable.

## Theory anchor

We build on two upstream observations:

- **Coverage Principle**: high-quality response mass is a central prerequisite for post-training and test-time extraction.
- **TPO**: detached target matching on scored sampled completions is a strong local policy-improvement primitive.

Our contribution is the finite-support link between them:

```math
V_full - V_C = lambda log(1 / rho_lambda(C_K; x)).
```

So experiments should not merely show "coverage matters". They must show that **DDO-specific coverage** controls **target-matching reliability** under learned rewards.

---

## Experiment 0: Existing DPO reference

**Status:** already available in the public repo.

**Purpose:** direct-preference reference, not the main fair baseline.

**Methods:** DPO, DDO-RM.

**Metrics:** pair accuracy, AUC, mean margin.

**Deliverable:** table reproduced from existing result.

---

## Experiment 1: Real coverage-performance curve

**Purpose:** validate the support-restriction theorem on real LLM candidate sets.

**Protocol:**

1. For each prompt, generate a large pool `P_M(x)` with `M >> K`.
2. Score all candidates using the reward model.
3. Approximate `q_lambda^*` on `P_M`.
4. Select smaller supports `C_K` using several strategies.
5. Compute `rho_lambda(C_K; x)`.
6. Run target matching on `C_K`.
7. Plot performance vs coverage.

**Baselines for predictors:**

- max reward in support
- mean reward in support
- reward variance
- support entropy
- lexical diversity
- embedding diversity
- average policy log probability
- DDO-specific coverage `rho_lambda`

**Success criterion:** `rho_lambda` predicts target-matching performance better than generic support statistics.

**Owner:** TBD.

---

## Experiment 2: Vanilla sampled support vs coverage-aware support

**Purpose:** test whether support construction improves target matching beyond vanilla TPO.

**Methods:**

- random/nucleus sampled support + TPO/DDO target matching
- beam support + target matching
- diverse sampling + target matching
- reward-diverse support + target matching
- submodular surrogate selected support + target matching
- OPD-guided support + target matching, optional

**Fixed component:** target-matching operator.

**Variable:** candidate support construction.

**Success criterion:** coverage-aware support improves both coverage and downstream target-matching performance.

**Owner:** TBD.

---

## Experiment 3: Same-RM PPO-RLHF comparison

**Purpose:** matched-information comparison for reward-model-based policy improvement.

**Protocol:**

Use the same:

- SFT checkpoint
- reward model
- prompts
- generation budget
- KL/reference budget
- evaluation set

Compare:

- SFT
- PPO-RLHF
- vanilla TPO/DDO target matching
- coverage-aware DDO-RM
- RM-argmax / Best-of-K
- DPO as reference only

**Metrics:** reward score, pair accuracy, AUC, margin, KL drift, GPU-hours, reward improvement per GPU-hour.

**Success criterion:** DDO/TPO-style target matching gives better KL-controlled reward improvement or lower variance than PPO-RLHF.

**Owner:** TBD.

---

## Experiment 4: RM-argmax / Best-of-K ablation

**Purpose:** show DDO-RM is not merely inference-time reranking.

**Protocol:**

- RM-argmax selects highest reward candidate, no training.
- DDO-RM trains policy toward target distribution.
- Evaluate both on new prompts and new candidate sets.

**Success criterion:** trained DDO-RM policy generalizes beyond same-support reranking.

**Owner:** TBD.

---

## Experiment 5: Reward noise / calibration sweep

**Purpose:** validate reward-error term.

**Protocol:** corrupt reward scores with:

- additive Gaussian noise
- scale distortion
- monotone transformation
- adversarial top-candidate flip

**Metrics:** target KL, reward regret, performance degradation, stability.

**Success criterion:** degradation tracks reward-error size and reveals calibration sensitivity.

**Owner:** TBD.

---

## Experiment 6: Mode-aware reasoning support

**Purpose:** connect DDO-RM to Mirror Horizon / verified-mode coverage.

**Protocol:**

- generate reasoning traces
- verify final answer
- cluster/label semantic modes
- compute verified-mode coverage
- apply target matching on selected supports

**Success criterion:** verified-mode coverage predicts reasoning improvement better than pass@K alone.

**Owner:** TBD.
