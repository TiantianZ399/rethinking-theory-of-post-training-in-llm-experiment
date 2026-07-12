# Collaborator Workload Split

## Team A: Core math and diagnostics
- Maintain `src/ddorm_coverage/targets.py` and `coverage.py`.
- Implement real-pool coverage estimation.
- Produce coverage-vs-performance plots.

## Team B: Candidate generation
- Implement nucleus, beam, diverse, reward-diverse, and submodular selection.
- Standardize candidate metadata format.
- Deliver comparison tables for candidate support quality.

## Team C: PPO-RLHF baseline
- Implement or integrate a reliable PPO-RLHF trainer.
- Use same SFT checkpoint, RM, prompts, and generation budget.
- Report KL-controlled reward improvement.

## Team D: Reranking and DPO references
- Maintain DPO reproduction.
- Implement RM-argmax and Best-of-K ablations.
- Verify generalization on new prompts/candidate sets.

## Team E: Reward noise and calibration
- Build controlled reward corruption suite.
- Estimate calibration sensitivity.
- Connect results to reward-error theorem.

## Team F: Reasoning / OPD optional
- Implement OPD-guided candidate generation.
- Implement reasoning trace candidate supports.
- Evaluate verified-mode coverage.
