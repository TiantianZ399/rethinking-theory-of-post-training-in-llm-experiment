"""
Coverage evaluation for DDO-RM.

Validates the oracle inequality:
    V_full - V_C = λ log(1/ρ)

by computing coverage and correlating with downstream performance.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..theory import estimate_coverage, coverage_curve, batch_coverage


@dataclass
class CoverageEvalResult:
    """Coverage evaluation results."""
    mean_rho: float
    std_rho: float
    mean_gap: float
    correlation_rho_reward: float
    correlation_rho_accuracy: float
    coverage_by_k: Optional[list[dict]] = None


def evaluate_coverage(
    reference_policy_scores: torch.Tensor,
    reference_rewards: torch.Tensor,
    candidate_indices: torch.Tensor,
    lam: float,
    tau: float = 1.0,
    downstream_rewards: Optional[torch.Tensor] = None,
    downstream_accuracy: Optional[torch.Tensor] = None,
) -> CoverageEvalResult:
    """
    Evaluate coverage of candidate sets and correlate with performance.

    Args:
        reference_policy_scores: [B, M] pool policy scores
        reference_rewards: [B, M] pool rewards
        candidate_indices: [B, K] selected candidate indices
        lam: entropy temperature
        tau: policy temperature
        downstream_rewards: [B] DDO-RM expected rewards (optional)
        downstream_accuracy: [B] DDO-RM pair accuracies (optional)

    Returns:
        CoverageEvalResult
    """
    rho = batch_coverage(
        reference_policy_scores, reference_rewards,
        candidate_indices, lam, tau,
    )

    rho_np = rho.cpu().numpy()
    gaps = lam * np.log(1.0 / np.maximum(rho_np, 1e-12))

    corr_reward = 0.0
    corr_acc = 0.0

    if downstream_rewards is not None:
        dr = downstream_rewards.cpu().numpy()
        if len(set(rho_np)) > 1 and len(set(dr)) > 1:
            corr_reward = float(np.corrcoef(rho_np, dr)[0, 1])

    if downstream_accuracy is not None:
        da = downstream_accuracy.cpu().numpy()
        if len(set(rho_np)) > 1 and len(set(da)) > 1:
            corr_acc = float(np.corrcoef(rho_np, da)[0, 1])

    return CoverageEvalResult(
        mean_rho=float(rho_np.mean()),
        std_rho=float(rho_np.std()),
        mean_gap=float(gaps.mean()),
        correlation_rho_reward=corr_reward,
        correlation_rho_accuracy=corr_acc,
    )


def coverage_sweep(
    reference_policy_scores: torch.Tensor,
    reference_rewards: torch.Tensor,
    lam: float,
    K_values: list[int],
    tau: float = 1.0,
    num_trials: int = 20,
    selection_methods: list[str] = None,
) -> dict:
    """
    Run coverage curve experiments for varying K and selection methods.

    Args:
        reference_policy_scores: [M] or [B, M]
        reference_rewards: [M] or [B, M]
        lam: entropy temperature
        K_values: candidate sizes to test
        tau: policy temperature
        num_trials: random trials per K
        selection_methods: list of selection strategies

    Returns:
        dict mapping method -> list of coverage results
    """
    if selection_methods is None:
        selection_methods = ["random", "top_reward", "top_policy", "diverse"]

    # Handle batched input: use first prompt or average
    if reference_policy_scores.dim() == 2:
        scores = reference_policy_scores[0]
        rewards = reference_rewards[0]
    else:
        scores = reference_policy_scores
        rewards = reference_rewards

    results = {}
    for method in selection_methods:
        results[method] = coverage_curve(
            scores, rewards, lam, K_values,
            tau=tau, num_trials=num_trials, selection=method,
        )

    return results
