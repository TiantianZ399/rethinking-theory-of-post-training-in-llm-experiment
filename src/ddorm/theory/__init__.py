"""
Coverage diagnostics for DDO-RM.

Implements:
- Candidate coverage ρ(C_K) = Σ_{y ∈ C_K} q*(y|x)
- Support-restriction gap: λ log(1/ρ)
- Coverage estimation from large reference pools
- Coverage vs performance correlation
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CoverageResult:
    """Coverage diagnostic result."""
    rho: float                    # ρ(C_K): coverage mass
    gap: float                    # λ log(1/ρ): support-restriction gap
    q_target_mass: torch.Tensor   # [K] mass of each candidate under q*
    uncovered_mass: float         # 1 - ρ
    effective_support: int        # number of candidates with > 1% mass


def estimate_coverage(
    reference_policy_scores: torch.Tensor,
    reference_rewards: torch.Tensor,
    candidate_indices: torch.Tensor,
    lam: float,
    tau: float = 1.0,
) -> CoverageResult:
    """
    Estimate ρ(C_K) using a large reference pool.

    The reference pool P_M = {z_1, ..., z_M} approximates the full space.
    The ideal target on the pool is:
        q_M*(z_m) ∝ p_M(z_m) · exp(r_m / λ)

    Coverage is:
        ρ(C_K) = Σ_{m ∈ candidate_indices} q_M*(z_m)

    Args:
        reference_policy_scores: [M] policy scores for reference pool
        reference_rewards: [M] reward scores for reference pool
        candidate_indices: [K] indices of selected candidates in the pool
        lam: entropy temperature λ
        tau: policy temperature τ

    Returns:
        CoverageResult
    """
    # Compute q* on reference pool
    log_p = F.log_softmax(reference_policy_scores / tau, dim=-1)
    r_centered = reference_rewards - reference_rewards.mean()
    log_q_unnorm = log_p + r_centered / lam
    log_q = log_q_unnorm - torch.logsumexp(log_q_unnorm, dim=-1, keepdim=False)
    q_star = log_q.exp()

    # Coverage
    q_candidates = q_star[candidate_indices]
    rho = q_candidates.sum().item()
    rho = max(rho, 1e-12)  # avoid log(0)

    gap = lam * np.log(1.0 / rho)
    effective_support = (q_candidates > 0.01).sum().item()

    return CoverageResult(
        rho=rho,
        gap=gap,
        q_target_mass=q_candidates,
        uncovered_mass=1.0 - rho,
        effective_support=effective_support,
    )


def coverage_curve(
    reference_policy_scores: torch.Tensor,
    reference_rewards: torch.Tensor,
    lam: float,
    K_values: list[int],
    tau: float = 1.0,
    num_trials: int = 10,
    selection: str = "random",
) -> list[dict]:
    """
    Compute coverage for varying K to produce the coverage curve.

    For each K, sample candidate sets and estimate coverage.

    Args:
        reference_policy_scores: [M] policy scores
        reference_rewards: [M] reward scores
        lam: entropy temperature
        K_values: list of candidate set sizes to test
        tau: policy temperature
        num_trials: number of random samples per K
        selection: "random", "top_reward", "top_policy", "diverse"

    Returns:
        list of dicts with {K, rho_mean, rho_std, gap_mean, gap_std}
    """
    M = reference_policy_scores.shape[0]

    # Compute q* once
    log_p = F.log_softmax(reference_policy_scores / tau, dim=-1)
    r_centered = reference_rewards - reference_rewards.mean()
    log_q_unnorm = log_p + r_centered / lam
    log_q = log_q_unnorm - torch.logsumexp(log_q_unnorm, dim=-1, keepdim=False)
    q_star = log_q.exp()

    results = []
    for K in K_values:
        rhos = []
        for _ in range(num_trials):
            if selection == "random":
                indices = torch.randperm(M)[:K]
            elif selection == "top_reward":
                indices = reference_rewards.argsort(descending=True)[:K]
            elif selection == "top_policy":
                indices = reference_policy_scores.argsort(descending=True)[:K]
            elif selection == "diverse":
                # Simple diverse: sample proportional to q*
                probs = q_star.clone()
                probs = probs / probs.sum()
                indices = torch.multinomial(probs, K, replacement=False)
            else:
                indices = torch.randperm(M)[:K]

            rho = q_star[indices].sum().item()
            rhos.append(rho)

        rhos = np.array(rhos)
        gaps = lam * np.log(1.0 / np.maximum(rhos, 1e-12))

        results.append({
            "K": K,
            "rho_mean": rhos.mean(),
            "rho_std": rhos.std(),
            "gap_mean": gaps.mean(),
            "gap_std": gaps.std(),
        })

    return results


def batch_coverage(
    reference_policy_scores: torch.Tensor,
    reference_rewards: torch.Tensor,
    candidate_indices: torch.Tensor,
    lam: float,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Batch coverage for multiple prompts.

    Args:
        reference_policy_scores: [B, M]
        reference_rewards: [B, M]
        candidate_indices: [B, K]
        lam: float
        tau: float

    Returns:
        rho: [B] coverage per prompt
    """
    B, M = reference_policy_scores.shape

    log_p = F.log_softmax(reference_policy_scores / tau, dim=-1)
    r_centered = reference_rewards - reference_rewards.mean(dim=-1, keepdim=True)
    log_q = log_p + r_centered / lam
    log_q = log_q - torch.logsumexp(log_q, dim=-1, keepdim=True)
    q_star = log_q.exp()  # [B, M]

    # Gather candidate masses
    q_cand = q_star.gather(-1, candidate_indices)  # [B, K]
    rho = q_cand.sum(dim=-1)  # [B]

    return rho
