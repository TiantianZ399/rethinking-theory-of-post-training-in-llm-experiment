"""
Variance analysis: DDO-RM vs PPO gradient estimators.

DDO-RM computes the exact gradient on C_K:
    ∂L/∂s_i = (1/τ)(p_i - q_i*)

PPO estimates it from sampled rollouts.

This module measures and compares gradient variance.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class VarianceAnalysis:
    """Gradient variance comparison."""
    ddorm_grad_var: float       # variance of DDO-RM gradient
    ppo_grad_var: float         # variance of PPO gradient estimate
    variance_ratio: float       # PPO / DDO-RM
    ddorm_grad_norm: float
    ppo_grad_norm: float
    snr_ddorm: float            # signal-to-noise ratio
    snr_ppo: float


def ddorm_gradient_variance(
    policy_scores: torch.Tensor,
    reward_scores: torch.Tensor,
    lam: float,
    tau: float = 1.0,
    num_trials: int = 100,
    subsample_ratio: float = 0.5,
) -> dict:
    """
    Estimate DDO-RM gradient variance via subsampling.

    Since DDO-RM computes the exact gradient on C_K,
    variance comes only from the candidate set C_K.
    We measure this by subsampling candidates.

    Args:
        policy_scores: [B, K] policy scores
        reward_scores: [B, K] reward scores
        lam: entropy temperature
        tau: policy temperature
        num_trials: number of subsampling trials
        subsample_ratio: fraction of candidates to keep

    Returns:
        dict with gradient statistics
    """
    from ddorm.objectives import ddorm_target

    B, K = policy_scores.shape
    k_sub = max(2, int(K * subsample_ratio))

    all_grads = []
    for _ in range(num_trials):
        # Subsample candidates
        idx = torch.stack([torch.randperm(K)[:k_sub] for _ in range(B)])
        sub_scores = policy_scores.gather(-1, idx)
        sub_rewards = reward_scores.gather(-1, idx)

        p, q = ddorm_target(sub_scores, sub_rewards, lam, tau)
        grad = (p - q) / tau  # [B, k_sub]

        # Pad back to K
        full_grad = torch.zeros(B, K, device=policy_scores.device)
        for b in range(B):
            full_grad[b, idx[b]] = grad[b]

        all_grads.append(full_grad)

    grads = torch.stack(all_grads)  # [T, B, K]
    mean_grad = grads.mean(dim=0)
    var_grad = grads.var(dim=0)

    return {
        "mean_grad_norm": float(mean_grad.norm(dim=-1).mean()),
        "mean_var": float(var_grad.mean()),
        "max_var": float(var_grad.max()),
        "snr": float(mean_grad.norm(dim=-1).mean() / (var_grad.mean().sqrt() + 1e-8)),
    }


def ppo_gradient_variance(
    policy_scores: torch.Tensor,
    reward_scores: torch.Tensor,
    num_trials: int = 100,
    num_samples: int = 1,
) -> dict:
    """
    Estimate PPO-style gradient variance.

    PPO samples one trajectory and uses REINFORCE.
    Gradient: r(y) * ∇ log π(y)

    We simulate this by sampling from the current policy
    and computing REINFORCE gradients.

    Args:
        policy_scores: [B, K] policy scores
        reward_scores: [B, K] reward scores
        num_trials: number of gradient estimates
        num_samples: rollouts per gradient estimate

    Returns:
        dict with gradient statistics
    """
    B, K = policy_scores.shape
    log_probs = F.log_softmax(policy_scores, dim=-1)
    probs = log_probs.exp()

    all_grads = []
    for _ in range(num_trials):
        # Sample actions from current policy
        actions = torch.multinomial(probs, num_samples, replacement=True)  # [B, S]

        # REINFORCE gradient for each sample
        trial_grad = torch.zeros(B, K, device=policy_scores.device)
        for s in range(num_samples):
            a = actions[:, s]  # [B]
            r = reward_scores.gather(-1, a.unsqueeze(-1)).squeeze(-1)  # [B]

            # ∇ log π(a) for softmax: e_a - p
            one_hot = F.one_hot(a, K).float()
            grad_log_pi = one_hot - probs  # [B, K]
            trial_grad += r.unsqueeze(-1) * grad_log_pi

        trial_grad /= num_samples
        all_grads.append(trial_grad)

    grads = torch.stack(all_grads)  # [T, B, K]
    mean_grad = grads.mean(dim=0)
    var_grad = grads.var(dim=0)

    return {
        "mean_grad_norm": float(mean_grad.norm(dim=-1).mean()),
        "mean_var": float(var_grad.mean()),
        "max_var": float(var_grad.max()),
        "snr": float(mean_grad.norm(dim=-1).mean() / (var_grad.mean().sqrt() + 1e-8)),
    }


def compare_gradient_variance(
    policy_scores: torch.Tensor,
    reward_scores: torch.Tensor,
    lam: float,
    tau: float = 1.0,
    num_trials: int = 100,
) -> VarianceAnalysis:
    """
    Compare DDO-RM and PPO gradient variance.

    Args:
        policy_scores: [B, K]
        reward_scores: [B, K]
        lam: DDO-RM temperature
        tau: policy temperature
        num_trials: trials per method

    Returns:
        VarianceAnalysis
    """
    ddorm_stats = ddorm_gradient_variance(
        policy_scores, reward_scores, lam, tau, num_trials
    )
    ppo_stats = ppo_gradient_variance(
        policy_scores, reward_scores, num_trials
    )

    ddorm_var = ddorm_stats["mean_var"]
    ppo_var = ppo_stats["mean_var"]

    return VarianceAnalysis(
        ddorm_grad_var=ddorm_var,
        ppo_grad_var=ppo_var,
        variance_ratio=ppo_var / max(ddorm_var, 1e-12),
        ddorm_grad_norm=ddorm_stats["mean_grad_norm"],
        ppo_grad_norm=ppo_stats["mean_grad_norm"],
        snr_ddorm=ddorm_stats["snr"],
        snr_ppo=ppo_stats["snr"],
    )
