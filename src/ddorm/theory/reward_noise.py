"""
Reward noise analysis for DDO-RM.

Validates Proposition 3 (reward-model perturbation):
    F_{r*}(q*_C) - F_{r*}(q_hat) ≤ 2ε_RM

Implements controlled noise injection and degradation measurement.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class RewardNoiseResult:
    """Result of reward noise experiment."""
    noise_level: float
    noise_type: str
    ddorm_reward_clean: float        # E[r*] under DDO-RM with r*
    ddorm_reward_noisy: float        # E[r*] under DDO-RM with r_hat
    value_gap: float                 # F(q*) - F(q_hat)
    theoretical_bound: float         # 2 * epsilon_RM
    rank_correlation: float          # Spearman between r* and r_hat
    pair_accuracy_clean: float
    pair_accuracy_noisy: float


def inject_noise(
    rewards: torch.Tensor,
    noise_type: str,
    noise_level: float,
    seed: int = 42,
) -> torch.Tensor:
    """
    Inject controlled noise into reward scores.

    Args:
        rewards: [B, K] clean reward scores
        noise_type: type of noise
        noise_level: σ or magnitude of noise
        seed: random seed

    Returns:
        noisy_rewards: [B, K]
    """
    gen = torch.Generator(device=rewards.device).manual_seed(seed)

    if noise_type == "gaussian":
        noise = torch.randn_like(rewards, generator=gen) * noise_level
        return rewards + noise

    elif noise_type == "scale":
        # Multiply by (1 + ε), distorts magnitudes but preserves ranking
        scale = 1.0 + noise_level * torch.randn(
            rewards.shape[0], 1, device=rewards.device, generator=gen
        )
        return rewards * scale

    elif noise_type == "shift":
        # Add constant shift per prompt, preserves ranking
        shift = noise_level * torch.randn(
            rewards.shape[0], 1, device=rewards.device, generator=gen
        )
        return rewards + shift

    elif noise_type == "rank_preserving":
        # Monotone distortion: r_hat = r + ε * f(r) where f is monotone
        noise = noise_level * torch.sign(rewards) * rewards.abs().sqrt()
        return rewards + noise

    elif noise_type == "adversarial_top_flip":
        # Swap the top candidate's reward with a random one
        B, K = rewards.shape
        noisy = rewards.clone()
        for b in range(B):
            if torch.rand(1, generator=gen).item() < noise_level:
                top_idx = rewards[b].argmax()
                random_idx = torch.randint(K, (1,), generator=gen).item()
                noisy[b, top_idx], noisy[b, random_idx] = (
                    noisy[b, random_idx].clone(),
                    noisy[b, top_idx].clone(),
                )
        return noisy

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def reward_noise_experiment(
    policy_scores: torch.Tensor,
    clean_rewards: torch.Tensor,
    noise_type: str,
    noise_level: float,
    lam: float,
    tau: float = 1.0,
    seed: int = 42,
) -> RewardNoiseResult:
    """
    Run a single reward noise experiment.

    Computes DDO-RM targets under clean and noisy rewards,
    then measures value gap under the clean objective.

    Args:
        policy_scores: [B, K]
        clean_rewards: [B, K] ground-truth rewards
        noise_type: type of noise
        noise_level: magnitude
        lam: entropy temperature
        tau: policy temperature
        seed: random seed

    Returns:
        RewardNoiseResult
    """
    from ddorm.objectives import ddorm_target

    noisy_rewards = inject_noise(clean_rewards, noise_type, noise_level, seed)

    # DDO-RM target under clean rewards
    p, q_clean = ddorm_target(policy_scores, clean_rewards, lam, tau)
    # DDO-RM target under noisy rewards
    _, q_noisy = ddorm_target(policy_scores, noisy_rewards, lam, tau)

    # Value under clean objective: F(q) = <q, r> - λ KL(q||p)
    log_p = F.log_softmax(policy_scores / tau, dim=-1)

    def value_fn(q):
        reward_term = (q * clean_rewards).sum(dim=-1)
        log_q = (q + 1e-12).log()
        kl = (q * (log_q - log_p)).sum(dim=-1)
        return (reward_term - lam * kl).mean().item()

    v_clean = value_fn(q_clean)
    v_noisy = value_fn(q_noisy)
    value_gap = v_clean - v_noisy

    # Theoretical bound: 2 * max|r_hat - r*|
    eps_rm = (noisy_rewards - clean_rewards).abs().max().item()
    theoretical_bound = 2 * eps_rm

    # Rank correlation
    from scipy.stats import spearmanr
    r_flat = clean_rewards.cpu().numpy().flatten()
    rn_flat = noisy_rewards.cpu().numpy().flatten()
    rank_corr, _ = spearmanr(r_flat, rn_flat)

    # Pair accuracy
    def pair_acc(q, rewards):
        B, K = q.shape
        correct = 0
        total = 0
        for b in range(B):
            for i in range(K):
                for j in range(i + 1, K):
                    if rewards[b, i] != rewards[b, j]:
                        total += 1
                        if (q[b, i] > q[b, j]) == (rewards[b, i] > rewards[b, j]):
                            correct += 1
        return correct / max(total, 1)

    pa_clean = pair_acc(q_clean, clean_rewards)
    pa_noisy = pair_acc(q_noisy, clean_rewards)

    return RewardNoiseResult(
        noise_level=noise_level,
        noise_type=noise_type,
        ddorm_reward_clean=(q_clean * clean_rewards).sum(dim=-1).mean().item(),
        ddorm_reward_noisy=(q_noisy * clean_rewards).sum(dim=-1).mean().item(),
        value_gap=value_gap,
        theoretical_bound=theoretical_bound,
        rank_correlation=rank_corr,
        pair_accuracy_clean=pa_clean,
        pair_accuracy_noisy=pa_noisy,
    )


def reward_noise_sweep(
    policy_scores: torch.Tensor,
    clean_rewards: torch.Tensor,
    noise_type: str,
    noise_levels: list[float],
    lam: float,
    tau: float = 1.0,
) -> list[RewardNoiseResult]:
    """Run noise experiment across multiple noise levels."""
    return [
        reward_noise_experiment(
            policy_scores, clean_rewards, noise_type, sigma, lam, tau
        )
        for sigma in noise_levels
    ]
