"""
PPO-RLHF: Proximal Policy Optimization for RLHF.

Standard PPO with KL penalty to reference policy.
Reward: r_total = r_hat(x,y) - β · KL(π_θ || π_ref)
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class PPORollout:
    """Single rollout data for PPO."""
    query_ids: torch.Tensor       # [B, L_q]
    response_ids: torch.Tensor    # [B, L_r]
    logprobs: torch.Tensor        # [B, L_r] log π_θ_old
    ref_logprobs: torch.Tensor    # [B, L_r] log π_ref
    rewards: torch.Tensor         # [B] reward model scores
    values: torch.Tensor          # [B, L_r] value estimates


@dataclass
class PPOOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    kl: torch.Tensor
    clip_fraction: torch.Tensor


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation.

    For RLHF, the reward is typically only at the last token.

    Args:
        rewards: [B, T] per-token rewards (usually 0 except last)
        values: [B, T] value estimates
        gamma: discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages: [B, T]
        returns: [B, T]
    """
    T = rewards.shape[1]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(rewards.shape[0], device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = torch.zeros_like(values[:, 0])
        else:
            next_value = values[:, t + 1]
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_gae = delta + gamma * gae_lambda * last_gae
        advantages[:, t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    new_values: torch.Tensor,
    clip_range: float = 0.2,
    vf_coeff: float = 0.5,
    mask: Optional[torch.Tensor] = None,
) -> PPOOutput:
    """
    Compute PPO clipped objective.

    Args:
        new_logprobs: [B, T] current policy log-probs
        old_logprobs: [B, T] old policy log-probs
        advantages: [B, T] GAE advantages
        returns: [B, T] GAE returns
        new_values: [B, T] current value estimates
        clip_range: PPO clip range
        vf_coeff: value function loss coefficient
        mask: [B, T] valid token mask

    Returns:
        PPOOutput
    """
    if mask is None:
        mask = torch.ones_like(new_logprobs)

    # Normalize advantages
    adv = (advantages - advantages[mask.bool()].mean()) / (advantages[mask.bool()].std() + 1e-8)

    # Policy loss
    ratio = (new_logprobs - old_logprobs).exp()
    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * ratio.clamp(1 - clip_range, 1 + clip_range)
    policy_loss = (torch.max(pg_loss1, pg_loss2) * mask).sum() / mask.sum()

    # Value loss
    vf_loss = ((new_values - returns) ** 2 * mask).sum() / mask.sum()

    # Entropy
    entropy = -(new_logprobs.exp() * new_logprobs * mask).sum() / mask.sum()

    # KL divergence (approx)
    kl = ((old_logprobs - new_logprobs) * mask).sum() / mask.sum()

    # Clip fraction
    clip_fraction = ((ratio - 1).abs() > clip_range).float().mean()

    loss = policy_loss + vf_coeff * vf_loss

    return PPOOutput(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=vf_loss,
        entropy=entropy,
        kl=kl,
        clip_fraction=clip_fraction,
    )


def compute_kl_penalty(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    kl_coeff: float = 0.1,
) -> torch.Tensor:
    """
    Per-token KL penalty: β · (log π_θ - log π_ref).

    Args:
        logprobs: [B, T]
        ref_logprobs: [B, T]
        kl_coeff: KL coefficient β

    Returns:
        kl_penalty: [B, T]
    """
    return kl_coeff * (logprobs - ref_logprobs)
