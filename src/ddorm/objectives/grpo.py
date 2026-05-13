"""
GRPO: Group Relative Policy Optimization.

For K candidate responses, compute group-normalized advantages
and use them as weights for policy gradient.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GRPOOutput:
    loss: torch.Tensor
    advantages: torch.Tensor
    mean_reward: torch.Tensor


def grpo_loss(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    rewards: torch.Tensor,
    kl_coeff: float = 0.1,
    clip_range: float = 0.2,
) -> GRPOOutput:
    """
    GRPO loss over a group of candidate responses.

    Args:
        logprobs: [B, K] sum of log-probs for each candidate
        ref_logprobs: [B, K] reference policy log-probs
        rewards: [B, K] reward/verifier scores
        kl_coeff: KL penalty coefficient
        clip_range: clipping range

    Returns:
        GRPOOutput
    """
    # Group-relative advantages
    mean_r = rewards.mean(dim=-1, keepdim=True)
    std_r = rewards.std(dim=-1, keepdim=True).clamp(min=1e-8)
    advantages = (rewards - mean_r) / std_r

    # Policy ratio
    ratio = (logprobs - ref_logprobs).exp()

    # Clipped surrogate
    pg1 = -advantages * ratio
    pg2 = -advantages * ratio.clamp(1 - clip_range, 1 + clip_range)
    policy_loss = torch.max(pg1, pg2).mean()

    # KL penalty
    kl = (logprobs - ref_logprobs).mean()
    loss = policy_loss + kl_coeff * kl

    return GRPOOutput(
        loss=loss,
        advantages=advantages.detach(),
        mean_reward=rewards.mean().detach(),
    )
