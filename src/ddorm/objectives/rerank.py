"""
Reranking baselines: RM-argmax and Best-of-K.

These are inference-time methods (no training) for comparison.
"""

import torch
from dataclasses import dataclass


@dataclass
class RerankOutput:
    selected_indices: torch.Tensor  # [B] index of selected candidate
    selected_rewards: torch.Tensor  # [B] reward of selected candidate
    selected_logprobs: torch.Tensor  # [B] log-prob of selected candidate


def rm_argmax(
    reward_scores: torch.Tensor,
    policy_logprobs: torch.Tensor,
) -> RerankOutput:
    """
    RM-argmax: select candidate with highest reward.

    Args:
        reward_scores: [B, K]
        policy_logprobs: [B, K] (for diagnostics)

    Returns:
        RerankOutput
    """
    indices = reward_scores.argmax(dim=-1)
    return RerankOutput(
        selected_indices=indices,
        selected_rewards=reward_scores.gather(-1, indices.unsqueeze(-1)).squeeze(-1),
        selected_logprobs=policy_logprobs.gather(-1, indices.unsqueeze(-1)).squeeze(-1),
    )


def best_of_k(
    reward_scores: torch.Tensor,
    policy_logprobs: torch.Tensor,
    k: int = -1,
) -> RerankOutput:
    """
    Best-of-K: sample K candidates, pick highest reward.
    When all K are provided, identical to rm_argmax.
    When k < K, subsample first.

    Args:
        reward_scores: [B, K]
        policy_logprobs: [B, K]
        k: number to subsample (-1 = use all)

    Returns:
        RerankOutput
    """
    B, K = reward_scores.shape
    if 0 < k < K:
        # Random subsample
        perm = torch.stack([torch.randperm(K, device=reward_scores.device)[:k] for _ in range(B)])
        reward_sub = reward_scores.gather(-1, perm)
        logprob_sub = policy_logprobs.gather(-1, perm)
        sub_idx = reward_sub.argmax(dim=-1)
        indices = perm.gather(-1, sub_idx.unsqueeze(-1)).squeeze(-1)
    else:
        indices = reward_scores.argmax(dim=-1)

    return RerankOutput(
        selected_indices=indices,
        selected_rewards=reward_scores.gather(-1, indices.unsqueeze(-1)).squeeze(-1),
        selected_logprobs=policy_logprobs.gather(-1, indices.unsqueeze(-1)).squeeze(-1),
    )
