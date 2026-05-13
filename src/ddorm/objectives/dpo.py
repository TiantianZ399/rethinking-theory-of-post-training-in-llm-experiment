"""
DPO: Direct Preference Optimization baseline.

L_DPO = -log σ(β (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class DPOOutput:
    loss: torch.Tensor
    chosen_rewards: torch.Tensor
    rejected_rewards: torch.Tensor
    accuracy: torch.Tensor


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> DPOOutput:
    """
    Compute DPO loss.

    Args:
        policy_chosen_logps: [B] log π_θ(y_w|x)
        policy_rejected_logps: [B] log π_θ(y_l|x)
        ref_chosen_logps: [B] log π_ref(y_w|x)
        ref_rejected_logps: [B] log π_ref(y_l|x)
        beta: temperature parameter

    Returns:
        DPOOutput
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    logits = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(logits).mean()
    accuracy = (logits > 0).float().mean()

    return DPOOutput(
        loss=loss,
        chosen_rewards=chosen_rewards.detach(),
        rejected_rewards=rejected_rewards.detach(),
        accuracy=accuracy.detach(),
    )
