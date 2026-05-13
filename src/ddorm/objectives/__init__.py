"""
DDO-RM: Support-Restricted Entropy Policy Improvement.

Core formula:
    q_i* ∝ p_i · exp(r_i / λ)

where p_i = softmax(s_i / τ) is the policy restricted to C_K(x),
r_i = r(x, y_i) is the reward score, and λ is the entropy temperature.

The distillation loss is:
    L = -Σ_i q_i* log p_θ,i

with gradient:
    ∂L/∂s_θ(x,y_i) = (1/τ)(p_θ,i - q_i*)
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class DDORMOutput:
    """Output of DDO-RM target computation."""
    p: torch.Tensor           # [B, K] restricted policy
    q_target: torch.Tensor    # [B, K] DDO-RM target
    loss: torch.Tensor        # scalar distillation loss
    reward_weighted: torch.Tensor  # [B] expected reward under q*
    kl_pq: torch.Tensor      # [B] KL(q* || p)


def ddorm_target(
    policy_scores: torch.Tensor,
    reward_scores: torch.Tensor,
    lam: float,
    tau: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the DDO-RM target distribution.

    q_i* ∝ p_i · exp(r_i / λ)

    Numerically stable: work in log-space.
        log q_i = log p_i + r_i/λ - log Z

    Args:
        policy_scores: [B, K] raw policy scores s_i
        reward_scores: [B, K] reward scores r_i
        lam: entropy temperature λ > 0
        tau: policy temperature τ > 0

    Returns:
        p: [B, K] restricted policy softmax(s/τ)
        q: [B, K] DDO-RM target
    """
    # Restricted policy: p_i = softmax(s_i / τ)
    log_p = F.log_softmax(policy_scores / tau, dim=-1)

    # DDO-RM target in log-space: log q_i = log p_i + r_i/λ - log Z
    # Center rewards for numerical stability
    r_centered = reward_scores - reward_scores.mean(dim=-1, keepdim=True)
    log_q_unnorm = log_p + r_centered / lam
    log_q = log_q_unnorm - torch.logsumexp(log_q_unnorm, dim=-1, keepdim=True)

    p = log_p.exp()
    q = log_q.exp()
    return p, q


def ddorm_loss(
    policy_scores_new: torch.Tensor,
    q_target: torch.Tensor,
    tau: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    DDO-RM distillation loss: L = -Σ_i q_i* log p_θ,i

    Args:
        policy_scores_new: [B, K] updated policy scores
        q_target: [B, K] DDO-RM target (detached)
        tau: policy temperature
        reduction: "mean" or "sum" or "none"

    Returns:
        loss: scalar or [B] cross-entropy loss
    """
    log_p_new = F.log_softmax(policy_scores_new / tau, dim=-1)
    # Cross-entropy: -Σ_i q_i log p_i
    per_sample = -(q_target * log_p_new).sum(dim=-1)

    if reduction == "mean":
        return per_sample.mean()
    elif reduction == "sum":
        return per_sample.sum()
    return per_sample


def ddorm_forward(
    policy_scores: torch.Tensor,
    reward_scores: torch.Tensor,
    lam: float,
    tau: float = 1.0,
    policy_scores_new: Optional[torch.Tensor] = None,
) -> DDORMOutput:
    """
    Full DDO-RM forward pass: compute target and loss.

    If policy_scores_new is None, uses policy_scores for the loss
    (single-step, target computed from current policy).

    Args:
        policy_scores: [B, K] policy scores for target computation
        reward_scores: [B, K] reward scores
        lam: entropy temperature λ
        tau: policy temperature τ
        policy_scores_new: [B, K] optional updated policy scores for loss

    Returns:
        DDORMOutput with p, q_target, loss, reward_weighted, kl_pq
    """
    p, q_target = ddorm_target(policy_scores, reward_scores, lam, tau)

    if policy_scores_new is None:
        policy_scores_new = policy_scores

    loss = ddorm_loss(policy_scores_new, q_target.detach(), tau)

    # Diagnostics
    reward_weighted = (q_target * reward_scores).sum(dim=-1)
    log_p = F.log_softmax(policy_scores / tau, dim=-1)
    log_q = (q_target + 1e-12).log()
    kl_pq = (q_target * (log_q - log_p)).sum(dim=-1)

    return DDORMOutput(
        p=p,
        q_target=q_target,
        loss=loss,
        reward_weighted=reward_weighted,
        kl_pq=kl_pq,
    )


def ddorm_gradient(
    policy_scores: torch.Tensor,
    q_target: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Analytic DDO-RM gradient w.r.t. policy scores.

    ∂L/∂s_i = (1/τ)(p_θ,i - q_i*)

    Args:
        policy_scores: [B, K]
        q_target: [B, K]
        tau: float

    Returns:
        grad: [B, K] gradient w.r.t. policy scores
    """
    p = F.softmax(policy_scores / tau, dim=-1)
    return (p - q_target) / tau
