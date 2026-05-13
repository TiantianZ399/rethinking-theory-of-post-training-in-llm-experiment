"""
OPD: On-Policy Distillation.

Token-level teacher distillation on student-generated prefixes.
Supports top-K local support matching.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class OPDOutput:
    loss: torch.Tensor
    kl_divergence: torch.Tensor
    top_k_coverage: torch.Tensor


def opd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    top_k: Optional[int] = None,
    temperature: float = 1.0,
) -> OPDOutput:
    """
    On-policy distillation loss.

    Without top_k: standard KL divergence over full vocabulary.
    With top_k: restrict to teacher's top-K tokens at each position.

    Args:
        student_logits: [B, T, V] student logits
        teacher_logits: [B, T, V] teacher logits
        mask: [B, T] valid token mask
        top_k: if set, restrict to teacher's top-K tokens
        temperature: softmax temperature

    Returns:
        OPDOutput
    """
    if mask is None:
        mask = torch.ones(student_logits.shape[:2], device=student_logits.device)

    if top_k is not None:
        # Top-K local support matching
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        topk_vals, topk_idx = teacher_probs.topk(top_k, dim=-1)  # [B, T, K]

        # Gather student logits at those positions
        student_topk = student_logits.gather(-1, topk_idx)  # [B, T, K]
        teacher_topk = teacher_logits.gather(-1, topk_idx)  # [B, T, K]

        # Renormalize on the top-K support
        teacher_log_probs = F.log_softmax(teacher_topk / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_topk / temperature, dim=-1)

        # KL(teacher || student) on top-K support
        teacher_probs_k = teacher_log_probs.exp()
        kl = (teacher_probs_k * (teacher_log_probs - student_log_probs)).sum(dim=-1)

        # Coverage: mass of teacher in top-K
        coverage = topk_vals.sum(dim=-1).mean()
    else:
        # Full vocabulary KL
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

        teacher_probs_full = teacher_log_probs.exp()
        kl = (teacher_probs_full * (teacher_log_probs - student_log_probs)).sum(dim=-1)
        coverage = torch.ones(1, device=student_logits.device)

    # Apply mask and reduce
    loss = (kl * mask).sum() / mask.sum()
    kl_mean = (kl * mask).sum() / mask.sum()

    return OPDOutput(
        loss=loss,
        kl_divergence=kl_mean.detach(),
        top_k_coverage=coverage.detach(),
    )
