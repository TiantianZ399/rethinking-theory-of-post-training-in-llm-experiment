"""
Pairwise evaluation metrics.

Used for all experiments comparing methods on pairwise preference data.
Reports: pair accuracy, AUC, mean chosen-minus-rejected margin.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass
from typing import Optional


@dataclass
class PairwiseResult:
    """Pairwise evaluation results."""
    pair_accuracy: float
    auc: float
    mean_margin: float
    std_margin: float
    num_pairs: int
    margins: Optional[np.ndarray] = None


def pairwise_eval(
    chosen_scores: torch.Tensor,
    rejected_scores: torch.Tensor,
) -> PairwiseResult:
    """
    Evaluate pairwise preference accuracy.

    Args:
        chosen_scores: [N] scores for chosen responses
        rejected_scores: [N] scores for rejected responses

    Returns:
        PairwiseResult
    """
    chosen = chosen_scores.cpu().numpy()
    rejected = rejected_scores.cpu().numpy()
    margins = chosen - rejected

    pair_accuracy = (margins > 0).mean()

    # AUC: treat as binary classification
    scores = np.concatenate([chosen, rejected])
    labels = np.concatenate([np.ones(len(chosen)), np.zeros(len(rejected))])
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.5

    return PairwiseResult(
        pair_accuracy=float(pair_accuracy),
        auc=float(auc),
        mean_margin=float(margins.mean()),
        std_margin=float(margins.std()),
        num_pairs=len(margins),
        margins=margins,
    )


def pairwise_eval_from_model(
    model,
    tokenizer,
    prompts: list[str],
    chosen: list[str],
    rejected: list[str],
    score_fn=None,
) -> PairwiseResult:
    """
    Evaluate model on pairwise preferences.

    Args:
        model: policy model
        tokenizer: tokenizer
        prompts: prompt strings
        chosen: chosen response strings
        rejected: rejected response strings
        score_fn: optional custom scoring function

    Returns:
        PairwiseResult
    """
    if score_fn is not None:
        chosen_scores = torch.tensor([
            score_fn(model, tokenizer, p, c) for p, c in zip(prompts, chosen)
        ])
        rejected_scores = torch.tensor([
            score_fn(model, tokenizer, p, r) for p, r in zip(prompts, rejected)
        ])
    else:
        from ..models.policy import PolicyModel
        policy = PolicyModel.__new__(PolicyModel)
        policy.model = model
        policy.tokenizer = tokenizer

        chosen_scores = torch.tensor([
            policy.score_candidates(p, [c]).item()
            for p, c in zip(prompts, chosen)
        ])
        rejected_scores = torch.tensor([
            policy.score_candidates(p, [r]).item()
            for p, r in zip(prompts, rejected)
        ])

    return pairwise_eval(chosen_scores, rejected_scores)
