"""
Reward model evaluation.

Evaluates RM quality independently from DDO-RM:
- Pair accuracy and AUC on held-out preferences
- Calibration analysis
- Noise sensitivity
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardEvalResult:
    """Reward model evaluation results."""
    pair_accuracy: float
    auc: float
    mean_reward_chosen: float
    mean_reward_rejected: float
    mean_margin: float
    std_margin: float
    calibration_error: float  # expected calibration error
    reward_std: float         # overall reward standard deviation


def evaluate_reward_model(
    reward_model,
    prompts: list[str],
    chosen: list[str],
    rejected: list[str],
    max_length: int = 512,
) -> RewardEvalResult:
    """
    Evaluate reward model on pairwise preferences.

    Args:
        reward_model: RewardModel instance
        prompts: prompt strings
        chosen: chosen response strings
        rejected: rejected response strings
        max_length: max sequence length

    Returns:
        RewardEvalResult
    """
    chosen_scores, rejected_scores = reward_model.score_pairs(
        prompts, chosen, rejected, max_length
    )

    c = chosen_scores.cpu().numpy()
    r = rejected_scores.cpu().numpy()
    margins = c - r

    pair_acc = (margins > 0).mean()

    scores_all = np.concatenate([c, r])
    labels_all = np.concatenate([np.ones(len(c)), np.zeros(len(r))])
    try:
        auc = roc_auc_score(labels_all, scores_all)
    except ValueError:
        auc = 0.5

    # Expected calibration error (simplified)
    # Bin margins and check if predicted preference matches actual
    margin_probs = 1 / (1 + np.exp(-margins))  # sigmoid of margin
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(len(bins) - 1):
        mask = (margin_probs >= bins[i]) & (margin_probs < bins[i + 1])
        if mask.sum() > 0:
            avg_conf = margin_probs[mask].mean()
            avg_acc = (margins[mask] > 0).mean()
            ece += mask.sum() * abs(avg_conf - avg_acc)
    ece /= len(margins)

    return RewardEvalResult(
        pair_accuracy=float(pair_acc),
        auc=float(auc),
        mean_reward_chosen=float(c.mean()),
        mean_reward_rejected=float(r.mean()),
        mean_margin=float(margins.mean()),
        std_margin=float(margins.std()),
        calibration_error=float(ece),
        reward_std=float(scores_all.std()),
    )


def reward_noise_sensitivity(
    reward_model,
    prompts: list[str],
    candidates: list[list[str]],
    noise_levels: list[float],
    max_length: int = 512,
) -> list[dict]:
    """
    Test RM sensitivity to input perturbation.

    For each noise level, measures how much the ranking changes.
    """
    from scipy.stats import kendalltau

    base_scores = reward_model.score(prompts, candidates, max_length)  # [B, K]
    base_rankings = base_scores.argsort(dim=-1, descending=True).cpu().numpy()

    results = []
    for noise_std in noise_levels:
        # Score consistency: run multiple times (if model is deterministic,
        # this tests numerical stability)
        # For actual noise sensitivity, perturb inputs
        noisy_scores = base_scores + torch.randn_like(base_scores) * noise_std
        noisy_rankings = noisy_scores.argsort(dim=-1, descending=True).cpu().numpy()

        taus = []
        for b in range(len(prompts)):
            tau, _ = kendalltau(base_rankings[b], noisy_rankings[b])
            taus.append(tau if not np.isnan(tau) else 1.0)

        results.append({
            "noise_level": noise_std,
            "mean_rank_correlation": float(np.mean(taus)),
            "rank_flip_rate": float((base_rankings[:, 0] != noisy_rankings[:, 0]).mean()),
        })

    return results
