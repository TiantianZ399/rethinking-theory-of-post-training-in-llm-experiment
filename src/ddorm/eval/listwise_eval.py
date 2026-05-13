"""
Listwise evaluation metrics.

Used for multi-candidate experiments (K > 2).
Reports: NDCG@K, Kendall tau, Spearman rho, top-1 accuracy, expected reward, regret.
"""

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
from dataclasses import dataclass
from typing import Optional


@dataclass
class ListwiseResult:
    """Listwise evaluation results."""
    ndcg_at_k: float
    top1_accuracy: float
    kendall_tau: float
    spearman_rho: float
    expected_reward: float
    regret: float      # gap from best candidate
    num_prompts: int


def dcg(scores: np.ndarray) -> float:
    """Discounted cumulative gain."""
    return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))


def ndcg_at_k(predicted_ranking: np.ndarray, true_scores: np.ndarray, k: int = -1) -> float:
    """
    Compute NDCG@K.

    Args:
        predicted_ranking: indices in predicted order (best first)
        true_scores: ground-truth relevance scores
        k: cutoff (-1 = use all)
    """
    if k > 0:
        predicted_ranking = predicted_ranking[:k]

    ideal_order = np.argsort(-true_scores)
    ideal_scores = true_scores[ideal_order[:len(predicted_ranking)]]
    pred_scores = true_scores[predicted_ranking]

    ideal_dcg = dcg(ideal_scores)
    if ideal_dcg == 0:
        return 1.0
    return dcg(pred_scores) / ideal_dcg


def listwise_eval(
    predicted_scores: torch.Tensor,
    true_scores: torch.Tensor,
) -> ListwiseResult:
    """
    Evaluate listwise ranking quality.

    Args:
        predicted_scores: [B, K] model's predicted scores/probabilities
        true_scores: [B, K] ground-truth scores

    Returns:
        ListwiseResult
    """
    B, K = predicted_scores.shape
    pred = predicted_scores.cpu().numpy()
    true = true_scores.cpu().numpy()

    ndcgs = []
    top1_correct = 0
    taus = []
    rhos = []
    expected_rewards = []
    regrets = []

    for b in range(B):
        pred_ranking = np.argsort(-pred[b])
        true_ranking = np.argsort(-true[b])

        # NDCG
        ndcgs.append(ndcg_at_k(pred_ranking, true[b]))

        # Top-1 accuracy
        if pred_ranking[0] == true_ranking[0]:
            top1_correct += 1

        # Rank correlation
        if len(set(true[b])) > 1:
            tau, _ = kendalltau(pred[b], true[b])
            rho, _ = spearmanr(pred[b], true[b])
            taus.append(tau if not np.isnan(tau) else 0.0)
            rhos.append(rho if not np.isnan(rho) else 0.0)

        # Expected reward under predicted distribution (softmax of scores)
        p = np.exp(pred[b] - pred[b].max())
        p = p / p.sum()
        exp_r = (p * true[b]).sum()
        expected_rewards.append(exp_r)

        # Regret
        best_r = true[b].max()
        regrets.append(best_r - exp_r)

    return ListwiseResult(
        ndcg_at_k=float(np.mean(ndcgs)),
        top1_accuracy=float(top1_correct / B),
        kendall_tau=float(np.mean(taus)) if taus else 0.0,
        spearman_rho=float(np.mean(rhos)) if rhos else 0.0,
        expected_reward=float(np.mean(expected_rewards)),
        regret=float(np.mean(regrets)),
        num_prompts=B,
    )
