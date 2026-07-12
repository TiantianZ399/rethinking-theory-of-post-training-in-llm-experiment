"""Core DDO-RM target-matching utilities.

These routines are deliberately framework-light. They use NumPy so that the
math can be tested on CPU. Production trainers should implement the same
functions in PyTorch/JAX.
"""
import numpy as np


def logsumexp(x, axis=-1, keepdims=False):
    m = np.max(x, axis=axis, keepdims=True)
    z = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return z if keepdims else np.squeeze(z, axis=axis)


def softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def restricted_policy(policy_scores, tau=1.0):
    """Compute p_C from candidate policy scores.

    Args:
        policy_scores: array [..., K]
        tau: candidate softmax temperature
    """
    return softmax(np.asarray(policy_scores, dtype=float) / tau, axis=-1)


def ddo_target(policy_scores, reward_scores, lam=1.0, tau=1.0, center=True):
    """Compute q_i ∝ p_i exp(r_i / lambda) on a finite support.

    Returns:
        p: restricted current policy
        q: support-restricted entropy-improved target
    """
    scores = np.asarray(policy_scores, dtype=float)
    rewards = np.asarray(reward_scores, dtype=float)
    p = restricted_policy(scores, tau=tau)
    r = rewards.copy()
    if center:
        r = r - np.sum(p * r, axis=-1, keepdims=True)
    log_q_unnorm = np.log(np.maximum(p, 1e-300)) + r / lam
    q = softmax(log_q_unnorm, axis=-1)
    return p, q


def ddo_cross_entropy(policy_scores_new, q_target, tau=1.0):
    """Cross-entropy -sum_i q_i log p_theta,i."""
    p_new = restricted_policy(policy_scores_new, tau=tau)
    return -np.sum(q_target * np.log(np.maximum(p_new, 1e-300)), axis=-1)


def score_gradient(policy_scores_new, q_target, tau=1.0):
    """Gradient d CE(q, p_theta) / d score_i = (p_i - q_i) / tau."""
    p_new = restricted_policy(policy_scores_new, tau=tau)
    return (p_new - q_target) / tau
