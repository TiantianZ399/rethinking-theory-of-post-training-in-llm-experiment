"""Coverage diagnostics for finite-support target matching."""
import numpy as np
from .targets import softmax, restricted_policy


def approximate_full_target(pool_policy_scores, pool_rewards, lam=1.0, tau=1.0):
    """Approximate q_lambda^* on a large finite reference pool.

    This approximates q*(y) ∝ pi(y) exp(r(y)/lambda) on P_M.
    """
    p_pool = restricted_policy(pool_policy_scores, tau=tau)
    log_q = np.log(np.maximum(p_pool, 1e-300)) + np.asarray(pool_rewards) / lam
    q_pool = softmax(log_q, axis=-1)
    return p_pool, q_pool


def support_coverage(q_pool, selected_indices):
    """Compute rho(C_K)=sum_{i in C_K} q_pool_i.

    Args:
        q_pool: array [B, M] or [M]
        selected_indices: list/array of selected indices, shape [K] or [B,K]
    """
    q = np.asarray(q_pool)
    idx = np.asarray(selected_indices)
    if q.ndim == 1:
        return float(np.sum(q[idx]))
    return np.array([np.sum(q[b, idx[b]]) for b in range(q.shape[0])])


def support_gap_from_coverage(rho, lam=1.0, eps=1e-12):
    """lambda log(1/rho)."""
    rho = np.clip(np.asarray(rho), eps, 1.0)
    return lam * np.log(1.0 / rho)
