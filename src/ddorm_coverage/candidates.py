"""Candidate-selection utilities for coverage diagnostics."""
import numpy as np


def random_support(M, K, rng=None):
    rng = np.random.default_rng(rng)
    return rng.choice(M, size=K, replace=False)


def top_reward_support(rewards, K):
    return np.argsort(rewards)[-K:][::-1]


def greedy_submodular_support(coverage_matrix, weights, K):
    """Greedy coverage selection.

    Args:
        coverage_matrix: bool/0-1 array [num_candidates, num_reference_modes]
        weights: nonnegative array [num_reference_modes]
        K: budget
    """
    C = []
    covered = np.zeros_like(weights, dtype=bool)
    for _ in range(K):
        best, best_gain = None, -np.inf
        for i in range(coverage_matrix.shape[0]):
            if i in C:
                continue
            new = np.logical_or(covered, coverage_matrix[i].astype(bool))
            gain = np.sum(weights[new]) - np.sum(weights[covered])
            if gain > best_gain:
                best, best_gain = i, gain
        C.append(best)
        covered = np.logical_or(covered, coverage_matrix[best].astype(bool))
    return np.array(C, dtype=int)
