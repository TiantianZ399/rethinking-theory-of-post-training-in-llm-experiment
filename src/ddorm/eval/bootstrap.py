"""
Bootstrap confidence intervals for experiment metrics.

Provides statistical significance testing across seeds and samples.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class BootstrapResult:
    """Bootstrap confidence interval result."""
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_bootstrap: int


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of observations
        statistic: function to compute (default: mean)
        n_bootstrap: number of bootstrap samples
        ci_level: confidence level (e.g., 0.95)
        seed: random seed

    Returns:
        BootstrapResult
    """
    rng = np.random.RandomState(seed)
    n = len(data)

    bootstrap_stats = np.array([
        statistic(data[rng.choice(n, n, replace=True)])
        for _ in range(n_bootstrap)
    ])

    alpha = (1 - ci_level) / 2
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha))

    return BootstrapResult(
        mean=float(statistic(data)),
        std=float(bootstrap_stats.std()),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
    )


def multi_seed_summary(
    results_by_seed: dict[int, dict],
    metric_keys: list[str],
) -> dict[str, BootstrapResult]:
    """
    Summarize results across multiple seeds.

    Args:
        results_by_seed: {seed: {metric: value, ...}, ...}
        metric_keys: which metrics to summarize

    Returns:
        dict mapping metric name to BootstrapResult
    """
    summaries = {}
    for key in metric_keys:
        values = np.array([
            results_by_seed[seed][key]
            for seed in results_by_seed
            if key in results_by_seed[seed]
        ])
        if len(values) > 0:
            summaries[key] = bootstrap_ci(values)
    return summaries


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap test: is method A better than method B?

    Args:
        scores_a: per-sample scores for method A
        scores_b: per-sample scores for method B

    Returns:
        dict with p_value, mean_diff, ci
    """
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    assert len(scores_b) == n

    diffs = scores_a - scores_b
    observed_diff = diffs.mean()

    bootstrap_diffs = np.array([
        diffs[rng.choice(n, n, replace=True)].mean()
        for _ in range(n_bootstrap)
    ])

    # Two-sided p-value
    p_value = (np.abs(bootstrap_diffs) >= np.abs(observed_diff)).mean()

    return {
        "mean_diff": float(observed_diff),
        "p_value": float(p_value),
        "ci_lower": float(np.percentile(bootstrap_diffs, 2.5)),
        "ci_upper": float(np.percentile(bootstrap_diffs, 97.5)),
        "significant_at_05": bool(p_value < 0.05),
    }
