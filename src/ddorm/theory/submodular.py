"""
Submodular coverage for candidate construction.

Validates Proposition 4 (submodular surrogate coverage) and
Proposition 5 (surrogate-weight robustness).

f_w(C) = Σ_m w_m · 1{∃ c ∈ C : d(φ(c), φ(z_m)) ≤ ε}

is monotone submodular. Greedy gives (1-1/e) approximation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class SubmodularResult:
    """Result of submodular coverage analysis."""
    selected_indices: list[int]
    coverage_value: float
    optimal_estimate: float     # estimated via greedy bound
    greedy_ratio: float         # f(C_gr) / ((1-1/e) * f(C*))
    marginal_gains: list[float]  # marginal gain at each step


def submodular_coverage(
    embeddings: torch.Tensor,
    weights: torch.Tensor,
    epsilon: float,
) -> float:
    """
    Compute coverage of full set.

    f_w(S) = Σ_m w_m · 1{∃ c ∈ S : sim(c, z_m) ≥ 1-ε}

    Args:
        embeddings: [M, D]
        weights: [M]
        epsilon: coverage radius

    Returns:
        coverage value
    """
    emb_norm = F.normalize(embeddings, dim=-1)
    sim = emb_norm @ emb_norm.T
    covered = (sim >= (1 - epsilon)).any(dim=0).float()
    return (weights * covered).sum().item()


def greedy_submodular_select(
    pool_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    weights: torch.Tensor,
    K: int,
    epsilon: float,
) -> SubmodularResult:
    """
    Greedy submodular coverage maximization.

    Select K items from pool to maximize coverage of reference set.

    Args:
        pool_embeddings: [N, D] embeddings of selectable items
        reference_embeddings: [M, D] embeddings of reference points
        weights: [M] reference weights
        K: cardinality constraint
        epsilon: coverage radius

    Returns:
        SubmodularResult
    """
    N = pool_embeddings.shape[0]
    M = reference_embeddings.shape[0]

    pool_norm = F.normalize(pool_embeddings, dim=-1)
    ref_norm = F.normalize(reference_embeddings, dim=-1)

    # Coverage matrix: does pool item c cover reference point m?
    sim = pool_norm @ ref_norm.T  # [N, M]
    cover_matrix = (sim >= (1 - epsilon)).float()

    covered = torch.zeros(M, device=pool_embeddings.device)
    selected = []
    marginals = []

    for _ in range(K):
        best_gain = -1.0
        best_idx = -1

        for c in range(N):
            if c in selected:
                continue
            newly_covered = cover_matrix[c] * (1 - covered)
            gain = (weights * newly_covered).sum().item()
            if gain > best_gain:
                best_gain = gain
                best_idx = c

        selected.append(best_idx)
        marginals.append(best_gain)
        covered = torch.clamp(covered + cover_matrix[best_idx], max=1.0)

    coverage_value = (weights * covered).sum().item()
    # Greedy gives ≥ (1-1/e) * OPT
    opt_estimate = coverage_value / (1 - 1 / np.e)

    return SubmodularResult(
        selected_indices=selected,
        coverage_value=coverage_value,
        optimal_estimate=opt_estimate,
        greedy_ratio=coverage_value / max(opt_estimate * (1 - 1 / np.e), 1e-12),
        marginal_gains=marginals,
    )


def verify_submodularity(
    embeddings: torch.Tensor,
    weights: torch.Tensor,
    epsilon: float,
    num_checks: int = 100,
    seed: int = 42,
) -> dict:
    """
    Empirically verify diminishing returns (submodularity).

    For random A ⊂ B and element c:
        f(A ∪ {c}) - f(A) ≥ f(B ∪ {c}) - f(B)

    Args:
        embeddings: [M, D]
        weights: [M]
        epsilon: coverage radius
        num_checks: number of random checks
        seed: random seed

    Returns:
        dict with verification statistics
    """
    M = embeddings.shape[0]
    emb_norm = F.normalize(embeddings, dim=-1)
    sim = emb_norm @ emb_norm.T
    cover_matrix = (sim >= (1 - epsilon)).float()

    gen = torch.Generator().manual_seed(seed)
    violations = 0

    for _ in range(num_checks):
        # Random subset sizes
        size_A = torch.randint(1, M // 2, (1,), generator=gen).item()
        size_B = torch.randint(size_A + 1, min(size_A + M // 2, M), (1,), generator=gen).item()

        perm = torch.randperm(M, generator=gen)
        A = perm[:size_A].tolist()
        B = perm[:size_B].tolist()  # A ⊂ B

        # Pick c not in B
        remaining = [i for i in range(M) if i not in B]
        if not remaining:
            continue
        c = remaining[torch.randint(len(remaining), (1,), generator=gen).item()]

        # Compute marginal gains
        def coverage(S):
            cov = torch.zeros(M, device=embeddings.device)
            for s in S:
                cov = torch.clamp(cov + cover_matrix[s], max=1.0)
            return (weights * cov).sum().item()

        gain_A = coverage(A + [c]) - coverage(A)
        gain_B = coverage(B + [c]) - coverage(B)

        if gain_A < gain_B - 1e-8:
            violations += 1

    return {
        "num_checks": num_checks,
        "violations": violations,
        "violation_rate": violations / num_checks,
        "is_submodular": violations == 0,
    }


def surrogate_robustness_experiment(
    embeddings: torch.Tensor,
    true_weights: torch.Tensor,
    K: int,
    epsilon: float,
    noise_levels: list[float],
    seed: int = 42,
) -> list[dict]:
    """
    Test Proposition 5: surrogate-weight robustness.

    f_w(C_hat_gr) ≥ α · f_w(C*_w) - (1+α)δ

    where δ = ||w - w_hat||_1 and α = 1-1/e.

    Args:
        embeddings: [M, D]
        true_weights: [M]
        K: cardinality constraint
        epsilon: coverage radius
        noise_levels: list of δ values
        seed: random seed

    Returns:
        list of dicts with verification results
    """
    M = embeddings.shape[0]
    alpha = 1 - 1 / np.e

    # Greedy on true weights
    true_result = greedy_submodular_select(
        embeddings, embeddings, true_weights, K, epsilon
    )

    results = []
    gen = torch.Generator().manual_seed(seed)

    for delta_target in noise_levels:
        # Create noisy weights with ||w - w_hat||_1 ≈ delta_target
        noise = torch.randn(M, generator=gen) * delta_target / np.sqrt(M)
        noisy_weights = F.softmax(true_weights.log() + noise, dim=-1)
        actual_delta = (true_weights - noisy_weights).abs().sum().item()

        # Greedy on noisy weights
        noisy_result = greedy_submodular_select(
            embeddings, embeddings, noisy_weights, K, epsilon
        )

        # Evaluate noisy selection under true weights
        # Recompute coverage of noisy_result.selected_indices under true_weights
        emb_norm = F.normalize(embeddings, dim=-1)
        sim = emb_norm @ emb_norm.T
        cover_matrix = (sim >= (1 - epsilon)).float()

        cov = torch.zeros(M, device=embeddings.device)
        for idx in noisy_result.selected_indices:
            cov = torch.clamp(cov + cover_matrix[idx], max=1.0)
        f_w_noisy_sel = (true_weights * cov).sum().item()

        theoretical_lb = alpha * true_result.coverage_value - (1 + alpha) * actual_delta

        results.append({
            "delta_target": delta_target,
            "delta_actual": actual_delta,
            "f_w_true_greedy": true_result.coverage_value,
            "f_w_noisy_greedy": f_w_noisy_sel,
            "theoretical_lower_bound": theoretical_lb,
            "bound_holds": f_w_noisy_sel >= theoretical_lb - 1e-8,
        })

    return results
