"""
Advanced candidate generators:
- Reward-diverse sampling: sample candidates that are diverse in reward-relevant space
- Submodular coverage selection: greedy selection maximizing surrogate coverage
"""

import torch
import torch.nn.functional as F
import numpy as np
from . import CandidateGenerator, CandidateOutput
from typing import Optional


class RewardDiverseGenerator(CandidateGenerator):
    """
    Generate a large pool, then select K candidates that are
    diverse in a reward-relevant embedding space.

    Uses DPP-style diversity or simple k-medoids.
    """

    def __init__(
        self,
        pool_size: int = 64,
        temperature: float = 1.0,
        diversity_weight: float = 0.5,
    ):
        self.pool_size = pool_size
        self.temperature = temperature
        self.diversity_weight = diversity_weight

    def generate(self, prompts, model, tokenizer, K, max_length=256, **kwargs):
        """Generate a large pool then select diverse subset."""
        reward_model = kwargs.get("reward_model", None)
        embedding_fn = kwargs.get("embedding_fn", None)

        # Step 1: Generate large pool using temperature sampling
        from .sampling import TemperatureSamplingGenerator
        pool_gen = TemperatureSamplingGenerator(self.temperature)
        pool = pool_gen.generate(prompts, model, tokenizer, self.pool_size, max_length)

        if reward_model is None or embedding_fn is None:
            # Fallback: random selection
            B = len(prompts)
            indices = [torch.randperm(self.pool_size)[:K] for _ in range(B)]
            selected_candidates = [
                [pool.candidates[b][i] for i in idx] for b, idx in enumerate(indices)
            ]
            return CandidateOutput(
                candidates=selected_candidates,
                candidate_ids=pool.candidate_ids[:, :K],
                gen_logprobs=pool.gen_logprobs[:, :K] if pool.gen_logprobs is not None else None,
                metadata={"generator": "reward_diverse", "selected": "random_fallback"},
            )

        # Step 2: Get embeddings and rewards for pool
        all_selected = []
        for b in range(len(prompts)):
            cands = pool.candidates[b]
            embeddings = embedding_fn(prompts[b], cands)  # [M, D]
            rewards = reward_model.score([prompts[b]] * len(cands), [[c] for c in cands]).squeeze()

            # Step 3: Greedy diverse selection
            selected = self._greedy_diverse_select(
                embeddings, rewards, K, self.diversity_weight
            )
            all_selected.append(selected)

        selected_candidates = [
            [pool.candidates[b][i] for i in sel] for b, sel in enumerate(all_selected)
        ]

        return CandidateOutput(
            candidates=selected_candidates,
            candidate_ids=pool.candidate_ids[:, :K],  # approximate
            gen_logprobs=None,
            metadata={"generator": "reward_diverse", "pool_size": self.pool_size},
        )

    @staticmethod
    def _greedy_diverse_select(
        embeddings: torch.Tensor,
        rewards: torch.Tensor,
        K: int,
        diversity_weight: float,
    ) -> list[int]:
        """
        Greedy selection balancing reward and diversity.

        score(c | S) = (1-w) · r(c) + w · min_{s in S} d(c, s)
        """
        M = embeddings.shape[0]
        # Normalize
        emb_norm = F.normalize(embeddings, dim=-1)
        dist_matrix = 1 - emb_norm @ emb_norm.T  # cosine distance

        r_norm = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)

        selected = []
        # First: pick highest reward
        selected.append(rewards.argmax().item())

        for _ in range(K - 1):
            best_score = -float("inf")
            best_idx = -1
            for c in range(M):
                if c in selected:
                    continue
                min_dist = min(dist_matrix[c, s].item() for s in selected)
                score = (1 - diversity_weight) * r_norm[c].item() + diversity_weight * min_dist
                if score > best_score:
                    best_score = score
                    best_idx = c
            selected.append(best_idx)

        return selected


class SubmodularCoverageSelector(CandidateGenerator):
    """
    Greedy submodular coverage selection.

    Given a reference pool with weights w_m, select K candidates
    to maximize:
        f(C) = Σ_m w_m · 1{∃ c ∈ C : d(c, z_m) ≤ ε}

    This is monotone submodular → greedy gives (1-1/e) approximation.
    """

    def __init__(
        self,
        pool_size: int = 64,
        epsilon: float = 0.5,
        temperature: float = 1.0,
    ):
        self.pool_size = pool_size
        self.epsilon = epsilon
        self.temperature = temperature

    def generate(self, prompts, model, tokenizer, K, max_length=256, **kwargs):
        """Generate pool, compute coverage weights, greedy select."""
        reward_model = kwargs.get("reward_model", None)
        embedding_fn = kwargs.get("embedding_fn", None)

        from .sampling import TemperatureSamplingGenerator
        pool_gen = TemperatureSamplingGenerator(self.temperature)
        pool = pool_gen.generate(prompts, model, tokenizer, self.pool_size, max_length)

        if reward_model is None or embedding_fn is None:
            # Random fallback
            selected_candidates = [cands[:K] for cands in pool.candidates]
            return CandidateOutput(
                candidates=selected_candidates,
                candidate_ids=pool.candidate_ids[:, :K],
                gen_logprobs=None,
                metadata={"generator": "submodular", "selected": "random_fallback"},
            )

        all_selected = []
        for b in range(len(prompts)):
            cands = pool.candidates[b]
            embeddings = embedding_fn(prompts[b], cands)  # [M, D]

            # Compute DDO-RM weights as surrogate for q*
            if pool.gen_logprobs is not None:
                log_p = F.log_softmax(pool.gen_logprobs[b], dim=-1)
            else:
                log_p = torch.zeros(len(cands), device=embeddings.device) - np.log(len(cands))

            rewards = reward_model.score(
                [prompts[b]] * len(cands), [[c] for c in cands]
            ).squeeze()
            lam = kwargs.get("lam", 1.0)
            log_w = log_p + rewards / lam
            weights = F.softmax(log_w, dim=-1)

            selected = self._greedy_submodular_select(
                embeddings, weights, K, self.epsilon
            )
            all_selected.append(selected)

        selected_candidates = [
            [pool.candidates[b][i] for i in sel] for b, sel in enumerate(all_selected)
        ]

        return CandidateOutput(
            candidates=selected_candidates,
            candidate_ids=pool.candidate_ids[:, :K],
            gen_logprobs=None,
            metadata={"generator": "submodular", "pool_size": self.pool_size, "epsilon": self.epsilon},
        )

    @staticmethod
    def _greedy_submodular_select(
        embeddings: torch.Tensor,
        weights: torch.Tensor,
        K: int,
        epsilon: float,
    ) -> list[int]:
        """
        Greedy submodular coverage maximization.

        f(C) = Σ_m w_m · 1{∃ c ∈ C : d(φ(c), φ(z_m)) ≤ ε}

        Args:
            embeddings: [M, D] embeddings of pool items
            weights: [M] coverage weights (approximating q*)
            K: number to select
            epsilon: coverage radius

        Returns:
            selected indices
        """
        M = embeddings.shape[0]
        emb_norm = F.normalize(embeddings, dim=-1)
        sim_matrix = emb_norm @ emb_norm.T  # [M, M] cosine similarity
        cover_matrix = (sim_matrix >= (1 - epsilon)).float()  # [M, M]

        covered = torch.zeros(M, device=embeddings.device)
        selected = []

        for _ in range(K):
            # Marginal gain: weight of newly covered points
            marginal = torch.zeros(M, device=embeddings.device)
            for c in range(M):
                if c in selected:
                    marginal[c] = -1
                    continue
                newly_covered = cover_matrix[c] * (1 - covered)
                marginal[c] = (weights * newly_covered).sum()

            best = marginal.argmax().item()
            selected.append(best)
            covered = torch.clamp(covered + cover_matrix[best], max=1.0)

        return selected
