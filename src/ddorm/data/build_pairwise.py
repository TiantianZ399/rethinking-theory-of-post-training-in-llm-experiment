"""
Build pairwise preference data for DPO and evaluation.

Converts multi-candidate data into chosen/rejected pairs,
optionally using reward-model scores for pair construction.
"""

import torch
from torch.utils.data import Dataset
from typing import Optional
import random


class PairwiseFromMultiCandidate(Dataset):
    """
    Convert multi-candidate dataset into pairwise preference pairs.

    Strategies:
    - "best_worst": pair highest-rated vs lowest-rated
    - "adjacent": pair adjacent ratings
    - "all_pairs": enumerate all (chosen, rejected) pairs with r_c > r_r
    - "rm_scored": use reward model to construct pairs
    """

    def __init__(
        self,
        multi_candidate_dataset,
        strategy: str = "best_worst",
        reward_model=None,
        margin_threshold: float = 0.0,
    ):
        self.base = multi_candidate_dataset
        self.strategy = strategy
        self.reward_model = reward_model
        self.margin_threshold = margin_threshold
        self.pairs = self._build_pairs()

    def _build_pairs(self) -> list[dict]:
        pairs = []
        for idx in range(len(self.base)):
            item = self.base[idx]
            prompt = item["prompt"]
            candidates = item["candidates"]
            ratings = item["ratings"]

            if self.strategy == "best_worst":
                best_idx = ratings.argmax().item()
                worst_idx = ratings.argmin().item()
                if ratings[best_idx] > ratings[worst_idx] + self.margin_threshold:
                    pairs.append({
                        "prompt": prompt,
                        "chosen": candidates[best_idx],
                        "rejected": candidates[worst_idx],
                        "margin": float(ratings[best_idx] - ratings[worst_idx]),
                    })

            elif self.strategy == "adjacent":
                sorted_idx = ratings.argsort(descending=True)
                for i in range(len(sorted_idx) - 1):
                    ci, ri = sorted_idx[i].item(), sorted_idx[i + 1].item()
                    if ratings[ci] > ratings[ri] + self.margin_threshold:
                        pairs.append({
                            "prompt": prompt,
                            "chosen": candidates[ci],
                            "rejected": candidates[ri],
                            "margin": float(ratings[ci] - ratings[ri]),
                        })

            elif self.strategy == "all_pairs":
                K = len(candidates)
                for i in range(K):
                    for j in range(K):
                        if ratings[i] > ratings[j] + self.margin_threshold:
                            pairs.append({
                                "prompt": prompt,
                                "chosen": candidates[i],
                                "rejected": candidates[j],
                                "margin": float(ratings[i] - ratings[j]),
                            })

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class PairwiseDataset(Dataset):
    """Generic pairwise preference dataset wrapper."""

    def __init__(self, data: list[dict]):
        """
        Args:
            data: list of dicts with keys 'prompt', 'chosen', 'rejected'
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @classmethod
    def from_lists(
        cls,
        prompts: list[str],
        chosen: list[str],
        rejected: list[str],
    ) -> "PairwiseDataset":
        data = [
            {"prompt": p, "chosen": c, "rejected": r}
            for p, c, r in zip(prompts, chosen, rejected)
        ]
        return cls(data)

    def split(self, test_ratio: float = 0.1, seed: int = 42):
        """Split into train/test."""
        rng = random.Random(seed)
        indices = list(range(len(self.data)))
        rng.shuffle(indices)
        split_idx = int(len(indices) * (1 - test_ratio))
        train_data = [self.data[i] for i in indices[:split_idx]]
        test_data = [self.data[i] for i in indices[split_idx:]]
        return PairwiseDataset(train_data), PairwiseDataset(test_data)
