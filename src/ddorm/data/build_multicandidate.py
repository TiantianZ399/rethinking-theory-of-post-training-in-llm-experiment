"""
Build multi-candidate datasets for DDO-RM.

Wraps prompt-only datasets by generating candidates on-the-fly
or from cached pools.
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional


class MultiCandidateDataset(Dataset):
    """
    Prompt dataset augmented with K candidates per prompt.

    Candidates can be:
    - Pre-generated and cached
    - Generated on-the-fly during training (via candidate_generator)
    - From a dataset that already has multiple completions
    """

    def __init__(
        self,
        prompts: list[str],
        candidates: Optional[list[list[str]]] = None,
        rewards: Optional[list[list[float]]] = None,
    ):
        self.prompts = prompts
        self.candidates = candidates
        self.rewards = rewards

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        item = {"prompt": self.prompts[idx]}
        if self.candidates is not None:
            item["candidates"] = self.candidates[idx]
        if self.rewards is not None:
            item["ratings"] = torch.tensor(self.rewards[idx], dtype=torch.float32)
        return item

    @classmethod
    def from_cached(cls, cache_path: str) -> "MultiCandidateDataset":
        """Load from a JSON cache file."""
        with open(cache_path) as f:
            data = json.load(f)
        prompts = [d["prompt"] for d in data]
        candidates = [d["candidates"] for d in data]
        rewards = [d.get("rewards", [0.0] * len(d["candidates"])) for d in data]
        return cls(prompts, candidates, rewards)

    def save_cache(self, cache_path: str):
        """Save to JSON cache."""
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for i in range(len(self)):
            item = {"prompt": self.prompts[i]}
            if self.candidates is not None:
                item["candidates"] = self.candidates[i]
            if self.rewards is not None:
                item["rewards"] = self.rewards[i]
            data.append(item)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def generate_candidate_pool(
    prompts: list[str],
    candidate_generator,
    model,
    tokenizer,
    K: int,
    reward_model=None,
    cache_path: Optional[str] = None,
) -> MultiCandidateDataset:
    """
    Generate a candidate pool for a list of prompts.

    Args:
        prompts: list of prompt strings
        candidate_generator: CandidateGenerator instance
        model: language model for generation
        tokenizer: tokenizer
        K: candidates per prompt
        reward_model: optional RM for scoring
        cache_path: optional path to save cache

    Returns:
        MultiCandidateDataset
    """
    gen_output = candidate_generator.generate(prompts, model, tokenizer, K)
    candidates = gen_output.candidates

    rewards = None
    if reward_model is not None:
        rewards = []
        for prompt, cands in zip(prompts, candidates):
            scores = reward_model.score([prompt], [cands])
            rewards.append(scores[0].cpu().tolist())

    dataset = MultiCandidateDataset(prompts, candidates, rewards)

    if cache_path is not None:
        dataset.save_cache(cache_path)

    return dataset
