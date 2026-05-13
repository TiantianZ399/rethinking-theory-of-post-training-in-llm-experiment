"""
Load UltraFeedback datasets for DDO-RM experiments.

Supports:
- UltraFeedback Binarized (pairwise chosen/rejected)
- Original UltraFeedback (4 candidates with ratings)
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Optional


class UltraFeedbackBinarized(Dataset):
    """
    UltraFeedback Binarized: pairwise chosen/rejected format.

    Each item has: prompt, chosen response, rejected response.
    """

    def __init__(
        self,
        split: str = "train_prefs",
        tokenizer=None,
        max_prompt_length: int = 256,
        max_response_length: int = 256,
    ):
        self.dataset = load_dataset(
            "HuggingFaceH4/ultrafeedback_binarized",
            split=split,
        )
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = item["prompt"] if "prompt" in item else item["chosen"][0]["content"]

        if isinstance(item["chosen"], list):
            chosen = item["chosen"][-1]["content"]
            rejected = item["rejected"][-1]["content"]
        else:
            chosen = item["chosen"]
            rejected = item["rejected"]

        result = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

        if self.tokenizer is not None:
            for key in ["prompt", "chosen", "rejected"]:
                max_len = (
                    self.max_prompt_length if key == "prompt"
                    else self.max_response_length
                )
                encoded = self.tokenizer(
                    result[key],
                    max_length=max_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                result[f"{key}_ids"] = encoded["input_ids"].squeeze(0)
                result[f"{key}_mask"] = encoded["attention_mask"].squeeze(0)

        return result


class UltraFeedbackMultiCandidate(Dataset):
    """
    Original UltraFeedback: 4 candidates with ratings per prompt.

    Native multi-candidate format for DDO-RM.
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer=None,
        max_prompt_length: int = 256,
        max_response_length: int = 256,
        num_candidates: int = 4,
    ):
        self.dataset = load_dataset("openbmb/UltraFeedback", split=split)
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.num_candidates = num_candidates

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = item["instruction"]
        completions = item["completions"]

        candidates = []
        ratings = []
        for comp in completions[:self.num_candidates]:
            candidates.append(comp["response"])
            rating = comp.get("overall_score", comp.get("rating", 0))
            ratings.append(float(rating))

        # Pad if fewer candidates
        while len(candidates) < self.num_candidates:
            candidates.append("")
            ratings.append(0.0)

        result = {
            "prompt": prompt,
            "candidates": candidates,
            "ratings": torch.tensor(ratings, dtype=torch.float32),
        }

        if self.tokenizer is not None:
            prompt_enc = self.tokenizer(
                prompt,
                max_length=self.max_prompt_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            result["prompt_ids"] = prompt_enc["input_ids"].squeeze(0)

            cand_ids = []
            cand_masks = []
            for c in candidates:
                enc = self.tokenizer(
                    c,
                    max_length=self.max_response_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                cand_ids.append(enc["input_ids"].squeeze(0))
                cand_masks.append(enc["attention_mask"].squeeze(0))

            result["candidate_ids"] = torch.stack(cand_ids)
            result["candidate_masks"] = torch.stack(cand_masks)

        return result
