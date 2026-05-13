"""
Base candidate generator interface.

All generators produce C_K(x) = {y_1, ..., y_K} for a given prompt x.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class CandidateOutput:
    """Output of candidate generation."""
    candidates: list[list[str]]       # [B][K] candidate strings
    candidate_ids: torch.Tensor       # [B, K, L] token ids (padded)
    gen_logprobs: Optional[torch.Tensor]  # [B, K] total log-prob of each candidate
    metadata: dict


class CandidateGenerator(ABC):
    """Abstract candidate generator."""

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        model,
        tokenizer,
        K: int,
        max_length: int = 256,
        **kwargs,
    ) -> CandidateOutput:
        """
        Generate K candidates for each prompt.

        Args:
            prompts: list of prompt strings
            model: language model
            tokenizer: tokenizer
            K: number of candidates per prompt
            max_length: max response length

        Returns:
            CandidateOutput
        """
        ...
