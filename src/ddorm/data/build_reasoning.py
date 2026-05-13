"""
Build reasoning datasets for GRPO / DDO-RM reasoning experiments.

Supports GSM8K and similar math reasoning benchmarks.
Provides verifier reward functions.
"""

import re
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Optional, Callable


class GSM8KDataset(Dataset):
    """
    GSM8K math reasoning dataset.

    Each item has a question and a numerical answer.
    The verifier checks if the generated response contains the correct answer.
    """

    def __init__(self, split: str = "train", max_samples: Optional[int] = None):
        self.dataset = load_dataset("openai/gsm8k", "main", split=split)
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "prompt": item["question"],
            "answer": self._extract_answer(item["answer"]),
            "full_solution": item["answer"],
        }

    @staticmethod
    def _extract_answer(solution: str) -> str:
        """Extract the final numerical answer from GSM8K solution."""
        match = re.search(r"####\s*(.+)", solution)
        if match:
            return match.group(1).strip().replace(",", "")
        return ""


def gsm8k_verifier(prompt: str, response: str, answer: str) -> float:
    """
    Binary verifier for GSM8K: 1.0 if response contains correct answer, else 0.0.

    Args:
        prompt: question
        response: model-generated response
        answer: ground-truth answer string

    Returns:
        reward: 0.0 or 1.0
    """
    # Try to find the answer in the response
    # Look for #### pattern first
    match = re.search(r"####\s*(.+?)(?:\s|$)", response)
    if match:
        extracted = match.group(1).strip().replace(",", "")
        return 1.0 if extracted == answer else 0.0

    # Look for "the answer is" pattern
    match = re.search(r"(?:answer|result)\s*(?:is|=)\s*(.+?)(?:\.|,|\s|$)", response, re.I)
    if match:
        extracted = match.group(1).strip().replace(",", "").replace("$", "")
        return 1.0 if extracted == answer else 0.0

    # Check if answer appears anywhere in the last line
    last_line = response.strip().split("\n")[-1]
    answer_clean = answer.replace(",", "").replace("$", "")
    return 1.0 if answer_clean in last_line else 0.0


def continuous_verifier(prompt: str, response: str, answer: str) -> float:
    """
    Soft verifier: rewards partial progress toward correct answer.

    Heuristic scoring:
    - 1.0: correct final answer
    - 0.5: contains intermediate computation
    - 0.2: has some mathematical content
    - 0.0: no relevant content
    """
    if gsm8k_verifier(prompt, response, answer) == 1.0:
        return 1.0

    # Check for mathematical content
    has_numbers = bool(re.search(r"\d+", response))
    has_operators = bool(re.search(r"[+\-*/=]", response))
    has_steps = len(response.strip().split("\n")) > 2

    score = 0.0
    if has_numbers:
        score += 0.1
    if has_operators:
        score += 0.1
    if has_steps:
        score += 0.1
    if "step" in response.lower() or "first" in response.lower():
        score += 0.1
    if answer.replace(",", "") in response:
        score += 0.3

    return min(score, 0.8)


class ReasoningCandidateDataset(Dataset):
    """
    Reasoning dataset with pre-generated candidate traces and verifier scores.

    For DDO-RM over reasoning traces.
    """

    def __init__(
        self,
        questions: list[str],
        answers: list[str],
        candidate_traces: Optional[list[list[str]]] = None,
        verifier_scores: Optional[list[list[float]]] = None,
    ):
        self.questions = questions
        self.answers = answers
        self.candidate_traces = candidate_traces
        self.verifier_scores = verifier_scores

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        item = {
            "prompt": self.questions[idx],
            "answer": self.answers[idx],
        }
        if self.candidate_traces is not None:
            item["candidates"] = self.candidate_traces[idx]
        if self.verifier_scores is not None:
            item["ratings"] = torch.tensor(
                self.verifier_scores[idx], dtype=torch.float32
            )
        return item

    @classmethod
    def from_gsm8k(
        cls,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> "ReasoningCandidateDataset":
        """Create from GSM8K dataset."""
        gsm = GSM8KDataset(split, max_samples)
        questions = [gsm[i]["prompt"] for i in range(len(gsm))]
        answers = [gsm[i]["answer"] for i in range(len(gsm))]
        return cls(questions, answers)
