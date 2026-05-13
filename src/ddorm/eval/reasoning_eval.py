"""
Reasoning evaluation metrics.

For Experiment 8 (GRPO / DDO-RM over reasoning traces).
Reports: pass@1, pass@K, verifier reward, compute efficiency.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class ReasoningEvalResult:
    """Reasoning evaluation results."""
    pass_at_1: float
    pass_at_k: dict[int, float]  # {K: pass@K}
    mean_verifier_reward: float
    solve_rate: float
    mean_attempts_to_solve: float
    num_problems: int


def pass_at_k_estimator(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator for pass@k.

    pass@k = 1 - C(n-c, k) / C(n, k)

    where n = total samples, c = correct samples.

    Args:
        n: total number of samples
        c: number of correct samples
        k: k value
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def evaluate_reasoning(
    problems: list[dict],
    model,
    tokenizer,
    verifier_fn: Callable,
    K_values: list[int] = [1, 4, 8, 16],
    num_samples: int = 16,
    max_response_length: int = 512,
    temperature: float = 0.8,
) -> ReasoningEvalResult:
    """
    Evaluate reasoning model on a set of problems.

    Args:
        problems: list of dicts with 'prompt' and 'answer'
        model: language model
        tokenizer: tokenizer
        verifier_fn: callable(prompt, response, answer) -> float
        K_values: K values for pass@K
        num_samples: samples per problem
        max_response_length: max tokens
        temperature: sampling temperature

    Returns:
        ReasoningEvalResult
    """
    import torch

    model.eval()
    device = next(model.parameters()).device

    all_correct_counts = []
    all_rewards = []
    problems_solved = 0
    total_attempts = 0

    for problem in problems:
        prompt = problem["prompt"]
        answer = problem["answer"]

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        correct_count = 0
        problem_rewards = []

        for _ in range(num_samples):
            with torch.no_grad():
                gen = model.generate(
                    input_ids,
                    max_new_tokens=max_response_length,
                    do_sample=True,
                    temperature=temperature,
                )

            response = tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)
            reward = verifier_fn(prompt, response, answer)
            problem_rewards.append(reward)

            if reward >= 1.0:
                correct_count += 1

        all_correct_counts.append(correct_count)
        all_rewards.extend(problem_rewards)

        if correct_count > 0:
            problems_solved += 1
            total_attempts += num_samples  # simplified

    # Compute pass@K for each K
    pass_at_k = {}
    for k in K_values:
        estimates = [
            pass_at_k_estimator(num_samples, c, min(k, num_samples))
            for c in all_correct_counts
        ]
        pass_at_k[k] = float(np.mean(estimates))

    return ReasoningEvalResult(
        pass_at_1=pass_at_k.get(1, 0.0),
        pass_at_k=pass_at_k,
        mean_verifier_reward=float(np.mean(all_rewards)),
        solve_rate=problems_solved / max(len(problems), 1),
        mean_attempts_to_solve=(
            total_attempts / max(problems_solved, 1) if problems_solved > 0 else float("inf")
        ),
        num_problems=len(problems),
    )


def evaluate_reasoning_from_candidates(
    problems: list[dict],
    candidate_responses: list[list[str]],
    verifier_fn: Callable,
    K_values: list[int] = [1, 4, 8, 16],
) -> ReasoningEvalResult:
    """
    Evaluate pre-generated candidate reasoning traces.

    Args:
        problems: list of dicts with 'prompt' and 'answer'
        candidate_responses: [N][K] pre-generated responses
        verifier_fn: callable(prompt, response, answer) -> float
        K_values: K values for pass@K
    """
    all_correct_counts = []
    all_rewards = []
    problems_solved = 0

    for problem, candidates in zip(problems, candidate_responses):
        prompt = problem["prompt"]
        answer = problem["answer"]

        correct_count = 0
        for response in candidates:
            reward = verifier_fn(prompt, response, answer)
            all_rewards.append(reward)
            if reward >= 1.0:
                correct_count += 1

        all_correct_counts.append(correct_count)
        if correct_count > 0:
            problems_solved += 1

    num_samples = len(candidate_responses[0]) if candidate_responses else 1
    pass_at_k = {}
    for k in K_values:
        estimates = [
            pass_at_k_estimator(num_samples, c, min(k, num_samples))
            for c in all_correct_counts
        ]
        pass_at_k[k] = float(np.mean(estimates))

    return ReasoningEvalResult(
        pass_at_1=pass_at_k.get(1, 0.0),
        pass_at_k=pass_at_k,
        mean_verifier_reward=float(np.mean(all_rewards)),
        solve_rate=problems_solved / max(len(problems), 1),
        mean_attempts_to_solve=float("inf"),
        num_problems=len(problems),
    )
