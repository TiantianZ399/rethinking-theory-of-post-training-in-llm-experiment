"""
KL divergence evaluation.

Measures KL(π_θ || π_ref) for trained policies.
Important for fair comparison: DDO-RM vs PPO should have similar KL budgets.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class KLEvalResult:
    """KL evaluation results."""
    mean_kl: float
    std_kl: float
    median_kl: float
    max_kl: float
    kl_per_token: float
    num_samples: int


def evaluate_kl(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> KLEvalResult:
    """
    Evaluate KL divergence between policy and reference.

    KL(π_θ || π_ref) = E_{π_θ}[log π_θ - log π_ref]

    Args:
        policy_logprobs: [B, T] per-token log-probs under π_θ
        ref_logprobs: [B, T] per-token log-probs under π_ref
        mask: [B, T] valid token mask

    Returns:
        KLEvalResult
    """
    if mask is None:
        mask = torch.ones_like(policy_logprobs)

    # Per-token KL
    kl_per_token = (policy_logprobs - ref_logprobs) * mask
    # Per-sequence KL
    seq_kl = kl_per_token.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    kl_np = seq_kl.cpu().numpy()

    total_tokens = mask.sum().item()
    total_kl = kl_per_token.sum().item()

    return KLEvalResult(
        mean_kl=float(kl_np.mean()),
        std_kl=float(kl_np.std()),
        median_kl=float(np.median(kl_np)),
        max_kl=float(kl_np.max()),
        kl_per_token=total_kl / max(total_tokens, 1),
        num_samples=len(kl_np),
    )


def evaluate_kl_from_models(
    policy_model,
    ref_model,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    max_length: int = 512,
) -> KLEvalResult:
    """
    Compute KL from model objects.

    Args:
        policy_model: trained policy
        ref_model: reference model (ReferenceModel instance)
        tokenizer: tokenizer
        prompts: prompt strings
        responses: response strings
        max_length: max sequence length

    Returns:
        KLEvalResult
    """
    all_policy_lps = []
    all_ref_lps = []

    device = next(policy_model.parameters()).device

    for prompt, response in zip(prompts, responses):
        prompt_enc = tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_enc["input_ids"].shape[1]

        full = tokenizer(
            prompt + response,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # Policy log-probs
        with torch.no_grad():
            policy_logits = policy_model(**full).logits
        shift_logits = policy_logits[:, prompt_len - 1:-1, :]
        shift_labels = full["input_ids"][:, prompt_len:]
        policy_lps = F.log_softmax(shift_logits, dim=-1)
        policy_token_lps = policy_lps.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Reference log-probs
        ref_token_lps = ref_model.log_probs(full["input_ids"], full["attention_mask"], prompt_len)

        all_policy_lps.append(policy_token_lps.squeeze(0))
        all_ref_lps.append(ref_token_lps.squeeze(0))

    # Pad and stack
    max_len = max(t.shape[0] for t in all_policy_lps)
    policy_padded = torch.zeros(len(prompts), max_len, device=device)
    ref_padded = torch.zeros(len(prompts), max_len, device=device)
    mask = torch.zeros(len(prompts), max_len, device=device)

    for i in range(len(prompts)):
        L = all_policy_lps[i].shape[0]
        policy_padded[i, :L] = all_policy_lps[i]
        ref_padded[i, :L] = all_ref_lps[i]
        mask[i, :L] = 1.0

    return evaluate_kl(policy_padded, ref_padded, mask)
