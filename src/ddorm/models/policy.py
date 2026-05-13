"""
Policy model wrapper for DDO-RM.

Provides sequence-level scoring for candidate evaluation:
    s_θ(x, y) = Σ_t log π_θ(y_t | x, y_{<t})
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class PolicyModel:
    """
    Wrapper around a causal LM for candidate-level scoring.

    Computes length-normalized log-probs for complete sequences.
    """

    def __init__(
        self,
        model_name_or_path: str,
        dtype: str = "bfloat16",
        device: str = "auto",
    ):
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype_map.get(dtype, torch.bfloat16),
            device_map=device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def device(self):
        return next(self.model.parameters()).device

    def score_candidates(
        self,
        prompt: str,
        candidates: list[str],
        max_length: int = 512,
        length_normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute sequence-level scores for each candidate.

        s(x, y) = (1/|y|) Σ_t log π(y_t | x, y_{<t})

        Args:
            prompt: prompt string
            candidates: list of candidate response strings
            max_length: max total sequence length
            length_normalize: if True, divide by response length

        Returns:
            scores: [K] sequence scores
        """
        scores = []
        for cand in candidates:
            full_text = prompt + cand
            encoding = self.tokenizer(
                full_text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            ).to(self.device)

            prompt_enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            )
            prompt_len = prompt_enc["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits  # [1, L, V]

            # Shift: predict token t from position t-1
            shift_logits = logits[:, prompt_len - 1:-1, :]
            shift_labels = encoding["input_ids"][:, prompt_len:]

            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(
                -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            total_lp = token_log_probs.sum().item()
            resp_len = shift_labels.shape[1]

            if length_normalize and resp_len > 0:
                scores.append(total_lp / resp_len)
            else:
                scores.append(total_lp)

        return torch.tensor(scores, device=self.device)

    def batch_score(
        self,
        prompts: list[str],
        candidates_per_prompt: list[list[str]],
        max_length: int = 512,
        length_normalize: bool = True,
    ) -> torch.Tensor:
        """
        Batch scoring: [B, K] scores.

        Args:
            prompts: [B] prompts
            candidates_per_prompt: [B][K] candidates

        Returns:
            scores: [B, K]
        """
        all_scores = []
        for prompt, cands in zip(prompts, candidates_per_prompt):
            s = self.score_candidates(prompt, cands, max_length, length_normalize)
            all_scores.append(s)
        return torch.stack(all_scores)

    def get_trainable_model(self):
        """Return the underlying model for training."""
        return self.model
