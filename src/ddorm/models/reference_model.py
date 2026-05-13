"""
Reference model for KL penalty computation.

Frozen copy of the SFT checkpoint used as the anchor
for KL(π_θ || π_ref) in PPO-RLHF and DPO.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class ReferenceModel:
    """
    Frozen reference policy for KL computation.

    All parameters are frozen; provides log-prob computation only.
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
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def device(self):
        return next(self.model.parameters()).device

    def log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute per-token log-probs for the response portion.

        Args:
            input_ids: [B, L] full sequence (prompt + response)
            attention_mask: [B, L]
            prompt_length: if set, only return log-probs for tokens after this position

        Returns:
            log_probs: [B, L-1] or [B, L-prompt_length] per-token log-probs
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # [B, L, V]

        # Shift for autoregressive: predict token t from position t-1
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        if prompt_length is not None and prompt_length > 0:
            token_log_probs = token_log_probs[:, prompt_length - 1:]

        return token_log_probs

    def sequence_log_prob(
        self,
        prompt: str,
        response: str,
        max_length: int = 512,
    ) -> float:
        """
        Compute total log-prob of response given prompt.

        Args:
            prompt: prompt string
            response: response string

        Returns:
            total log-prob (scalar)
        """
        prompt_enc = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_enc["input_ids"].shape[1]

        full_text = prompt + response
        enc = self.tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        lps = self.log_probs(enc["input_ids"], enc["attention_mask"], prompt_len)
        return lps.sum().item()

    def batch_sequence_log_probs(
        self,
        prompts: list[str],
        candidates: list[list[str]],
        max_length: int = 512,
    ) -> torch.Tensor:
        """
        Batch log-probs for candidate sets.

        Args:
            prompts: [B] prompts
            candidates: [B][K] candidate responses

        Returns:
            log_probs: [B, K]
        """
        all_lps = []
        for prompt, cands in zip(prompts, candidates):
            lps = [self.sequence_log_prob(prompt, c, max_length) for c in cands]
            all_lps.append(lps)
        return torch.tensor(all_lps, device=self.device)
