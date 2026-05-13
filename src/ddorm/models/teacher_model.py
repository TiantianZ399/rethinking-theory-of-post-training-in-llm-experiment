"""
Teacher model wrapper for OPD experiments.

Frozen larger model providing target distributions
for knowledge distillation.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class TeacherModel:
    """
    Frozen teacher model for distillation.

    Provides:
    - Token-level logits for OPD
    - Top-K support sets per position
    - Sequence-level scoring for candidate evaluation
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

    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get teacher logits for a sequence."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return outputs.logits

    def get_top_k_support(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        top_k: int = 32,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get teacher's top-K tokens and probabilities per position.

        Returns:
            topk_indices: [B, T, K] token indices
            topk_probs: [B, T, K] probabilities
            topk_logits: [B, T, K] logits
        """
        logits = self.get_logits(input_ids, attention_mask)
        probs = F.softmax(logits / temperature, dim=-1)
        topk_probs, topk_indices = probs.topk(top_k, dim=-1)
        topk_logits = logits.gather(-1, topk_indices)
        return topk_indices, topk_probs, topk_logits

    def score_sequence(
        self,
        prompt: str,
        response: str,
        max_length: int = 512,
    ) -> float:
        """Compute total log-prob of response under teacher."""
        prompt_enc = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_enc["input_ids"].shape[1]

        full = self.tokenizer(
            prompt + response,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        logits = self.get_logits(full["input_ids"], full["attention_mask"])
        shift_logits = logits[:, prompt_len - 1:-1, :]
        shift_labels = full["input_ids"][:, prompt_len:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_lps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        return token_lps.sum().item()
