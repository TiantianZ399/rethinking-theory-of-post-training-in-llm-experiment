"""
Reward model wrapper for DDO-RM.

Supports:
- AutoModelForSequenceClassification (standard RM)
- Custom reward head on causal LM
- Batched scoring of candidate sets
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from typing import Optional


class RewardModel:
    """
    Reward model: r(x, y) → scalar.

    Wraps AutoModelForSequenceClassification or adds a value head
    to a causal LM backbone.
    """

    def __init__(
        self,
        model_name_or_path: str,
        model_type: str = "auto",
        dtype: str = "bfloat16",
        device: str = "auto",
    ):
        """
        Args:
            model_name_or_path: HF model name or local path
            model_type: "sequence_classification", "causal_lm_with_head", or "auto"
                        "auto" will detect from rm_config.json if present
            dtype: weight dtype
            device: device map
        """
        import os
        import json

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        td = dtype_map.get(dtype, torch.bfloat16)

        # Auto-detect model type from saved config
        if model_type == "auto":
            rm_config_path = os.path.join(model_name_or_path, "rm_config.json")
            if os.path.exists(rm_config_path):
                with open(rm_config_path) as f:
                    rm_cfg = json.load(f)
                model_type = rm_cfg.get("model_type", "sequence_classification")
            else:
                model_type = "sequence_classification"

        self.model_type = model_type

        if model_type == "sequence_classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                torch_dtype=td,
                device_map=device,
                num_labels=1,
            )
        else:
            backbone = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=td,
                device_map=device,
            )
            self.model = RewardHeadModel(backbone)
            # Load value head weights if available
            value_head_path = os.path.join(model_name_or_path, "value_head.pt")
            if os.path.exists(value_head_path):
                value_head_state = torch.load(value_head_path, map_location="cpu")
                self.model.value_head.load_state_dict(value_head_state)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @property
    def device(self):
        return next(self.model.parameters()).device

    def score(
        self,
        prompts: list[str],
        candidates: list[list[str]],
        max_length: int = 512,
    ) -> torch.Tensor:
        """
        Score candidate responses.

        Args:
            prompts: [B] prompt strings
            candidates: [B][K] candidate strings

        Returns:
            scores: [B, K] reward scores
        """
        B = len(prompts)
        K = len(candidates[0])
        all_scores = []

        for b in range(B):
            batch_texts = []
            for k in range(K):
                text = f"{prompts[b]}\n{candidates[b][k]}"
                batch_texts.append(text)

            encoding = self.tokenizer(
                batch_texts,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoding)

            if self.model_type == "sequence_classification":
                scores = outputs.logits.squeeze(-1)  # [K]
            else:
                scores = outputs  # [K]

            all_scores.append(scores)

        return torch.stack(all_scores)  # [B, K]

    def score_pairs(
        self,
        prompts: list[str],
        chosen: list[str],
        rejected: list[str],
        max_length: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Score chosen/rejected pairs for RM training evaluation.

        Returns:
            chosen_scores: [B]
            rejected_scores: [B]
        """
        candidates = [[c, r] for c, r in zip(chosen, rejected)]
        scores = self.score(prompts, candidates, max_length)
        return scores[:, 0], scores[:, 1]

    def get_trainable_model(self):
        return self.model


class RewardHeadModel(nn.Module):
    """Causal LM with a scalar value head for reward modeling."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Use last hidden state of last token
        hidden = outputs.hidden_states[-1]  # [B, L, H]

        if attention_mask is not None:
            # Find last non-pad token
            lengths = attention_mask.sum(dim=-1) - 1  # [B]
            last_hidden = hidden[torch.arange(hidden.shape[0]), lengths]
        else:
            last_hidden = hidden[:, -1]

        reward = self.value_head(last_hidden).squeeze(-1)  # [B]
        return reward
