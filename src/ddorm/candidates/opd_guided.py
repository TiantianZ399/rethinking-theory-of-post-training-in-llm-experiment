"""
OPD-guided candidate generator.

Uses teacher model logits to guide student sampling,
producing candidates that lie in the teacher's high-probability region.
"""

import torch
import torch.nn.functional as F
from . import CandidateGenerator, CandidateOutput
from typing import Optional


class OPDGuidedGenerator(CandidateGenerator):
    """
    Generate candidates using teacher-guided sampling.

    At each token position, restrict sampling to the teacher's top-K tokens,
    then sample from the student's distribution over that restricted set.
    This produces candidates that are both student-likely and teacher-likely.
    """

    def __init__(
        self,
        teacher_model=None,
        teacher_top_k: int = 32,
        temperature: float = 1.0,
        interpolation: float = 0.5,
    ):
        """
        Args:
            teacher_model: teacher language model
            teacher_top_k: restrict to teacher's top-K tokens per position
            temperature: sampling temperature
            interpolation: weight for teacher (0=student only, 1=teacher only)
        """
        self.teacher_model = teacher_model
        self.teacher_top_k = teacher_top_k
        self.temperature = temperature
        self.interpolation = interpolation

    def generate(self, prompts, model, tokenizer, K, max_length=256, **kwargs):
        teacher = kwargs.get("teacher_model", self.teacher_model)
        if teacher is None:
            # Fall back to standard sampling
            from .sampling import TemperatureSamplingGenerator
            fallback = TemperatureSamplingGenerator(self.temperature)
            return fallback.generate(prompts, model, tokenizer, K, max_length)

        model.eval()
        teacher.eval()
        device = next(model.parameters()).device
        all_candidates = []
        all_ids = []

        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            candidates = []
            ids_list = []

            for _ in range(K):
                generated = input_ids.clone()

                for step in range(max_length):
                    with torch.no_grad():
                        student_logits = model(generated).logits[:, -1, :]
                        teacher_logits = teacher(generated).logits[:, -1, :]

                    # Teacher's top-K tokens
                    teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
                    topk_vals, topk_idx = teacher_probs.topk(self.teacher_top_k, dim=-1)

                    # Student distribution on teacher's support
                    student_logits_k = student_logits.gather(-1, topk_idx)
                    teacher_logits_k = teacher_logits.gather(-1, topk_idx)

                    # Interpolated logits
                    mixed_logits = (
                        (1 - self.interpolation) * student_logits_k / self.temperature
                        + self.interpolation * teacher_logits_k / self.temperature
                    )
                    mixed_probs = F.softmax(mixed_logits, dim=-1)

                    # Sample
                    token_idx = torch.multinomial(mixed_probs, 1)
                    next_token = topk_idx.gather(-1, token_idx)

                    generated = torch.cat([generated, next_token], dim=-1)

                    if next_token.item() == tokenizer.eos_token_id:
                        break

                gen_ids = generated[0, input_ids.shape[1]:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                candidates.append(text)
                ids_list.append(gen_ids)

            all_candidates.append(candidates)

            max_len = max(ids.shape[0] for ids in ids_list) if ids_list else 1
            padded = torch.zeros(K, max_len, dtype=torch.long, device=device)
            for i, ids in enumerate(ids_list):
                padded[i, :ids.shape[0]] = ids
            all_ids.append(padded)

        max_seq = max(t.shape[1] for t in all_ids)
        batch_ids = torch.zeros(len(prompts), K, max_seq, dtype=torch.long, device=device)
        for i, t in enumerate(all_ids):
            batch_ids[i, :, :t.shape[1]] = t

        return CandidateOutput(
            candidates=all_candidates,
            candidate_ids=batch_ids,
            gen_logprobs=None,
            metadata={
                "generator": "opd_guided",
                "teacher_top_k": self.teacher_top_k,
                "interpolation": self.interpolation,
            },
        )
