"""
Diverse beam search candidate generator.

Uses Hamming diversity penalty to encourage distinct candidates.
Each group produces one beam; diversity penalty discourages overlap
across groups, yielding K candidates with higher semantic coverage.
"""

import torch
from . import CandidateGenerator, CandidateOutput


class DiverseBeamSearchGenerator(CandidateGenerator):
    """Generate candidates via diverse beam search (DBS).

    Groups beams and applies a Hamming diversity penalty across groups,
    so the final K candidates span more of the output space than
    standard beam search.

    Args:
        num_beam_groups: number of diversity groups
        diversity_penalty: strength of cross-group penalty (higher = more diverse)
        num_beams_per_group: beams within each group
        length_penalty: length normalization exponent
    """

    def __init__(
        self,
        num_beam_groups: int = 4,
        diversity_penalty: float = 1.0,
        num_beams_per_group: int = 4,
        length_penalty: float = 1.0,
    ):
        self.num_beam_groups = num_beam_groups
        self.diversity_penalty = diversity_penalty
        self.num_beams_per_group = num_beams_per_group
        self.length_penalty = length_penalty

    def generate(self, prompts, model, tokenizer, K, max_length=256, **kwargs):
        """Generate K diverse candidates per prompt.

        Args:
            prompts: list of prompt strings
            model: HuggingFace CausalLM
            tokenizer: tokenizer
            K: number of candidates to return per prompt
            max_length: max new tokens to generate

        Returns:
            CandidateOutput with candidates, log_probs, metadata
        """
        model.eval()
        all_candidates = []
        all_ids = []

        total_beams = self.num_beam_groups * self.num_beams_per_group

        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_length,
                    num_beams=total_beams,
                    num_beam_groups=self.num_beam_groups,
                    diversity_penalty=self.diversity_penalty,
                    num_return_sequences=min(K, total_beams),
                    length_penalty=self.length_penalty,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            # Decode candidates
            prompt_len = input_ids.shape[1]
            gen_ids = output.sequences[:, prompt_len:]
            candidates = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            # Trim or pad to exactly K
            if len(candidates) < K:
                # Duplicate last candidate to fill
                candidates = candidates + [candidates[-1]] * (K - len(candidates))
                gen_ids_list = list(gen_ids) + [gen_ids[-1]] * (K - len(list(gen_ids)))
            else:
                candidates = candidates[:K]
                gen_ids_list = list(gen_ids[:K])

            all_candidates.append(candidates)
            all_ids.append(torch.stack(gen_ids_list) if isinstance(gen_ids_list[0], torch.Tensor) else None)

        # Compute log-probs via forward pass
        log_probs = self._compute_log_probs(prompts, all_candidates, model, tokenizer)

        return CandidateOutput(
            candidates=all_candidates,
            log_probs=log_probs,
            metadata={
                "generator": "diverse_beam_search",
                "num_beam_groups": self.num_beam_groups,
                "diversity_penalty": self.diversity_penalty,
                "num_beams_per_group": self.num_beams_per_group,
                "K": K,
            },
        )

    def _compute_log_probs(self, prompts, all_candidates, model, tokenizer):
        """Compute sequence log-probs for each prompt-candidate pair."""
        B = len(prompts)
        K = len(all_candidates[0])
        log_probs = torch.zeros(B, K)

        for i, (prompt, candidates) in enumerate(zip(prompts, all_candidates)):
            for j, cand in enumerate(candidates):
                full_text = prompt + cand
                inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
                prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
                prompt_len = prompt_ids.shape[1]

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits[0, prompt_len - 1:-1]  # shift by 1
                    target_ids = inputs["input_ids"][0, prompt_len:]

                    token_log_probs = torch.log_softmax(logits, dim=-1)
                    seq_log_prob = token_log_probs.gather(
                        1, target_ids.unsqueeze(1)
                    ).squeeze(1).sum()

                log_probs[i, j] = seq_log_prob.cpu()

        return log_probs
