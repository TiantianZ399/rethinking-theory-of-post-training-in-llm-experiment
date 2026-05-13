"""
Beam search candidate generator.

Produces high-probability candidates via standard beam search.
Lower diversity but higher average quality per candidate.
"""

import torch
from . import CandidateGenerator, CandidateOutput


class BeamSearchGenerator(CandidateGenerator):
    """Generate candidates via beam search."""

    def __init__(self, num_beams: int = 16, length_penalty: float = 1.0):
        self.num_beams = num_beams
        self.length_penalty = length_penalty

    def generate(self, prompts, model, tokenizer, K, max_length=256, **kwargs):
        model.eval()
        all_candidates = []
        all_ids = []

        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_length,
                    num_beams=max(self.num_beams, K),
                    num_return_sequences=K,
                    length_penalty=self.length_penalty,
                    early_stopping=True,
                    return_dict_in_generate=True,
                )

            candidates = []
            ids_list = []
            for seq in output.sequences:
                gen_ids = seq[input_ids.shape[1]:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                candidates.append(text)
                ids_list.append(gen_ids)

            candidates = candidates[:K]
            ids_list = ids_list[:K]
            all_candidates.append(candidates)

            max_len = max(ids.shape[0] for ids in ids_list) if ids_list else 1
            padded = torch.zeros(K, max_len, dtype=torch.long, device=model.device)
            for i, ids in enumerate(ids_list):
                padded[i, :ids.shape[0]] = ids
            all_ids.append(padded)

        max_seq = max(t.shape[1] for t in all_ids)
        batch_ids = torch.zeros(len(prompts), K, max_seq, dtype=torch.long, device=model.device)
        for i, t in enumerate(all_ids):
            batch_ids[i, :, :t.shape[1]] = t

        return CandidateOutput(
            candidates=all_candidates,
            candidate_ids=batch_ids,
            gen_logprobs=None,
            metadata={"generator": "beam_search", "num_beams": self.num_beams},
        )


class DiverseBeamSearchGenerator(CandidateGenerator):
    """Generate candidates via diverse beam search (group beam search)."""

    def __init__(
        self,
        num_beams: int = 16,
        num_beam_groups: int = 4,
        diversity_penalty: float = 1.0,
        length_penalty: float = 1.0,
    ):
        self.num_beams = num_beams
        self.num_beam_groups = num_beam_groups
        self.diversity_penalty = diversity_penalty
        self.length_penalty = length_penalty

    def generate(self, prompts, model, tokenizer, K, max_length=256, **kwargs):
        model.eval()
        all_candidates = []
        all_ids = []

        # Ensure num_beams is divisible by num_beam_groups
        num_beams = max(self.num_beams, K)
        num_groups = self.num_beam_groups
        while num_beams % num_groups != 0:
            num_beams += 1

        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_length,
                    num_beams=num_beams,
                    num_beam_groups=num_groups,
                    num_return_sequences=K,
                    diversity_penalty=self.diversity_penalty,
                    length_penalty=self.length_penalty,
                    return_dict_in_generate=True,
                )

            candidates = []
            ids_list = []
            for seq in output.sequences[:K]:
                gen_ids = seq[input_ids.shape[1]:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                candidates.append(text)
                ids_list.append(gen_ids)

            all_candidates.append(candidates)

            max_len = max(ids.shape[0] for ids in ids_list) if ids_list else 1
            padded = torch.zeros(K, max_len, dtype=torch.long, device=model.device)
            for i, ids in enumerate(ids_list):
                padded[i, :ids.shape[0]] = ids
            all_ids.append(padded)

        max_seq = max(t.shape[1] for t in all_ids)
        batch_ids = torch.zeros(len(prompts), K, max_seq, dtype=torch.long, device=model.device)
        for i, t in enumerate(all_ids):
            batch_ids[i, :, :t.shape[1]] = t

        return CandidateOutput(
            candidates=all_candidates,
            candidate_ids=batch_ids,
            gen_logprobs=None,
            metadata={"generator": "diverse_beam_search", "num_groups": num_groups},
        )
