"""
Sampling-based candidate generators: temperature and nucleus.
"""

import torch
from . import CandidateGenerator, CandidateOutput


class TemperatureSamplingGenerator(CandidateGenerator):
    """Generate candidates via temperature sampling."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def generate(self, prompts, model, tokenizer, K, max_length=256, **kwargs):
        model.eval()
        all_candidates = []
        all_ids = []
        all_logprobs = []

        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            candidates = []
            ids_list = []
            lps = []

            for _ in range(K):
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_new_tokens=max_length,
                        do_sample=True,
                        temperature=self.temperature,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                gen_ids = output.sequences[0, input_ids.shape[1]:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                candidates.append(text)
                ids_list.append(gen_ids)

                # Compute total log-prob
                if output.scores:
                    scores = torch.stack(output.scores, dim=0)  # [T, 1, V]
                    log_probs = torch.log_softmax(scores / self.temperature, dim=-1)
                    token_lps = log_probs[
                        range(len(gen_ids)), 0, gen_ids
                    ]
                    lps.append(token_lps.sum().item())
                else:
                    lps.append(0.0)

            all_candidates.append(candidates)
            all_logprobs.append(lps)

            # Pad ids to same length
            max_len = max(ids.shape[0] for ids in ids_list)
            padded = torch.zeros(K, max_len, dtype=torch.long, device=model.device)
            for i, ids in enumerate(ids_list):
                padded[i, :ids.shape[0]] = ids
            all_ids.append(padded)

        # Stack across batch
        max_seq = max(t.shape[1] for t in all_ids)
        batch_ids = torch.zeros(len(prompts), K, max_seq, dtype=torch.long, device=model.device)
        for i, t in enumerate(all_ids):
            batch_ids[i, :, :t.shape[1]] = t

        return CandidateOutput(
            candidates=all_candidates,
            candidate_ids=batch_ids,
            gen_logprobs=torch.tensor(all_logprobs, device=model.device),
            metadata={"generator": "temperature", "temperature": self.temperature},
        )


class NucleusSamplingGenerator(CandidateGenerator):
    """Generate candidates via nucleus (top-p) sampling."""

    def __init__(self, temperature: float = 1.0, top_p: float = 0.9):
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, prompts, model, tokenizer, K, max_length=256, **kwargs):
        model.eval()
        all_candidates = []
        all_ids = []
        all_logprobs = []

        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            candidates = []
            ids_list = []
            lps = []

            for _ in range(K):
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_new_tokens=max_length,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                gen_ids = output.sequences[0, input_ids.shape[1]:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                candidates.append(text)
                ids_list.append(gen_ids)

                if output.scores:
                    scores = torch.stack(output.scores, dim=0)
                    log_probs = torch.log_softmax(scores / self.temperature, dim=-1)
                    token_lps = log_probs[range(len(gen_ids)), 0, gen_ids]
                    lps.append(token_lps.sum().item())
                else:
                    lps.append(0.0)

            all_candidates.append(candidates)
            all_logprobs.append(lps)

            max_len = max(ids.shape[0] for ids in ids_list)
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
            gen_logprobs=torch.tensor(all_logprobs, device=model.device),
            metadata={"generator": "nucleus", "temperature": self.temperature, "top_p": self.top_p},
        )
