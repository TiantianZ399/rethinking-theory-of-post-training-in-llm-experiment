"""
GRPO training loop.

Group Relative Policy Optimization for reasoning/verifier tasks.
Generates K candidate traces, normalizes rewards within group,
uses as policy gradient weights.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

from ..objectives.grpo import grpo_loss
from ..models.reference_model import ReferenceModel
from .utils import (
    TrainConfig, set_seed, build_optimizer, build_scheduler,
    MetricTracker, save_checkpoint,
)


def train_grpo(
    model_name: str,
    ref_model_path: str,
    dataset,
    config: TrainConfig,
    output_dir: str = "results/grpo",
    num_candidates: int = 8,
    kl_coeff: float = 0.1,
    clip_range: float = 0.2,
    reward_fn=None,
):
    """
    Train policy via GRPO.

    Args:
        model_name: SFT checkpoint path
        ref_model_path: reference model path
        dataset: reasoning dataset with prompts and verifier
        config: training config
        output_dir: save directory
        num_candidates: K candidates per prompt
        kl_coeff: KL coefficient
        clip_range: PPO-style clip range
        reward_fn: callable(prompt, response) -> float
    """
    set_seed(config.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ref = ReferenceModel(ref_model_path)

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
    )

    optimizer = build_optimizer(model, config.learning_rate, config.weight_decay)
    num_steps = len(dataloader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = build_scheduler(optimizer, num_steps, config.warmup_ratio)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    tracker = MetricTracker(output_dir, f"grpo_seed{config.seed}", config.use_wandb)
    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()

        for batch_items in dataloader:
            if isinstance(batch_items, dict):
                prompts = batch_items["prompt"]
            else:
                prompts = [item["prompt"] for item in batch_items]

            if isinstance(prompts, str):
                prompts = [prompts]

            with accelerator.accumulate(model):
                unwrapped = accelerator.unwrap_model(model)

                all_logprobs = []
                all_ref_logprobs = []
                all_rewards = []

                for prompt in prompts:
                    query_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                        accelerator.device
                    )

                    # Generate K candidates
                    cand_logprobs = []
                    cand_ref_logprobs = []
                    cand_rewards = []

                    for _ in range(num_candidates):
                        with torch.no_grad():
                            gen = unwrapped.generate(
                                query_ids,
                                max_new_tokens=256,
                                do_sample=True,
                                temperature=1.0,
                                return_dict_in_generate=True,
                                output_scores=True,
                            )

                        resp_ids = gen.sequences[:, query_ids.shape[1]:]
                        resp_text = tokenizer.decode(resp_ids[0], skip_special_tokens=True)

                        # Compute log-probs (simplified: sum of token log-probs)
                        full_ids = gen.sequences
                        with torch.set_grad_enabled(True):
                            logits = unwrapped(full_ids).logits
                        shift = logits[:, query_ids.shape[1] - 1:-1, :]
                        labels = full_ids[:, query_ids.shape[1]:]
                        lps = F.log_softmax(shift, dim=-1)
                        token_lps = lps.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                        seq_lp = token_lps.sum(dim=-1)

                        ref_lp_val = ref.sequence_log_prob(prompt, resp_text)

                        if reward_fn is not None:
                            r = reward_fn(prompt, resp_text)
                        else:
                            r = 0.0

                        cand_logprobs.append(seq_lp)
                        cand_ref_logprobs.append(torch.tensor([ref_lp_val], device=seq_lp.device))
                        cand_rewards.append(r)

                    all_logprobs.append(torch.cat(cand_logprobs))
                    all_ref_logprobs.append(torch.cat(cand_ref_logprobs))
                    all_rewards.append(torch.tensor(cand_rewards, device=accelerator.device))

                # Stack: [B, K]
                logprobs = torch.stack(all_logprobs)
                ref_logprobs = torch.stack(all_ref_logprobs)
                rewards = torch.stack(all_rewards)

                grpo_out = grpo_loss(
                    logprobs, ref_logprobs, rewards,
                    kl_coeff=kl_coeff, clip_range=clip_range,
                )

                accelerator.backward(grpo_out.loss)
                if config.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % config.logging_steps == 0:
                tracker.log({
                    "loss": grpo_out.loss.item(),
                    "mean_reward": grpo_out.mean_reward.item(),
                    "advantage_std": grpo_out.advantages.std().item(),
                    "epoch": epoch,
                }, step=global_step)
                tracker.print_last(["loss", "mean_reward"])

    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    tracker.save()

    return {"method": "grpo", "seed": config.seed, "total_steps": global_step}
