"""
DPO training loop.

Direct Preference Optimization baseline.
Uses frozen reference model for KL-implicit reward.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

from ..objectives.dpo import dpo_loss
from ..models.reference_model import ReferenceModel
from .utils import (
    TrainConfig, set_seed, build_optimizer, build_scheduler,
    MetricTracker, save_checkpoint,
)


def _get_response_logprobs(model, tokenizer, prompt, response, max_length=512):
    """Compute sum of log-probs for response tokens given prompt."""
    device = next(model.parameters()).device
    prompt_enc = tokenizer(prompt, return_tensors="pt")
    prompt_len = prompt_enc["input_ids"].shape[1]

    full = tokenizer(
        prompt + response, max_length=max_length,
        truncation=True, return_tensors="pt",
    ).to(device)

    with torch.set_grad_enabled(model.training):
        logits = model(**full).logits

    shift_logits = logits[:, prompt_len - 1:-1, :]
    shift_labels = full["input_ids"][:, prompt_len:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_lps.sum(dim=-1).squeeze()


def train_dpo(
    model_name: str,
    dataset,
    config: TrainConfig,
    ref_model_path: str = None,
    output_dir: str = "results/dpo",
    beta: float = 0.1,
):
    """
    Train policy via DPO.

    Args:
        model_name: HF model name (or SFT checkpoint path)
        ref_model_path: path to frozen reference model
        dataset: pairwise preference dataset
        config: training config
        output_dir: save directory
        beta: DPO temperature
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

    ref = ReferenceModel(ref_model_path or model_name)

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
    )

    optimizer = build_optimizer(model, config.learning_rate, config.weight_decay)
    num_steps = len(dataloader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = build_scheduler(optimizer, num_steps, config.warmup_ratio)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    tracker = MetricTracker(output_dir, f"dpo_seed{config.seed}", config.use_wandb)
    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()

        for batch_items in dataloader:
            # batch_items is a list of dataset items
            if isinstance(batch_items, dict):
                prompts = batch_items["prompt"]
                chosens = batch_items["chosen"]
                rejecteds = batch_items["rejected"]
            else:
                prompts = [item["prompt"] for item in batch_items]
                chosens = [item["chosen"] for item in batch_items]
                rejecteds = [item["rejected"] for item in batch_items]

            with accelerator.accumulate(model):
                unwrapped = accelerator.unwrap_model(model)

                # Compute log-probs
                policy_chosen_lps = torch.stack([
                    _get_response_logprobs(unwrapped, tokenizer, p, c)
                    for p, c in zip(prompts, chosens)
                ])
                policy_rejected_lps = torch.stack([
                    _get_response_logprobs(unwrapped, tokenizer, p, r)
                    for p, r in zip(prompts, rejecteds)
                ])

                with torch.no_grad():
                    ref_chosen_lps = torch.stack([
                        _get_response_logprobs(ref.model, tokenizer, p, c)
                        for p, c in zip(prompts, chosens)
                    ])
                    ref_rejected_lps = torch.stack([
                        _get_response_logprobs(ref.model, tokenizer, p, r)
                        for p, r in zip(prompts, rejecteds)
                    ])

                output = dpo_loss(
                    policy_chosen_lps, policy_rejected_lps,
                    ref_chosen_lps, ref_rejected_lps,
                    beta=beta,
                )

                accelerator.backward(output.loss)
                if config.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % config.logging_steps == 0:
                tracker.log({
                    "loss": output.loss.item(),
                    "accuracy": output.accuracy.item(),
                    "chosen_reward": output.chosen_rewards.mean().item(),
                    "rejected_reward": output.rejected_rewards.mean().item(),
                    "margin": (output.chosen_rewards - output.rejected_rewards).mean().item(),
                    "epoch": epoch,
                }, step=global_step)
                tracker.print_last(["loss", "accuracy", "margin"])

            if global_step % config.save_steps == 0:
                unwrapped = accelerator.unwrap_model(model)
                save_checkpoint(unwrapped, optimizer, global_step, output_dir, "dpo")

    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    tracker.save()

    return {"method": "dpo", "seed": config.seed, "total_steps": global_step}
