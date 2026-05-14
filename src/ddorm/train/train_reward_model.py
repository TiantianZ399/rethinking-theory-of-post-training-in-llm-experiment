"""
Reward model training loop.

Trains a reward model on pairwise preferences:
    L = -log σ(r(x, y_w) - r(x, y_l))

Produces the shared RM used by PPO-RLHF, DDO-RM, and baselines.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from accelerate import Accelerator

from ..models.reward_model import RewardModel
from .utils import (
    TrainConfig, set_seed, build_optimizer, build_scheduler,
    MetricTracker, save_checkpoint, save_results,
)


def train_reward_model(
    model_name: str,
    dataset,
    config: TrainConfig,
    output_dir: str = "results/reward_model",
    eval_dataset=None,
):
    """
    Train reward model from pairwise preferences.

    Args:
        model_name: HF model name (base for RM)
        dataset: dataset with 'prompt', 'chosen', 'rejected'
        config: training config
        output_dir: save directory
        eval_dataset: optional eval set
    """
    set_seed(config.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    rm = RewardModel(model_name, model_type="causal_lm_with_head")
    model = rm.get_trainable_model()
    tokenizer = rm.tokenizer

    def collate_fn(batch):
        chosen_texts = [item["prompt"] + "\n" + item["chosen"] for item in batch]
        rejected_texts = [item["prompt"] + "\n" + item["rejected"] for item in batch]

        chosen_enc = tokenizer(
            chosen_texts, max_length=512, truncation=True,
            padding=True, return_tensors="pt",
        )
        rejected_enc = tokenizer(
            rejected_texts, max_length=512, truncation=True,
            padding=True, return_tensors="pt",
        )
        return {
            "chosen_ids": chosen_enc["input_ids"],
            "chosen_mask": chosen_enc["attention_mask"],
            "rejected_ids": rejected_enc["input_ids"],
            "rejected_mask": rejected_enc["attention_mask"],
        }

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_fn,
    )

    optimizer = build_optimizer(model, config.learning_rate, config.weight_decay)
    num_steps = len(dataloader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = build_scheduler(optimizer, num_steps, config.warmup_ratio)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    tracker = MetricTracker(output_dir, f"rm_seed{config.seed}", config.use_wandb)
    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()

        for batch in dataloader:
            with accelerator.accumulate(model):
                # Score chosen
                r_chosen = model(
                    input_ids=batch["chosen_ids"],
                    attention_mask=batch["chosen_mask"],
                )
                # Score rejected
                r_rejected = model(
                    input_ids=batch["rejected_ids"],
                    attention_mask=batch["rejected_mask"],
                )

                # Bradley-Terry loss: -log σ(r_w - r_l)
                loss = -F.logsigmoid(r_chosen - r_rejected).mean()
                accuracy = ((r_chosen > r_rejected).float()).mean()

                accelerator.backward(loss)
                if config.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % config.logging_steps == 0:
                tracker.log({
                    "loss": loss.item(),
                    "accuracy": accuracy.item(),
                    "margin": (r_chosen - r_rejected).mean().item(),
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }, step=global_step)
                tracker.print_last(["loss", "accuracy", "margin"])

            if global_step % config.save_steps == 0:
                unwrapped = accelerator.unwrap_model(model)
                save_checkpoint(unwrapped, optimizer, global_step, output_dir, "rm")

    # Save final
    unwrapped = accelerator.unwrap_model(model)
    final_dir = f"{output_dir}/final"
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    # Save the backbone in HuggingFace format
    unwrapped.backbone.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    # Save the value head weights alongside
    torch.save(unwrapped.value_head.state_dict(), f"{final_dir}/value_head.pt")
    # Save a marker so RewardModel knows to load as causal_lm_with_head
    import json
    with open(f"{final_dir}/rm_config.json", "w") as f:
        json.dump({"model_type": "causal_lm_with_head"}, f)
    tracker.save()

    return {
        "method": "reward_model",
        "seed": config.seed,
        "total_steps": global_step,
    }
