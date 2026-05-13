"""
SFT training loop.

Standard supervised fine-tuning on chosen responses.
Produces the base checkpoint for all downstream methods.
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

from .utils import (
    TrainConfig, set_seed, build_optimizer, build_scheduler,
    MetricTracker, save_checkpoint, save_results,
)


def train_sft(
    model_name: str,
    dataset,
    config: TrainConfig,
    output_dir: str = "results/sft",
):
    """
    Train SFT model on chosen responses.

    Args:
        model_name: HF model name
        dataset: dataset with 'prompt' and 'chosen' fields
        config: training config
        output_dir: where to save
    """
    set_seed(config.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare data
    def collate_fn(batch):
        texts = [item["prompt"] + item["chosen"] for item in batch]
        prompt_texts = [item["prompt"] for item in batch]

        encoding = tokenizer(
            texts, max_length=512, truncation=True,
            padding=True, return_tensors="pt",
        )
        prompt_enc = tokenizer(
            prompt_texts, max_length=256, truncation=True,
            padding=True, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "prompt_lengths": prompt_enc["attention_mask"].sum(dim=-1),
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

    tracker = MetricTracker(output_dir, f"sft_seed{config.seed}", config.use_wandb)
    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0

        for batch in dataloader:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["input_ids"],
                )

                # Mask prompt tokens from loss
                # (simplified: use full sequence loss for now,
                #  proper implementation would mask prompt positions)
                loss = outputs.loss

                accelerator.backward(loss)
                if config.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % config.logging_steps == 0:
                tracker.log({
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }, step=global_step)
                tracker.print_last(["loss", "lr"])

            if global_step % config.save_steps == 0:
                unwrapped = accelerator.unwrap_model(model)
                save_checkpoint(unwrapped, optimizer, global_step, output_dir, "sft")

    # Save final
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    tracker.save()

    return {
        "method": "sft",
        "seed": config.seed,
        "final_loss": epoch_loss / len(dataloader),
        "total_steps": global_step,
    }
