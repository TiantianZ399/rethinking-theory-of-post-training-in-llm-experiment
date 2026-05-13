"""
OPD training loop.

On-Policy Distillation: token-level teacher distillation
on student-generated prefixes.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

from ..objectives.opd import opd_loss
from .utils import (
    TrainConfig, set_seed, build_optimizer, build_scheduler,
    MetricTracker, save_checkpoint,
)


def train_opd(
    student_model_name: str,
    teacher_model_name: str,
    dataset,
    config: TrainConfig,
    output_dir: str = "results/opd",
    top_k: int = 32,
    temperature: float = 1.0,
    max_response_length: int = 256,
):
    """
    Train student via on-policy distillation.

    Args:
        student_model_name: student checkpoint path
        teacher_model_name: teacher model path
        dataset: prompt dataset
        config: training config
        output_dir: save directory
        top_k: restrict to teacher's top-K tokens (None = full vocab)
        temperature: distillation temperature
        max_response_length: max response tokens
    """
    set_seed(config.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    student = AutoModelForCausalLM.from_pretrained(
        student_model_name, torch_dtype=torch.bfloat16
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name, torch_dtype=torch.bfloat16
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
    )

    optimizer = build_optimizer(student, config.learning_rate, config.weight_decay)
    num_steps = len(dataloader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = build_scheduler(optimizer, num_steps, config.warmup_ratio)

    student, teacher, optimizer, dataloader, scheduler = accelerator.prepare(
        student, teacher, optimizer, dataloader, scheduler
    )

    tracker = MetricTracker(output_dir, f"opd_seed{config.seed}", config.use_wandb)
    global_step = 0

    for epoch in range(config.num_epochs):
        student.train()

        for batch_items in dataloader:
            if isinstance(batch_items, dict):
                prompts = batch_items["prompt"]
            else:
                prompts = [item["prompt"] for item in batch_items]
            if isinstance(prompts, str):
                prompts = [prompts]

            with accelerator.accumulate(student):
                unwrapped_student = accelerator.unwrap_model(student)
                unwrapped_teacher = accelerator.unwrap_model(teacher)

                # Step 1: Generate from student (on-policy)
                prompt_enc = tokenizer(
                    prompts, return_tensors="pt", padding=True,
                    max_length=256, truncation=True,
                ).to(accelerator.device)

                with torch.no_grad():
                    gen = unwrapped_student.generate(
                        **prompt_enc,
                        max_new_tokens=max_response_length,
                        do_sample=True,
                        temperature=1.0,
                    )

                full_ids = gen
                prompt_len = prompt_enc["input_ids"].shape[1]
                full_mask = (full_ids != tokenizer.pad_token_id).long()

                # Step 2: Get teacher logits
                with torch.no_grad():
                    teacher_logits = unwrapped_teacher(
                        full_ids, attention_mask=full_mask
                    ).logits

                # Step 3: Get student logits
                student_logits = unwrapped_student(
                    full_ids, attention_mask=full_mask
                ).logits

                # Only distill on response tokens
                resp_student = student_logits[:, prompt_len:, :]
                resp_teacher = teacher_logits[:, prompt_len:, :]
                resp_mask = full_mask[:, prompt_len:]

                opd_out = opd_loss(
                    student_logits=resp_student,
                    teacher_logits=resp_teacher,
                    mask=resp_mask.float(),
                    top_k=top_k,
                    temperature=temperature,
                )

                accelerator.backward(opd_out.loss)
                if config.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(student.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % config.logging_steps == 0:
                tracker.log({
                    "loss": opd_out.loss.item(),
                    "kl_divergence": opd_out.kl_divergence.item(),
                    "top_k_coverage": opd_out.top_k_coverage.item(),
                    "epoch": epoch,
                }, step=global_step)
                tracker.print_last(["loss", "kl_divergence", "top_k_coverage"])

    unwrapped = accelerator.unwrap_model(student)
    unwrapped.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    tracker.save()

    return {"method": "opd", "seed": config.seed, "total_steps": global_step}
