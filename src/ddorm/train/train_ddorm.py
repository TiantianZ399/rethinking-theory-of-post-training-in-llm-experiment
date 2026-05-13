"""
DDO-RM training loop.

Support-restricted entropy policy improvement:
1. Generate candidates C_K(x)
2. Score with reward model: r_i = r(x, y_i)
3. Compute policy scores: p_i = softmax(s_i / τ)
4. Compute target: q_i* ∝ p_i exp(r_i / λ)
5. Distill: L = -Σ q_i* log p_θ,i
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

from ..objectives import ddorm_forward
from ..models.reward_model import RewardModel
from ..models.policy import PolicyModel
from .utils import (
    TrainConfig, set_seed, build_optimizer, build_scheduler,
    MetricTracker, save_checkpoint,
)


def train_ddorm(
    model_name: str,
    reward_model_path: str,
    dataset,
    config: TrainConfig,
    output_dir: str = "results/ddorm",
    lam: float = 1.0,
    tau: float = 1.0,
    num_candidates: int = 4,
    candidate_generator=None,
):
    """
    Train policy via DDO-RM.

    Args:
        model_name: SFT checkpoint path
        reward_model_path: reward model path
        dataset: dataset with prompts (and optionally pre-generated candidates)
        config: training config
        output_dir: save directory
        lam: entropy temperature λ
        tau: policy temperature τ
        num_candidates: K (ignored if dataset has pre-generated candidates)
        candidate_generator: optional CandidateGenerator instance
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
        model.config.pad_token_id = tokenizer.pad_token_id

    rm = RewardModel(reward_model_path)
    policy = PolicyModel(model_name)

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
    )

    optimizer = build_optimizer(model, config.learning_rate, config.weight_decay)
    num_steps = len(dataloader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = build_scheduler(optimizer, num_steps, config.warmup_ratio)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    tracker = MetricTracker(output_dir, f"ddorm_seed{config.seed}", config.use_wandb)
    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()

        for batch_items in dataloader:
            # Get prompts and candidates
            if isinstance(batch_items, dict):
                prompts = batch_items["prompt"]
                if "candidates" in batch_items:
                    candidates = batch_items["candidates"]
                    ratings = batch_items.get("ratings", None)
                else:
                    # Generate candidates on-the-fly
                    candidates = None
                    ratings = None
            else:
                prompts = [item["prompt"] for item in batch_items]
                candidates = None
                ratings = None

            # Make prompts a list if it's a single string
            if isinstance(prompts, str):
                prompts = [prompts]

            with accelerator.accumulate(model):
                unwrapped = accelerator.unwrap_model(model)

                # Step 1: Get or generate candidates
                if candidates is not None:
                    # Pre-generated candidates from dataset
                    cand_list = candidates
                    if isinstance(cand_list[0], str):
                        # Single candidate per item, wrap
                        cand_list = [[c] for c in cand_list]
                elif candidate_generator is not None:
                    gen_out = candidate_generator.generate(
                        prompts, unwrapped, tokenizer, num_candidates
                    )
                    cand_list = gen_out.candidates
                else:
                    # Fallback: use chosen/rejected as 2 candidates
                    if isinstance(batch_items, dict) and "chosen" in batch_items:
                        cand_list = [
                            [c, r] for c, r in
                            zip(batch_items["chosen"], batch_items["rejected"])
                        ]
                    else:
                        raise ValueError(
                            "No candidates available. Provide dataset with candidates, "
                            "or a candidate_generator, or pairwise data."
                        )

                K = len(cand_list[0])
                B = len(prompts)

                # Step 2: Compute policy scores s_i for each candidate
                policy_scores = policy.batch_score(prompts, cand_list)  # [B, K]

                # Step 3: Compute reward scores r_i
                if ratings is not None:
                    reward_scores = ratings.to(policy_scores.device)
                else:
                    reward_scores = rm.score(prompts, cand_list)  # [B, K]

                # Step 4 & 5: DDO-RM forward (target + loss)
                ddorm_out = ddorm_forward(
                    policy_scores=policy_scores.detach(),
                    reward_scores=reward_scores.detach(),
                    lam=lam,
                    tau=tau,
                    policy_scores_new=policy_scores,  # gradients flow here
                )

                accelerator.backward(ddorm_out.loss)
                if config.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % config.logging_steps == 0:
                tracker.log({
                    "loss": ddorm_out.loss.item(),
                    "reward_weighted": ddorm_out.reward_weighted.mean().item(),
                    "kl_pq": ddorm_out.kl_pq.mean().item(),
                    "q_entropy": -(ddorm_out.q_target * (ddorm_out.q_target + 1e-12).log()).sum(-1).mean().item(),
                    "epoch": epoch,
                }, step=global_step)
                tracker.print_last(["loss", "reward_weighted", "kl_pq"])

            if global_step % config.save_steps == 0:
                unwrapped = accelerator.unwrap_model(model)
                save_checkpoint(unwrapped, optimizer, global_step, output_dir, "ddorm")

    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    tracker.save()

    return {"method": "ddorm", "seed": config.seed, "total_steps": global_step}
