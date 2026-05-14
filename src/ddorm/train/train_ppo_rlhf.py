"""
PPO-RLHF training loop.

Proper PPO with:
- Value head for advantage estimation
- Rollout buffer with KL penalty
- GAE for advantage computation
- Clipped surrogate objective
- Reward normalization
- Matched compute budget tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from dataclasses import dataclass

from ..objectives.ppo_rlhf import compute_gae, ppo_loss, compute_kl_penalty
from ..models.reward_model import RewardModel
from ..models.reference_model import ReferenceModel
from .utils import (
    TrainConfig, set_seed, build_optimizer, build_scheduler,
    MetricTracker, save_checkpoint,
)


class PolicyWithValueHead(nn.Module):
    """Causal LM with a learned value head for PPO."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = outputs.logits
        hidden = outputs.hidden_states[-1]
        values = self.value_head(hidden).squeeze(-1)
        return logits, values

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)


@dataclass
class RolloutBatch:
    """A batch of PPO rollout data."""
    query_ids: torch.Tensor         # [B, L_q]
    response_ids: torch.Tensor      # [B, L_r]
    old_logprobs: torch.Tensor      # [B, L_r]
    ref_logprobs: torch.Tensor      # [B, L_r]
    rewards: torch.Tensor           # [B]
    old_values: torch.Tensor        # [B, L_r]
    advantages: torch.Tensor        # [B, L_r]
    returns: torch.Tensor           # [B, L_r]
    response_mask: torch.Tensor     # [B, L_r]


def collect_rollouts(
    model: PolicyWithValueHead,
    ref_model: ReferenceModel,
    reward_model: RewardModel,
    tokenizer,
    prompts: list[str],
    max_response_length: int = 256,
    kl_coeff: float = 0.1,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> RolloutBatch:
    """
    Collect rollouts: generate responses, compute rewards, compute advantages.

    Args:
        model: policy with value head
        ref_model: frozen reference model
        reward_model: reward model for scoring
        tokenizer: tokenizer
        prompts: list of prompt strings
        max_response_length: max tokens to generate
        kl_coeff: KL penalty coefficient
        gamma: discount factor
        gae_lambda: GAE lambda

    Returns:
        RolloutBatch
    """
    model.eval()
    device = next(model.parameters()).device

    all_query_ids = []
    all_response_ids = []
    all_old_logprobs = []
    all_ref_logprobs = []
    all_rewards = []
    all_old_values = []
    all_masks = []

    for prompt in prompts:
        # Tokenize prompt
        query_enc = tokenizer(prompt, return_tensors="pt").to(device)
        query_ids = query_enc["input_ids"]

        # Generate response
        with torch.no_grad():
            gen_output = model.generate(
                query_ids,
                max_new_tokens=max_response_length,
                do_sample=True,
                temperature=1.0,
                return_dict_in_generate=True,
                output_scores=True,
            )

        response_ids = gen_output.sequences[:, query_ids.shape[1]:]
        resp_len = response_ids.shape[1]

        if resp_len == 0:
            continue

        # Full sequence
        full_ids = gen_output.sequences

        # Get old logprobs and values
        with torch.no_grad():
            logits, values = model(full_ids)

        # Response portion logprobs
        shift_logits = logits[:, query_ids.shape[1] - 1:-1, :]
        shift_labels = full_ids[:, query_ids.shape[1]:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        old_lps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        old_vals = values[:, query_ids.shape[1]:]
        if old_vals.shape[1] > resp_len:
            old_vals = old_vals[:, :resp_len]

        # Reference logprobs
        ref_lps = ref_model.log_probs(full_ids, prompt_length=query_ids.shape[1])
        if ref_lps.shape[1] > resp_len:
            ref_lps = ref_lps[:, :resp_len]

        # Reward
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        rm_score = reward_model.score([prompt], [[response_text]])[0, 0]

        # Per-token rewards: KL penalty at each token, RM reward at last token
        kl_penalty = compute_kl_penalty(old_lps, ref_lps, kl_coeff)
        per_token_rewards = -kl_penalty
        per_token_rewards[:, -1] += rm_score

        mask = torch.ones_like(old_lps)

        all_query_ids.append(query_ids.squeeze(0))
        all_response_ids.append(response_ids.squeeze(0))
        all_old_logprobs.append(old_lps.squeeze(0))
        all_ref_logprobs.append(ref_lps.squeeze(0))
        all_rewards.append(per_token_rewards.squeeze(0))
        all_old_values.append(old_vals.squeeze(0))
        all_masks.append(mask.squeeze(0))

    if not all_response_ids:
        raise RuntimeError("No valid rollouts collected")

    # Pad to same length
    max_resp = max(r.shape[0] for r in all_response_ids)

    def pad_1d(tensors, max_len, pad_val=0):
        return torch.stack([
            F.pad(t, (0, max_len - t.shape[0]), value=pad_val) for t in tensors
        ])

    response_ids = pad_1d(all_response_ids, max_resp)
    old_logprobs = pad_1d(all_old_logprobs, max_resp)
    ref_logprobs = pad_1d(all_ref_logprobs, max_resp)
    rewards = pad_1d(all_rewards, max_resp)
    old_values = pad_1d(all_old_values, max_resp)
    masks = pad_1d(all_masks, max_resp)

    # Compute GAE
    advantages, returns = compute_gae(rewards, old_values, gamma, gae_lambda)

    # Pad query ids
    max_query = max(q.shape[0] for q in all_query_ids)
    query_ids = pad_1d(all_query_ids, max_query, tokenizer.pad_token_id)

    return RolloutBatch(
        query_ids=query_ids,
        response_ids=response_ids,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
        rewards=all_rewards[0].new_tensor([r[-1].item() for r in all_rewards]),
        old_values=old_values,
        advantages=advantages,
        returns=returns,
        response_mask=masks,
    )


def train_ppo_rlhf(
    model_name: str,
    reward_model_path: str,
    ref_model_path: str,
    dataset,
    config: TrainConfig,
    output_dir: str = "results/ppo_rlhf",
    kl_coeff: float = 0.1,
    clip_range: float = 0.2,
    vf_coeff: float = 0.5,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    num_rollouts_per_step: int = 32,
    num_ppo_epochs: int = 4,
):
    """
    Train policy via PPO-RLHF.

    Args:
        model_name: SFT checkpoint path
        reward_model_path: RM path
        ref_model_path: frozen reference model path
        dataset: prompt dataset
        config: training config
        output_dir: save directory
        kl_coeff: KL penalty coefficient
        clip_range: PPO clip range
        vf_coeff: value function coefficient
        gamma: discount factor
        gae_lambda: GAE lambda
        num_rollouts_per_step: rollouts per PPO step
        num_ppo_epochs: PPO epochs per rollout batch
    """
    set_seed(config.seed)
    accelerator = Accelerator()

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    )
    model = PolicyWithValueHead(base_model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model.config.pad_token_id = tokenizer.pad_token_id

    rm = RewardModel(reward_model_path)
    ref = ReferenceModel(ref_model_path)

    optimizer = build_optimizer(model, config.learning_rate, config.weight_decay)

    # Prompt dataloader
    prompt_list = []
    for item in dataset:
        if isinstance(item, dict):
            prompt_list.append(item["prompt"])
        else:
            prompt_list.append(str(item))

    tracker = MetricTracker(output_dir, f"ppo_seed{config.seed}", config.use_wandb)
    global_step = 0
    prompt_idx = 0

    num_total_steps = config.num_epochs * (len(prompt_list) // num_rollouts_per_step)

    for step in range(num_total_steps):
        # Sample prompts
        batch_prompts = []
        for _ in range(num_rollouts_per_step):
            batch_prompts.append(prompt_list[prompt_idx % len(prompt_list)])
            prompt_idx += 1

        # Collect rollouts
        rollout = collect_rollouts(
            model, ref, rm, tokenizer, batch_prompts,
            kl_coeff=kl_coeff, gamma=gamma, gae_lambda=gae_lambda,
        )

        # PPO update epochs
        for ppo_epoch in range(num_ppo_epochs):
            model.train()

            # Full sequence for forward pass
            full_ids = torch.cat([rollout.query_ids, rollout.response_ids], dim=1)
            full_mask = torch.ones_like(full_ids)

            logits, values = model(full_ids, full_mask)

            # Response portion
            q_len = rollout.query_ids.shape[1]
            resp_logits = logits[:, q_len - 1:-1, :]
            resp_labels = full_ids[:, q_len:]
            new_log_probs = F.log_softmax(resp_logits, dim=-1)
            new_lps = new_log_probs.gather(-1, resp_labels.unsqueeze(-1)).squeeze(-1)

            new_values = values[:, q_len:]
            if new_values.shape[1] > rollout.response_mask.shape[1]:
                new_values = new_values[:, :rollout.response_mask.shape[1]]
            if new_lps.shape[1] > rollout.response_mask.shape[1]:
                new_lps = new_lps[:, :rollout.response_mask.shape[1]]

            ppo_out = ppo_loss(
                new_logprobs=new_lps,
                old_logprobs=rollout.old_logprobs,
                advantages=rollout.advantages,
                returns=rollout.returns,
                new_values=new_values,
                clip_range=clip_range,
                vf_coeff=vf_coeff,
                mask=rollout.response_mask,
            )

            ppo_out.loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        global_step += 1

        if global_step % config.logging_steps == 0:
            tracker.log({
                "loss": ppo_out.loss.item(),
                "policy_loss": ppo_out.policy_loss.item(),
                "value_loss": ppo_out.value_loss.item(),
                "kl": ppo_out.kl.item(),
                "entropy": ppo_out.entropy.item(),
                "clip_fraction": ppo_out.clip_fraction.item(),
                "mean_reward": rollout.rewards.mean().item(),
            }, step=global_step)
            tracker.print_last(["loss", "mean_reward", "kl", "clip_fraction"])

        if global_step % config.save_steps == 0:
            save_checkpoint(model, optimizer, global_step, output_dir, "ppo")

    # Save final
    model.base_model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    tracker.save()

    return {"method": "ppo_rlhf", "seed": config.seed, "total_steps": global_step}
