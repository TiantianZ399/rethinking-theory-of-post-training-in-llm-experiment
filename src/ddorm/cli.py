"""
DDO-RM CLI — unified entry point for all training and evaluation.

Usage:
    python -m ddorm train sft --model_name ... --dataset ...
    python -m ddorm train ddorm --model_name ... --reward_model_path ...
    python -m ddorm train ppo --model_name ... --reward_model_path ...
    python -m ddorm eval --model_path ... --dataset ...
    python -m ddorm coverage --model_name ... --dataset ...
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import yaml


def load_yaml_config(path: str) -> dict:
    """Load YAML config, returning empty dict if not found."""
    if path and os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def merge_configs(*configs):
    """Merge multiple config dicts, later ones override earlier ones."""
    merged = {}
    for c in configs:
        if c:
            merged.update(c)
    return merged


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def load_dataset_by_name(name: str, split: str = "train", max_samples: int = -1):
    """Load a dataset by short name."""
    from .data.load_ultrafeedback import (
        UltraFeedbackBinarized, UltraFeedbackMultiCandidate,
    )

    if name in ("ultrafeedback_binarized", "uf_binarized"):
        ds = UltraFeedbackBinarized(split=split)
    elif name in ("ultrafeedback_4candidate", "uf_4cand"):
        ds = UltraFeedbackMultiCandidate(split=split, num_candidates=4)
    else:
        # Try as a HuggingFace dataset path
        from datasets import load_dataset as hf_load
        ds = hf_load(name, split=split)

    if max_samples > 0:
        ds = ds[:max_samples] if hasattr(ds, '__getitem__') else list(ds)[:max_samples]
    return ds


def get_candidate_generator(name: str, **kwargs):
    """Instantiate a candidate generator by name."""
    from .candidates.sampling import TemperatureSamplingGenerator, NucleusSamplingGenerator
    from .candidates.beam import BeamSearchGenerator
    from .candidates.diverse_beam import DiverseBeamSearchGenerator
    from .candidates.reward_diverse import RewardDiverseGenerator, SubmodularCoverageSelector

    generators = {
        "temperature": TemperatureSamplingGenerator,
        "nucleus": NucleusSamplingGenerator,
        "beam": BeamSearchGenerator,
        "diverse_beam": DiverseBeamSearchGenerator,
        "reward_diverse": RewardDiverseGenerator,
        "submodular": SubmodularCoverageSelector,
    }
    cls = generators.get(name)
    if cls is None:
        raise ValueError(f"Unknown generator: {name}. Options: {list(generators.keys())}")
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Train subcommands
# ---------------------------------------------------------------------------

def cmd_train_sft(args):
    """Train SFT."""
    from .train.train_sft import train_sft
    from .train.utils import TrainConfig

    config = TrainConfig(
        method="sft",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
    )

    split = args.train_split or "train_prefs"
    dataset = load_dataset_by_name(args.dataset, split=split, max_samples=args.max_samples)
    result = train_sft(args.model_name, dataset, config, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))


def cmd_train_reward_model(args):
    """Train reward model."""
    from .train.train_reward_model import train_reward_model
    from .train.utils import TrainConfig

    config = TrainConfig(
        method="reward_model",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
    )

    split = args.train_split or "train_prefs"
    dataset = load_dataset_by_name(args.dataset, split=split, max_samples=args.max_samples)
    result = train_reward_model(args.model_name, dataset, config, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))


def cmd_train_dpo(args):
    """Train DPO."""
    from .train.train_dpo import train_dpo
    from .train.utils import TrainConfig

    config = TrainConfig(
        method="dpo",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
    )

    split = args.train_split or "train_prefs"
    dataset = load_dataset_by_name(args.dataset, split=split, max_samples=args.max_samples)
    ref_path = args.ref_model_path or args.model_name
    result = train_dpo(
        args.model_name, dataset, config,
        ref_model_path=ref_path,
        output_dir=args.output_dir,
        beta=args.beta,
    )
    print(json.dumps(result, indent=2))


def cmd_train_ddorm(args):
    """Train DDO-RM."""
    from .train.train_ddorm import train_ddorm
    from .train.utils import TrainConfig

    config = TrainConfig(
        method="ddorm",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
    )

    split = args.train_split or "train_prefs"
    dataset = load_dataset_by_name(args.dataset, split=split, max_samples=args.max_samples)

    cand_gen = None
    if args.candidate_generator:
        gen_kwargs = {}
        if args.gen_temperature:
            gen_kwargs["temperature"] = args.gen_temperature
        if args.gen_top_p:
            gen_kwargs["top_p"] = args.gen_top_p
        cand_gen = get_candidate_generator(args.candidate_generator, **gen_kwargs)

    result = train_ddorm(
        args.model_name,
        args.reward_model_path,
        dataset,
        config,
        output_dir=args.output_dir,
        lam=args.lambda_temp,
        tau=args.tau,
        num_candidates=args.num_candidates,
        candidate_generator=cand_gen,
    )
    print(json.dumps(result, indent=2))


def cmd_train_ppo(args):
    """Train PPO-RLHF."""
    from .train.train_ppo_rlhf import train_ppo_rlhf
    from .train.utils import TrainConfig

    config = TrainConfig(
        method="ppo_rlhf",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
    )

    split = args.train_split or "train_prefs"
    dataset = load_dataset_by_name(args.dataset, split=split, max_samples=args.max_samples)
    ref_path = args.ref_model_path or args.model_name

    result = train_ppo_rlhf(
        model_name=args.model_name,
        reward_model_path=args.reward_model_path,
        ref_model_path=ref_path,
        dataset=dataset,
        config=config,
        output_dir=args.output_dir,
        kl_coeff=args.kl_coeff,
        clip_range=args.clip_range,
        value_clip_range=args.value_clip_range,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ppo_epochs=args.ppo_epochs,
        max_gen_len=args.max_gen_len,
    )
    print(json.dumps(result, indent=2))


def cmd_train_grpo(args):
    """Train GRPO."""
    from .train.train_grpo import train_grpo
    from .train.utils import TrainConfig

    config = TrainConfig(
        method="grpo",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
    )

    split = args.train_split or "train"
    dataset = load_dataset_by_name(args.dataset, split=split, max_samples=args.max_samples)

    result = train_grpo(
        args.model_name, dataset, config,
        output_dir=args.output_dir,
        group_size=args.num_candidates,
        kl_coeff=args.kl_coeff,
        clip_range=args.clip_range,
    )
    print(json.dumps(result, indent=2))


def cmd_train_opd(args):
    """Train OPD."""
    from .train.train_opd import train_opd
    from .train.utils import TrainConfig

    config = TrainConfig(
        method="opd",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
    )

    split = args.train_split or "train_prefs"
    dataset = load_dataset_by_name(args.dataset, split=split, max_samples=args.max_samples)

    result = train_opd(
        args.model_name,
        args.teacher_model_path,
        dataset,
        config,
        output_dir=args.output_dir,
        top_k=args.opd_top_k,
        tau=args.tau,
    )
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Eval subcommand
# ---------------------------------------------------------------------------

def cmd_eval(args):
    """Run evaluation on a trained model."""
    from .eval.pairwise_eval import pairwise_accuracy, pairwise_auc, pairwise_margin
    from .eval.listwise_eval import ndcg_at_k, kendall_tau, top1_accuracy, expected_reward
    from .eval.coverage_eval import coverage_evaluation
    from .eval.kl_eval import kl_evaluation
    from .eval.bootstrap import bootstrap_ci
    from .models.policy import PolicyModel
    from .models.reward_model import RewardModel

    split = args.eval_split or "test_prefs"
    dataset = load_dataset_by_name(args.dataset, split=split, max_samples=args.max_samples)

    policy = PolicyModel(args.model_path)
    results = {}

    # Pairwise metrics
    if hasattr(dataset[0], 'get') and 'chosen' in dataset[0]:
        prompts = [d["prompt"] for d in dataset]
        chosen = [d["chosen"] for d in dataset]
        rejected = [d["rejected"] for d in dataset]

        chosen_scores = policy.batch_score(prompts, [[c] for c in chosen]).squeeze(-1)
        rejected_scores = policy.batch_score(prompts, [[r] for r in rejected]).squeeze(-1)

        import torch
        pair_acc = pairwise_accuracy(chosen_scores, rejected_scores)
        auc = pairwise_auc(chosen_scores, rejected_scores)
        margin = pairwise_margin(chosen_scores, rejected_scores)

        results["pair_accuracy"] = pair_acc
        results["auc"] = auc
        results["margin"] = margin.mean().item()

        # Bootstrap CIs
        acc_values = (chosen_scores > rejected_scores).float().cpu().numpy()
        results["pair_accuracy_ci"] = bootstrap_ci(acc_values)

    # Reward model eval
    if args.reward_model_path:
        rm = RewardModel(args.reward_model_path)
        # Could score and compute reward-based metrics here

    # KL eval
    if args.ref_model_path:
        kl_result = kl_evaluation(
            policy_model_name=args.model_path,
            ref_model_name=args.ref_model_path,
            dataset=dataset,
        )
        results["kl_divergence"] = kl_result

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps(results, indent=2, default=str))


# ---------------------------------------------------------------------------
# Coverage subcommand
# ---------------------------------------------------------------------------

def cmd_coverage(args):
    """Run coverage diagnostics."""
    from .theory import estimate_coverage, coverage_curve
    from .models.policy import PolicyModel
    from .models.reward_model import RewardModel
    import torch

    dataset = load_dataset_by_name(
        args.dataset,
        split=args.eval_split or "test_prefs",
        max_samples=args.max_samples,
    )

    policy = PolicyModel(args.model_name)
    rm = RewardModel(args.reward_model_path)

    results = {"curves": []}

    for K in args.k_values:
        # Generate large reference pool
        # For each prompt, sample M candidates
        M = args.reference_pool_size

        for i, item in enumerate(dataset):
            if i >= args.max_prompts:
                break

            prompt = item["prompt"] if isinstance(item, dict) else item

            # Score reference pool
            # (In practice this would generate M candidates and score them)
            # Here we use a placeholder computation
            ref_scores = torch.randn(M)  # placeholder
            ref_rewards = torch.randn(M)  # placeholder
            candidate_indices = list(range(K))

            cov = estimate_coverage(ref_scores, ref_rewards, candidate_indices, lam=args.lambda_temp)
            results["curves"].append({
                "K": K, "prompt_idx": i, "coverage": cov,
            })

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "coverage_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps(results, indent=2, default=str))


# ---------------------------------------------------------------------------
# Reward noise subcommand
# ---------------------------------------------------------------------------

def cmd_reward_noise(args):
    """Run reward noise sweep experiment."""
    from .theory.reward_noise import run_reward_noise_experiment
    import torch

    dataset = load_dataset_by_name(
        args.dataset,
        split=args.eval_split or "test_prefs",
        max_samples=args.max_samples,
    )

    # Load policy scores and rewards from a pre-trained model
    # In practice, these come from the model; here we set up the sweep
    results = run_reward_noise_experiment(
        policy_scores=torch.randn(100, args.num_candidates),  # placeholder
        reward_scores=torch.randn(100, args.num_candidates),
        lam=args.lambda_temp,
        sigma_range=[0.0, 0.1, 0.2, 0.5, 1.0, 2.0],
        noise_types=["gaussian", "uniform", "scale", "rank_preserving", "adversarial_top"],
    )

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "reward_noise_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps(results, indent=2, default=str))


# ---------------------------------------------------------------------------
# Run experiment from config
# ---------------------------------------------------------------------------

def cmd_run_experiment(args):
    """Run a full experiment from a YAML config file."""
    cfg = load_yaml_config(args.config)
    if not cfg:
        print(f"ERROR: Could not load config from {args.config}")
        sys.exit(1)

    experiment = cfg.get("experiment", "unknown")
    methods = cfg.get("methods", [])
    seeds = cfg.get("seeds", [42])
    model = cfg.get("model_name", cfg.get("model", ""))
    dataset = cfg.get("dataset", cfg.get("data", "ultrafeedback_binarized"))
    output_base = cfg.get("output_dir", f"results/{experiment}")

    print(f"Running experiment: {experiment}")
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}")
    print(f"Model: {model}")

    all_results = []

    for seed in seeds:
        for method in methods:
            output_dir = f"{output_base}/{method}_seed{seed}"
            print(f"\n{'='*60}")
            print(f"Method: {method} | Seed: {seed}")
            print(f"{'='*60}")

            # Build method-specific args
            method_cfg = cfg.get(f"{method}_config", {})
            cmd_args = [
                "train", method,
                "--model_name", model,
                "--dataset", dataset,
                "--seed", str(seed),
                "--output_dir", output_dir,
            ]
            for k, v in method_cfg.items():
                cmd_args.extend([f"--{k}", str(v)])

            # Parse and run
            sub_args = build_parser().parse_args(cmd_args)
            try:
                sub_args.func(sub_args)
                all_results.append({
                    "method": method, "seed": seed, "status": "success",
                    "output_dir": output_dir,
                })
            except Exception as e:
                print(f"ERROR in {method} seed={seed}: {e}")
                all_results.append({
                    "method": method, "seed": seed, "status": "failed",
                    "error": str(e),
                })

    out_path = Path(output_base)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "experiment_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nExperiment {experiment} complete. Results in {output_base}/")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def add_common_train_args(parser):
    """Add common training arguments."""
    parser.add_argument("--model_name", required=True, help="Model name or path")
    parser.add_argument("--dataset", default="ultrafeedback_binarized")
    parser.add_argument("--train_split", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--wandb", action="store_true")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ddorm",
        description="DDO-RM: Support-Restricted Entropy Policy Improvement",
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # ---- train ----
    train_parser = sub.add_parser("train", help="Train a model")
    train_sub = train_parser.add_subparsers(dest="method", help="Training method")

    # SFT
    p = train_sub.add_parser("sft", help="Supervised fine-tuning")
    add_common_train_args(p)
    p.set_defaults(func=cmd_train_sft)

    # Reward model
    p = train_sub.add_parser("reward_model", aliases=["rm"], help="Train reward model")
    add_common_train_args(p)
    p.set_defaults(func=cmd_train_reward_model)

    # DPO
    p = train_sub.add_parser("dpo", help="Direct Preference Optimization")
    add_common_train_args(p)
    p.add_argument("--ref_model_path", default=None)
    p.add_argument("--beta", type=float, default=0.1)
    p.set_defaults(func=cmd_train_dpo)

    # DDO-RM
    p = train_sub.add_parser("ddorm", help="DDO-RM training")
    add_common_train_args(p)
    p.add_argument("--reward_model_path", required=True)
    p.add_argument("--lambda_temp", type=float, default=1.0, help="Entropy temperature λ")
    p.add_argument("--tau", type=float, default=1.0, help="Policy temperature τ")
    p.add_argument("--num_candidates", type=int, default=4, help="Candidate set size K")
    p.add_argument("--candidate_generator", default=None,
                    choices=["temperature", "nucleus", "beam", "diverse_beam",
                             "reward_diverse", "submodular"])
    p.add_argument("--gen_temperature", type=float, default=None)
    p.add_argument("--gen_top_p", type=float, default=None)
    p.set_defaults(func=cmd_train_ddorm)

    # PPO-RLHF
    p = train_sub.add_parser("ppo", aliases=["ppo_rlhf"], help="PPO-RLHF training")
    add_common_train_args(p)
    p.add_argument("--reward_model_path", required=True)
    p.add_argument("--ref_model_path", default=None)
    p.add_argument("--kl_coeff", type=float, default=0.1)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--value_clip_range", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--max_gen_len", type=int, default=256)
    p.set_defaults(func=cmd_train_ppo)

    # GRPO
    p = train_sub.add_parser("grpo", help="Group Relative Policy Optimization")
    add_common_train_args(p)
    p.add_argument("--num_candidates", type=int, default=8)
    p.add_argument("--kl_coeff", type=float, default=0.1)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.set_defaults(func=cmd_train_grpo)

    # OPD
    p = train_sub.add_parser("opd", help="On-Policy Distillation")
    add_common_train_args(p)
    p.add_argument("--teacher_model_path", required=True)
    p.add_argument("--opd_top_k", type=int, default=50)
    p.add_argument("--tau", type=float, default=1.0)
    p.set_defaults(func=cmd_train_opd)

    # ---- eval ----
    p = sub.add_parser("eval", help="Evaluate a model")
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset", default="ultrafeedback_binarized")
    p.add_argument("--eval_split", default=None)
    p.add_argument("--reward_model_path", default=None)
    p.add_argument("--ref_model_path", default=None)
    p.add_argument("--output_dir", default="results/eval")
    p.add_argument("--max_samples", type=int, default=-1)
    p.set_defaults(func=cmd_eval)

    # ---- coverage ----
    p = sub.add_parser("coverage", help="Run coverage diagnostics")
    p.add_argument("--model_name", required=True)
    p.add_argument("--reward_model_path", required=True)
    p.add_argument("--dataset", default="ultrafeedback_binarized")
    p.add_argument("--eval_split", default=None)
    p.add_argument("--lambda_temp", type=float, default=1.0)
    p.add_argument("--k_values", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--reference_pool_size", type=int, default=128)
    p.add_argument("--max_prompts", type=int, default=50)
    p.add_argument("--output_dir", default="results/coverage")
    p.add_argument("--max_samples", type=int, default=-1)
    p.set_defaults(func=cmd_coverage)

    # ---- reward_noise ----
    p = sub.add_parser("reward_noise", help="Run reward noise sweep")
    p.add_argument("--dataset", default="ultrafeedback_binarized")
    p.add_argument("--eval_split", default=None)
    p.add_argument("--lambda_temp", type=float, default=1.0)
    p.add_argument("--num_candidates", type=int, default=4)
    p.add_argument("--output_dir", default="results/reward_noise")
    p.add_argument("--max_samples", type=int, default=-1)
    p.set_defaults(func=cmd_reward_noise)

    # ---- run (experiment from config) ----
    p = sub.add_parser("run", help="Run experiment from YAML config")
    p.add_argument("--config", required=True, help="Path to experiment YAML config")
    p.set_defaults(func=cmd_run_experiment)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if not hasattr(args, 'func'):
        parser.parse_args([args.command, "--help"])
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
