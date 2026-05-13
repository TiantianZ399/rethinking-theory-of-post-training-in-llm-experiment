"""
Shared training utilities for all methods.

Provides: optimizer setup, LR scheduling, logging, checkpointing,
metric tracking, and seed management.
"""

import os
import json
import time
import random
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


@dataclass
class TrainConfig:
    """Base training configuration."""
    method: str = "ddorm"
    learning_rate: float = 5e-6
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    output_dir: str = "results"
    run_name: str = ""
    use_wandb: bool = False

    # Method-specific (set by subclasses)
    extra: dict = field(default_factory=dict)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, lr: float, weight_decay: float = 0.01):
    """Build AdamW optimizer with weight decay exclusion for bias/norm."""
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    params = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)


def build_scheduler(
    optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
):
    """Build linear warmup + cosine decay scheduler."""
    warmup_steps = int(num_training_steps * warmup_ratio)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=num_training_steps - warmup_steps)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


class MetricTracker:
    """Track and log training metrics."""

    def __init__(self, output_dir: str, run_name: str = "", use_wandb: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.use_wandb = use_wandb and HAS_WANDB
        self.metrics = []
        self.step = 0

        if self.use_wandb:
            wandb.init(project="ddorm", name=run_name)

    def log(self, metrics: dict, step: Optional[int] = None):
        """Log metrics at a given step."""
        if step is not None:
            self.step = step
        metrics["step"] = self.step
        metrics["timestamp"] = time.time()
        self.metrics.append(metrics)

        if self.use_wandb:
            wandb.log(metrics, step=self.step)

        self.step += 1

    def save(self, filename: str = "metrics.json"):
        """Save all metrics to JSON."""
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def print_last(self, keys: Optional[list[str]] = None):
        """Print the last logged metrics."""
        if not self.metrics:
            return
        last = self.metrics[-1]
        if keys:
            last = {k: v for k, v in last.items() if k in keys}
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in last.items()]
        print(f"[Step {last.get('step', '?')}] " + " | ".join(parts))


def save_checkpoint(model, optimizer, step: int, output_dir: str, name: str = "checkpoint"):
    """Save model checkpoint."""
    path = Path(output_dir) / f"{name}_step{step}"
    path.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path)
    else:
        torch.save(model.state_dict(), path / "model.pt")

    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    torch.save({"step": step}, path / "train_state.pt")
    print(f"Saved checkpoint to {path}")


def save_results(results: dict, output_dir: str, filename: str = "results.json"):
    """Save experiment results."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / filename, "w") as f:
        json.dump(results, f, indent=2)
