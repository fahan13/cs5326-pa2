"""Training plumbing shared by SFT and DPO: seeding, optimizer, schedule, logging, saving."""
from __future__ import annotations

import json
import math
import random
import subprocess
import time
from pathlib import Path

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(model, name: str, lr: float, weight_decay: float):
    """AdamW or RMSprop. Weight decay skips 1-D tensors (biases, norm scales): shrinking
    those toward zero does not regularise anything, it just distorts the activations."""
    decay, no_decay = [], []
    for param in model.parameters():
        if param.requires_grad:
            (decay if param.ndim >= 2 else no_decay).append(param)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    on_cuda = next(model.parameters()).is_cuda
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=on_cuda)
    if name == "rmsprop":
        return torch.optim.RMSprop(groups, lr=lr, alpha=0.99, eps=1e-8)
    raise ValueError(f"unknown optimizer {name}")


def warmup_cosine(total_steps: int, warmup_steps: int, final_ratio: float = 0.0):
    """LR multiplier: linear 0 -> 1 over warmup, then cosine 1 -> final_ratio."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return factor


def constant_with_warmup(warmup_steps: int):
    def factor(step: int) -> float:
        return min(1.0, (step + 1) / max(warmup_steps, 1)) if warmup_steps else 1.0
    return factor


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        return "unknown"


class MetricsLog:
    """Appends one JSON object per line and flushes immediately, so a Kaggle disconnect
    keeps every metric written up to that moment (the manual asks for exactly this)."""

    def __init__(self, path: Path):
        self.path = path
        self.start = time.time()

    def write(self, **record):
        record["elapsed_s"] = round(time.time() - self.start, 1)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record


def save_checkpoint(model, tokenizer, path: Path, info: dict) -> None:
    """Weights + tokenizer + a small JSON saying which step/metric produced this checkpoint."""
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    (path / "checkpoint_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def gpu_memory_gb() -> float:
    return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0
