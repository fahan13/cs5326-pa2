"""Supervised fine-tuning (Task 3). Our own loop: no HF Trainer, no TRL.

Full run on Kaggle (one T4, guide configuration):
    python scripts/train_sft.py --run_name sft

Pilot on a laptop (tiny model, 60 steps, just to prove the loop works):
    python scripts/train_sft.py --run_name pilot_sft --model HuggingFaceTB/SmolLM2-135M \
        --max_steps 60 --eval_every 20 --val_limit 64

What one optimizer step does
    1. Take grad_accum micro-batches (micro_batch rows each) -> effective batch of 32 rows.
    2. Count the response tokens in ALL of them first, so the loss is the mean over every
       response token in the effective batch (our documented reduction), not a mean of means.
    3. Forward each micro-batch under fp16 autocast, scale the loss, backprop (grads add up).
    4. Unscale, clip the global grad norm to 1.0, optimizer step, LR schedule step.

Writes to outputs/<run_name>/: config.json, metrics.jsonl, best/, final/.
Small files (config + metrics) are also copied to results/runs/<run_name>/ for git.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from alignment import core  # noqa: E402
from alignment.data import read_jsonl_gz  # noqa: E402
from alignment.prompts import PROMPT_TEMPLATE, clean_generation, format_prompt  # noqa: E402
from alignment.training import (  # noqa: E402
    MetricsLog, git_commit, gpu_memory_gb, make_optimizer, save_checkpoint, set_seed, warmup_cosine,
)

SAMPLE_INSTRUCTIONS = [
    "Give three tips for staying focused while studying.",
    "What is the capital of Australia?",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--train_file", default=str(REPO_ROOT / "data/processed/sft_train.jsonl.gz"))
    p.add_argument("--val_file", default=str(REPO_ROOT / "data/processed/sft_val.jsonl.gz"))
    p.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    # guide configuration (manual section 9.2)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--micro_batch", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--optimizer", choices=["adamw", "rmsprop"], default="adamw")
    p.add_argument("--seed", type=int, default=42)
    # evaluation / logging
    p.add_argument("--eval_every", type=int, default=125, help="optimizer steps between validations")
    p.add_argument("--eval_batch", type=int, default=8)
    p.add_argument("--log_every", type=int, default=10)
    # pilots and memory
    p.add_argument("--max_steps", type=int, help="stop early (pilot runs)")
    p.add_argument("--train_limit", type=int, help="use only the first N training rows")
    p.add_argument("--val_limit", type=int, help="use only the first N validation rows")
    p.add_argument("--gradient_checkpointing", action="store_true", help="less memory, ~30%% slower")
    p.add_argument("--no_amp", action="store_true", help="disable fp16 autocast (debugging)")
    return p.parse_args()


def micro_batches(rows, tokenizer, size, max_length, generator):
    """Shuffle once per epoch with a seeded generator, then cut into micro-batches."""
    order = torch.randperm(len(rows), generator=generator).tolist()
    for i in range(0, len(order), size):
        chunk = [rows[j] for j in order[i : i + size]]
        yield core.build_lm_batch(
            tokenizer, [format_prompt(r["prompt"]) for r in chunk], [r["response"] for r in chunk], max_length
        )


@torch.no_grad()
def validate(model, tokenizer, rows, args, amp):
    """Token-mean response NLL over the whole validation set (same reduction as training)."""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for i in range(0, len(rows), args.eval_batch):
        chunk = rows[i : i + args.eval_batch]
        batch = core.build_lm_batch(
            tokenizer, [format_prompt(r["prompt"]) for r in chunk], [r["response"] for r in chunk], args.max_length
        )
        with amp():
            logps, counts = core.response_logps(model, batch)
        total_nll += -logps.sum().item()
        total_tokens += counts.sum().item()
    model.train()
    return total_nll / max(total_tokens, 1)


@torch.no_grad()
def sample(model, tokenizer, device, amp, max_new_tokens=60):
    """Two short greedy answers, printed at every validation, to watch the model learn."""
    model.eval()
    out = []
    for instruction in SAMPLE_INSTRUCTIONS:
        enc = tokenizer(format_prompt(instruction), return_tensors="pt", add_special_tokens=False).to(device)
        with amp():
            ids = model.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        text = tokenizer.decode(ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        out.append(clean_generation(text)[0])
    model.train()
    return out


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and not args.no_amp

    def amp():
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)

    run_dir = Path(args.output_root) / args.run_name
    if (run_dir / "final").exists():
        sys.exit(f"{run_dir} already has a final checkpoint; pick a new --run_name")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.jsonl").unlink(missing_ok=True)

    train_rows = read_jsonl_gz(args.train_file)[: args.train_limit]
    val_rows = read_jsonl_gz(args.val_file)[: args.val_limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # fp32 master weights; autocast runs the matmuls in fp16 (manual 9.1, points 5-6)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    effective_batch = args.micro_batch * args.grad_accum
    steps_per_epoch = len(train_rows) // effective_batch     # a final partial batch (<32 rows) is dropped
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = max(1, round(args.warmup_ratio * total_steps))

    optimizer = make_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(total_steps, warmup_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {
        **vars(args), "effective_batch": effective_batch, "total_steps": total_steps, "warmup_steps": warmup_steps,
        "train_rows": len(train_rows), "val_rows": len(val_rows), "loss_reduction": "token_mean_over_effective_batch",
        "schedule": "linear warmup then cosine to 0", "precision": "fp32 weights + fp16 autocast + GradScaler" if use_amp else "fp32",
        "prompt_template": PROMPT_TEMPLATE, "git_commit": git_commit(REPO_ROOT), "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log = MetricsLog(run_dir / "metrics.jsonl")
    print(f"{len(train_rows)} train / {len(val_rows)} val rows | {total_steps} optimizer steps of {effective_batch} rows "
          f"| warmup {warmup_steps} | device {config['device']} | amp {use_amp}", flush=True)

    val_loss = validate(model, tokenizer, val_rows, args, amp)
    log.write(step=0, split="val", loss=val_loss, ppl=math.exp(val_loss))
    print(f"step 0 | val loss {val_loss:.4f}", flush=True)
    best_val, best_step = val_loss, 0

    generator = torch.Generator().manual_seed(args.seed)
    step, window, start = 0, [], time.time()
    stream = (b for _ in range(args.epochs) for b in micro_batches(train_rows, tokenizer, args.micro_batch, args.max_length, generator))

    for batch in stream:
        window.append(batch)
        if len(window) < args.grad_accum:
            continue

        # ---- one optimizer step over the effective batch ----
        window_tokens = sum(b["labels"][:, 1:].ne(core.IGNORE_INDEX).sum().item() for b in window)
        window_nll = 0.0
        for b in window:
            with amp():
                logps, _ = core.response_logps(model, b)
                loss = -logps.sum() / max(window_tokens, 1)   # this micro-batch's share of the token mean
            scaler.scale(loss).backward()
            window_nll += -logps.sum().item()
        window = []

        scaler.unscale_(optimizer)                            # back to true gradient scale before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)                                # skipped automatically if grads are inf/nan
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1
        skipped = use_amp and scaler.get_scale() < scale_before

        if step % args.log_every == 0 or step == 1:
            rate = step / (time.time() - start)
            rec = log.write(step=step, split="train", loss=window_nll / max(window_tokens, 1),
                            lr=scheduler.get_last_lr()[0], grad_norm=grad_norm, tokens=window_tokens,
                            skipped_step=bool(skipped), gpu_mem_gb=gpu_memory_gb())
            eta_min = (total_steps - step) / rate / 60
            print(f"step {step}/{total_steps} | loss {rec['loss']:.4f} | lr {rec['lr']:.2e} | grad {grad_norm:.2f} "
                  f"| mem {rec['gpu_mem_gb']} GB | {rate * 60:.1f} steps/min | ETA {eta_min:.0f} min", flush=True)

        if step % args.eval_every == 0 or step == total_steps:
            val_loss = validate(model, tokenizer, val_rows, args, amp)
            improved = val_loss < best_val
            log.write(step=step, split="val", loss=val_loss, ppl=math.exp(val_loss), best=improved)
            print(f"step {step} | val loss {val_loss:.4f} {'(best)' if improved else ''}", flush=True)
            for instruction, answer in zip(SAMPLE_INSTRUCTIONS, sample(model, tokenizer, device, amp)):
                print(f"   Q: {instruction}\n   A: {answer[:200]!r}", flush=True)
            if improved:
                best_val, best_step = val_loss, step
                save_checkpoint(model, tokenizer, run_dir / "best", {"step": step, "val_loss": val_loss})

        if step >= total_steps:
            break

    if step % args.eval_every != 0 and step != total_steps:   # stream ended early: make sure final is validated
        val_loss = validate(model, tokenizer, val_rows, args, amp)
        log.write(step=step, split="val", loss=val_loss, ppl=math.exp(val_loss), best=False)
    save_checkpoint(model, tokenizer, run_dir / "final", {"step": step, "val_loss": val_loss})
    summary = log.write(step=step, split="summary", best_step=best_step, best_val_loss=best_val,
                        final_val_loss=val_loss, minutes=round((time.time() - start) / 60, 1), gpu_mem_gb=gpu_memory_gb())
    print(f"done | best val {best_val:.4f} at step {best_step} | {summary['minutes']} min", flush=True)

    keep = REPO_ROOT / "results" / "runs" / args.run_name
    keep.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "metrics.jsonl"):
        shutil.copy(run_dir / name, keep / name)


if __name__ == "__main__":
    main()
