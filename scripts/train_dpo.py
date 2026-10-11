"""From-scratch DPO (Task 4). One script for both preference datasets; no TRL.

Full runs on Kaggle (policy on GPU 0, frozen reference on GPU 1):
    python scripts/train_dpo.py --run_name dpo_hh --data hh --sft_checkpoint <path to SFT best/>
    python scripts/train_dpo.py --run_name dpo_uf --data uf --sft_checkpoint <path to SFT best/>

Laptop pilot (tiny SFT checkpoint from the SFT pilot, both models on one GPU):
    python scripts/train_dpo.py --run_name pilot_dpo --data hh --sft_checkpoint outputs/pilot_sft/best \
        --ref_device cuda:0 --max_steps 20 --eval_every 10 --val_limit 64

What one optimizer step does
    For each of grad_accum micro-batches (1 pair each by default -> 32 pairs per step):
      1. Reference scores chosen + rejected under no_grad (launched first on its own GPU,
         so it computes while the policy works on the other GPU).
      2. Policy scores chosen + rejected in one forward pass (core.pair_logps).
      3. DPO loss = -log sigmoid(beta * (policy gap - reference gap)); divide by grad_accum
         so the accumulated gradient is the mean over all pairs; backprop the policy only.
    Then unscale, clip to 1.0, optimizer step. The learning rate is constant (guide recipe).

Validation tracks DPO loss, reward accuracy (strict r_w > r_l) and reward margin; best/
is chosen by validation reward accuracy. Before any update the policy equals the
reference, so step 0 must show loss = log 2 = 0.693 and accuracy = 0: a built-in check.
"""
from __future__ import annotations

import argparse
import json
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
from alignment.prompts import PROMPT_TEMPLATE, format_prompt  # noqa: E402
from alignment.training import (  # noqa: E402
    MetricsLog, constant_with_warmup, git_commit, gpu_memory_gb, make_optimizer, save_checkpoint, set_seed,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--data", choices=["hh", "uf"], required=True)
    p.add_argument("--sft_checkpoint", required=True, help="folder of the SFT best checkpoint")
    p.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    # guide configuration (manual sections 10.3-10.4)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--warmup_steps", type=int, default=0, help="guide: none")
    p.add_argument("--optimizer", choices=["rmsprop", "adamw"], default="rmsprop")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--micro_batch", type=int, default=1, help="pairs per forward pass")
    p.add_argument("--grad_accum", type=int, default=32)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    # devices
    p.add_argument("--policy_device", default="cuda:0")
    p.add_argument("--ref_device", default="cuda:1")
    p.add_argument("--ref_dtype", choices=["float32", "float16"], default="float32",
                   help="float32 makes step-0 rewards exactly 0; float16 saves ~1 GB")
    # evaluation / logging
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_batch", type=int, default=4, help="pairs per validation forward")
    p.add_argument("--log_every", type=int, default=5)
    # pilots
    p.add_argument("--max_steps", type=int)
    p.add_argument("--train_limit", type=int)
    p.add_argument("--val_limit", type=int)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def load_model(path, device, dtype, trainable):
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device)
    model.config.use_cache = False
    if trainable:
        model.train()
    else:
        model.eval()
        model.requires_grad_(False)          # frozen: no gradients can ever reach it
    return model


def score_pairs(policy, reference, tokenizer, rows, args, amp, ref_amp, with_grad):
    """Returns core.dpo_stats for these rows. The reference call is issued first so its GPU
    work overlaps with the policy forward (CUDA launches are asynchronous)."""
    prompts = [format_prompt(r["prompt"]) for r in rows]
    chosen, rejected = [r["chosen"] for r in rows], [r["rejected"] for r in rows]
    with torch.no_grad(), ref_amp():
        ref_c, ref_r = core.pair_logps(reference, tokenizer, prompts, chosen, rejected, args.max_length)
    with torch.set_grad_enabled(with_grad), amp():
        pol_c, pol_r = core.pair_logps(policy, tokenizer, prompts, chosen, rejected, args.max_length)
    return core.dpo_stats(pol_c, pol_r, ref_c, ref_r, args.beta)


@torch.no_grad()
def validate(policy, reference, tokenizer, rows, args, amp, ref_amp):
    policy.eval()
    losses, correct, margins, r_w, r_l = [], [], [], [], []
    for i in range(0, len(rows), args.eval_batch):
        chunk = rows[i : i + args.eval_batch]
        s = score_pairs(policy, reference, tokenizer, chunk, args, amp, ref_amp, with_grad=False)
        losses.append(s["loss"].item() * len(chunk))
        correct += s["correct"].tolist()
        margins += s["margin"].tolist()
        r_w += s["reward_chosen"].tolist()
        r_l += s["reward_rejected"].tolist()
    policy.train()
    n = len(rows)
    return {
        "loss": sum(losses) / n,
        "reward_acc": sum(correct) / n,
        "reward_margin": sum(margins) / n,
        "reward_chosen": sum(r_w) / n,
        "reward_rejected": sum(r_l) / n,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    on_cuda = torch.cuda.is_available()
    if not on_cuda:
        args.policy_device = args.ref_device = "cpu"
    use_amp = on_cuda and not args.no_amp

    def amp():
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)

    ref_amp = amp   # the reference runs through exactly the same autocast path as the policy

    run_dir = Path(args.output_root) / args.run_name
    if (run_dir / "final").exists():
        sys.exit(f"{run_dir} already has a final checkpoint; pick a new --run_name")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.jsonl").unlink(missing_ok=True)

    data_dir = REPO_ROOT / "data" / "processed"
    train_rows = read_jsonl_gz(data_dir / f"{args.data}_train.jsonl.gz")[: args.train_limit]
    val_rows = read_jsonl_gz(data_dir / f"{args.data}_val.jsonl.gz")[: args.val_limit]

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # policy: fp32 master weights + fp16 autocast + GradScaler (manual 10 memory tip)
    policy = load_model(args.sft_checkpoint, args.policy_device, torch.float32, trainable=True)
    ref_dtype = torch.float32 if args.ref_dtype == "float32" else torch.float16
    reference = load_model(args.sft_checkpoint, args.ref_device, ref_dtype, trainable=False)

    effective_batch = args.micro_batch * args.grad_accum
    total_steps = (len(train_rows) // effective_batch) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    optimizer = make_optimizer(policy, args.optimizer, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, constant_with_warmup(args.warmup_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {
        **vars(args), "effective_batch": effective_batch, "total_steps": total_steps,
        "train_rows": len(train_rows), "val_rows": len(val_rows),
        "reference_checkpoint": str(Path(args.sft_checkpoint).resolve()),
        "precision": "policy fp32 + fp16 autocast + GradScaler" if use_amp else "fp32",
        "checkpoint_selection": "best validation reward accuracy (strict r_w > r_l)",
        "prompt_template": PROMPT_TEMPLATE, "git_commit": git_commit(REPO_ROOT),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if on_cuda else ["cpu"],
        "torch": torch.__version__, "transformers": transformers.__version__,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log = MetricsLog(run_dir / "metrics.jsonl")
    print(f"[{args.data}] {len(train_rows)} train / {len(val_rows)} val pairs | {total_steps} steps of "
          f"{effective_batch} pairs | beta {args.beta} | lr {args.lr} | {args.optimizer} | "
          f"policy {args.policy_device} ref {args.ref_device} ({args.ref_dtype})", flush=True)

    val = validate(policy, reference, tokenizer, val_rows, args, amp, ref_amp)
    log.write(step=0, split="val", **val)
    print(f"step 0 | val loss {val['loss']:.4f} (expect 0.6931) | acc {val['reward_acc']:.3f} (expect 0) "
          f"| margin {val['reward_margin']:.4f}", flush=True)
    best_acc, best_step = -1.0, 0

    generator = torch.Generator().manual_seed(args.seed)
    order = [i for _ in range(args.epochs) for i in torch.randperm(len(train_rows), generator=generator).tolist()]
    step, start, cursor = 0, time.time(), 0

    while step < total_steps:
        win = {"loss": 0.0, "correct": [], "margin": [], "r_w": [], "r_l": []}
        for _ in range(args.grad_accum):
            rows = [train_rows[i] for i in order[cursor : cursor + args.micro_batch]]
            cursor += args.micro_batch
            s = score_pairs(policy, reference, tokenizer, rows, args, amp, ref_amp, with_grad=True)
            scaler.scale(s["loss"] / args.grad_accum).backward()   # mean over all pairs in the step
            win["loss"] += s["loss"].item() / args.grad_accum
            win["correct"] += s["correct"].tolist()
            win["margin"] += s["margin"].tolist()
            win["r_w"] += s["reward_chosen"].tolist()
            win["r_l"] += s["reward_rejected"].tolist()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm).item()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1

        if step % args.log_every == 0 or step == 1:
            n = len(win["correct"])
            rec = log.write(
                step=step, split="train", loss=win["loss"], reward_acc=sum(win["correct"]) / n,
                reward_margin=sum(win["margin"]) / n, reward_chosen=sum(win["r_w"]) / n,
                reward_rejected=sum(win["r_l"]) / n, grad_norm=grad_norm, lr=scheduler.get_last_lr()[0],
                skipped_step=bool(use_amp and scaler.get_scale() < scale_before), gpu_mem_gb=gpu_memory_gb(),
            )
            rate = step / (time.time() - start)
            print(f"step {step}/{total_steps} | loss {rec['loss']:.4f} | acc {rec['reward_acc']:.2f} "
                  f"| margin {rec['reward_margin']:.3f} | r_w {rec['reward_chosen']:.3f} r_l {rec['reward_rejected']:.3f} "
                  f"| grad {grad_norm:.2f} | mem {rec['gpu_mem_gb']} GB | {rate * 60:.2f} steps/min "
                  f"| ETA {(total_steps - step) / rate / 60:.0f} min", flush=True)

        if step % args.eval_every == 0 or step == total_steps:
            val = validate(policy, reference, tokenizer, val_rows, args, amp, ref_amp)
            improved = val["reward_acc"] > best_acc
            log.write(step=step, split="val", best=improved, **val)
            print(f"step {step} | val loss {val['loss']:.4f} | acc {val['reward_acc']:.3f} "
                  f"| margin {val['reward_margin']:.4f} {'(best)' if improved else ''}", flush=True)
            if improved:
                best_acc, best_step = val["reward_acc"], step
                save_checkpoint(policy, tokenizer, run_dir / "best", {"step": step, **val})

    save_checkpoint(policy, tokenizer, run_dir / "final", {"step": step, **val})
    summary = log.write(step=step, split="summary", best_step=best_step, best_val_reward_acc=best_acc,
                        final_val=val, minutes=round((time.time() - start) / 60, 1), gpu_mem_gb=gpu_memory_gb())
    print(f"done | best val acc {best_acc:.3f} at step {best_step} | {summary['minutes']} min", flush=True)

    keep = REPO_ROOT / "results" / "runs" / args.run_name
    keep.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "metrics.jsonl"):
        shutil.copy(run_dir / name, keep / name)


if __name__ == "__main__":
    main()
