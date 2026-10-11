"""Task 5: "Idiot Sandwich" persona tuning with TRL DPOTrainer + PEFT LoRA.

This is the one track where a library does the training loop (the manual allows TRL here).

    python scripts/train_lora_dpo.py --run_name lora_dpo

LoRA in one paragraph: instead of updating a weight matrix W (d x d), we freeze it and learn a
low-rank correction B @ A (d x r and r x d, with r << d), so the layer computes W x + (alpha/r) B A x.
Only A and B train (~1-2% of the parameters). With peft_config given, TRL also gets the frozen
reference model for free: it is the same network with the LoRA adapter switched off, so no
second copy of the model is loaded.

Writes outputs/<run_name>/{adapter,merged}/ and results/runs/<run_name>/{config.json,metrics.jsonl}.
The merged folder is a normal checkpoint that scripts/generate.py can load (--chat_template).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import peft  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
import trl  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import DPOConfig, DPOTrainer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from alignment.training import git_commit  # noqa: E402

DATA = REPO_ROOT / "data" / "persona_lora" / "ramsay_style_preferences.json"
# Every Linear layer in a Llama-style block: attention projections + MLP. Style lives partly in
# word choice (MLP, which stores vocabulary-ish features) and partly in phrasing (attention).
ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="lora_dpo")
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--output_root", default=str(REPO_ROOT / "outputs"))
    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", nargs="+", default=ALL_LINEAR)
    # DPO / optimisation
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--val_size", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_steps", type=int, default=10)
    return p.parse_args()


def to_conversational(row):
    """TRL conversational preference format: role/content message lists."""
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def known_fields(cls, wanted: dict) -> dict:
    """Keep only arguments this TRL version's DPOConfig accepts (names change between releases)."""
    names = {f.name for f in dataclasses.fields(cls)}
    dropped = sorted(set(wanted) - names)
    if dropped:
        print(f"note: this TRL version has no {dropped}; skipping them")
    return {k: v for k, v in wanted.items() if k in names}


def main():
    args = parse_args()
    run_dir = Path(args.output_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(DATA.read_text(encoding="utf-8"))
    random.Random(args.seed).shuffle(rows)
    val_rows, train_rows = rows[: args.val_size], rows[args.val_size :]
    train_ds = Dataset.from_list([to_conversational(r) for r in train_rows])
    val_ds = Dataset.from_list([to_conversational(r) for r in val_rows])

    on_cuda = torch.cuda.is_available()
    bf16 = on_cuda and torch.cuda.is_bf16_supported()   # RTX 40xx: yes; Kaggle T4: no -> fp16
    fp16 = on_cuda and not bf16

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)

    lora = LoraConfig(
        task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=args.target_modules, bias="none",
    )
    config = DPOConfig(**known_fields(DPOConfig, dict(
        output_dir=str(run_dir / "trainer"),
        beta=args.beta,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        bf16=bf16,
        fp16=fp16,
        logging_steps=2,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )))
    trainer = DPOTrainer(
        model=model, ref_model=None, args=config, train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tokenizer, peft_config=lora,
    )
    trainable, total = trainer.model.get_nb_trainable_parameters()
    print(f"LoRA r={args.lora_r} alpha={args.lora_alpha} on {args.target_modules}: "
          f"{trainable:,} trainable of {total:,} ({100 * trainable / total:.2f}%)", flush=True)

    before = trainer.evaluate()          # step 0: adapter is zero-initialised, so policy == reference
    print("step 0 eval:", {k: round(v, 4) for k, v in before.items() if "rewards" in k or k == "eval_loss"}, flush=True)
    trainer.train()
    after = trainer.evaluate()
    print("final eval:", {k: round(v, 4) for k, v in after.items() if "rewards" in k or k == "eval_loss"}, flush=True)

    trainer.model.save_pretrained(run_dir / "adapter")            # small: only A and B matrices
    merged = trainer.model.merge_and_unload()                      # fold B@A into W for plain loading
    merged.save_pretrained(run_dir / "merged", safe_serialization=True)
    tokenizer.save_pretrained(run_dir / "merged")

    keep = REPO_ROOT / "results" / "runs" / args.run_name
    keep.mkdir(parents=True, exist_ok=True)
    with open(keep / "metrics.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"step": 0, **before}) + "\n")
        for record in trainer.state.log_history:
            f.write(json.dumps(record) + "\n")
    (keep / "config.json").write_text(json.dumps({
        **vars(args), "train_pairs": len(train_rows), "val_pairs": len(val_rows),
        "trainable_params": trainable, "total_params": total,
        "precision": "bf16" if bf16 else "fp16" if fp16 else "fp32",
        "optimizer_steps": trainer.state.global_step, "effective_batch": args.batch_size * args.grad_accum,
        "reference": "same base model with the LoRA adapter disabled (TRL + peft_config)",
        "git_commit": git_commit(REPO_ROOT), "trl": trl.__version__, "peft": peft.__version__,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if on_cuda else "cpu",
    }, indent=2), encoding="utf-8")
    print(f"saved adapter + merged model to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
