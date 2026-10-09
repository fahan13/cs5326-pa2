"""Build every fixed data subset used in the assignment. Run once, commit the outputs.

    python scripts/prepare_data.py

Writes (all deterministic for a given seed):
    data/processed/sft_{train,val}.jsonl.gz   UltraChat 200k, first turn only     30k / 1k
    data/processed/hh_{train,val}.jsonl.gz    Anthropic HH-RLHF, single-turn       15k / 1k
    data/processed/uf_{train,val}.jsonl.gz    UltraFeedback binarized              15k / 1k
    data/processed/stats.json                 rows scanned and drop reasons (for the report)
    eval_sets/alpaca_eval_300_seed42.json     fixed AlpacaEval subset

Validation rows come from each dataset's official test split, so they never overlap training.
Steps whose output already exists are skipped; pass --overwrite to rebuild.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from datasets import load_dataset  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from alignment.data import (  # noqa: E402
    LengthFilter,
    parse_hh,
    parse_ultrachat,
    parse_ultrafeedback,
    select_examples,
    write_jsonl_gz,
)

MODEL = "Qwen/Qwen2.5-0.5B"

# name -> (HF dataset, config, train split, val split, parser, max_prompt_length, train size)
SOURCES = {
    "sft": ("HuggingFaceH4/ultrachat_200k", None, "train_sft", "test_sft", parse_ultrachat, None, 30_000),
    "hh": ("Anthropic/hh-rlhf", None, "train", "test", parse_hh, 256, 15_000),
    "uf": ("HuggingFaceH4/ultrafeedback_binarized", None, "train_prefs", "test_prefs", parse_ultrafeedback, 256, 15_000),
}


def build_alpaca_eval_subset(path, n, seed):
    source = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    with open(source, encoding="utf-8") as f:
        rows = json.load(f)
    indices = sorted(random.Random(seed).sample(range(len(rows)), n))
    subset = [
        {
            "id": f"alpaca_eval-{i:03d}",
            "instruction": rows[i]["instruction"],
            "dataset": "alpaca_eval",
            "source": rows[i].get("dataset", ""),
        }
        for i in indices
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"source_rows": len(rows), "kept": n, "seed": seed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "data" / "processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_size", type=int, default=1_000)
    parser.add_argument("--only", nargs="*", choices=[*SOURCES, "alpaca"], help="build just these")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    stats_path = out_dir / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    wanted = set(args.only or [*SOURCES, "alpaca"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    for name, (repo, config, train_split, val_split, parse, max_prompt, train_size) in SOURCES.items():
        if name not in wanted:
            continue
        train_path, val_path = out_dir / f"{name}_train.jsonl.gz", out_dir / f"{name}_val.jsonl.gz"
        if train_path.exists() and val_path.exists() and not args.overwrite:
            print(f"[{name}] exists, skipping")
            continue

        print(f"[{name}] loading {repo} ...", flush=True)
        dataset = load_dataset(repo, config)
        length_filter = LengthFilter(tokenizer, args.max_length, max_prompt)
        stats[name] = {"source": repo, "seed": args.seed, "max_length": args.max_length, "max_prompt_length": max_prompt}

        train_prompts = set()
        for split_name, split, size, path in (
            ("train", train_split, train_size, train_path),
            ("val", val_split, args.val_size, val_path),
        ):
            # The manual asks for 1k validation pairs "when available": a short val split is
            # kept as-is (and recorded in stats.json), a short train split is an error.
            examples, split_stats = select_examples(
                dataset[split], parse, length_filter, size, args.seed, f"{name}-{split}",
                allow_fewer=(split_name == "val"),
                exclude_prompts=train_prompts,   # empty for train; train's prompts for val
            )
            if split_name == "train":
                train_prompts = {example["prompt"] for example in examples}
            write_jsonl_gz(examples, path)
            stats[name][split_name] = {"split": split, **split_stats}
            print(f"[{name}] {split_name}: kept {split_stats['kept']} of {split_stats['scanned']} scanned, "
                  f"dropped {split_stats['dropped']}", flush=True)

        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")   # save after every dataset

    alpaca_path = REPO_ROOT / "eval_sets" / f"alpaca_eval_300_seed{args.seed}.json"
    if "alpaca" in wanted and (args.overwrite or not alpaca_path.exists()):
        stats["alpaca_eval"] = build_alpaca_eval_subset(alpaca_path, 300, args.seed)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"[alpaca] wrote {alpaca_path.relative_to(REPO_ROOT)}")

    print("done")


if __name__ == "__main__":
    main()
