"""Generate answers for a fixed evaluation set (Task 2 baseline, Task 6 SFT/DPO outputs).

    python scripts/generate.py --model Qwen/Qwen2.5-0.5B --generator qwen2.5-0.5b-base --eval_set alpaca

--model is a Hugging Face id or a local checkpoint folder. --eval_set is "alpaca", "hh",
or a path to a JSON list of {"id", "instruction", ...}.

Decoding is greedy, and every model must be generated with the SAME flags so a
comparison measures the checkpoint, not the decoding. The exact settings are written
next to the outputs (<output>.config.json) for the report.

The output file is rewritten after every batch. Rerunning the same command resumes
and only generates prompts that are missing.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from alignment.prompts import PROMPT_TEMPLATE, STOP_MARKER, clean_generation, format_prompt  # noqa: E402

EVAL_SETS = {
    "alpaca": REPO_ROOT / "eval_sets" / "alpaca_eval_300_seed42.json",
    "hh": REPO_ROOT / "eval_sets" / "hh_rlhf_eval_300_seed42.json",
    "persona": REPO_ROOT / "eval_sets" / "persona_style_eval.json",
}
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def load_eval_set(name_or_path):
    path = EVAL_SETS.get(name_or_path, Path(name_or_path))
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    for row in rows:
        if "id" not in row or "instruction" not in row:
            raise ValueError(f"{path}: every row needs 'id' and 'instruction', got keys {sorted(row)}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{path}: ids are not unique")
    return rows


def write_json_atomic(data, path):
    """Write to a temp file, then rename: a crash mid-write never leaves a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def build_prompt(tokenizer, instruction, use_chat_template):
    """Our ### template for the Qwen pipeline; the model's own chat template for instruct models
    (the LoRA persona track trains SmolLM2-Instruct through TRL, which uses its chat template)."""
    if use_chat_template:
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return format_prompt(instruction)


def stop_token_ids(tokenizer):
    ids = {tokenizer.eos_token_id}
    for token in ("<|im_end|>", "<|endoftext|>"):      # chat models end turns with <|im_end|>
        tid = tokenizer.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            ids.add(tid)
    return sorted(i for i in ids if i is not None)


def decode_one(tokenizer, new_ids, stop_ids):
    """New token ids -> (answer text, finish reason, number of tokens the model produced)."""
    ids = new_ids.tolist()
    first_stop = min((ids.index(t) for t in stop_ids if t in ids), default=None)
    hit_eos = first_stop is not None
    if hit_eos:
        ids = ids[:first_stop]        # finished rows are padded with EOS, so cut at the first stop token
    text, hit_marker = clean_generation(tokenizer.decode(ids, skip_special_tokens=True))
    reason = "stop_marker" if hit_marker else "eos" if hit_eos else "max_new_tokens"
    return text, reason, len(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--generator", required=True, help="name stored in every row, e.g. qwen2.5-0.5b-sft")
    parser.add_argument("--eval_set", required=True)
    parser.add_argument("--output", help="default: results/generations/<generator>__<eval_set>.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    # 1.1 chosen from a 20-prompt pilot on the base model (results/sanity/): plain greedy hit the
    # 512-token cap on 8/20 prompts by looping; 1.1 removed every loop. Used for ALL models.
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--chat_template", action="store_true",
                        help="format prompts with the tokenizer's chat template instead of our ### template")
    parser.add_argument("--limit", type=int, help="only the first N prompts (sanity checks)")
    parser.add_argument("--show", type=int, default=3, help="print this many outputs at the end")
    args = parser.parse_args()

    eval_name = Path(args.eval_set).stem if args.eval_set not in EVAL_SETS else args.eval_set
    out_path = Path(args.output) if args.output else (
        REPO_ROOT / "results" / "generations" / f"{args.generator}__{eval_name}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_eval_set(args.eval_set)[: args.limit]
    done = {}
    if out_path.exists():
        done = {row["id"]: row for row in json.loads(out_path.read_text(encoding="utf-8"))}
    todo = [row for row in rows if row["id"] not in done]
    print(f"{len(rows)} prompts, {len(rows) - len(todo)} already done, {len(todo)} to generate -> {out_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = DTYPES[args.dtype] if device == "cuda" else torch.float32
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"          # decoder-only models must be padded on the left to generate
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()

    gen_kwargs = dict(
        do_sample=False,                       # greedy
        temperature=None, top_p=None, top_k=None,   # silence sampling-only defaults from generation_config
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        eos_token_id=stop_token_ids(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        stop_strings=[STOP_MARKER],
        tokenizer=tokenizer,
    )

    config = {
        "model": args.model, "generator": args.generator, "eval_set": args.eval_set,
        "prompt_template": "tokenizer chat template" if args.chat_template else PROMPT_TEMPLATE,
        "stop_marker": STOP_MARKER,
        "decoding": {k: v for k, v in gen_kwargs.items() if k not in ("tokenizer",)},
        "dtype": str(dtype), "device": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "torch": torch.__version__, "transformers": transformers.__version__,
    }
    write_json_atomic(config, out_path.with_suffix(".config.json"))

    # Longest prompts first: similar lengths share a batch (less padding), and any
    # out-of-memory error happens in the first batch instead of at the very end.
    todo.sort(key=lambda row: len(tokenizer.encode(build_prompt(tokenizer, row["instruction"], args.chat_template))),
              reverse=True)
    order = [row["id"] for row in rows]
    start = time.time()

    for i in range(0, len(todo), args.batch_size):
        batch = todo[i : i + args.batch_size]
        enc = tokenizer(
            [build_prompt(tokenizer, row["instruction"], args.chat_template) for row in batch],
            return_tensors="pt", padding=True, add_special_tokens=False,
        ).to(device)
        with torch.inference_mode():
            out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], **gen_kwargs)
        new_tokens = out[:, enc["input_ids"].shape[1]:]

        for row, ids in zip(batch, new_tokens):
            text, reason, n = decode_one(tokenizer, ids, gen_kwargs["eos_token_id"])
            done[row["id"]] = {
                "id": row["id"],
                "instruction": row["instruction"],
                "output": text,
                "generator": args.generator,
                "dataset": row.get("dataset", eval_name),
                **({"category": row["category"]} if "category" in row else {}),
                "finish_reason": reason,
                "new_tokens": n,
            }
        write_json_atomic([done[rid] for rid in order if rid in done], out_path)
        finished = min(i + args.batch_size, len(todo))
        print(f"  {finished}/{len(todo)}  {time.time() - start:.0f}s", flush=True)

    results = [done[rid] for rid in order if rid in done]
    reasons = Counter(row["finish_reason"] for row in results)
    mean_tokens = sum(row["new_tokens"] for row in results) / max(len(results), 1)
    print(f"\nfinish reasons: {dict(reasons)} | mean new tokens: {mean_tokens:.0f}")
    for row in results[: args.show]:
        print(f"\n--- {row['id']} [{row['finish_reason']}, {row['new_tokens']} tok]\n"
              f"Q: {row['instruction'][:300]}\nA: {row['output'][:600]}")


if __name__ == "__main__":
    main()
