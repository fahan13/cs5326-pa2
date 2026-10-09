"""Parse, filter, and save the fixed train/validation subsets.

Every stored example keeps the RAW instruction; the prompt template is applied
only when batching (alignment.prompts.format_prompt), so the files stay readable
and the template lives in exactly one place.
"""
from __future__ import annotations

import gzip
import json
import random
from collections import Counter
from pathlib import Path

from alignment.prompts import format_prompt


# ---------- file helpers ----------

def write_jsonl_gz(rows, path):
    """mtime=0 makes the .gz byte-identical on every rerun (clean git diffs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        for row in rows:
            gz.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------- length filter ----------

class LengthFilter:
    """Keeps only examples that fit WITHOUT truncation, so every training row ends in EOS.

    Token counts are measured exactly the way core.build_lm_batch builds rows:
    template-formatted prompt and response tokenized separately, plus 1 for EOS.
    """

    def __init__(self, tokenizer, max_length=512, max_prompt_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

    def _n(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def reject_reason(self, instruction, *responses):
        prompt_tokens = self._n(format_prompt(instruction))
        if self.max_prompt_length is not None and prompt_tokens > self.max_prompt_length:
            return "prompt_too_long"
        for response in responses:
            if prompt_tokens + self._n(response) + 1 > self.max_length:
                return "too_long"
        return None


# ---------- per-dataset parsers ----------
# Each returns a tuple of clean strings, or a short string naming why the row was dropped.

def parse_ultrachat(row):
    """First user->assistant exchange of a UltraChat conversation (single-turn SFT)."""
    messages = row["messages"]
    if len(messages) < 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return "bad_roles"
    instruction = messages[0]["content"].strip()
    response = messages[1]["content"].strip()
    if not instruction or not response:
        return "empty"
    return instruction, response


HH_HUMAN, HH_ASSISTANT = "\n\nHuman:", "\n\nAssistant:"


def split_hh_transcript(text):
    """'\\n\\nHuman: x\\n\\nAssistant: y' -> (x, y); None if it is not exactly one turn each."""
    if text.lstrip().startswith("Human:"):
        text = "\n\n" + text.lstrip()
    if text.count(HH_HUMAN) != 1 or text.count(HH_ASSISTANT) != 1 or not text.startswith(HH_HUMAN):
        return None
    head, _, response = text.partition(HH_ASSISTANT)
    return head[len(HH_HUMAN):].strip(), response.strip()


def parse_hh(row):
    chosen = split_hh_transcript(row["chosen"])
    rejected = split_hh_transcript(row["rejected"])
    if chosen is None or rejected is None:
        return "multi_turn"
    if chosen[0] != rejected[0]:
        return "prompt_mismatch"
    if not chosen[0] or not chosen[1] or not rejected[1]:
        return "empty"
    if chosen[1] == rejected[1]:
        return "identical"
    return chosen[0], chosen[1], rejected[1]


def parse_ultrafeedback(row):
    chosen, rejected = row["chosen"], row["rejected"]
    if len(chosen) != 2 or len(rejected) != 2:
        return "multi_turn"
    if chosen[-1]["role"] != "assistant" or rejected[-1]["role"] != "assistant":
        return "bad_roles"
    if row.get("score_chosen") is not None and row["score_chosen"] <= row["score_rejected"]:
        return "tied_scores"
    instruction = row["prompt"].strip()
    chosen_text, rejected_text = chosen[-1]["content"].strip(), rejected[-1]["content"].strip()
    if not instruction or not chosen_text or not rejected_text:
        return "empty"
    if chosen_text == rejected_text:
        return "identical"
    return instruction, chosen_text, rejected_text


# ---------- deterministic selection ----------

def select_examples(rows, parse, length_filter, n, seed, id_prefix, allow_fewer=False, exclude_prompts=None):
    """Shuffle indices with a fixed seed, keep the first n rows that parse and fit.

    Returns (examples, stats). stats records how many rows were scanned and why
    each dropped row was dropped -- this goes straight into the report.
    exclude_prompts: prompts already used elsewhere (e.g. in train), so validation never repeats them.
    """
    exclude_prompts = exclude_prompts or set()
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)

    kept, dropped, scanned = [], Counter(), 0
    for index in order:
        scanned += 1
        parsed = parse(rows[index])
        if isinstance(parsed, str):
            dropped[parsed] += 1
            continue
        if parsed[0] in exclude_prompts:
            dropped["prompt_in_train"] += 1
            continue
        reason = length_filter.reject_reason(*parsed)
        if reason:
            dropped[reason] += 1
            continue
        example = {"id": f"{id_prefix}-{index}", "prompt": parsed[0]}
        if len(parsed) == 2:
            example["response"] = parsed[1]
        else:
            example["chosen"], example["rejected"] = parsed[1], parsed[2]
        kept.append(example)
        if len(kept) == n:
            break

    if len(kept) < n and not allow_fewer:
        raise RuntimeError(f"{id_prefix}: only {len(kept)} usable rows, wanted {n}. Dropped: {dict(dropped)}")
    return kept, {"source_rows": len(rows), "scanned": scanned, "kept": len(kept), "dropped": dict(dropped)}
