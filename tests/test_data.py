"""Offline checks for the dataset parsers and the deterministic selector."""
from __future__ import annotations

from alignment.data import (
    LengthFilter,
    parse_hh,
    parse_ultrachat,
    parse_ultrafeedback,
    read_jsonl_gz,
    select_examples,
    split_hh_transcript,
    write_jsonl_gz,
)


class CharTokenizer:
    """One token per character, any character (the toy tokenizer has no '#' or newline)."""
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def msg(role, content):
    return {"role": role, "content": content}


def test_hh_single_turn_is_split_cleanly():
    assert split_hh_transcript("\n\nHuman: Hi there?\n\nAssistant: Hello!") == ("Hi there?", "Hello!")


def test_hh_multi_turn_is_rejected():
    text = "\n\nHuman: a\n\nAssistant: b\n\nHuman: c\n\nAssistant: d"
    assert parse_hh({"chosen": text, "rejected": text}) == "multi_turn"


def test_hh_pair_parses_and_rejects_identical_responses():
    good = {"chosen": "\n\nHuman: q\n\nAssistant: yes", "rejected": "\n\nHuman: q\n\nAssistant: no"}
    assert parse_hh(good) == ("q", "yes", "no")
    same = {"chosen": good["chosen"], "rejected": good["chosen"]}
    assert parse_hh(same) == "identical"


def test_ultrachat_takes_first_exchange_only():
    row = {"messages": [msg("user", " q1 "), msg("assistant", "a1"), msg("user", "q2"), msg("assistant", "a2")]}
    assert parse_ultrachat(row) == ("q1", "a1")


def test_ultrafeedback_parses_and_drops_ties():
    row = {
        "prompt": "q",
        "chosen": [msg("user", "q"), msg("assistant", "good")],
        "rejected": [msg("user", "q"), msg("assistant", "bad")],
        "score_chosen": 8.0,
        "score_rejected": 3.0,
    }
    assert parse_ultrafeedback(row) == ("q", "good", "bad")
    assert parse_ultrafeedback({**row, "score_rejected": 8.0}) == "tied_scores"


def test_length_filter_counts_template_and_eos():
    tok = CharTokenizer()
    lf = LengthFilter(tok, max_length=10_000, max_prompt_length=5)
    assert lf.reject_reason("q", "a") == "prompt_too_long"           # template alone is > 5 tokens
    exact = LengthFilter(tok, max_length=len(tok.encode("### Instruction:\nq\n\n### Response:\n")) + 2)
    assert exact.reject_reason("q", "a") is None                     # prompt + 'a' + EOS fits exactly
    assert exact.reject_reason("q", "ab") == "too_long"


def test_selection_is_deterministic_and_counts_drops(tmp_path):
    rows = [{"messages": [msg("user", f"q{i}"), msg("assistant", "a" if i % 3 else "")]} for i in range(30)]
    lf = LengthFilter(CharTokenizer(), max_length=10_000)
    first, stats = select_examples(rows, parse_ultrachat, lf, 10, seed=42, id_prefix="t")
    second, _ = select_examples(rows, parse_ultrachat, lf, 10, seed=42, id_prefix="t")
    assert first == second
    assert stats["kept"] == 10 and stats["dropped"].get("empty", 0) > 0

    path = tmp_path / "x.jsonl.gz"
    write_jsonl_gz(first, path)
    assert read_jsonl_gz(path) == first
    first_bytes = path.read_bytes()
    write_jsonl_gz(first, path)
    assert path.read_bytes() == first_bytes                          # byte-reproducible


def test_selection_can_exclude_prompts_already_used():
    rows = [{"messages": [msg("user", f"q{i}"), msg("assistant", "a")]} for i in range(10)]
    lf = LengthFilter(CharTokenizer(), max_length=10_000)
    kept, stats = select_examples(rows, parse_ultrachat, lf, 5, seed=0, id_prefix="v",
                                  exclude_prompts={"q1", "q2", "q3"})
    assert not {"q1", "q2", "q3"} & {row["prompt"] for row in kept}
    assert stats["dropped"].get("prompt_in_train", 0) >= 0
