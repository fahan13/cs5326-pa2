"""Judge pipeline: parsing, A/B -> win/loss mapping, blinding, retries and resume (offline mock judge)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from judge_pairwise_kaggle import build_prompt, main, normalise_verdict, outcome, parse_reply, read_jsonl  # noqa: E402


def make_pairs(path, n=7, mode="general"):
    rows = []
    for i in range(n):
        cand_pos = "A" if i % 2 == 0 else "B"
        short, long_ = f"short {i}", f"a much longer answer number {i}"
        a, b = (short, long_) if cand_pos == "A" else (long_, short)   # candidate always gives the short answer
        rows.append({"pair_id": f"secretdpo_vs_secretsft-{i:03d}", "prompt_id": f"p{i}", "eval_set": "x",
                     "category": "c1" if i < 4 else "c2", "expected_behavior": "be good", "judge_mode": mode,
                     "instruction": f"question {i}", "response_a": a, "response_b": b,
                     "hidden": {"comparison": "cmp", "candidate": "secretdpo", "baseline": "secretsft",
                                "candidate_position": cand_pos, "model_a": "secretdpo" if cand_pos == "A" else "secretsft",
                                "model_b": "secretsft" if cand_pos == "A" else "secretdpo",
                                "tokens_a": len(a), "tokens_b": len(b)}})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return rows


def test_parse_reply_handles_fences_and_variants():
    reply = 'Here you go:\n```json\n[{"pair": 1, "reason": "x", "winner": "Response A"}, {"pair": 2, "winner": "TIE"}, {"pair": 3, "winner": "maybe"}]\n```'
    out = parse_reply(reply, 3, "general")
    assert [out[i]["winner"] for i in (1, 2, 3)] == ["A", "tie", None]
    with pytest.raises(ValueError):
        parse_reply("I refuse.", 3, "general")
    assert normalise_verdict("b") == "B" and normalise_verdict("draw") == "tie"


def test_outcome_maps_position_to_candidate():
    assert outcome("A", "A") == "win" and outcome("A", "B") == "loss"
    assert outcome("tie", "A") == "tie" and outcome("unknown", "B") == "unknown"


def test_prompt_is_blinded(tmp_path):
    pairs = make_pairs(tmp_path / "p.jsonl")
    for mode in ("general", "safety", "persona"):
        text = build_prompt(pairs[:5], mode)
        assert "secret" not in text and "pair_id" not in text
        assert text.count("[Response A]") == 5


def test_mock_run_with_failures_then_resume(tmp_path):
    pairs_path, out = tmp_path / "p.jsonl", tmp_path / "j.jsonl"
    make_pairs(pairs_path)
    common = ["--pairs_path", str(pairs_path), "--output_path", str(out), "--backend", "mock", "--sleep_seconds", "0",
              "--batch_size", "3"]
    main(common + ["--mock_fail_rate", "1.0", "--max_retries", "2"])          # every call fails
    assert read_jsonl(out) == [] and len(read_jsonl(tmp_path / "j.failures.jsonl")) == 3
    s = main(common + ["--reuse_judgments"])                                  # healthy rerun resumes
    assert s["n_judged"] == 7 and s["failures"] == 0
    assert s["headline"]["wins"] == 7                                         # mock prefers shorter = candidate
    assert set(s["per_category"]["winner"]) == {"c1", "c2"}
    with pytest.raises(SystemExit):
        main(common)                                                          # output exists -> must choose a flag
    s2 = main(common + ["--reuse_judgments"])                                 # nothing left: no new requests
    assert s2["run_stats"].get("requests", 0) == 0


def test_persona_mode_has_four_criteria(tmp_path):
    pairs_path, out = tmp_path / "p.jsonl", tmp_path / "j.jsonl"
    make_pairs(pairs_path, mode="persona")
    s = main(["--pairs_path", str(pairs_path), "--output_path", str(out), "--backend", "mock", "--sleep_seconds", "0"])
    assert s["headline"]["criterion"] == "overall"
    assert all(k in s for k in ("style", "usefulness", "safety"))
