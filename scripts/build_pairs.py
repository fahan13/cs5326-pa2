"""Build blinded A/B pair files from saved generations (Task 6 -> Task 7).

    python scripts/build_pairs.py                 # every comparison in COMPARISONS
    python scripts/build_pairs.py --only dpo_uf_vs_sft__alpaca

Each comparison pairs a CANDIDATE model (e.g. DPO) against a BASELINE (e.g. SFT) on the
same prompts. For every prompt the order is randomised (seeded, so rebuilding gives the
same file) and the position -> model mapping is stored under "hidden". The judge script
only ever sends instruction / response_a / response_b to the judge model.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_DIR = REPO_ROOT / "results" / "generations"
EVAL_SETS = {
    "alpaca": REPO_ROOT / "eval_sets" / "alpaca_eval_300_seed42.json",
    "hh": REPO_ROOT / "eval_sets" / "hh_rlhf_eval_300_seed42.json",
    "persona": REPO_ROOT / "eval_sets" / "persona_style_eval.json",
}
JUDGE_MODE = {"alpaca": "general", "hh": "safety", "persona": "persona"}

# (name, candidate generator, baseline generator, eval set). Required ones first.
COMPARISONS = [
    ("dpo_uf_vs_sft__alpaca", "qwen2.5-0.5b-dpo-uf", "qwen2.5-0.5b-sft", "alpaca"),   # required
    ("dpo_hh_vs_sft__hh", "qwen2.5-0.5b-dpo-hh", "qwen2.5-0.5b-sft", "hh"),           # required
    ("lora_dpo_vs_instruct__persona", "smollm2-360m-lora-dpo", "smollm2-360m-instruct", "persona"),
    ("lora_dpo_lr1e-4_vs_instruct__persona", "smollm2-360m-lora-dpo-lr1e-4", "smollm2-360m-instruct", "persona"),
    ("dpo_hh_vs_sft__alpaca", "qwen2.5-0.5b-dpo-hh", "qwen2.5-0.5b-sft", "alpaca"),   # cross: helpfulness cost of HH
    ("dpo_uf_vs_sft__hh", "qwen2.5-0.5b-dpo-uf", "qwen2.5-0.5b-sft", "hh"),           # cross: safety effect of UF
    ("sft_vs_base__alpaca", "qwen2.5-0.5b-sft", "qwen2.5-0.5b-base", "alpaca"),       # did SFT help at all
]

# Decoding keys that must match inside a comparison (the model path/name obviously differs).
DECODING_KEYS = ("do_sample", "max_new_tokens", "repetition_penalty", "no_repeat_ngram_size", "temperature", "top_p", "top_k")


def load_generations(generator, eval_name):
    path = GEN_DIR / f"{generator}__{eval_name}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    config_path = path.with_suffix(".config.json")
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return {row["id"]: row for row in rows}, config, path


def check_same_decoding(cand_cfg, base_cfg, name):
    a, b = cand_cfg.get("decoding", {}), base_cfg.get("decoding", {})
    diffs = {k: (a.get(k), b.get(k)) for k in DECODING_KEYS if a.get(k) != b.get(k)}
    if diffs:
        raise ValueError(f"{name}: candidate and baseline were generated with different decoding: {diffs}")
    if cand_cfg.get("prompt_template") != base_cfg.get("prompt_template"):
        raise ValueError(f"{name}: candidate and baseline used different prompt templates")


def build(name, candidate, baseline, eval_name, seed):
    eval_rows = json.loads(EVAL_SETS[eval_name].read_text(encoding="utf-8"))
    cand, cand_cfg, cand_path = load_generations(candidate, eval_name)
    base, base_cfg, base_path = load_generations(baseline, eval_name)
    check_same_decoding(cand_cfg, base_cfg, name)

    rng = random.Random(f"{seed}:{name}")      # string seed: stable across Python versions
    pairs = []
    for idx, ev in enumerate(eval_rows):
        pid = ev["id"]
        if pid not in cand or pid not in base:
            raise ValueError(f"{name}: prompt {pid} missing from {'candidate' if pid not in cand else 'baseline'}")
        c, b = cand[pid], base[pid]
        if c["instruction"] != ev["instruction"] or b["instruction"] != ev["instruction"]:
            raise ValueError(f"{name}: instruction text differs for {pid}")
        cand_pos = rng.choice("AB")
        a_row, b_row = (c, b) if cand_pos == "A" else (b, c)
        pairs.append({
            "pair_id": f"{name}-{idx:03d}",
            "prompt_id": pid,
            "eval_set": eval_name,
            "category": ev.get("category", "all"),
            "expected_behavior": ev.get("expected_behavior"),
            "judge_mode": JUDGE_MODE[eval_name],
            "instruction": ev["instruction"],
            "response_a": a_row["output"],
            "response_b": b_row["output"],
            # Never shown to the judge. Used only after judging to map A/B back to models.
            "hidden": {
                "comparison": name,
                "candidate": candidate,
                "baseline": baseline,
                "candidate_position": cand_pos,
                "model_a": a_row["generator"],
                "model_b": b_row["generator"],
                "tokens_a": a_row.get("new_tokens"),
                "tokens_b": b_row.get("new_tokens"),
                "finish_a": a_row.get("finish_reason"),
                "finish_b": b_row.get("finish_reason"),
            },
        })
    sources = {"candidate_file": str(cand_path.relative_to(REPO_ROOT)), "baseline_file": str(base_path.relative_to(REPO_ROOT))}
    return pairs, sources


def write_swapped(out_dir):
    """Mirror every existing pair file: same pairs, A and B exchanged. Judging both orders and
    combining them cancels the judge's position bias instead of only averaging it out."""
    manifest = json.loads((out_dir / "pairs_manifest.json").read_text(encoding="utf-8"))
    swapped_manifest = []
    for entry in manifest:
        src = REPO_ROOT / entry["pairs_file"]
        dst = src.with_name(src.name.replace("_pairs.jsonl", "_swapped_pairs.jsonl"))
        rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
        with dst.open("w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                h = dict(r["hidden"])
                h.update(candidate_position="B" if h["candidate_position"] == "A" else "A",
                         model_a=h["model_b"], model_b=h["model_a"], tokens_a=h["tokens_b"], tokens_b=h["tokens_a"],
                         finish_a=h.get("finish_b"), finish_b=h.get("finish_a"), swapped=True)
                r = {**r, "response_a": r["response_b"], "response_b": r["response_a"], "hidden": h}
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        swapped_manifest.append({**entry, "pairs_file": str(dst.relative_to(REPO_ROOT)),
                                 "candidate_in_A": entry["candidate_in_B"], "candidate_in_B": entry["candidate_in_A"],
                                 "swapped_from": entry["pairs_file"]})
        print(f"{entry['name']:40s} swapped -> {dst.relative_to(REPO_ROOT)}")
    (out_dir / "pairs_manifest_swapped.json").write_text(json.dumps(swapped_manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--swap", action="store_true", help="write A/B-mirrored copies of the existing pair files")
    parser.add_argument("--only", nargs="*", help="comparison names to build (default: all)")
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "results" / "eval"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.swap:
        write_swapped(out_dir)
        return
    manifest = []
    for name, candidate, baseline, eval_name in COMPARISONS:
        if args.only and name not in args.only:
            continue
        pairs, sources = build(name, candidate, baseline, eval_name, args.seed)
        path = out_dir / f"{name}_pairs.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for row in pairs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        pos = Counter(p["hidden"]["candidate_position"] for p in pairs)
        manifest.append({"name": name, "candidate": candidate, "baseline": baseline, "eval_set": eval_name,
                         "judge_mode": JUDGE_MODE[eval_name], "n_pairs": len(pairs),
                         "candidate_in_A": pos["A"], "candidate_in_B": pos["B"],
                         "pairs_file": str(path.relative_to(REPO_ROOT)), "seed": args.seed, **sources})
        print(f"{name:40s} {len(pairs):4d} pairs  candidate A/B = {pos['A']}/{pos['B']}  -> {path.relative_to(REPO_ROOT)}")
    if not args.only:
        (out_dir / "pairs_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
