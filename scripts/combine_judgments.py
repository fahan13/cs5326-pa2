"""Combine the original and A/B-swapped judging passes into the final evaluation tables.

    python scripts/combine_judgments.py

Every pair was judged twice, once in each order. Per pair, each pass gives the candidate a
score (win = 1, tie = 0.5, loss = 0) and the two scores are summed:
    win + win / win + tie   -> win        loss + loss / loss + tie -> loss
    win + loss / tie + tie  -> tie        (the judge contradicted itself, or saw no difference)
A verdict that flips when only the order changes is position bias, not evidence, so it
cancels to a tie. The "consistency" column is the share of pairs where both passes agree.

Writes results/eval/final_results.json and results/eval/final_results.md.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "results" / "eval"
SCORE = {"win": 1.0, "tie": 0.5, "loss": 0.0}
HEADLINE = {"general": "winner", "safety": "winner", "persona": "overall"}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def combine_outcomes(o1, o2):
    if "unknown" in (o1, o2):
        return "unknown"
    total = SCORE[o1] + SCORE[o2]
    return "win" if total > 1 else "loss" if total < 1 else "tie"


def binom_two_sided_p(wins, losses):
    """Sign test: probability of a split at least this lopsided if both models were equal."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(wins, n, z=1.96):
    if n == 0:
        return None
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(centre - half, 4), round(centre + half, 4)]


def stats(counts):
    w, t, l, u = (counts.get(k, 0) for k in ("win", "tie", "loss", "unknown"))
    decided, judged = w + l, w + t + l
    return {
        "wins": w, "ties": t, "losses": l, "unknown": u,
        "win_rate_excl_ties": round(w / decided, 4) if decided else None,
        "win_rate_excl_ties_ci95": wilson(w, decided),
        "win_rate_ties_half": round((w + 0.5 * t) / judged, 4) if judged else None,
        "sign_test_p": (round(binom_two_sided_p(w, l), 6) if decided else None),
    }


def one_pass_rate(rows, crit):
    c = Counter(r["outcomes"][crit] for r in rows)
    return stats(c)


def main():
    manifest = json.loads((EVAL_DIR / "pairs_manifest.json").read_text(encoding="utf-8"))
    results = []
    for entry in manifest:
        name = entry["name"]
        pairs = read_jsonl(REPO_ROOT / entry["pairs_file"])
        p1 = {r["pair_id"]: r for r in read_jsonl(EVAL_DIR / f"{name}_judged.jsonl")}
        p2 = {r["pair_id"]: r for r in read_jsonl(EVAL_DIR / f"{name}_swapped_judged.jsonl")}
        mode = pairs[0]["judge_mode"]
        criteria = list(next(iter(p1.values()))["outcomes"])
        criteria = [c for c in criteria if c != "lean"]           # forced-choice field: reported per pass only
        head = HEADLINE[mode]

        missing = [p["pair_id"] for p in pairs if p["pair_id"] not in p1 or p["pair_id"] not in p2]
        both = [p for p in pairs if p["pair_id"] in p1 and p["pair_id"] in p2]

        res = {"comparison": name, "candidate": entry["candidate"], "baseline": entry["baseline"],
               "eval_set": entry["eval_set"], "judge_mode": mode,
               "judge_model": next(iter(p1.values()))["judge_model"],
               "n_pairs": len(pairs), "n_judged_both_orders": len(both), "missing": len(missing),
               "judge_calls": None}
        for crit in criteria:
            combined = Counter(combine_outcomes(p1[p["pair_id"]]["outcomes"][crit], p2[p["pair_id"]]["outcomes"][crit])
                               for p in both)
            agree = sum(p1[p["pair_id"]]["outcomes"][crit] == p2[p["pair_id"]]["outcomes"][crit] for p in both)
            block = stats(combined)
            block["order_consistency"] = round(agree / len(both), 4) if both else None
            block["pass1_original_order"] = one_pass_rate(list(p1.values()), crit)
            block["pass2_swapped_order"] = one_pass_rate(list(p2.values()), crit)
            res["headline" if crit == head else crit] = {"criterion": crit, **block}

            cats = defaultdict(Counter)
            for p in both:
                cats[p["category"]][combine_outcomes(p1[p["pair_id"]]["outcomes"][crit], p2[p["pair_id"]]["outcomes"][crit])] += 1
            if len(cats) > 1:
                res.setdefault("per_category", {})[crit] = {c: stats(v) for c, v in sorted(cats.items())}

        # Judge position bias: share of decided verdicts that went to whichever response was shown first.
        picks = Counter(r["verdicts"][head] for r in list(p1.values()) + list(p2.values()))
        res["judge_picked_A_rate"] = round(picks["A"] / (picks["A"] + picks["B"]), 4) if picks["A"] + picks["B"] else None
        # Length: in pairs the combined verdict decided, how often did the longer response win?
        lw = ld = 0
        for p in both:
            o = combine_outcomes(p1[p["pair_id"]]["outcomes"][head], p2[p["pair_id"]]["outcomes"][head])
            h = p["hidden"]
            ta, tb = h.get("tokens_a"), h.get("tokens_b")
            if o in ("win", "loss") and ta is not None and tb is not None and ta != tb:
                cand_tokens = ta if h["candidate_position"] == "A" else tb
                base_tokens = tb if h["candidate_position"] == "A" else ta
                ld += 1
                lw += (o == "win") == (cand_tokens > base_tokens)
        res["longer_response_won_rate"] = round(lw / ld, 4) if ld else None
        cand_tok = [p["hidden"]["tokens_a"] if p["hidden"]["candidate_position"] == "A" else p["hidden"]["tokens_b"] for p in pairs]
        base_tok = [p["hidden"]["tokens_b"] if p["hidden"]["candidate_position"] == "A" else p["hidden"]["tokens_a"] for p in pairs]
        res["mean_tokens"] = {"candidate": round(sum(cand_tok) / len(cand_tok), 1), "baseline": round(sum(base_tok) / len(base_tok), 1)}
        calls = 0
        for suffix in ("_judged.summary.json", "_swapped_judged.summary.json"):
            s = json.loads((EVAL_DIR / f"{name}{suffix}").read_text(encoding="utf-8"))
            calls += (s.get("run_stats") or {}).get("requests", 0)
        res["judge_calls"] = calls
        results.append(res)

    (EVAL_DIR / "final_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_markdown(results)
    for r in results:
        h = r["headline"]
        print(f"{r['comparison']:38s} W/T/L {h['wins']:3d}/{h['ties']:3d}/{h['losses']:3d}  "
              f"wr={h['win_rate_excl_ties']}  CI={h['win_rate_excl_ties_ci95']}  half={h['win_rate_ties_half']}  "
              f"p={h['sign_test_p']}  consistent={h['order_consistency']}  A-bias={r['judge_picked_A_rate']}")


def fmt_p(p):
    return "–" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def write_markdown(results):
    lines = ["# Pairwise evaluation (final)", "",
             "Judge: `google/gemini-3.6-flash` via kaggle_benchmarks, temperature 0, 5 pairs/request, >=5 s between requests, "
             "3 attempts per batch. Every pair judged in both A/B orders; per-pair scores summed (win=1, tie=0.5, loss=0), "
             "so a verdict that flips with order becomes a tie. Win rate excludes ties; CI is Wilson 95%; p is a two-sided sign test.", "",
             "| Comparison (candidate vs baseline) | Eval set | n | W | T | L | Win rate (excl. ties) | 95% CI | Ties = ½ | p | Order-consistent | Judge picked A | Mean tokens cand/base |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        h = r["headline"]
        ci = h["win_rate_excl_ties_ci95"] or [None, None]
        lines.append(f"| {r['candidate']} vs {r['baseline']} | {r['eval_set']} | {r['n_judged_both_orders']} | {h['wins']} | {h['ties']} | "
                     f"{h['losses']} | {h['win_rate_excl_ties']:.3f} | {ci[0]:.3f}–{ci[1]:.3f} | {h['win_rate_ties_half']:.3f} | "
                     f"{fmt_p(h['sign_test_p'])} | {h['order_consistency']:.2f} | {r['judge_picked_A_rate']:.2f} | "
                     f"{r['mean_tokens']['candidate']:.0f} / {r['mean_tokens']['baseline']:.0f} |")
    lines += ["", "## Single-pass results (each order separately)", "",
              "| Comparison | Pass | W/T/L | Win rate (excl. ties) |", "|---|---|---|---|"]
    for r in results:
        for label, key in (("original order", "pass1_original_order"), ("swapped order", "pass2_swapped_order")):
            b = r["headline"][key]
            lines.append(f"| {r['comparison']} | {label} | {b['wins']}/{b['ties']}/{b['losses']} | {b['win_rate_excl_ties']:.3f} |")
    persona = [r for r in results if r["judge_mode"] == "persona"]
    if persona:
        lines += ["", "## Persona LoRA-DPO vs SmolLM2-360M-Instruct (combined over both orders)", "",
                  "| Model | Criterion | W | T | L | Win rate (excl. ties) | Order-consistent |", "|---|---|---|---|---|---|---|"]
        for r in persona:
            for key in ("style", "usefulness", "safety", "headline"):
                b = r[key]
                wr = "–" if b["win_rate_excl_ties"] is None else f"{b['win_rate_excl_ties']:.3f}"
                lines.append(f"| {r['candidate']} | {b['criterion']} | {b['wins']} | {b['ties']} | {b['losses']} | {wr} | {b['order_consistency']:.2f} |")
    for r in results:
        if r["eval_set"] == "hh" and "per_category" in r:
            lines += ["", f"## Per category: {r['candidate']} vs {r['baseline']} (HH eval)", "",
                      "| Category | W | T | L | Win rate (excl. ties) |", "|---|---|---|---|---|"]
            for c, b in r["per_category"]["winner"].items():
                wr = "–" if b["win_rate_excl_ties"] is None else f"{b['win_rate_excl_ties']:.3f}"
                lines.append(f"| {c} | {b['wins']} | {b['ties']} | {b['losses']} | {wr} |")
    (EVAL_DIR / "final_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
