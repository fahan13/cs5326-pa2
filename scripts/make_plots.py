"""Training and evaluation figures for the report.

    python scripts/make_plots.py          # -> results/figures/*.pdf and *.png

Reads only committed run records (results/runs/*/metrics.jsonl) and the combined
judgments (results/eval/final_results.json), so the figures are reproducible from the repo.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "results" / "runs"
OUT = REPO_ROOT / "results" / "figures"

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9a94"     # categorical slots 1-2 + neutral
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e6e5e0"
EMA = 0.9                                                  # smoothing factor for train curves

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.6, "savefig.dpi": 200,
    "font.family": "DejaVu Sans",
})


def read(run):
    return [json.loads(line) for line in (RUNS / run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def ema(values, alpha=EMA):
    out, s = [], None
    for v in values:
        s = v if s is None else alpha * s + (1 - alpha) * v
        out.append(s)
    return out


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def plot_sft():
    m = read("sft")
    cfg = json.loads((RUNS / "sft" / "config.json").read_text())
    tr = [r for r in m if r["split"] == "train"]
    va = [r for r in m if r["split"] == "val"]
    best = next(r for r in m if r["split"] == "summary")["best_step"]
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    ax.plot([r["step"] for r in tr], [r["loss"] for r in tr], color=BLUE, alpha=0.22, linewidth=0.8, label="train (raw, logged every 10 steps)")
    ax.plot([r["step"] for r in tr], ema([r["loss"] for r in tr]), color=BLUE, label=f"train (EMA {EMA})")
    ax.plot([r["step"] for r in va], [r["loss"] for r in va], color=ORANGE, marker="o", markersize=4, label="validation (1,000 examples)")
    bv = next(r for r in va if r["step"] == best)
    ax.annotate(f"best = final\nstep {best}, {bv['loss']:.3f}", (best, bv["loss"]), xytext=(-62, 22), textcoords="offset points",
                fontsize=7, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.axvline(cfg["warmup_steps"], color=GRAY, linewidth=0.8, linestyle=":")
    ax.text(cfg["warmup_steps"] + 8, 1.02, "end of warmup (step %d)" % cfg["warmup_steps"], fontsize=7, color=INK2)
    ax.set_xlabel("optimizer step (effective batch 32)")
    ax.set_ylabel("response-token NLL (mean)")
    ax.set_ylim(1.0, 1.62)
    ax.set_title("SFT on Qwen2.5-0.5B: 30k UltraChat examples, 1 epoch")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    save(fig, "sft_loss")


def plot_dpo():
    runs = {"HH-RLHF": ("dpo_hh", BLUE), "UltraFeedback": ("dpo_uf", ORANGE)}
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.1))
    axes = axes.ravel()
    for label, (run, color) in runs.items():
        m = read(run)
        tr = [r for r in m if r["split"] == "train"]
        va = [r for r in m if r["split"] == "val"]
        best = next(r for r in m if r["split"] == "summary")["best_step"]
        vs = [r["step"] for r in va]
        a = axes[0]
        a.plot([r["step"] for r in tr], ema([r["loss"] for r in tr]), color=color, alpha=0.35, linewidth=1.0)
        a.plot(vs, [r["loss"] for r in va], color=color, marker="o", markersize=3, label=label)
        a = axes[1]
        a.plot(vs[1:], [r["reward_acc"] for r in va][1:], color=color, marker="o", markersize=3, label=label)
        bv = next(r for r in va if r["step"] == best)
        a.plot([best], [bv["reward_acc"]], marker="*", markersize=11, color=color, markeredgecolor="white", markeredgewidth=0.6, zorder=5)
        axes[2].plot(vs, [r["reward_margin"] for r in va], color=color, marker="o", markersize=3, label=label)
        a = axes[3]
        a.plot(vs, [r["reward_chosen"] for r in va], color=color, linewidth=1.6)
        a.plot(vs, [r["reward_rejected"] for r in va], color=color, linewidth=1.2, linestyle="--")
    axes[0].axhline(0.6931, color=GRAY, linewidth=0.8, linestyle=":")
    axes[0].text(5, 0.6965, "log 2", fontsize=6.5, color=INK2)
    axes[0].set_title("DPO loss")
    axes[0].set_ylabel("loss")
    axes[1].set_title("val reward accuracy (★ = selected)")
    axes[1].set_ylim(0.55, 0.72)
    axes[2].set_title("val reward margin  (r_w − r_l)")
    axes[2].axhline(0, color=GRAY, linewidth=0.8)
    axes[3].set_title("val implicit rewards  β·log(π/π_ref)")
    axes[3].axhline(0, color=GRAY, linewidth=0.8)
    axes[3].plot([], [], color=INK2, linewidth=1.6, label="chosen  r_w")
    axes[3].plot([], [], color=INK2, linewidth=1.2, linestyle="--", label="rejected  r_l")
    handles = [h for h in axes[3].get_legend_handles_labels()[0][-2:]]
    axes[3].legend(handles=handles, loc="lower left")
    for a in axes[2:]:
        a.set_xlabel("optimizer step (effective batch 32)")
    h0, l0 = axes[0].get_legend_handles_labels()
    fig.legend(h0, l0, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(h_pad=1.2, w_pad=1.5, rect=(0, 0, 1, 0.96))
    save(fig, "dpo_curves")


def plot_lora():
    runs = {"lr 5e-5 (submitted)": ("lora_dpo", BLUE), "lr 1e-4 (ablation)": ("lora_dpo_lr1e-4", ORANGE)}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 1.9))
    for label, (run, color) in runs.items():
        m = read(run)
        tr = [r for r in m if "loss" in r and "eval_loss" not in r and "train_runtime" not in r]
        ev = [r for r in m if "eval_loss" in r]
        axes[0].plot([r["step"] for r in tr], [r["loss"] for r in tr], color=color, alpha=0.3, linewidth=0.9)
        axes[0].plot([r["step"] for r in ev], [r["eval_loss"] for r in ev], color=color, marker="o", markersize=3, label=label)
        axes[1].plot([r["step"] for r in ev], [r["eval_rewards/margins"] for r in ev], color=color, marker="o", markersize=3, label=label)
    axes[0].set_title("LoRA-DPO loss (markers: 20 held-out pairs; faint: train)")
    axes[0].set_ylabel("loss")
    axes[1].set_title("held-out reward margin")
    for a in axes:
        a.set_xlabel("optimizer step (3 epochs × 25 steps, batch 8)")
    axes[0].legend(loc="upper right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "lora_dpo_curves")


def plot_results():
    res = {r["comparison"]: r for r in json.loads((REPO_ROOT / "results" / "eval" / "final_results.json").read_text())}
    rows = [
        ("dpo_uf_vs_sft__alpaca", "DPO-UF vs SFT, Alpaca*"),
        ("dpo_hh_vs_sft__hh", "DPO-HH vs SFT, HH*"),
        ("dpo_hh_vs_sft__alpaca", "DPO-HH vs SFT, Alpaca"),
        ("dpo_uf_vs_sft__hh", "DPO-UF vs SFT, HH"),
        ("sft_vs_base__alpaca", "SFT vs base, Alpaca"),
    ]
    fig, ax = plt.subplots(figsize=(5.0, 1.85))
    for i, (key, label) in enumerate(rows):
        h = res[key]["headline"]
        n = h["wins"] + h["ties"] + h["losses"]
        left = 0
        for val, color, name in ((h["wins"], BLUE, "candidate wins"), (h["ties"], "#d9d8d2", "tie"), (h["losses"], ORANGE, "baseline wins")):
            ax.barh(i, val / n, left=left, color=color, height=0.62, edgecolor="white", linewidth=1.5, label=name if i == 0 else None)
            if val / n > 0.07:
                ax.text(left + val / n / 2, i, str(val), ha="center", va="center", fontsize=7, color="white" if color != "#d9d8d2" else INK)
            left += val / n
        p = h["sign_test_p"]
        ptxt = "p<0.001" if p < 0.001 else f"p={p:.2f}"
        ax.text(1.01, i, f"{h['win_rate_excl_ties']:.1%}  ({ptxt})", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of 300 pairs (both A/B orders combined)   * required comparison")
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3)
    ax.text(1.01, -0.75, "win rate excl. ties", fontsize=7, color=INK2)
    save(fig, "pairwise_results")


if __name__ == "__main__":
    plot_sft()
    plot_dpo()
    plot_lora()
    plot_results()
