"""Per-seed plots for the RoBERTa-faithful n=20 reproduction.

Unlike plot_repro_paper_figures.py (mean + std band), this draws every seed as
its own line. For a bimodal outcome (one seed generalises, the others stay flat)
the mean+band view is misleading; the per-seed view shows the runs as they are.

Outputs into runs/repro_paper_n20_roberta/:
  fig1_perseed.png   - unrestricted: small multiples, one panel per seed,
                       each with ER / 2Chain / 2Clique exact match vs step.
  fig7_perseed.png   - 2Chain exact match, one line per (condition, seed).
  fig11_perseed.png  - 2Clique exact match, one line per (condition, seed).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path("runs/repro_paper_n20_roberta")
SEEDS = [1000, 2000, 3000]
CONDS = [("unrestricted", "Unrestricted", "#ff7f0e"),
         ("diam9", "Restrict D≤9", "#1f77b4")]
SEED_STYLE = {1000: "-", 2000: "--", 3000: ":"}


def load(cond: str, seed: int) -> dict:
    with (RUNS / f"n20_p008_{cond}_seed{seed}" / "history.json").open() as f:
        return json.load(f)


def fig1_small_multiples() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        h = load("unrestricted", seed)
        s = h["steps"]
        ax.plot(s, h["val_er_exact"], lw=2, color="#1f77b4", label="Erdős–Rényi (in-dist)")
        ax.plot(s, h["val_2chain_exact"], lw=2, color="#2ca02c", label="Two Chains")
        ax.plot(s, h["val_2clique_exact"], lw=2, color="#ff7f0e", label="Two Cliques")
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("Training step")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Exact-match accuracy")
    axes[0].legend(loc="center left", fontsize=9)
    fig.suptitle("RoBERTa-faithful, unrestricted ER$(n=20,p=0.08)$ — per seed", y=1.02)
    fig.tight_layout()
    out = RUNS / "fig1_perseed.png"
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def metric_perseed(metric: str, title: str, out_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond, clabel, color in CONDS:
        for seed in SEEDS:
            h = load(cond, seed)
            ax.plot(h["steps"], h[metric], color=color, ls=SEED_STYLE[seed],
                    lw=2, alpha=0.95, label=f"{clabel}, seed {seed}")
    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Exact-match accuracy")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    out = RUNS / out_name
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    if not RUNS.exists():
        raise SystemExit(f"{RUNS} not found")
    fig1_small_multiples()
    metric_perseed("val_2chain_exact",
                   "RoBERTa-faithful — Two Chains exact match (per seed)",
                   "fig7_perseed.png")
    metric_perseed("val_2clique_exact",
                   "RoBERTa-faithful — Two Cliques exact match (per seed)",
                   "fig11_perseed.png")


if __name__ == "__main__":
    main()
