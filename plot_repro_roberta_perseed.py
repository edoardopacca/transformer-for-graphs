"""Per-seed plots for the RoBERTa-faithful n=20 reproduction.

Draws every seed as its own line (no mean, no band): for a bimodal outcome
(a few seeds generalise, most stay flat) the mean+band view is misleading.
Seeds are discovered automatically from the run directories, so this picks up
the full sweep (8 seeds/condition) once all runs have a history.json.

Outputs into runs/repro_paper_n20_roberta/:
  fig1_perseed.png   - unrestricted: small multiples, one panel per seed
                       (ER / 2Chain / 2Clique exact match vs step).
  fig7_perseed.png   - 2Chain exact match, one line per seed, colour by condition.
  fig11_perseed.png  - 2Clique exact match, one line per seed, colour by condition.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RUNS = Path("runs/repro_paper_n20_roberta")
CONDS = [("unrestricted", "Unrestricted", "#ff7f0e"),
         ("diam9", "Restrict D≤9", "#1f77b4")]
DIR_RE = re.compile(r"n\d+_p\d+_(?P<cond>unrestricted|diam9)_seed(?P<seed>\d+)$")


def discover():
    """{cond: sorted [seeds with a history.json]}."""
    out = {c: [] for c, _, _ in CONDS}
    for d in sorted(RUNS.iterdir()):
        m = DIR_RE.match(d.name) if d.is_dir() else None
        if m and (d / "history.json").exists() and m.group("cond") in out:
            out[m.group("cond")].append(int(m.group("seed")))
    return {c: sorted(s) for c, s in out.items()}


def load(cond: str, seed: int) -> dict:
    with (RUNS / f"n20_p008_{cond}_seed{seed}" / "history.json").open() as f:
        return json.load(f)


def fig1_small_multiples(seeds_by_cond) -> None:
    seeds = seeds_by_cond["unrestricted"]
    if not seeds:
        return
    ncols = min(4, len(seeds)); nrows = math.ceil(len(seeds) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows),
                             sharey=True, squeeze=False)
    flat = axes.flatten()
    for ax, seed in zip(flat, seeds):
        h = load("unrestricted", seed); s = h["steps"]
        ax.plot(s, h["val_er_exact"], lw=1.8, color="#1f77b4", label="ER (in-dist)")
        ax.plot(s, h["val_2chain_exact"], lw=1.8, color="#2ca02c", label="Two Chains")
        ax.plot(s, h["val_2clique_exact"], lw=1.8, color="#ff7f0e", label="Two Cliques")
        ax.set_title(f"seed {seed}"); ax.set_ylim(-0.03, 1.05); ax.grid(alpha=0.3)
        ax.set_xlabel("step")
    for ax in flat[len(seeds):]:
        ax.axis("off")
    flat[0].set_ylabel("exact-match acc"); flat[0].legend(fontsize=8, loc="center left")
    fig.suptitle("RoBERTa-faithful, unrestricted ER$(n=20,p=0.08)$ — per seed", y=1.0)
    fig.tight_layout()
    out = RUNS / "fig1_perseed.png"; fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}  ({len(seeds)} seeds)")


def metric_perseed(metric: str, title: str, out_name: str, seeds_by_cond) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond, _label, color in CONDS:
        for seed in seeds_by_cond[cond]:
            h = load(cond, seed)
            ax.plot(h["steps"], h[metric], color=color, lw=1.3, alpha=0.55)
    ax.set_title(title); ax.set_xlabel("Training step")
    ax.set_ylabel("Exact-match accuracy"); ax.set_ylim(-0.03, 1.05); ax.grid(alpha=0.3)
    handles = [Line2D([0], [0], color=c, lw=2,
                      label=f"{lab} ({len(seeds_by_cond[cd])} seeds)")
               for cd, lab, c in CONDS]
    ax.legend(handles=handles, loc="upper left")
    fig.tight_layout()
    out = RUNS / out_name; fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    if not RUNS.exists():
        raise SystemExit(f"{RUNS} not found")
    seeds = discover()
    for c, lab, _ in CONDS:
        print(f"  {lab}: {len(seeds[c])} seeds {seeds[c]}")
    fig1_small_multiples(seeds)
    metric_perseed("val_2chain_exact",
                   "RoBERTa-faithful — Two Chains exact match (per seed)",
                   "fig7_perseed.png", seeds)
    metric_perseed("val_2clique_exact",
                   "RoBERTa-faithful — Two Cliques exact match (per seed)",
                   "fig11_perseed.png", seeds)


if __name__ == "__main__":
    main()
