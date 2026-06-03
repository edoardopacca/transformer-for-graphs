"""Generalisation-rate analysis for the RoBERTa-faithful n=20 reproduction.

Aggregates every seed in runs/repro_paper_n20_roberta/ and, for each condition
(unrestricted vs restricted D<=9), reports the fraction of seeds that recover a
generalising solution, with 95% Wilson confidence intervals. A "success" is
defined a priori as final exact-match accuracy > THRESHOLD on the OOD family.

This converts the single-seed anecdote (the paper's Fig 7/11) into a statistic:
if the restricted condition generalises at a significantly higher rate than the
unrestricted one, the data lever helps; if the intervals overlap heavily, the
clean separation is not distinguishable from seed luck at this sample size.

Usage:
    python plot_repro_rate.py [--runs_dir runs/repro_paper_n20_roberta]
                              [--threshold 0.5]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR_RE = re.compile(r"n\d+_p\d+_(?P<cond>unrestricted|diam\d+)_seed(?P<seed>\d+)")
COND_LABEL = {"unrestricted": "Unrestricted", "diam9": "Restrict D<=9"}


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs/repro_paper_n20_roberta")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="a-priori success threshold on final OOD exact match")
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)
    thr = args.threshold

    data: dict[str, list] = {}
    for d in sorted(runs_dir.iterdir()):
        m = DIR_RE.match(d.name) if d.is_dir() else None
        hist = d / "history.json"
        if not m or not hist.exists():
            continue
        h = json.load(hist.open())
        data.setdefault(m.group("cond"), []).append({
            "seed": int(m.group("seed")),
            "er": h["val_er_exact"][-1],
            "2chain": h["val_2chain_exact"][-1],
            "2clique": h["val_2clique_exact"][-1],
        })

    if not data:
        raise SystemExit(f"No runs under {runs_dir}")

    conds = [c for c in ("unrestricted", "diam9") if c in data]
    families = ["2chain", "2clique", "any"]
    print(f"Success threshold: final exact match > {thr}\n")
    summary = {}
    for cond in conds:
        runs = data[cond]
        n = len(runs)
        seeds = sorted(r["seed"] for r in runs)
        print(f"== {COND_LABEL[cond]} == ({n} seeds: {seeds})")
        summary[cond] = {}
        for fam in families:
            if fam == "any":
                k = sum(1 for r in runs if max(r["2chain"], r["2clique"]) > thr)
            else:
                k = sum(1 for r in runs if r[fam] > thr)
            p, lo, hi = wilson(k, n)
            summary[cond][fam] = (k, n, p, lo, hi)
            print(f"   {fam:8s}: {k}/{n} generalise  "
                  f"rate={p:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
        print()

    # ── Bar plot with Wilson CIs ──
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(families))
    width = 0.38
    colors = {"unrestricted": "#ff7f0e", "diam9": "#1f77b4"}
    for i, cond in enumerate(conds):
        rates = [summary[cond][f][2] for f in families]
        los = [summary[cond][f][2] - summary[cond][f][3] for f in families]
        his = [summary[cond][f][4] - summary[cond][f][2] for f in families]
        offs = [xi + (i - (len(conds) - 1) / 2) * width for xi in x]
        ax.bar(offs, rates, width, color=colors[cond], label=COND_LABEL[cond],
               yerr=[los, his], capsize=5)
        for xi, r, (k, nn, *_ ) in zip(offs, rates, [summary[cond][f] for f in families]):
            ax.text(xi, r + 0.02, f"{k}/{nn}", ha="center", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(["2Chain", "2Clique", "either OOD"])
    ax.set_ylabel(f"Generalisation rate (final exact > {thr})")
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    ax.set_title("RoBERTa-faithful, n=20: fraction of seeds that generalise\n"
                 "(error bars: 95% Wilson CI)")
    ax.legend()
    fig.tight_layout()
    out = runs_dir / "generalisation_rate.png"
    fig.savefig(out, dpi=170); plt.close(fig)
    print(f"saved figure: {out}")
    json.dump({c: {f: summary[c][f] for f in families} for c in conds},
              (runs_dir / "generalisation_rate.json").open("w"), indent=2)
    print(f"saved data:   {runs_dir / 'generalisation_rate.json'}")


if __name__ == "__main__":
    main()
