"""Restricted vs. unrestricted, side by side, for a few seeds --- to show that
the data lever ``working'' (the paper's Figure 7/11) is a per-seed accident.

For each of three seeds (1000, 6000, 7000) we overlay the unrestricted and the
restricted (D<=9) training curves, on 2Chain (top row) and 2Clique (bottom row).
Reading across:
  - seed 6000: restricted lifts both OOD families while unrestricted does not
    -> looks exactly like the paper;
  - seed 1000: mixed (restricted wins on 2Chain, unrestricted wins on 2Clique);
  - seed 7000: unrestricted generalises and restricted *fails* -> the lever hurts.
So whether restricting helps is set by the seed, not by the filter.

    python plot_repro_seed_examples.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path("runs/report3/repro_paper_n20_roberta")
SEEDS = [1000, 6000, 7000]
CONDS = [("unrestricted", "Unrestricted", "#ff7f0e"),
         ("diam9", "Restricted D≤9", "#1f77b4")]
ROWS = [("val_2chain_exact", "2Chain"), ("val_2clique_exact", "2Clique")]


def load(cond: str, seed: int) -> dict:
    with (RUNS / f"n20_p008_{cond}_seed{seed}" / "history.json").open() as f:
        return json.load(f)


def main() -> None:
    fig, axes = plt.subplots(len(ROWS), len(SEEDS),
                             figsize=(4.3 * len(SEEDS), 3.4 * len(ROWS)),
                             sharex=True, sharey=True, squeeze=False)
    for r, (metric, mlabel) in enumerate(ROWS):
        for c, seed in enumerate(SEEDS):
            ax = axes[r][c]
            for cond, clabel, color in CONDS:
                h = load(cond, seed)
                ax.plot(h["steps"], h[metric], color=color, lw=2, label=clabel)
            ax.set_ylim(-0.03, 1.05); ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(f"seed {seed}")
            if c == 0:
                ax.set_ylabel(f"{mlabel}\nexact-match acc")
            if r == len(ROWS) - 1:
                ax.set_xlabel("training step")
    axes[0][0].legend(loc="center left", fontsize=8)
    fig.suptitle("Restricted vs. unrestricted, per seed (RoBERTa-faithful, n=20): "
                 "the data lever helps only by luck of the seed", y=1.01)
    fig.tight_layout()
    out = RUNS / "seed_examples.png"
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
