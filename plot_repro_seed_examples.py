"""Illustrative per-seed training curves for the RoBERTa-faithful, restricted
(D<=9) reproduction: exact-match accuracy on 2Chain and 2Clique vs training step,
for a few hand-picked seeds. Shows that even WITH the data lever the outcome is a
seed lottery: one seed looks like the paper's Figure 7/11, others do not.

Default panels (restricted D<=9): seed 6000 (both OOD families rise --- the kind
of single run the paper shows), seed 7000 (stays flat --- fails), seed 1000
(2Chain rises but 2Clique does not --- partial).

    python plot_repro_seed_examples.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path("runs/repro_paper_n20_roberta")
# (condition_dir_tag, seed, short label)
PANELS = [("diam9", 6000, "seed 6000 — both rise"),
          ("diam9", 7000, "seed 7000 — fails"),
          ("diam9", 1000, "seed 1000 — partial")]


def load(cond: str, seed: int) -> dict:
    with (RUNS / f"n20_p008_{cond}_seed{seed}" / "history.json").open() as f:
        return json.load(f)


def main() -> None:
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.6 * len(PANELS), 4.0),
                             sharey=True, squeeze=False)
    for ax, (cond, seed, title) in zip(axes[0], PANELS):
        h = load(cond, seed); s = h["steps"]
        ax.plot(s, h["val_2chain_exact"], lw=2, color="#2ca02c", label="2Chain")
        ax.plot(s, h["val_2clique_exact"], lw=2, color="#ff7f0e", label="2Clique")
        ax.set_title(title); ax.set_xlabel("training step")
        ax.set_ylim(-0.03, 1.05); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("OOD exact-match accuracy")
    axes[0][0].legend(loc="center left")
    fig.suptitle("Restricted D≤9 (RoBERTa-faithful): same data lever, "
                 "different seeds", y=1.02)
    fig.tight_layout()
    out = RUNS / "seed_examples.png"
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
