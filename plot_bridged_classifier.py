"""Report V: figures for the dedicated bridged-vs-split classifier.

Reads the per-run histories under runs/report5/bridged_clf/ (written by
experiments2/train_bridged_classifier.py) and produces, per node count present
(n=20 and, if run, n=40):

  * classifier_trajectory_n{n}.png -- the main figure: accuracy vs TRAINING STEP,
    one line per clique size (colour = clique size), one panel per random-c seed.
    Shows the whole learning dynamics, not a single snapshot: small cliques are
    mastered first and large ones last, and every size eventually reaches 1.0.

  * classifier_fixedc_loss.png -- the fixed-clique-size runs (separate model per
    clique size): training loss at step 2000 vs clique size, n=20 and n=40 overlaid.
    The "density in optimisation" signal: a denser clique is slower to fit (loss
    rises towards the chance value ln2), though accuracy still reaches 1.0 with more
    steps.

No GPU, no checkpoints: regenerates from the committed history.json files.

  python plot_bridged_classifier.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parent
CLF = ROOT / "runs" / "report5" / "bridged_clf"
OUT = ROOT / "runs" / "report5" / "report5_figs"

RAND_SEEDS = [1000, 2000, 3000, 4000]
FIXED = {20: [3, 6, 10], 40: [5, 10, 15, 20]}   # fixed clique sizes per n
FIXED_SEEDS = [1000, 2000]
CMAP = plt.get_cmap("viridis")


def load(run: str) -> dict | None:
    p = CLF / run / "history.json"
    return json.load(open(p)) if p.exists() else None


def make_trajectory_fig(n: int) -> None:
    runs = {s: load(f"bridged_clf_n{n}_rand_seed{s}") for s in RAND_SEEDS}
    runs = {s: h for s, h in runs.items() if h is not None}
    if not runs:
        print(f"n={n}: no random-c runs, skipping trajectory")
        return

    # clique sizes present (from any run's by_c dict)
    any_h = next(iter(runs.values()))
    cs = sorted(int(c) for c in any_h["val_acc_by_c"][0])
    norm = Normalize(vmin=min(cs), vmax=max(cs))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    for ax, (s, h) in zip(axes.ravel(), runs.items()):
        steps = h["steps"]
        for c in cs:
            ys = [d.get(str(c), float("nan")) for d in h["val_acc_by_c"]]
            ax.plot(steps, ys, color=CMAP(norm(c)), lw=1.2)
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_ylim(0.45, 1.03)
        ax.set_xscale("log")
        ax.set_xlim(left=min(steps))
        ax.grid(alpha=0.3, which="both")
        ax.set_title(f"seed {s}", fontsize=10)
    for ax in axes[-1, :]:
        ax.set_xlabel("training step")
    for ax in axes[:, 0]:
        ax.set_ylabel("classification accuracy")

    sm = ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.04, pad=0.02)
    cbar.set_label("clique size c")
    fig.suptitle(f"Bridged-vs-split classifier (n={n}): accuracy by training step, "
                 f"one line per clique size\n(small cliques learned first, large ones last; "
                 f"all sizes reach 1.0)")
    out = OUT / f"classifier_trajectory_n{n}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out}")


def make_fixedc_loss_fig() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    colors = {20: "tab:blue", 40: "tab:red"}
    for n, fixed_cs in FIXED.items():
        by_c: dict[int, list[float]] = {}
        for c in fixed_cs:
            for s in FIXED_SEEDS:
                h = load(f"bridged_clf_n{n}_c{c}_seed{s}")
                if h is not None:
                    by_c.setdefault(c, []).append(h["train_loss"][0])  # loss @ step 2000
        if not by_c:
            continue
        cs = sorted(by_c)
        means = [sum(by_c[c]) / len(by_c[c]) for c in cs]
        ax.plot(cs, means, marker="s", lw=1.8, color=colors[n], label=f"n={n}")
        for c in cs:
            for v in by_c[c]:
                ax.plot(c, v, marker="o", ms=4, color=colors[n], alpha=0.35)
    ax.axhline(math.log(2), color="gray", ls=":", lw=1)
    ax.text(0.02, math.log(2) + 0.01, "chance loss (ln 2)", fontsize=8,
            transform=ax.get_yaxis_transform())
    ax.set_xlabel("clique size c (fixed)")
    ax.set_ylabel("training loss at step 2000")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_title("Fixed-size runs: denser clique = slower to fit\n"
                 "(accuracy still reaches 1.0 for every c with more steps)")
    fig.tight_layout()
    out = OUT / "classifier_fixedc_loss.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"saved -> {out}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for n in (20, 40):
        make_trajectory_fig(n)
    make_fixedc_loss_fig()


if __name__ == "__main__":
    main()
