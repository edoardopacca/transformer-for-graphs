"""Report V: aggregate figure for the dedicated bridged-vs-split classifier.

Reads the per-run histories under runs/report5/bridged_clf/ (written by
experiments2/train_bridged_classifier.py) and produces one summary figure:

  * left  -- accuracy by clique size EARLY in training (step 2000), one line per
             random-clique-size seed. This is where the difficulty ordering shows:
             larger cliques are mastered last, seed-dependently. (The FINAL curve is
             flat at 1.0 for every seed, so we draw it once as a reference line.)
  * right -- fixed-clique-size runs: early (step 2000) training loss vs clique size.
             Accuracy is already 1.0 at the first eval for every c; the loss still
             rises with the clique density -- the mild "density in optimisation"
             signal -- without delaying the model reaching 100%.

No GPU, no checkpoints: regenerates from the committed history.json files. Run
locally after pulling runs/report5/bridged_clf/.

  python plot_bridged_classifier.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CLF = ROOT / "runs" / "report5" / "bridged_clf"
OUT = ROOT / "runs" / "report5" / "report5_figs"


def load(run: str) -> dict:
    return json.load(open(CLF / run / "history.json"))


def acc_at_step(h: dict, step: int) -> dict:
    """accuracy-by-clique-size dict at the eval closest to `step`."""
    i = min(range(len(h["steps"])), key=lambda k: abs(h["steps"][k] - step))
    return {int(c): v for c, v in h["val_acc_by_c"][i].items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rand_seeds = [1000, 2000, 3000, 4000]
    fixed = [(3, 1000), (3, 2000), (6, 1000), (6, 2000), (10, 1000), (10, 2000)]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    # -- left: by-clique-size early (step 2000), per seed --
    for s in rand_seeds:
        h = load(f"bridged_clf_n20_rand_seed{s}")
        d = acc_at_step(h, 2000)
        cs = sorted(d)
        axL.plot(cs, [d[c] for c in cs], marker="o", lw=1.6, label=f"seed {s}")
    axL.axhline(1.0, color="green", ls="--", lw=1.2, label="final (all seeds)")
    axL.axhline(0.5, color="gray", ls=":", lw=1)
    axL.set_xlabel("clique size c")
    axL.set_ylabel("classification accuracy")
    axL.set_ylim(0.45, 1.03)
    axL.grid(alpha=0.3)
    axL.legend(fontsize=8)
    axL.set_title("early in training (step 2000)\nlarger cliques mastered last")

    # -- right: fixed-c early training loss vs clique size --
    by_c: dict[int, list[float]] = {}
    for c, s in fixed:
        h = load(f"bridged_clf_n20_c{c}_seed{s}")
        by_c.setdefault(c, []).append(h["train_loss"][0])  # loss at step 2000
    cs = sorted(by_c)
    means = [sum(by_c[c]) / len(by_c[c]) for c in cs]
    axR.plot(cs, means, marker="s", lw=1.8, color="purple")
    for c in cs:
        for v in by_c[c]:
            axR.plot(c, v, marker="o", ms=4, color="purple", alpha=0.4)
    axR.set_xlabel("clique size c (fixed)")
    axR.set_ylabel("training loss at step 2000")
    axR.grid(alpha=0.3)
    axR.set_title("fixed-size runs: denser clique = higher early loss\n"
                  "(accuracy already 1.0 at step 2000 for every c)")

    fig.suptitle("Bridged-vs-split classifier (n=20, minimal trunk, mean-pool + 1 logit)")
    fig.tight_layout()
    out = OUT / "classifier_by_clique_size.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
