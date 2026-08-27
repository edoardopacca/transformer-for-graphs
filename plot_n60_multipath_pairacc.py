"""Report X (personal working doc) -- per-pair accuracy heatmap for the n=60 multipath rescue
test (eval_n60_multipath_v2.py's new per_pair_acc field): since ground truth is "connected"
everywhere in Construction A/B, per-pair accuracy IS the fraction of test graphs where that
pair was predicted connected. Whole-graph exact match is near 0 (Edoardo, 2026-08-27) despite
target-pair accuracy 1.000 and pooled reach_route ~0.998 -- this shows WHICH pairs carry that
residual error instead of just the aggregate number.

Uses paper_zcolormap-style red gradient (white=worst, dark red=best) but on a plain [0,1]
accuracy scale, not the signed Z_ij quantity -- a different colorbar semantics, so NOT the
same colormap object as the Z_ij figures (that one is reserved for scale*cos+bias).

    python plot_n60_multipath_pairacc.py --seed 1000
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def panel(ax, acc, boundaries, target_pair, title):
    n = acc.shape[0]
    im = ax.imshow(acc, cmap="Reds_r", vmin=0.0, vmax=1.0)
    for b in boundaries:
        ax.axvline(b, color="k", ls="--", lw=1)
        ax.axhline(b, color="k", ls="--", lw=1)
    i, j = target_pair
    ax.plot(j, i, "b+", ms=10, mew=2)
    ax.plot(i, j, "b+", ms=10, mew=2)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([0, n - 1]); ax.set_xticklabels([1, n], fontsize=8)
    ax.set_yticks([0, n - 1]); ax.set_yticklabels([1, n], fontsize=8)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    root = Path(f"runs/report10/multipath_v2/n60_pathunion_k3_seed{args.seed}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im = None
    for ax, name, label in [(axes[0], "A_clean", "Construction A"),
                              (axes[1], "B_stitched", "Construction B")]:
        d = np.load(root / f"{name}_per_pair_acc.npz")
        im = panel(ax, d["acc"], d["boundaries"], d["target_pair"], label)
    fig.suptitle(f"n=60 multipath rescue -- per-pair predicted-connected rate (seed {args.seed})\n"
                 "1.0 = always predicted connected (correct); blue + marks the target pair/hubs",
                 fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="P(predicted connected)")

    out = Path(f"runs/report10/report10_figs/r10_n60_multipath_pairacc_seed{args.seed}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
