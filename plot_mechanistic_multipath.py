"""Local, no-GPU. Renders the real attention-score/alpha heatmaps produced by
mechanistic_multipath_heatmaps.py -- one figure per checkpoint, one row per k, columns
= [scores layer0, alpha layer0, scores layer1, alpha layer1]. Dashed lines mark the
route boundaries (s, t, then each parallel route in turn) so it is visible whether
attention concentrates within a single route or spreads across several as k grows.

  python plot_mechanistic_multipath.py \
      --npz runs/report9/heatmaps_multipath/<tag>/multipath_heatmap_data.npz \
      --title_tag "ER-trained, n=64, ell=13" \
      --out runs/report9/report9_figs/r9_multipath_attention_<tag>.png
"""
import argparse
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _draw_bounds(ax, bounds):
    for b in bounds[1:-1]:
        ax.axhline(b - 0.5, color="white", lw=0.6, alpha=0.7)
        ax.axvline(b - 0.5, color="white", lw=0.6, alpha=0.7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title_tag", default="")
    args = ap.parse_args()

    d = np.load(args.npz)
    ks = sorted(int(k) for k in d["ks_present"])
    n_full = {k: int(d[f"k{k}__n_full"][0]) for k in ks}
    n_active = {k: int(d[f"k{k}__n_active"][0]) for k in ks}

    fig, axes = plt.subplots(len(ks), 4, figsize=(4 * 3.2, len(ks) * 3.0), squeeze=False)
    cols = ["scores0", "alpha0", "scores1", "alpha1"]
    col_titles = [r"scores layer 0 ($S=QK^\top/\sqrt{d_h}$)", r"$\alpha$ layer 0",
                  r"scores layer 1", r"$\alpha$ layer 1"]
    for ri, k in enumerate(ks):
        na = n_active[k]
        bounds = [b for b in d[f"k{k}__route_bounds"] if b <= na]
        if bounds[-1] != na:
            bounds.append(na)
        for ci, (col, ctitle) in enumerate(zip(cols, col_titles)):
            ax = axes[ri][ci]
            mat = d[f"k{k}__{col}"][:na, :na]
            cmap = "RdBu_r" if col.startswith("scores") else "viridis"
            vmax = np.abs(mat).max() if col.startswith("scores") else mat.max()
            vmin = -vmax if col.startswith("scores") else 0.0
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
            _draw_bounds(ax, bounds)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if ri == 0:
                ax.set_title(ctitle, fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"k={k} ($n_{{full}}$={n_full[k]})\nnode index", fontsize=9)
    suptitle = "Real attention, multipath graphs" + (f" -- {args.title_tag}" if args.title_tag else "")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
