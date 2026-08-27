"""Report X (personal working doc) -- render the per-split H^(2) geometry heatmaps produced by
eval_beyond_threshold_heatmaps.py (already-downloaded *_Z.npz files, no HPC/GPU needed) into one
grid figure per training condition (full split-size distribution, odd/grid), one panel per
failing split a=12..23. Same colour convention as plot_n60_geometry_heatmaps_v2.py: every
Z<=0 cell (predicted disconnected) is flat dark blue regardless of magnitude, every Z>0 cell
(predicted connected) is a white-to-red gradient scaled by its own magnitude -- so a
wrongly-disconnected pair inside a path, or a wrongly-connected pair across the two paths,
shows up as a colour anomaly against the expected block-diagonal pattern (all-red within each
path, all-blue across them) instead of blending smoothly across zero.

    python plot_beyond_threshold_heatmaps.py \\
        --in_dir runs/report10/beyond_threshold_heatmaps/<tag> --title "..." --out out.png
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

DARK_BLUE = "#08306b"
CMAP = LinearSegmentedColormap.from_list(
    "disconnected_flat_vs_connected_gradient",
    [(0.0, DARK_BLUE), (0.499999, DARK_BLUE), (0.5, "#ffffff"), (1.0, "#67000d")])

SPLITS = list(range(12, 24))  # a = 12..23


def plot_grid(in_dir, title, out_png):
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    ims = []
    for ax, a in zip(axes.flat, SPLITS):
        f = Path(in_dir) / f"a{a}_Z.npz"
        d = np.load(f)
        Z, boundary, n = d["Z"], float(d["boundary"]), int(d["n"])
        vmax = float(np.abs(Z).max())
        im = ax.imshow(Z, cmap=CMAP, norm=Normalize(vmin=-vmax, vmax=vmax))
        ims.append(im)
        ax.axvline(boundary, color="k", ls="--", lw=1)
        ax.axhline(boundary, color="k", ls="--", lw=1)
        ax.set_xticks([0, a - 1, n - 1])
        ax.set_xticklabels([1, a, n])
        ax.set_yticks([0, a - 1, n - 1])
        ax.set_yticklabels([1, a, n])
        ax.tick_params(labelsize=8)
        ax.set_title(f"$a={a}$: split $({a},{n - a})$", fontsize=10)
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.text(0.5, 0.005,
              r"$Z_{ij}=\mathrm{scale}\cdot\cos(h_i,h_j)+\mathrm{bias}$  "
              "(flat dark blue = disconnected, any Z≤0; red gradient = connected, "
              "scaled by Z; dashed line = true component boundary)",
              ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    plot_grid(args.in_dir, args.title, args.out)


if __name__ == "__main__":
    main()
