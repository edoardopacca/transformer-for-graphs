"""paper_submission_prigm.tex Appendix figure 4 (fig:beyond-diameter-heatmap): per-split
H^(2) similarity geometry, full-distribution two-chain model, seed 1000. Paper version of
plot_beyond_threshold_heatmaps.py's grid: same source data, same Z_CMAP color convention
(paper_zcolormap.py, shared across every layer-geometry figure in the paper), but no
baked-in descriptive suptitle/caption text (2026-08-27, Edoardo: never put text above a
figure that just describes what it shows -- the LaTeX caption already does that; a short
"a=12" per-panel identifier is fine, same as Figure 1/3's panel labels, since it's just
telling panels apart, not describing the plot).

2026-09-02 (Edoardo, relaying the professor's request): originally every failing split
a=12..23 (12 panels); now odd a from 3 to 23 plus a=12 (the exact breakpoint) = 12 panels,
so the figure also shows part of the SUCCEEDING half of the sweep (a<12) instead of only
the failing one, without ballooning to all 23 splits or looking like a cherry-picked subset
(a systematic "every other split, plus the breakpoint" rule).

Source: runs/report10/beyond_threshold_heatmaps/n46_splitchains_full_seed1000/a{N}_Z.npz,
N in SPLITS below.

    python plot_paper_beyond_threshold_heatmap.py
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from paper_zcolormap import Z_CMAP

SPLITS = [3, 5, 7, 9, 11, 12, 13, 15, 17, 19, 21, 23]
IN_DIR = Path("runs/report10/beyond_threshold_heatmaps/n46_splitchains_full_seed1000")


def main():
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.6))
    im = None
    for ax, a in zip(axes.flat, SPLITS):
        d = np.load(IN_DIR / f"a{a}_Z.npz")
        Z, boundary, n = d["Z"], float(d["boundary"]), int(d["n"])
        vmax = float(np.abs(Z).max())
        im = ax.imshow(Z, cmap=Z_CMAP, norm=Normalize(vmin=-vmax, vmax=vmax))
        ax.axvline(boundary, color="white", ls="--", lw=1)
        ax.axhline(boundary, color="white", ls="--", lw=1)
        ax.set_xticks([0, a - 1, n - 1]); ax.set_xticklabels([1, a, n], fontsize=7.5)
        ax.set_yticks([0, a - 1, n - 1]); ax.set_yticklabels([1, a, n], fontsize=7.5)
        ax.set_title(f"$a={a}$", fontsize=9.5, pad=3)

    fig.tight_layout(rect=(0, 0, 0.93, 1), h_pad=1.2, w_pad=0.8)
    cax = fig.add_axes((0.945, 0.15, 0.014, 0.7))
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$Z_{ij}$ (own scale per panel)", fontsize=9)
    cbar.ax.tick_params(labelsize=7.5)

    out = Path("paper_figures/fig_beyond_threshold_heatmap_full_seed1000.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
