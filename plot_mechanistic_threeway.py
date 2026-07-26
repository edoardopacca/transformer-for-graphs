"""Report IX, Thread A.3 -- render the mechanistic battery (attention scores + layerwise cosine
geometry) for specific three-way split-size combinations (local, no GPU). Reads
runs/<report_root>/{heatmaps_threeway,stagewise_threeway}/<tag>/... produced by
mechanistic_threeway_heatmaps.py / stagewise_threeway.py. One pair of figures PER cell (unlike
the K-way plotting script, there is no single canonical "pair" to compare -- every cell here was
individually chosen because it is interesting on its own).

    python plot_mechanistic_threeway.py --tag n46_splitchains_seed1000 --report_root report9
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = ["H0", "Hattn1", "H1", "Hattn2", "H2"]
STAGE_LABELS = [r"$H^{(0)}$", r"$H_{\mathrm{attn}}^{(1)}$", r"$H^{(1)}$",
                r"$H_{\mathrm{attn}}^{(2)}$", r"$H^{(2)}$"]


def cell_boundaries(d, cell_key):
    c1 = d[f"{cell_key}__comp1_idx"]; c2 = d[f"{cell_key}__comp2_idx"]
    return len(c1) - 0.5, len(c1) + len(c2) - 0.5


def fig_attention_scores(heat_d, cell_key, sizes, out, suffix):
    b1, b2 = cell_boundaries(heat_d, cell_key)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for row, li in enumerate((0, 1)):
        S = heat_d[f"{cell_key}__scores{li}"]
        A = heat_d[f"{cell_key}__alpha{li}"]
        for col, (mat, title, cmap, diverging) in enumerate([
                (S, f"scores layer {li}", "RdBu_r", True),
                (A, rf"$\alpha$ layer {li}", "viridis", False)]):
            ax = axes[row, col]
            if diverging:
                v = np.abs(mat).max()
                im = ax.imshow(mat, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
            else:
                im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=mat.max(), aspect="auto")
            for b in (b1, b2):
                ax.axvline(b, color="white" if diverging else "red", ls="--", lw=1)
                ax.axhline(b, color="white" if diverging else "red", ls="--", lw=1)
            ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Attention scores $S=QK^\\top/\\sqrt{{d_h}}$ and normalised-ReLU $\\alpha$ -- "
                 f"sizes $(s_1,s_2,s_3)=({sizes[0]},{sizes[1]},{sizes[2]})$\n"
                 f"(dashed lines: component boundaries, node order = comp1 | comp2 | comp3)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = out / f"r9_threeway_attention_s{sizes[0]}_{sizes[1]}_{sizes[2]}{suffix}.png"
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def fig_cosine_geometry(stage_d, cell_key, sizes, out, suffix, scale, bias):
    b1, b2 = cell_boundaries(stage_d, cell_key)
    mats = [scale * stage_d[f"{cell_key}__G_{X}"] + bias for X in STAGES]
    vmax = max(abs(m).max() for m in mats)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
    for ax, m, lab in zip(axes, mats, STAGE_LABELS):
        im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for b in (b1, b2):
            ax.axvline(b, color="k", ls="--", lw=1)
            ax.axhline(b, color="k", ls="--", lw=1)
        ax.set_title(lab, fontsize=11)
    fig.suptitle(f"Layerwise similarity geometry $Z=\\mathrm{{scale}}\\cdot\\cos+\\mathrm{{bias}}$ -- "
                 f"sizes $(s_1,s_2,s_3)=({sizes[0]},{sizes[1]},{sizes[2]})$")
    fig.tight_layout(rect=[0, 0, 0.93, 0.88])
    cbar_ax = fig.add_axes([0.945, 0.12, 0.012, 0.72])
    fig.colorbar(im, cax=cbar_ax)
    p = out / f"r9_threeway_stagewise_cosine_s{sizes[0]}_{sizes[1]}_{sizes[2]}{suffix}.png"
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="n46_splitchains_seed1000")
    ap.add_argument("--report_root", default="report9")
    ap.add_argument("--suffix", default=None)
    args = ap.parse_args()
    suffix = f"_{args.suffix}" if args.suffix else f"_{args.tag}"

    heat_path = Path(f"runs/{args.report_root}/heatmaps_threeway/{args.tag}/heatmap_data.npz")
    stage_path = Path(f"runs/{args.report_root}/stagewise_threeway/{args.tag}/stagewise_geometry.npz")
    out = Path(f"runs/{args.report_root}/{args.report_root}_figs")
    out.mkdir(parents=True, exist_ok=True)

    heat_d = np.load(heat_path) if heat_path.exists() else None
    stage_d = np.load(stage_path) if stage_path.exists() else None
    if heat_d is None and stage_d is None:
        print(f"no data found at {heat_path} or {stage_path}"); return

    cell_keys = set()
    if heat_d is not None:
        cell_keys |= {k.split("__")[0] for k in heat_d.files if "__scores0" in k}
    if stage_d is not None:
        cell_keys |= {k.split("__")[0] for k in stage_d.files if "__G_H0" in k}

    scale = float(stage_d["scale"]) if stage_d is not None else None
    bias = float(stage_d["bias"]) if stage_d is not None else None

    for cell_key in sorted(cell_keys):
        sizes = [int(x) for x in cell_key[1:].split("_")]
        if heat_d is not None and f"{cell_key}__scores0" in heat_d.files:
            fig_attention_scores(heat_d, cell_key, sizes, out, suffix)
        if stage_d is not None and f"{cell_key}__G_H0" in stage_d.files:
            fig_cosine_geometry(stage_d, cell_key, sizes, out, suffix, scale, bias)


if __name__ == "__main__":
    main()
