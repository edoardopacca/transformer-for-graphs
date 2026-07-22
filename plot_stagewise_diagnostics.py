"""Report VIII -- render the stagewise "where is information combined"
diagnostics (local, no GPU). Reads runs/report8/stagewise/<tag>/stagewise_{
geometry.npz,metrics.csv,margins.csv,deltaz.csv} (from stagewise_diagnostics.py).

    python plot_stagewise_diagnostics.py --tag_glob "n40_pathunion_seed*" \\
        --heatmap_seed 1000 --title_tag "disjoint-paths-trained"
"""
import argparse, csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("runs/report8/stagewise")
OUT = Path("runs/report8/report8_figs")
STAGES = ["H0", "Hattn1", "H1", "Hattn2", "H2"]
STAGE_LABELS = [r"$H^{(0)}$", r"$H_{\mathrm{attn}}^{(1)}$", r"$H^{(1)}$",
                r"$H_{\mathrm{attn}}^{(2)}$", r"$H^{(2)}$"]
BRANCHES = ["dZ_attn1", "dZ_mlp1", "dZ_attn2", "dZ_mlp2"]
BRANCH_LABELS = ["attn 1", "MLP 1", "attn 2", "MLP 2"]


def load_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def fig_cosine_heatmaps(tag_glob, seed, split, suffix):
    seed_dirs = sorted(ROOT.glob(tag_glob))
    seed_dir = next((d for d in seed_dirs if f"seed{seed}" in d.name), None)
    if seed_dir is None:
        print(f"no stagewise data for seed {seed} under {ROOT}/{tag_glob}"); return
    d = np.load(seed_dir / "stagewise_geometry.npz")
    key = f"a{split}"
    if f"{key}__G_H0" not in d.files:
        print(f"split a={split} not found in {seed_dir.name}"); return
    S = d[f"{key}__short_idx"]
    mats = [d[f"{key}__G_{X}"] for X in STAGES]
    vmax = max(abs(m).max() for m in mats)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
    for ax, m, lab in zip(axes, mats, STAGE_LABELS):
        im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.axvline(len(S) - 0.5, color="k", ls="--", lw=1)
        ax.axhline(len(S) - 0.5, color="k", ls="--", lw=1)
        ax.set_title(lab, fontsize=11)
    fig.suptitle(f"Layerwise cosine geometry $G^X_{{ij}}=\\cos(x_i,x_j)$ -- "
                 f"{seed_dir.name}, split $a$={split}")
    fig.tight_layout(rect=[0, 0, 0.93, 0.90])
    cbar_ax = fig.add_axes([0.945, 0.12, 0.012, 0.72])
    fig.colorbar(im, cax=cbar_ax)
    p = OUT / f"r8_stagewise_cosine_a{split}{suffix}.png"
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def fig_stage_curve(tag_glob, splits, value_col, csv_name, quantity_key, quantity_val,
                     ylabel, title, out_name):
    per_split_stage = defaultdict(lambda: defaultdict(list))
    for f in sorted(ROOT.glob(f"{tag_glob}/{csv_name}")):
        rows = load_csv(f)
        for r in rows:
            if int(r["split_a"]) not in splits: continue
            if quantity_key is not None and r[quantity_key] != quantity_val: continue
            v = r[value_col]
            if v in ("", "nan"): continue
            v = float(v)
            if np.isfinite(v):
                per_split_stage[int(r["split_a"])][r["stage"]].append(v)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(splits)))
    for a, col in zip(splits, colors):
        ys = [np.mean(per_split_stage[a][s]) if per_split_stage[a][s] else np.nan for s in STAGES]
        es = [np.std(per_split_stage[a][s]) if per_split_stage[a][s] else np.nan for s in STAGES]
        ax.errorbar(range(len(STAGES)), ys, yerr=es, fmt="-o", ms=5, capsize=3,
                    color=col, label=f"$a$={a}")
    ax.set_xticks(range(len(STAGES))); ax.set_xticklabels(STAGE_LABELS)
    ax.set_ylabel(ylabel); ax.set_title(title + "\n(error bars: std across seeds)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = OUT / out_name
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def fig_deltaz_heatmaps(tag_glob, seed, split, suffix):
    seed_dirs = sorted(ROOT.glob(tag_glob))
    seed_dir = next((d for d in seed_dirs if f"seed{seed}" in d.name), None)
    if seed_dir is None:
        print(f"no stagewise data for seed {seed} under {ROOT}/{tag_glob}"); return
    d = np.load(seed_dir / "stagewise_geometry.npz")
    key = f"a{split}"
    if f"{key}__dZ_attn1" not in d.files:
        print(f"split a={split} not found in {seed_dir.name}"); return
    S = d[f"{key}__short_idx"]
    mats = [d[f"{key}__{b}"] for b in BRANCHES]
    vmax = max(abs(m).max() for m in mats)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
    for ax, m, lab in zip(axes, mats, BRANCH_LABELS):
        im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.axvline(len(S) - 0.5, color="k", ls="--", lw=1)
        ax.axhline(len(S) - 0.5, color="k", ls="--", lw=1)
        ax.set_title(lab, fontsize=11)
    fig.suptitle(f"Per-sub-block logit change $\\Delta Z$ -- {seed_dir.name}, split $a$={split}\n"
                 f"(positive = sub-block makes the pair MORE likely connected)")
    fig.tight_layout(rect=[0, 0, 0.93, 0.86])
    cbar_ax = fig.add_axes([0.945, 0.12, 0.014, 0.68])
    fig.colorbar(im, cax=cbar_ax)
    p = OUT / f"r8_stagewise_deltaz_a{split}{suffix}.png"
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def table_deltaz_categories(tag_glob, splits):
    per = defaultdict(lambda: defaultdict(list))  # (split,branch,category) -> values
    for f in sorted(ROOT.glob(f"{tag_glob}/stagewise_deltaz.csv")):
        for r in load_csv(f):
            a = int(r["split_a"])
            if a not in splits: continue
            v = r["value"]
            if v in ("", "nan"): continue
            per[a][(r["branch"], r["category"])].append(float(v))
    for a in splits:
        print(f"\n=== mean(dZ) by branch x category, split a={a} ===")
        cats = ["within_short", "within_long_near", "within_long_far", "cross"]
        print(f"{'branch':<12}" + "".join(f"{c:>18}" for c in cats))
        for b in BRANCHES:
            row = f"{b:<12}"
            for c in cats:
                vals = per[a].get((b, c), [])
                row += f"{np.mean(vals):>18.4f}" if vals else f"{'--':>18}"
            print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag_glob", default="n40_pathunion_seed*")
    ap.add_argument("--heatmap_seed", type=int, default=1000)
    ap.add_argument("--heatmap_splits", type=int, nargs="+", default=[4, 20])
    ap.add_argument("--curve_splits", type=int, nargs="+", default=[4, 7, 8, 10, 14, 20])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--title_tag", default="disjoint-paths-trained")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    for a in args.heatmap_splits:
        fig_cosine_heatmaps(args.tag_glob, args.heatmap_seed, a, args.suffix)
        fig_deltaz_heatmaps(args.tag_glob, args.heatmap_seed, a, args.suffix)

    fig_stage_curve(args.tag_glob, args.heatmap_splits, "value", "stagewise_metrics.csv",
                     "metric", "reach_long_far", "far reach (within-long, dist $>9$)",
                     f"Far reach vs. stage -- {args.title_tag}",
                     f"r8_stagewise_far_reach{args.suffix}.png")
    fig_stage_curve(args.tag_glob, args.heatmap_splits, "value", "stagewise_margins.csv",
                     "quantity", "M_far", "margin $M_{\\mathrm{far}}$ (mean cos, far-long $-$ cross)",
                     f"Far margin vs. stage -- {args.title_tag}",
                     f"r8_stagewise_margin{args.suffix}.png")

    print(f"\n=== stagewise reach/margin summary (mean over seeds), {args.tag_glob} ===")
    table_deltaz_categories(args.tag_glob, args.heatmap_splits)


if __name__ == "__main__":
    main()
