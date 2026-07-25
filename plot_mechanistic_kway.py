"""Report IX, Thread A.4 (mechanistic extension) -- render the K-way (K=5,6,7) mechanistic
battery (local, no GPU). Reads runs/<report_root>/{mechanistic_kway,heatmaps_kway,
stagewise_kway}/<tag>/... produced by mechanistic_kway.py / mechanistic_kway_heatmaps.py /
stagewise_kway.py, pooled over every seed matching --tag_glob.

    python plot_mechanistic_kway.py --tag_glob "n40_pathunion_seed*_kway" \\
        --report_root report9 --heatmap_seed 1000 --title_tag "path_union-trained, n=40"
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = ["H0", "Hattn1", "H1", "Hattn2", "H2"]
STAGE_LABELS = [r"$H^{(0)}$", r"$H_{\mathrm{attn}}^{(1)}$", r"$H^{(1)}$",
                r"$H_{\mathrm{attn}}^{(2)}$", r"$H^{(2)}$"]


def load_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def print_table(mech_root, tag_glob):
    per_cell = defaultdict(lambda: defaultdict(list))
    for f in sorted(mech_root.glob(f"{tag_glob}/metrics.csv")):
        for r in load_csv(f):
            if r["distance"] not in ("", None) or r["metric"] not in (
                    "exact", "reach_long", "reach_short", "cut", "pred_positive_rate"):
                continue
            per_cell[(int(r["k"]), int(r["small_size"]))][r["metric"]].append(float(r["value"]))
    cells = sorted(per_cell.keys())
    print(f"\n=== Table 1 equivalent: mean (std) over seeds, {tag_glob} ===")
    print(f"{'K':>3} {'small':>6} {'exact':>14} {'reach_long':>14} {'reach_short':>14} {'cut':>14}")
    for k, s in cells:
        row = per_cell[(k, s)]
        vals = []
        for m in ("exact", "reach_long", "reach_short", "cut"):
            v = row.get(m, [])
            vals.append(f"{np.mean(v):.3f} ({np.std(v):.3f})" if v else "--")
        print(f"{k:>3} {s:>6} " + " ".join(f"{v:>14}" for v in vals))
    return cells


def fig_sweep_and_logit_kway(mech_root, tag_glob, suffix, title_tag):
    metrics_rows, readout_rows = [], []
    for f in sorted(mech_root.glob(f"{tag_glob}/metrics.csv")):
        metrics_rows += load_csv(f)
    for f in sorted(mech_root.glob(f"{tag_glob}/readout.csv")):
        readout_rows += load_csv(f)
    ks = sorted({int(r["k"]) for r in metrics_rows})
    behav = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # k -> s -> metric -> [vals]
    keep = ("exact", "reach_long", "reach_short", "cut", "pred_positive_rate")
    for r in metrics_rows:
        if r["distance"] not in ("", None) or r["metric"] not in keep:
            continue
        behav[int(r["k"])][int(r["small_size"])][r["metric"]].append(float(r["value"]))
    read = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in readout_rows:
        if r["pair_type"] in ("within_long", "within_short", "cut"):
            read[int(r["k"])][int(r["small_size"])][r["pair_type"]].append(float(r["mean_cos"]))

    fig, axes = plt.subplots(2, len(ks), figsize=(5.5 * len(ks), 8), sharex=False)
    if len(ks) == 1:
        axes = axes.reshape(2, 1)
    for col, k in enumerate(ks):
        ss = sorted(behav[k].keys())
        ax1 = axes[0, col]
        for key, lab, color in [("exact", "exact match", "#222222"),
                                ("reach_long", "reach (long)", "#1b9e77"),
                                ("reach_short", "reach (short)", "#66a61e"),
                                ("cut", "cut (target: disconnected)", "#e7298a")]:
            ys = [np.mean(behav[k][s][key]) if behav[k][s][key] else np.nan for s in ss]
            es = [np.std(behav[k][s][key]) if behav[k][s][key] else np.nan for s in ss]
            ax1.errorbar(ss, ys, yerr=es, fmt="-o", ms=4, capsize=3, elinewidth=1, color=color, label=lab)
        pos = [np.mean(behav[k][s]["pred_positive_rate"]) for s in ss]
        pos_err = [np.std(behav[k][s]["pred_positive_rate"]) for s in ss]
        ax1.errorbar(ss, pos, yerr=pos_err, fmt="--", capsize=2, elinewidth=1,
                     color="#7570b3", label="predicted-positive rate")
        ax1.set_title(f"$K={k}$ ({k-1} short components)"); ax1.set_ylim(-0.02, 1.02)
        ax1.grid(alpha=0.3)
        if col == 0:
            ax1.set_ylabel("accuracy / rate"); ax1.legend(fontsize=7, loc="center left")

        ax2 = axes[1, col]
        for key, lab, color in [("within_long", "within long: mean cos", "#1b9e77"),
                                ("within_short", "within short: mean cos", "#66a61e"),
                                ("cut", "cut: mean cos", "#e7298a")]:
            ys = [np.mean(read[k][s][key]) if read[k][s][key] else np.nan for s in ss]
            es = [np.std(read[k][s][key]) if read[k][s][key] else np.nan for s in ss]
            ax2.errorbar(ss, ys, yerr=es, fmt="-o", ms=4, capsize=3, elinewidth=1, color=color, label=lab)
        ax2.axhline(0, color="k", lw=1)
        ax2.set_xlabel("short-component size $s$"); ax2.grid(alpha=0.3)
        if col == 0:
            ax2.set_ylabel("mean $\\cos(h_i,h_j)$ (pre-bias, pre-scale)")
            ax2.legend(fontsize=7, loc="center left")

    n_seeds = max((len(v) for k in behav for s in behav[k] for v in behav[k][s].values()), default=1)
    fig.suptitle(f"K-way (5/6/7 components) sweep + raw similarity logit -- {title_tag}\n"
                 f"(error bars: std across the {n_seeds} seeds; x-axis = short-component size, "
                 f"long component = $n-(K{{-}}1)\\cdot s$)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = Path(f"runs/report9/report9_figs/r9_kway_sweep_and_logit{suffix}.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def fig_attention_scores_kway(heat_root, tag_prefix, seed, cells, suffix):
    seed_dir = heat_root / f"{tag_prefix}_seed{seed}"
    if not seed_dir.exists():
        print(f"no heatmap data at {seed_dir}"); return
    d = np.load(seed_dir / "heatmap_data.npz")
    fig, axes = plt.subplots(2, len(cells), figsize=(5.5 * len(cells), 8.5))
    if len(cells) == 1:
        axes = axes.reshape(2, 1)
    for col, (k, s) in enumerate(cells):
        key = f"k{k}_s{s}"
        if f"{key}__alpha0" not in d.files:
            print(f"  cell {key} not found in {seed_dir.name}"); continue
        S = d[f"{key}__short_idx"]
        for row, li in enumerate((0, 1)):
            alpha = d[f"{key}__alpha{li}"]
            ax = axes[row, col]
            im = ax.imshow(alpha, cmap="viridis", aspect="auto")
            ax.axvline(len(S) - 0.5, color="w", ls="--", lw=1)
            ax.axhline(len(S) - 0.5, color="w", ls="--", lw=1)
            ax.set_title(f"$K={k}$, $s={s}$, layer {li} $\\alpha$", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Real normalised-ReLU attention $\\alpha$ -- {seed_dir.name}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = Path(f"runs/report9/report9_figs/r9_kway_heatmap_attention{suffix}.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def fig_cosine_geometry_kway(stage_root, tag_prefix, seed, cell, suffix):
    seed_dir = stage_root / f"{tag_prefix}_seed{seed}"
    if not seed_dir.exists():
        print(f"no stagewise data at {seed_dir}"); return
    d = np.load(seed_dir / "stagewise_geometry.npz")
    k, s = cell
    key = f"k{k}_s{s}"
    if f"{key}__G_H0" not in d.files:
        print(f"  cell {key} not found in {seed_dir.name}"); return
    S = d[f"{key}__short_idx"]
    raw = [d[f"{key}__G_{X}"] for X in STAGES]
    if "scale" in d.files and "bias" in d.files:
        scale, bias = float(d["scale"]), float(d["bias"])
        mats = [scale * m + bias for m in raw]
        title_quantity = (r"$Z^X_{ij}=\mathrm{scale}\cdot\cos(x_i,x_j)+\mathrm{bias}$ "
                          r"(connected $>0$, disconnected $<0$)")
    else:
        mats = raw
        title_quantity = r"$G^X_{ij}=\cos(x_i,x_j)$"
    vmax = max(abs(m).max() for m in mats)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
    for ax, m, lab in zip(axes, mats, STAGE_LABELS):
        im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.axvline(len(S) - 0.5, color="k", ls="--", lw=1)
        ax.axhline(len(S) - 0.5, color="k", ls="--", lw=1)
        ax.set_title(lab, fontsize=11)
    fig.suptitle(f"Layerwise similarity geometry {title_quantity} -- {seed_dir.name}, $K={k},s={s}$")
    fig.tight_layout(rect=[0, 0, 0.93, 0.90])
    cbar_ax = fig.add_axes([0.945, 0.12, 0.012, 0.72])
    fig.colorbar(im, cax=cbar_ax)
    p = Path(f"runs/report9/report9_figs/r9_kway_stagewise_cosine_k{k}s{s}{suffix}.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag_glob", default="n40_pathunion_seed*_kway")
    ap.add_argument("--tag_prefix", default="n40_pathunion",
                    help="prefix for heatmap/stagewise dirs, e.g. n40_pathunion_kway")
    ap.add_argument("--report_root", default="report9")
    ap.add_argument("--heatmap_seed", type=int, default=1000)
    ap.add_argument("--attn_cells", type=str, nargs="+", default=["5,5", "7,2"],
                    help="K,small_size pairs to show in the attention-scores figure")
    ap.add_argument("--cosine_cell", type=str, default="7,2",
                    help="K,small_size pair to show in the layerwise cosine geometry figure")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--title_tag", default="disjoint-paths-trained")
    args = ap.parse_args()

    mech_root = Path(f"runs/{args.report_root}/mechanistic_kway")
    heat_root = Path(f"runs/{args.report_root}/heatmaps_kway")
    stage_root = Path(f"runs/{args.report_root}/stagewise_kway")

    print_table(mech_root, args.tag_glob)
    fig_sweep_and_logit_kway(mech_root, args.tag_glob, args.suffix, args.title_tag)
    attn_cells = [tuple(int(x) for x in c.split(",")) for c in args.attn_cells]
    fig_attention_scores_kway(heat_root, args.tag_prefix, args.heatmap_seed, attn_cells, args.suffix)
    cosine_cell = tuple(int(x) for x in args.cosine_cell.split(","))
    fig_cosine_geometry_kway(stage_root, args.tag_prefix, args.heatmap_seed, cosine_cell, args.suffix)


if __name__ == "__main__":
    main()
