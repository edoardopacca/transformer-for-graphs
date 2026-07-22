"""Report VIII -- render the causal edge-ablation contribution (local, no GPU).

Reads runs/<report_root>/edge_contrib/<tag>/edge_contrib.npz (from
edge_ablation_contribution.py). Two outputs:
  * heatmaps of C_edge / C_logit / C_far at two representative splits (one
    representative seed -- like the attention-score heatmaps, these are
    patterns, not scalars, so pooling across seeds with different learned
    bases would blur genuine structure);
  * the edge/logit/far leak-fraction vs split, pooled over all seeds matching
    --tag_glob, WITH ERROR BARS (std across seeds -- istruzioni.md errore 61).

    python plot_edge_ablation.py --tag_glob "n40_pathunion_seed*" --n 40 \\
        --report_root report8 --suffix _n40_pathunion --title_tag "disjoint-paths-trained"
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EDGE_ROOT = Path("runs/report8/edge_contrib")
OUT = Path("runs/report8/report8_figs")
FIG_PREFIX = "r8"


def heat(ax, mat, title, cmap="viridis", vlim=None):
    vmax = vlim if vlim is not None else (mat.max() if mat.size else 1.0)
    im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def fig_edge_heatmaps(tag_glob, pair, seed, suffix, title_tag):
    seed_dirs = sorted(EDGE_ROOT.glob(tag_glob))
    seed_dir = next((d for d in seed_dirs if f"seed{seed}" in d.name), None)
    if seed_dir is None:
        print(f"no edge_contrib.npz found for seed {seed} under {EDGE_ROOT}/{tag_glob}"); return
    d = np.load(seed_dir / "edge_contrib.npz")
    fig, axes = plt.subplots(3, len(pair), figsize=(6 * len(pair), 13))
    if len(pair) == 1:
        axes = axes[:, None]
    for col, a in enumerate(pair):
        key = f"a{a}"
        if f"{key}__C_edge" not in d.files:
            print(f"  split a={a} not in {seed_dir.name}, skipping column"); continue
        S = d[f"{key}__short_idx"]
        heat(axes[0, col], d[f"{key}__C_edge"],
             rf"$C^{{\mathrm{{edge}}}}$ (final-embedding change), $a$={a}")
        heat(axes[1, col], d[f"{key}__C_logit"],
             rf"$C^{{\mathrm{{logit}}}}$ (mean $|\Delta z_{{ij}}|$), $a$={a}")
        heat(axes[2, col], d[f"{key}__C_far"],
             rf"$C^{{\mathrm{{far}}}}$ (within-long, dist$>9$ only), $a$={a}")
        for row in range(3):
            axes[row, col].axvline(len(S) - 0.5, color="red", ls=":", lw=1)
            axes[row, col].axhline(len(S) - 0.5, color="red", ls=":", lw=1)
    fig.suptitle(f"Causal edge-ablation contribution -- {seed_dir.name}, {title_tag}\n"
                 f"(row $i$ = query node, column $k$ = node whose incident edges were "
                 f"removed; dotted line = component boundary)")
    fig.tight_layout()
    p = OUT / f"{FIG_PREFIX}_edge_contrib_heatmaps{suffix}.png"
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def fig_edge_leak(tag_glob, suffix, title_tag):
    per_a = {"edge_leak": {}, "logit_leak": {}, "far_leak": {}}
    for f in sorted(EDGE_ROOT.glob(f"{tag_glob}/edge_contrib.npz")):
        d = np.load(f)
        keys = sorted({k.split("__")[0] for k in d.files})
        for key in keys:
            a = int(key[1:])
            for metric in per_a:
                fk = f"{key}__{metric}"
                if fk in d.files:
                    v = float(d[fk][0])
                    if np.isfinite(v):
                        per_a[metric].setdefault(a, []).append(v)
    splits = sorted(set().union(*[set(v.keys()) for v in per_a.values()]))
    if not splits:
        print(f"no leak-fraction data found under {EDGE_ROOT}/{tag_glob}"); return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = {"edge_leak": "#1b9e77", "logit_leak": "#d95f02", "far_leak": "#7570b3"}
    labels = {"edge_leak": r"edge leak ($C^{\mathrm{edge}}$)",
              "logit_leak": r"logit leak ($C^{\mathrm{logit}}$)",
              "far_leak": r"far leak ($C^{\mathrm{far}}$, within-long dist$>9$ only)"}
    for metric in per_a:
        xs = [a for a in splits if per_a[metric].get(a)]
        ys = [np.mean(per_a[metric][a]) for a in xs]
        es = [np.std(per_a[metric][a]) for a in xs]
        if xs:
            ax.errorbar(xs, ys, yerr=es, fmt="-o", ms=4, capsize=3, elinewidth=1,
                        color=colors[metric], label=labels[metric])
    ax.set_xlabel("split: short-component size $a$")
    ax.set_ylabel("fraction of the long component's causal mass\n"
                  "attributable to edges in the short component")
    ax.set_title(f"Edge-ablation leak fraction -- {title_tag}\n(error bars: std across seeds)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = OUT / f"{FIG_PREFIX}_edge_leak{suffix}.png"
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def main():
    global EDGE_ROOT, OUT, FIG_PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag_glob", default="n40_pathunion_seed*",
                    help="glob (relative to runs/<report_root>/edge_contrib/) selecting seed dirs")
    ap.add_argument("--seed", type=int, default=1000, help="representative seed for the heatmaps")
    ap.add_argument("--pair", type=int, nargs="+", default=[4, 20], help="splits to show as heatmaps")
    ap.add_argument("--suffix", default="", help="appended to output figure filenames")
    ap.add_argument("--title_tag", default="disjoint-paths-trained",
                    help="short description of the training distribution, used in figure titles")
    ap.add_argument("--report_root", default="report8")
    args = ap.parse_args()
    EDGE_ROOT = Path(f"runs/{args.report_root}/edge_contrib")
    OUT = Path(f"runs/{args.report_root}/{args.report_root}_figs")
    FIG_PREFIX = args.report_root.replace("report", "r")
    OUT.mkdir(parents=True, exist_ok=True)

    fig_edge_heatmaps(args.tag_glob, args.pair, args.seed, args.suffix, args.title_tag)
    fig_edge_leak(args.tag_glob, args.suffix, args.title_tag)


if __name__ == "__main__":
    main()
