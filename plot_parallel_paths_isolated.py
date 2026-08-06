"""Local, no-GPU. Renders the clean parallel-paths-isolated battery produced by
mechanistic_parallel_paths_isolated.py (one run per checkpoint/seed):
  * a behavioural figure: term_connect / reach_route / isolation_acc /
    pred_positive_rate vs k, one line per seed (mean+std if multiple seeds match
    --tag_glob).
  * an attention-heatmap figure per seed (real scores/alpha, layer 0/1, one row
    per k), reusing plot_mechanistic_multipath.py's exact layout/crop convention.

  python plot_parallel_paths_isolated.py --tag_glob "n46_splitchains_seed*" \\
      --report_root report9 --title_tag "split_chains-only, n=46"
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fig_behaviour(root, tag_glob, suffix, title_tag):
    by_seed = {}
    for f in sorted(root.glob(f"{tag_glob}/parallel_paths_isolated.json")):
        d = json.loads(f.read_text())
        by_seed[f.parent.name] = {c["k"]: c for c in d["cells"]}
    if not by_seed:
        print(f"no data matching {tag_glob}"); return
    ks = sorted({k for cells in by_seed.values() for k in cells})
    metrics = [("term_connect", "term_connect: $(s,t)$ predicted connected", "#222222"),
               ("reach_route", "reach_route: within-route pairs predicted connected", "#1b9e77"),
               ("isolation_acc", "isolation_acc: pairs touching an isolated node predicted "
                                 "DISconnected", "#e7298a"),
               ("pred_positive_rate", "predicted-positive rate (all pairs)", "#7570b3")]
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, lab, color in metrics:
        ys = [np.mean([by_seed[s][k][key] for s in by_seed if k in by_seed[s]]) for k in ks]
        es = [np.std([by_seed[s][k][key] for s in by_seed if k in by_seed[s]]) for k in ks]
        ax.errorbar(ks, ys, yerr=es, fmt="-o", ms=5, capsize=3, color=color, label=lab)
    ax.set_xlabel("number of parallel routes $k$ (path length fixed)")
    ax.set_ylabel("accuracy / rate"); ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(ks)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="center left")
    n_seeds = len(by_seed)
    ax.set_title(f"Parallel paths, isolated filler (no leaf padding) -- {title_tag}\n"
                 f"(mean$\\pm$std over {n_seeds} seed{'s' if n_seeds != 1 else ''})")
    fig.tight_layout()
    p = Path(f"runs/report9/report9_figs/r9_parallel_paths_isolated_sweep{suffix}.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def _draw_bounds(ax, bounds):
    for b in bounds[1:-1]:
        ax.axhline(b - 0.5, color="white", lw=0.6, alpha=0.7)
        ax.axvline(b - 0.5, color="white", lw=0.6, alpha=0.7)


def fig_attention(root, tag, suffix, title_tag):
    npz_path = root / tag / "parallel_paths_isolated_attn.npz"
    if not npz_path.exists():
        print(f"no attention data at {npz_path}"); return
    d = np.load(npz_path)
    ks = sorted(int(k) for k in d["ks_present"])
    fig, axes = plt.subplots(len(ks), 4, figsize=(4 * 3.2, len(ks) * 3.0), squeeze=False)
    cols = ["scores0", "alpha0", "scores1", "alpha1"]
    col_titles = [r"scores layer 0 ($S=QK^\top/\sqrt{d_h}$)", r"$\alpha$ layer 0",
                  r"scores layer 1", r"$\alpha$ layer 1"]
    for ri, k in enumerate(ks):
        bounds = d[f"k{k}__route_bounds"]
        na = int(bounds[-1])
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
                ax.set_ylabel(f"k={k}\nnode index", fontsize=9)
    suptitle = ("Real attention, parallel-paths-isolated graphs (no leaf padding)"
               + (f" -- {title_tag}" if title_tag else ""))
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = Path(f"runs/report9/report9_figs/r9_parallel_paths_isolated_attn{suffix}.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag_glob", default="n46_splitchains_seed*",
                    help="glob (under runs/<report_root>/parallel_paths_isolated/) pooled "
                         "for the behavioural figure")
    ap.add_argument("--attn_tag", default=None,
                    help="single tag (exact dir name) to render the attention figure for; "
                         "defaults to the first match of --tag_glob")
    ap.add_argument("--report_root", default="report9")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--title_tag", default="split_chains-only, n=46")
    args = ap.parse_args()

    root = Path(f"runs/{args.report_root}/parallel_paths_isolated")
    fig_behaviour(root, args.tag_glob, args.suffix, args.title_tag)

    attn_tag = args.attn_tag
    if attn_tag is None:
        matches = sorted(root.glob(args.tag_glob))
        attn_tag = matches[0].name if matches else None
    if attn_tag:
        fig_attention(root, attn_tag, args.suffix, f"{args.title_tag}, {attn_tag}")


if __name__ == "__main__":
    main()
