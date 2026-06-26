"""Report VI, Thread A1 -- aggregate and plot the multipath probe (local, no GPU).

Reads runs/report6/multipath/<tag>/multipath.json (tag = n{N}_{dist}_seed{S}, dist in
{er, pathunion, mixed}), pools across seeds, and produces, per (n, dist):
  * the RESCUE figure: pair-(s,t) accuracy vs the number of routes k, one line per route
    length ell (mean over seeds), with the matrix-exact context and the capacity ell=9
    marked. If more routes lift a beyond-capacity pair, the lines climb with k.
  * the MECHANISM figure: for each beyond-capacity (k, ell) cell, the distribution of how
    many routes are "intact" (0..k), so one can see whether a correct (s,t) comes from a
    single resolved route or the aggregate of several -- the question the report cares about.
Also prints a compact per-cell table (pair acc, matrix exact, mean intact routes).

    python plot_multipath.py            # all tags under runs/report6/multipath
    python plot_multipath.py --dists er pathunion
"""
import argparse, glob, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG_RE = re.compile(r"n(\d+)_([a-z]+)_seed(\d+)$")
CAP = 9  # 3^L at L=2


def load(root, dists):
    """-> data[(n, dist)][(k, ell)] = list of per-seed cell dicts."""
    data = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(Path(root) / "*" / "multipath.json"))):
        tag = Path(f).parent.name
        m = TAG_RE.match(tag)
        if not m:
            continue
        n, dist, _ = int(m.group(1)), m.group(2), int(m.group(3))
        if dists and dist not in dists:
            continue
        d = json.load(open(f))
        for c in d["cells"]:
            data[(n, dist)][(c["k"], c["ell"])].append(c)
    return data


def agg(cells, key):
    return float(np.mean([c[key] for c in cells]))


def plot_rescue(nd, cellmap, outdir):
    n, dist = nd
    ells = sorted({e for (k, e) in cellmap})
    ks = sorted({k for (k, e) in cellmap})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("viridis")
    for ax, (metric, title) in zip(axes, [("pair_acc", f"(s,t)-pair accuracy  [n={n}, {dist}]"),
                                          ("matrix_exact", f"whole-matrix exact  [n={n}, {dist}]")]):
        for j, e in enumerate(ells):
            xs, ys = [], []
            for k in ks:
                cs = cellmap.get((k, e))
                if cs:
                    xs.append(k); ys.append(agg(cs, metric))
            if xs:
                style = "--" if e > CAP else "-"
                ax.plot(xs, ys, style, marker="o", color=cmap(j / max(1, len(ells) - 1)),
                        label=f"ell={e}" + (" (>cap)" if e > CAP else ""))
        ax.set_xlabel("number of routes k"); ax.set_ylabel(metric); ax.set_title(title)
        ax.set_xticks(ks); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = Path(outdir) / f"multipath_rescue_n{n}_{dist}.png"
    fig.savefig(p, dpi=150); plt.close(fig); print("  saved", p)


def plot_mechanism(nd, cellmap, outdir):
    n, dist = nd
    beyond = sorted({(k, e) for (k, e) in cellmap if e > CAP and k >= 2})
    if not beyond:
        return
    ells = sorted({e for (k, e) in beyond})
    fig, axes = plt.subplots(1, len(ells), figsize=(4.2 * len(ells), 4), squeeze=False)
    for ax, e in zip(axes[0], ells):
        ks = sorted({k for (k, ee) in beyond if ee == e})
        bottoms = np.zeros(len(ks))
        maxk = max(ks)
        for j in range(maxk + 1):                       # number of intact routes = j
            frac = []
            for k in ks:
                cs = cellmap[(k, e)]
                tot = sum(sum(c["n_intact_hist"]) for c in cs)
                cnt = sum(c["n_intact_hist"][j] if j < len(c["n_intact_hist"]) else 0 for c in cs)
                frac.append(cnt / max(1, tot))
            frac = np.array(frac)
            ax.bar(range(len(ks)), frac, bottom=bottoms, label=f"{j} intact")
            bottoms += frac
        # overlay pair accuracy
        pa = [agg(cellmap[(k, e)], "pair_acc") for k in ks]
        ax.plot(range(len(ks)), pa, "k-o", lw=2, label="pair acc")
        ax.set_xticks(range(len(ks))); ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_title(f"ell={e} (>cap)  [n={n}, {dist}]"); ax.set_ylim(0, 1.02)
        ax.set_ylabel("fraction of graphs")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    p = Path(outdir) / f"multipath_mechanism_n{n}_{dist}.png"
    fig.savefig(p, dpi=150); plt.close(fig); print("  saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/report6/multipath")
    ap.add_argument("--dists", nargs="*", default=None,
                    help="subset of {er, pathunion, mixed}; default all present")
    ap.add_argument("--outdir", default="runs/report6/report6_figs")
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    data = load(args.root, args.dists)
    if not data:
        print(f"no multipath.json under {args.root} (pull the eval outputs first)"); return
    for nd in sorted(data):
        n, dist = nd
        cellmap = data[nd]
        nseed = max(len(v) for v in cellmap.values())
        print(f"\n=== n={n} {dist}  ({nseed} seeds) ===")
        print(f"{'k':>2} {'ell':>3} | {'pair':>5} {'mat_ex':>6} {'mat_pw':>6} "
              f"{'intact':>6}/{'k':<2}")
        for (k, e) in sorted(cellmap):
            cs = cellmap[(k, e)]
            print(f"{k:>2} {e:>3} | {agg(cs,'pair_acc'):>5.2f} {agg(cs,'matrix_exact'):>6.2f} "
                  f"{agg(cs,'matrix_pairwise'):>6.2f} {agg(cs,'mean_n_intact'):>6.2f}/{k:<2}")
        plot_rescue(nd, cellmap, args.outdir)
        plot_mechanism(nd, cellmap, args.outdir)


if __name__ == "__main__":
    main()
