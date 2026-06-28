"""Report VI, Thread B -- aggregate and plot the asymmetric two-chains probe (local, no GPU).

Reads runs/report6/asym_chains/<tag>/asym_chains.json (tag = n{N}_{cond}_seed{S}, cond in
{er, pathunion, mixed}), pools across seeds, and produces, per n:
  * the SPLIT figure: whole-graph exact match vs the split a, one line per condition (mean
    over seeds). This is the headline answering the Report-IV puzzle (is 4+36 easier than
    17+23?) as the full curve over splits, not two points.
  * the BLOCK figure (clean condition): the long/short within-block and the cut block
    exact-fractions vs split -> WHERE the exact-match breaks as the split changes.
  * the PER-DISTANCE figure: within-long-component reach by shortest-path distance for a
    few representative splits -> the mechanism (a near-full-length path recovers end-to-end
    while two medium paths sit in the post-capacity valley).
Also prints a compact per-split table per (n, cond).

    python plot_asym_chains.py                       # all tags under runs/report6/asym_chains
    python plot_asym_chains.py --conds pathunion er
"""
import argparse, glob, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG_RE = re.compile(r"n(\d+)_([a-z]+)_seed(\d+)$")
CAP = 9  # 3^L at L=2
COND_COLOR = {"pathunion": "#2c7fb8", "er": "#d95f02", "mixed": "#7570b3"}
COND_LABEL = {"pathunion": "path_union (clean)", "er": "ER (unseen)", "mixed": "mixed (baseline)"}


def load(root, conds):
    """-> data[(n, cond)][split] = list of per-seed cell dicts."""
    data = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(Path(root) / "*" / "asym_chains.json"))):
        tag = Path(f).parent.name
        m = TAG_RE.match(tag)
        if not m:
            continue
        n, cond = int(m.group(1)), m.group(2)
        if conds and cond not in conds:
            continue
        d = json.load(open(f))
        for c in d["cells"]:
            data[(n, cond)][c["split"]].append(c)
    return data


def agg(cells, key):
    vals = [c[key] for c in cells if c.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def plot_split(n, conds_present, data, outdir):
    """Exact match vs split a, one line per condition."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cond in conds_present:
        cm = data.get((n, cond))
        if not cm:
            continue
        splits = sorted(cm)
        ex = [agg(cm[a], "exact") for a in splits]
        ax.plot(splits, ex, "-o", ms=4, color=COND_COLOR.get(cond, None),
                label=COND_LABEL.get(cond, cond))
    ax.set_xlabel("split: short-component size a  (components a and n-a)")
    ax.set_ylabel("whole-graph exact match")
    ax.set_title(f"Asymmetric two-chains: exact match vs split  (n={n}, base linear L=2)")
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
    # mark the Report-IV reference splits at n=40
    if n == 40:
        for a in (4, 17):
            ax.axvline(a, color="gray", ls=":", lw=1)
            ax.text(a, 1.0, f"a={a}", fontsize=7, color="gray", ha="center", va="bottom")
    allsplits = sorted({a for cond in conds_present for a in data.get((n, cond), {})})
    if allsplits:
        ax.set_xticks(allsplits)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    p = Path(outdir) / f"asym_chains_exact_n{n}.png"
    fig.savefig(p, dpi=150); plt.close(fig); print("  saved", p)


def plot_blocks(n, cond, cm, outdir):
    """Long/short within-block and cut block exact-fractions vs split, for one condition."""
    splits = sorted(cm)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, lab, col in [("long_block_exact", "long path fully correct", "#1b9e77"),
                          ("short_block_exact", "short path fully correct", "#66a61e"),
                          ("cut_block_exact", "cut fully correct", "#e7298a"),
                          ("exact", "whole graph exact", "#222222")]:
        ys = [agg(cm[a], key) for a in splits]
        ax.plot(splits, ys, "-o", ms=4, color=col, label=lab)
    ax.set_xlabel("split: short-component size a")
    ax.set_ylabel("fraction of graphs entirely correct (block)")
    ax.set_title(f"Where the exact-match breaks vs split  (n={n}, {COND_LABEL.get(cond, cond)})")
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.set_xticks(splits)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    p = Path(outdir) / f"asym_chains_blocks_n{n}_{cond}.png"
    fig.savefig(p, dpi=150); plt.close(fig); print("  saved", p)


def plot_perdist(n, cond, cm, outdir):
    """Within-long-component reach by distance, for a few representative splits."""
    splits = sorted(cm)
    if not splits:
        return
    # representative splits: balanced, the two R4 reference points (if present), and small
    want = sorted(set([s for s in (4, 17, n // 2) if s in splits] + [splits[0]]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("viridis")
    for j, a in enumerate(want):
        cells = cm[a]
        # pool per-distance accuracy over seeds (counts identical across seeds)
        pd = defaultdict(list)
        for c in cells:
            for d, (acc, _) in c["per_dist_long"].items():
                pd[int(d)].append(acc)
        if not pd:
            continue
        ds = sorted(pd)
        ys = [float(np.mean(pd[d])) for d in ds]
        long_len = cells[0]["long_len"]
        ax.plot(ds, ys, "-", color=cmap(j / max(1, len(want) - 1)),
                label=f"a=({a},{n - a}); long={long_len}")
    ax.axvline(CAP, color="red", ls="--", lw=1, label="capacity 3^L=9")
    ax.set_xlabel("shortest-path distance d within the long component")
    ax.set_ylabel("reach (pairwise on connected pairs)")
    ax.set_title(f"Long-component reach profile by split  (n={n}, {COND_LABEL.get(cond, cond)})")
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    p = Path(outdir) / f"asym_chains_perdist_n{n}_{cond}.png"
    fig.savefig(p, dpi=150); plt.close(fig); print("  saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/report6/asym_chains")
    ap.add_argument("--conds", nargs="*", default=None,
                    help="subset of {er, pathunion, mixed}; default all present")
    ap.add_argument("--outdir", default="runs/report6/report6_figs")
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    data = load(args.root, args.conds)
    if not data:
        print(f"no asym_chains.json under {args.root} (pull the eval outputs first)"); return

    ns = sorted({n for (n, _) in data})
    conds_order = ["pathunion", "er", "mixed"]
    for n in ns:
        conds_present = [c for c in conds_order if (n, c) in data]
        for cond in conds_present:
            cm = data[(n, cond)]
            nseed = max(len(v) for v in cm.values())
            print(f"\n=== n={n} {cond}  ({nseed} seeds) ===")
            print(f"{'a':>3} {'long':>4} | {'exact':>5} {'reachL':>6} {'cut':>5} "
                  f"{'Lblock':>6} {'cutblk':>6}")
            for a in sorted(cm):
                cs = cm[a]
                print(f"{a:>3} {cs[0]['long_len']:>4} | {agg(cs,'exact'):>5.2f} "
                      f"{agg(cs,'reach_long'):>6.2f} {agg(cs,'cut'):>5.2f} "
                      f"{agg(cs,'long_block_exact'):>6.2f} {agg(cs,'cut_block_exact'):>6.2f}")
        plot_split(n, conds_present, data, args.outdir)
        # clean condition preferred for the block / per-distance mechanism figures
        clean = "pathunion" if (n, "pathunion") in data else conds_present[0]
        plot_blocks(n, clean, data[(n, clean)], args.outdir)
        plot_perdist(n, clean, data[(n, clean)], args.outdir)


if __name__ == "__main__":
    main()
