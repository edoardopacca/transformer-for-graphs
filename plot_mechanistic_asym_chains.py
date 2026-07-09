"""Report VII -- aggregate and plot the mechanistic asymmetric-chains audit
(local, no GPU). Reads the json/csv produced by mechanistic_asym_chains.py and
eval_three_way_split.py under runs/report7/, pools across the seeds matching
--tag_glob, and writes figures to runs/report7/report7_figs/ tagged with
--suffix (so different (n, training distribution) conditions don't overwrite
each other's figures).

    python plot_mechanistic_asym_chains.py                                            # n=40 path_union (default)
    python plot_mechanistic_asym_chains.py --tag_glob "n64_pathunion_seed*" --n 64 --suffix _n64_pathunion
    python plot_mechanistic_asym_chains.py --tag_glob "n64_er_seed*" --n 64 --suffix _n64_er
"""
import argparse, csv, glob, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

MECH_ROOT = Path("runs/report7/mechanistic")
THREEWAY_ROOT = Path("runs/report7/three_way")
OUT = Path("runs/report7/report7_figs")
CAP = 9


def load_metrics(tag_glob):
    rows = []
    for f in sorted(MECH_ROOT.glob(f"{tag_glob}/metrics.csv")):
        with f.open() as fh:
            rows += list(csv.DictReader(fh))
    return rows


def load_readout(tag_glob):
    rows = []
    for f in sorted(MECH_ROOT.glob(f"{tag_glob}/readout.csv")):
        with f.open() as fh:
            rows += list(csv.DictReader(fh))
    return rows


def load_attn(tag_glob):
    out = {}
    for f in sorted(MECH_ROOT.glob(f"{tag_glob}/attn_cache.npz")):
        out[f.parent.name] = np.load(f)
    return out


def dist_bounded_rate(n, a, cap=CAP):
    def reach_sum(m):
        return sum(min(i, cap) + min(m - 1 - i, cap) for i in range(m))
    b = n - a
    return (reach_sum(a) + reach_sum(b)) / (n * (n - 1))


def fig_sweep_and_logit(metrics_rows, readout_rows, n, suffix, title_tag):
    splits = sorted({int(r["split_a"]) for r in metrics_rows if r["mode"] == "random"})
    behav = defaultdict(lambda: defaultdict(list))
    keep_metrics = ("exact", "reach_long", "reach_short", "cut", "pred_positive_rate")
    for r in metrics_rows:
        if r["mode"] != "random" or r["metric"] not in keep_metrics:
            continue
        behav[int(r["split_a"])][r["metric"]].append(float(r["value"]))
    read = defaultdict(lambda: defaultdict(list))
    for r in readout_rows:
        if r["pair_type"] in ("within_long", "within_short", "cut"):
            read[int(r["split_a"])][r["pair_type"]].append(float(r["mean_hTw"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for key, lab, col in [("exact", "exact match", "#222222"),
                          ("reach_long", "reach (long)", "#1b9e77"),
                          ("reach_short", "reach (short)", "#66a61e"),
                          ("cut", "cut (target: disconnected)", "#e7298a")]:
        ys = [np.mean(behav[a][key]) if behav[a][key] else np.nan for a in splits]
        ax1.plot(splits, ys, "-o", ms=4, color=col, label=lab)
    pos = [np.mean(behav[a]["pred_positive_rate"]) for a in splits]
    oracle = [dist_bounded_rate(n, a) for a in splits]
    ax1.plot(splits, pos, "--", color="#7570b3", label="predicted-positive rate")
    ax1.plot(splits, oracle, ":", color="#7570b3", lw=2,
             label=r"distance$\,\leq 9$ oracle positive rate")
    ax1.axvline(9, color="red", ls="--", lw=1, label="capacity $3^L=9$")
    ax1.set_ylabel("accuracy / rate"); ax1.set_ylim(-0.02, 1.02)
    ax1.grid(alpha=0.3); ax1.legend(loc="center left", fontsize=7, ncol=1)
    ax1.set_title(f"Behavioural sweep (top) and raw read-out logit (bottom), n={n}, {title_tag}")

    for key, lab, col in [("within_long", r"within long: mean $h_i^\top w_j$", "#1b9e77"),
                          ("within_short", r"within short: mean $h_i^\top w_j$", "#66a61e"),
                          ("cut", r"cut: mean $h_i^\top w_j$", "#e7298a")]:
        ys = [np.mean(read[a][key]) if read[a][key] else np.nan for a in splits]
        ax2.plot(splits, ys, "-o", ms=4, color=col, label=lab)
    ax2.axhline(0, color="k", lw=1)
    ax2.axvline(9, color="red", ls="--", lw=1)
    ax2.set_xlabel(f"split: short-component size $a$ (components $a$ and ${n}-a$)")
    ax2.set_ylabel(r"mean $h_i^\top w_j$ (pre-bias logit)")
    ax2.grid(alpha=0.3); ax2.legend(loc="center left", fontsize=8)
    ax2.set_xticks(splits[::2] if len(splits) > 24 else splits)
    fig.tight_layout()
    p = OUT / f"r7_sweep_and_logit{suffix}.png"
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)
    return splits, behav


def fig_attention_leak(attn_by_run, n, suffix, title_tag):
    splits_all = set()
    per_a = defaultdict(list)
    for run, d in attn_by_run.items():
        keys = {k.split("__")[0] for k in d.files}
        for key in keys:
            a = int(key[1:])
            if f"{key}__contrib_exact_mean" not in d.files:
                continue
            contrib = d[f"{key}__contrib_exact_mean"]
            long_idx = d[f"{key}__long_idx"]; short_idx = d[f"{key}__short_idx"]
            if len(short_idx) == 0:
                continue
            sub = contrib[long_idx]
            mass_long = sub[:, long_idx].sum(axis=1)
            mass_short = sub[:, short_idx].sum(axis=1)
            leak = mass_short / (mass_long + mass_short + 1e-9)
            per_a[a].append(float(leak.mean()))
            splits_all.add(a)
    if not splits_all:
        print("no attention cache data found, skipping leak-fraction figure"); return
    splits = sorted(splits_all)
    means = [np.mean(per_a[a]) for a in splits]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a in splits:
        ax.scatter([a] * len(per_a[a]), per_a[a], color="#2c7fb8", s=14, alpha=0.6)
    ax.plot(splits, means, "-o", color="#08306b", ms=5, label="mean over seeds")
    ax.axvline(9, color="red", ls="--", lw=1, label="capacity $3^L=9$")
    ax.set_xlabel("split: short-component size $a$")
    ax.set_ylabel("real contribution mass reaching the wrong component")
    ax.set_title(f"real node-to-node contribution: how much of a long-path node's final\nembedding traces back to the short component (n={n}, {title_tag})")
    ax.set_ylim(-0.01, max(0.4, max(means) * 1.15) if means else 0.4)
    ax.grid(alpha=0.3); ax.set_xticks(splits)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = OUT / f"r7_attention_leak{suffix}.png"
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def table_threeway(tag_glob):
    cells_by_small = defaultdict(list)
    for f in sorted(THREEWAY_ROOT.glob(f"{tag_glob}/three_way_split.json")):
        d = json.load(f.open())
        for c in d["cells"]:
            cells_by_small[c["small_len"]].append(c)
    if not cells_by_small:
        print("no three_way_split.json found for this tag_glob"); return
    print(f"{'small':>5} {'large1':>6} {'large2':>6} {'reachL1':>8} {'reachL2':>8} "
          f"{'cut(S,L1)':>10} {'cut(S,L2)':>10} {'cut(L1,L2)':>11}")
    for s in sorted(cells_by_small):
        cs = cells_by_small[s]
        agg = lambda k: np.mean([c[k] for c in cs])
        print(f"{s:>5} {cs[0]['large1_len']:>6} {cs[0]['large2_len']:>6} "
              f"{agg('reach_large1'):>8.3f} {agg('reach_large2'):>8.3f} "
              f"{agg('cut_small_large1'):>10.3f} {agg('cut_small_large2'):>10.3f} "
              f"{agg('cut_large1_large2'):>11.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag_glob", default="n40_pathunion_seed*",
                    help="glob (relative to runs/report7/mechanistic/) selecting which seed dirs to pool")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--suffix", default="", help="appended to output figure filenames")
    ap.add_argument("--title_tag", default="disjoint-paths-trained",
                    help="short description of the training distribution, used in figure titles")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_rows = load_metrics(args.tag_glob)
    readout_rows = load_readout(args.tag_glob)
    attn_by_run = load_attn(args.tag_glob)
    if not metrics_rows:
        print(f"no metrics.csv found under {MECH_ROOT}/{args.tag_glob}"); return
    fig_sweep_and_logit(metrics_rows, readout_rows, args.n, args.suffix, args.title_tag)
    fig_attention_leak(attn_by_run, args.n, args.suffix, args.title_tag)
    print(f"\n=== three-way split falsification test (mean over seeds, {args.tag_glob}) ===")
    table_threeway(args.tag_glob)


if __name__ == "__main__":
    main()
