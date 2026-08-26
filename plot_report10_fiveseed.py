"""Report X -- reconstruct paper_draft.tex Sec.5.1's table+figure style at 5 seeds, and
apply the identical treatment to the two two-cycle conditions (never done in the paper).

Two source formats coexist in the repo for the same underlying sweep:
  * chain conditions   -- runs/report9/asym_chains_n46/<tag>_seed<S>/asym_chains.json
                           (eval_asym_chains.py; one JSON per seed, field names exact/
                           reach_long/cut).
  * cycle conditions    -- runs/report9/mechanistic/<tag>_seed<S>/metrics.csv
                           (mechanistic_asym_chains.py --topology cycle; long/tidy CSV,
                           mode=="random" is the primary condition, metric column names
                           exact/reach_long/cut). eval_asym_chains.py is chain-only, so
                           this is the only place the cycle sweep exists.

For each condition this script writes:
  * a LaTeX table fragment (same columns as paper_draft.tex Table 1 / the appendix
    table: split, exact, reach, cut, mean +/- std across seeds, consecutive rows grouped
    only where all three match to 3 decimals -- never grouped across a std change);
  * the exact-match-vs-a figure in the same style as fig_oddtrain_exactmatch_vs_a.pdf
    (thick blue mean line #0072B2, thin light-gray per-seed lines, filled/hollow markers
    for trained/OOD splits when the condition has a fixed training grid, dashed vertical
    line at the empirical break with an honest label, D_long annotation).

    python plot_report10_fiveseed.py
"""
import argparse, csv, json
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#0072B2"
GRAY = "#b0b0b0"
ORANGE = "#D55E00"
N = 46
CAP2L = 18  # 2*3^L at L=2


def load_chain(tag_prefix, seeds):
    """runs/report9/asym_chains_n46/<tag_prefix>_seed<S>/asym_chains.json"""
    per_split = {}
    for s in seeds:
        f = Path(f"runs/report9/asym_chains_n46/{tag_prefix}_seed{s}/asym_chains.json")
        d = json.load(open(f))
        for c in d["cells"]:
            per_split.setdefault(c["split"], {"exact": [], "reach": [], "cut": []})
            per_split[c["split"]]["exact"].append(c["exact"])
            per_split[c["split"]]["reach"].append(c["reach_long"])
            per_split[c["split"]]["cut"].append(c["cut"])
    return per_split


def load_cycle(tag_prefix, seeds):
    """runs/report9/mechanistic/<tag_prefix>_seed<S>/metrics.csv, mode=random."""
    per_split = {}
    for s in seeds:
        f = Path(f"runs/report9/mechanistic/{tag_prefix}_seed{s}/metrics.csv")
        rows = list(csv.DictReader(f.open()))
        vals = {}
        for r in rows:
            if r["mode"] != "random" or r["metric"] not in ("exact", "reach_long", "cut"):
                continue
            a = int(r["split_a"])
            vals.setdefault(a, {})[r["metric"]] = float(r["value"])
        for a, d in vals.items():
            per_split.setdefault(a, {"exact": [], "reach": [], "cut": []})
            per_split[a]["exact"].append(d["exact"])
            per_split[a]["reach"].append(d["reach_long"])
            per_split[a]["cut"].append(d["cut"])
    return per_split


def make_table(per_split, out_tex, caption, label):
    splits = sorted(per_split)
    rows = []
    for a in splits:
        e = np.array(per_split[a]["exact"]); r = np.array(per_split[a]["reach"]); c = np.array(per_split[a]["cut"])
        rows.append((a, e.mean(), e.std(), r.mean(), r.std(), c.mean(), c.std()))

    def fmt(mean, std, n_seeds):
        if std < 5e-4 or n_seeds < 2:
            return f"${mean:.3f}$"
        return f"${mean:.3f} \\pm {std:.3f}$"

    n_seeds = len(per_split[splits[0]]["exact"])
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\begin{tabular}{lccc}", r"\toprule",
             r"Test split $(a,\," + f"{N}-a)$ & Exact match & Reach & Cut \\\\", r"\midrule"]
    i = 0
    while i < len(rows):
        a0 = rows[i]
        j = i
        while j + 1 < len(rows):
            a1 = rows[j + 1]
            same = (abs(a0[1] - a1[1]) < 5e-4 and abs(a0[3] - a1[3]) < 5e-4 and abs(a0[5] - a1[5]) < 5e-4
                    and abs(a0[2] - a1[2]) < 5e-4 and abs(a0[4] - a1[4]) < 5e-4 and abs(a0[6] - a1[6]) < 5e-4)
            if not same:
                break
            j += 1
        a_lo, a_hi = rows[i][0], rows[j][0]
        span = f"$({a_lo},{N-a_lo})$" if a_lo == a_hi else f"$({a_lo},{N-a_lo})$ -- $({a_hi},{N-a_hi})$"
        _, em, es, rm, rs, cm, cs = rows[i]
        lines.append(f"{span} & {fmt(em,es,n_seeds)} & {fmt(rm,rs,n_seeds)} & {fmt(cm,cs,n_seeds)} \\\\")
        i = j + 1
    lines += [r"\bottomrule", r"\end{tabular}",
              f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table}"]
    Path(out_tex).write_text("\n".join(lines) + "\n")
    print("wrote", out_tex)


def make_figure(per_split, out_png, trained_sizes, title_note, break_a=None):
    splits = sorted(per_split)
    seeds_list = sorted(range(len(per_split[splits[0]]["exact"])))
    n_seeds = len(seeds_list)

    fig, ax = plt.subplots(figsize=(9, 5))
    if title_note:
        ax.set_title(title_note, fontsize=12)

    # per-seed thin lines
    for si in range(n_seeds):
        ys = [per_split[a]["exact"][si] for a in splits]
        ax.plot(splits, ys, "-", color=GRAY, lw=0.8, alpha=0.8, zorder=1)

    # thick mean line, filled/hollow markers if this condition has a training grid
    means = [np.mean(per_split[a]["exact"]) for a in splits]
    ax.plot(splits, means, "-", color=BLUE, lw=2.2, zorder=3)
    for a, m in zip(splits, means):
        filled = trained_sizes is not None and a in trained_sizes
        ax.plot(a, m, "o", ms=8, color=BLUE,
                 markerfacecolor=(BLUE if filled or trained_sizes is None else "white"),
                 markeredgecolor=BLUE, markeredgewidth=1.6, zorder=4)

    if break_a is not None:
        ax.axvline(break_a - 0.5, color="gray", ls="--", lw=1.2, zorder=2)
        ax.text(break_a - 0.5, 0.5, f"  exact match drops from $a={break_a}$",
                color="gray", fontsize=10, va="center")

    ax.set_xlabel("Short-component size $a$", fontsize=12)
    ax.set_ylabel("Whole-graph exact match", fontsize=12)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(splits[::1] if len(splits) <= 24 else splits[::2])
    ax.grid(axis="y", alpha=0.3)
    if trained_sizes is not None:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker="o", color=BLUE, markerfacecolor=BLUE,
                           markeredgecolor=BLUE, lw=2.2, label="split present in training"),
                   Line2D([0], [0], marker="o", color=BLUE, markerfacecolor="white",
                           markeredgecolor=BLUE, lw=2.2, label="split never present in training"),
                   Line2D([0], [0], color=GRAY, lw=0.8, label="individual seeds")]
        ax.legend(handles=handles, loc="center left", fontsize=10)
    else:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=BLUE, lw=2.2, label="mean over seeds"),
                   Line2D([0], [0], color=GRAY, lw=0.8, label="individual seeds")]
        ax.legend(handles=handles, loc="center left", fontsize=10)
    fig.text(0.5, -0.02, f"split $=(a,\\,{N}-a)$", ha="center", fontsize=10, color="#555555")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_png)


CONDITIONS = {
    "chains_grid": dict(loader=load_chain, tag="n46_splitchainsgrid", trained=[3, 5, 7, 9],
                         title=f"All splits satisfy $D_\\mathrm{{long}} > {CAP2L} = 2\\cdot3^L$",
                         break_a=12, seeds=[1000, 2000, 3000, 4000, 5000],
                         caption="Two-path model trained on the odd setting (n=46, similarity "
                                  "read-out, 5 seeds). Exact match, reach, and cut by test split.",
                         label="tab:beyond-threshold-5seed"),
    "chains_continuous": dict(loader=load_chain, tag="n46_splitchains", trained=None,
                               title=f"All splits satisfy $D_\\mathrm{{long}} > {CAP2L} = 2\\cdot3^L$"
                                     " (full split-size distribution)",
                               break_a=12, seeds=[1000, 2000, 3000, 4000, 5000],
                               caption="Two-path model trained on the full split-size "
                                        "distribution (n=46, similarity read-out, 5 seeds). "
                                        "Exact match, reach, and cut by test split.",
                               label="tab:full-dist-control-5seed"),
    "cycles_grid": dict(loader=load_cycle, tag="n46_splitcyclesgrid", trained=[3, 5, 7, 9],
                         title=f"Two-cycle, odd/grid training. All splits satisfy "
                               f"$D_\\mathrm{{long}} > {CAP2L} = 2\\cdot3^L$",
                         break_a=13, seeds=[1000, 2000, 3000, 4000, 5000,
                                            6000, 7000, 8000, 9000, 10000],
                         caption="Two-cycle model trained on the odd/grid setting (n=46, "
                                  "similarity read-out, 10 seeds). Exact match, reach, and cut "
                                  "by test split.",
                         label="tab:cycles-grid-10seed"),
    "cycles_continuous": dict(loader=load_cycle, tag="n46_splitcycles", trained=None,
                               title=f"Two-cycle, full split-size distribution training. All "
                                     f"splits satisfy $D_\\mathrm{{long}} > {CAP2L} = 2\\cdot3^L$",
                               break_a=12, seeds=[1000, 2000, 3000, 4000, 5000,
                                                  6000, 7000, 8000, 9000, 10000],
                               caption="Two-cycle model trained on the full split-size "
                                        "distribution (n=46, similarity read-out, 10 seeds). "
                                        "Exact match, reach, and cut by test split.",
                               label="tab:cycles-continuous-10seed"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--out_dir", default="runs/report10/report10_figs")
    ap.add_argument("--tex_dir", default="runs/report10/report10_figs")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for key in args.which:
        cfg = CONDITIONS[key]
        nseed_tag = f"{len(cfg['seeds'])}seed"
        per_split = cfg["loader"](cfg["tag"], cfg["seeds"])
        make_table(per_split, f"{args.tex_dir}/r10_table_{key}_{nseed_tag}.tex",
                   cfg["caption"], cfg["label"])
        make_figure(per_split, f"{args.out_dir}/r10_exactmatch_vs_a_{key}_{nseed_tag}.png",
                    cfg["trained"], cfg["title"], cfg["break_a"])


if __name__ == "__main__":
    main()
