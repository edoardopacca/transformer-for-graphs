"""Aggregate figure for the capstone (Report V): across the natural families, does the
base model follow the matrix-power oracle or the bounded-BFS oracle? Reads the per-
checkpoint JSONs written by eval_oracle_agreement_families.py under

    runs/report5/oracle_families/<tag>/oracle_families.json   (tag = n{N}_{set}_seed{S})

groups by (n, training set), and plots, on the DISAGREEING pairs only, the fraction of
the model that follows each oracle as the BFS budget varies -- pooled over families and
seeds. No GPU; run locally after pulling.

  python plot_oracle_families.py --root runs/report5/oracle_families \\
      --output_dir runs/report5/report5_figs
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG_RE = re.compile(r"n(\d+)_(er|mixed)_seed(\d+)")


def load_all(root):
    groups = defaultdict(list)
    for jf in sorted(Path(root).glob("*/oracle_families.json")):
        m = TAG_RE.search(jf.parent.name)
        if not m:
            continue
        groups[(int(m.group(1)), m.group(2))].append((int(m.group(3)), json.loads(jf.read_text())))
    return groups


def _curve(results, key):
    arrs = [np.array([np.nan if v is None else v for v in r["pooled"][key]], float)
            for _, r in results]
    return np.nanmean(np.vstack(arrs), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/report5/oracle_families")
    ap.add_argument("--output_dir", default="runs/report5/report5_figs")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    groups = load_all(args.root)
    if not groups:
        print(f"no JSONs under {args.root}"); return
    print("conditions:", {f"n{n}_{s}": len(v) for (n, s), v in groups.items()})

    # one panel per (n, set): model-follows-MP vs model-follows-BFS on disagreeing pairs,
    # pooled over families+seeds, vs budget.
    conds = sorted(groups.items())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    pos = {(20, "mixed"): (0, 0), (40, "mixed"): (0, 1),
           (20, "er"): (1, 0), (40, "er"): (1, 1)}
    for (n, fam), results in conds:
        if (n, fam) not in pos:
            continue
        ax = axes[pos[(n, fam)][0]][pos[(n, fam)][1]]
        b = results[0][1]["pooled"]["budgets"]
        fmp = _curve(results, "model_follows_mp_on_disagree")
        fbfs = _curve(results, "model_follows_bfs_on_disagree")
        dis = _curve(results, "disagree_frac")
        ax.plot(b, fmp, color="tab:green", lw=2.2, marker="o", ms=3, label="follows matrix-power")
        ax.plot(b, fbfs, color="tab:red", lw=2.2, marker="s", ms=3, label="follows bounded-BFS")
        ax.plot(b, dis, color="tab:gray", lw=1.4, ls=":", label="MP-vs-BFS disagree mass")
        ax.axhline(0.5, color="k", lw=0.7, ls=":")
        ax.set_ylim(-0.03, 1.05); ax.grid(alpha=0.3)
        ax.set_xlabel("bounded-BFS budget b (nodes)")
        ax.set_ylabel("fraction of disagreeing pairs")
        ax.set_title(f"n={n}, {fam}-trained ({len(results)} seeds), pooled over families")
        if (n, fam) == (20, "mixed"):
            ax.legend(fontsize=8, loc="center right")
    fig.suptitle("Capstone: on the pairs where the two algorithms disagree, which does the base model follow?\n"
                 "(pooled over the natural families; the dotted grey line is how discriminating each budget is)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    f = out / "oracle_families_follow.png"
    fig.savefig(f, dpi=150); plt.close(fig); print("wrote", f)


if __name__ == "__main__":
    main()
