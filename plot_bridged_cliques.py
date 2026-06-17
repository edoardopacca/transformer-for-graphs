"""Figures for the bridged-cliques vs split-cliques probe (Report V), pooled over
seeds. Reads the per-checkpoint JSONs written by eval_bridged_cliques.py under

    runs/report5/bridged_cliques/<tag>/bridged_cliques.json     (tag = n{N}_{set}_seed{S})

and groups them by condition (n, training set). No GPU; run locally after pulling.

  python plot_bridged_cliques.py --root runs/report5/bridged_cliques \
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
    groups = defaultdict(list)         # (n, set) -> list of (seed, result dict)
    for jf in sorted(Path(root).glob("*/bridged_cliques.json")):
        m = TAG_RE.search(jf.parent.name)
        if not m:
            continue
        n, fam, seed = int(m.group(1)), m.group(2), int(m.group(3))
        groups[(n, fam)].append((seed, json.loads(jf.read_text())))
    return groups


def _mean_curve(results, key_path):
    """Stack a per-seed list-of-floats curve, return (mean, std, all)."""
    arrs = []
    for _, r in results:
        d = r
        for k in key_path:
            d = d[k]
        arrs.append(np.array([np.nan if v is None else v for v in d], float))
    A = np.vstack(arrs)
    return np.nanmean(A, 0), np.nanstd(A, 0), A


def plot_clique_sweep(groups, out):
    """Model cross-block accuracy vs clique size, with the two oracle references."""
    for (n, fam), results in sorted(groups.items()):
        cs = results[0][1]["clique_size_sweep"]["clique_sizes"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        mean, std, A = _mean_curve(results, ["clique_size_sweep", "model_cross_acc"])
        for row in A:
            ax.plot(cs, row, color="tab:blue", alpha=0.25, lw=1)
        ax.plot(cs, mean, color="tab:blue", lw=2.5, marker="o", label="model (mean over seeds)")
        mp, _, _ = _mean_curve(results, ["clique_size_sweep", "oracle_mp_cross_acc"])
        bfs, _, _ = _mean_curve(results, ["clique_size_sweep", "oracle_bfs_cross_acc"])
        ax.plot(cs, mp, color="tab:green", lw=2, ls="--", label="matrix-power oracle (distance-bounded)")
        ax.plot(cs, bfs, color="tab:red", lw=2, ls=":", label="bounded-traversal oracle (visit-bounded, budget=c)")
        ax.set_xlabel("clique size c (each clique has c nodes)")
        ax.set_ylabel("cross-block accuracy on bridged cliques")
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
        ax.set_title(f"n={n}, {fam}-trained, linear read-out ({len(results)} seeds)\n"
                     "does the model see the single bridge as the cliques grow?")
        ax.legend(fontsize=8, loc="center left")
        fig.tight_layout()
        f = out / f"clique_sweep_n{n}_{fam}.png"
        fig.savefig(f, dpi=150); plt.close(fig); print("wrote", f)


def plot_blocks(groups, out):
    """Per-block accuracy (within-A / within-B / cross) for bridged + split, c=full."""
    for (n, fam), results in sorted(groups.items()):
        labels = ["within A", "within B", "cross"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        for ax, lab in zip(axes, ["bridged", "split"]):
            keys = ["within_A_acc", "within_B_acc", "cross_acc"]
            vals = np.array([[r["main_c_full"][lab][k] for k in keys] for _, r in results], float)
            mean = np.nanmean(vals, 0); sd = np.nanstd(vals, 0)
            x = np.arange(3)
            ax.bar(x, mean, yerr=sd, capsize=4, color=["tab:gray", "tab:gray", "tab:orange"])
            for xi, row in zip(x, vals.T):
                ax.scatter(np.full_like(row, xi, float), row, color="k", s=8, alpha=0.5, zorder=3)
            ax.set_xticks(x); ax.set_xticklabels(labels)
            ax.set_ylim(0, 1.02); ax.grid(alpha=0.3, axis="y")
            ax.set_title(lab); ax.axhline(1.0, color="g", lw=0.8, ls="--", alpha=0.5)
        axes[0].set_ylabel("pair accuracy")
        fig.suptitle(f"n={n}, {fam}-trained, linear read-out, c={n//2} ({len(results)} seeds)\n"
                     "where the model is right: within each clique vs across the bridge")
        fig.tight_layout()
        f = out / f"blocks_n{n}_{fam}.png"
        fig.savefig(f, dpi=150); plt.close(fig); print("wrote", f)


def plot_oracle_follow(groups, out):
    """On the entries where the two oracles disagree, which one the model follows."""
    for (n, fam), results in sorted(groups.items()):
        budgets = results[0][1]["oracle_agreement_bridged"]["budgets"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        mp, _, _ = _mean_curve(results, ["oracle_agreement_bridged", "model_follows_mp_on_disagree"])
        bfs, _, _ = _mean_curve(results, ["oracle_agreement_bridged", "model_follows_bfs_on_disagree"])
        ax.plot(budgets, mp, color="tab:green", lw=2, marker="o", label="model follows matrix-power")
        ax.plot(budgets, bfs, color="tab:red", lw=2, marker="s", label="model follows bounded-traversal")
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.set_xlabel("bounded-traversal budget (nodes per start)")
        ax.set_ylabel("fraction of disagreeing entries matched")
        ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
        ax.set_title(f"n={n}, {fam}-trained ({len(results)} seeds): on the cross entries where the\n"
                     "two oracles disagree, which algorithm does the model match?")
        ax.legend(fontsize=8)
        fig.tight_layout()
        f = out / f"oracle_follow_n{n}_{fam}.png"
        fig.savefig(f, dpi=150); plt.close(fig); print("wrote", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/report5/bridged_cliques")
    ap.add_argument("--output_dir", default="runs/report5/report5_figs")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    groups = load_all(args.root)
    if not groups:
        print(f"no JSONs found under {args.root}"); return
    print("conditions:", {f"n{n}_{s}": len(v) for (n, s), v in groups.items()})
    plot_clique_sweep(groups, out)
    plot_blocks(groups, out)
    plot_oracle_follow(groups, out)


if __name__ == "__main__":
    main()
