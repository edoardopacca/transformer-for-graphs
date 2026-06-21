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


def plot_combined_cross_sweep(groups, out):
    """The report figure base_bridged_cross_sweep.png: a 2x2 grid (rows = mixed/ER,
    cols = n20/n40) of model cross-block accuracy vs clique size, against ALL THREE
    reference oracles, so the three-way comparison is explicit:
      matrix-power (distance-bounded) -- flat at 1 (the bridge is <= 3 hops at every c);
      bounded-DFS  (visit-bounded, dives) -- RISES with c (crosses the bridge early
                    and reaches deeper into the far clique as the budget grows);
      bounded-BFS  (visit-bounded, ball) -- FALLS to ~0 (stuck in the near clique).
    The model falls from 1 to 0: it tracks matrix power up to a node budget (~6-7),
    then collapses to the stuck-BFS floor -- the OPPOSITE trend to DFS, which rejects
    the depth-first reading."""
    rows = [("mixed", "clean: dense cliques in training"),
            ("er", "confounded: degree-heuristic OOD")]
    ns = sorted({n for (n, _) in groups})
    fig, axes = plt.subplots(2, len(ns), figsize=(5.2 * len(ns), 8.4), squeeze=False)
    for ri, (fam, subtitle) in enumerate(rows):
        for ci, n in enumerate(ns):
            ax = axes[ri][ci]
            results = groups.get((n, fam))
            if not results:
                ax.set_visible(False); continue
            cs = results[0][1]["clique_size_sweep"]["clique_sizes"]
            mean, _, A = _mean_curve(results, ["clique_size_sweep", "model_cross_acc"])
            for row in A:
                ax.plot(cs, row, color="tab:blue", alpha=0.22, lw=1)
            ax.plot(cs, mean, color="tab:blue", lw=2.6, marker="o", ms=4,
                    label="model (mean over seeds)")
            mp, _, _ = _mean_curve(results, ["clique_size_sweep", "oracle_mp_cross_acc"])
            dfs, _, _ = _mean_curve(results, ["clique_size_sweep", "oracle_dfs_cross_acc"])
            bfs, _, _ = _mean_curve(results, ["clique_size_sweep", "oracle_bfs_cross_acc"])
            ax.plot(cs, mp, color="tab:green", lw=2, ls="--",
                    label="matrix-power oracle (distance-bounded)")
            ax.plot(cs, dfs, color="tab:orange", lw=2, ls="-.",
                    label="bounded-DFS oracle (dives, budget=c)")
            ax.plot(cs, bfs, color="tab:red", lw=2, ls=":",
                    label="bounded-BFS oracle (gets stuck, budget=c)")
            ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
            ax.set_xlabel("clique size c")
            if ci == 0:
                ax.set_ylabel(f"{fam}-trained\ncross-block accuracy")
            ax.set_title(f"n={n}, {fam}-trained ({len(results)} seeds)\n{subtitle}", fontsize=9)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7.5, loc="center left")
    fig.suptitle("Bridged cliques: does the model carry the single bridge across the cliques as they grow?\n"
                 "model FALLS like a stuck bounded-BFS, the OPPOSITE of bounded-DFS (which rises); "
                 "matrix power stays flat at 1", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    f = out / "base_bridged_cross_sweep.png"
    fig.savefig(f, dpi=150); plt.close(fig); print("wrote", f)


def plot_combined_oracle_follow(groups, out):
    """The report figure base_bridged_oracle_follow.png: for the MIXED model, on the
    cross entries where matrix-power and bounded-BFS disagree, the fraction the model
    matches to each, vs the BFS budget. (DFS is absent here: it is already rejected by
    the clique-size sweep, where its curve rises while the model's falls.)"""
    ns = sorted({n for (n, _) in groups})
    fig, axes = plt.subplots(1, len(ns), figsize=(5.6 * len(ns), 4.6), squeeze=False)
    for ci, n in enumerate(ns):
        ax = axes[0][ci]
        results = groups.get((n, "mixed"))
        if not results:
            ax.set_visible(False); continue
        budgets = results[0][1]["oracle_agreement_bridged"]["budgets"]
        mp, _, _ = _mean_curve(results, ["oracle_agreement_bridged", "model_follows_mp_on_disagree"])
        bfs, _, _ = _mean_curve(results, ["oracle_agreement_bridged", "model_follows_bfs_on_disagree"])
        ax.plot(budgets, mp, color="tab:green", lw=2, marker="o",
                label="model follows matrix-power")
        ax.plot(budgets, bfs, color="tab:red", lw=2, marker="s",
                label="model follows bounded-BFS")
        ax.axvline(n // 2, color="grey", ls="--", lw=1, label="one clique (budget = c)")
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
        ax.set_xlabel("bounded-BFS budget (nodes per start)")
        if ci == 0:
            ax.set_ylabel("fraction of contested cross pairs matched")
        ax.set_title(f"n={n}, mixed-trained ({len(results)} seeds)", fontsize=10)
        if ci == 0:
            ax.legend(fontsize=8)
    fig.suptitle("On the cross pairs where matrix-power and bounded-BFS disagree, the mixed model "
                 "tracks bounded-BFS, not matrix power", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    f = out / "base_bridged_oracle_follow.png"
    fig.savefig(f, dpi=150); plt.close(fig); print("wrote", f)


SIM_TAG_RE = re.compile(r"n(\d+)_(er|mixed)_sim_seed(\d+)")


def load_all_sim(root):
    """Same as load_all but for the SIMILARITY read-out checkpoints, whose tags
    carry an extra '_sim' (n{N}_{set}_sim_seed{S})."""
    groups = defaultdict(list)
    for jf in sorted(Path(root).glob("*/bridged_cliques.json")):
        m = SIM_TAG_RE.search(jf.parent.name)
        if not m:
            continue
        n, fam, seed = int(m.group(1)), m.group(2), int(m.group(3))
        groups[(n, fam)].append((seed, json.loads(jf.read_text())))
    return groups


def plot_similarity_vs_linear(lin_root, sim_root, out):
    """Report figure base_bridged_similarity_knee.png (Section 5.5): the MIXED model's
    cross-block accuracy vs clique size, LINEAR vs SIMILARITY read-out overlaid, one
    panel per n. Report IV showed the similarity read-out doubles the DISTANCE reach
    (3^L -> 2*3^L, a meet-in-the-middle of two neighbourhoods); here we ask whether it
    also moves the NODE budget of Section 5.2. The matrix-power oracle (flat at 1) is
    drawn as the distance-bounded reference. The knee moves right under similarity at
    n40 (where the canvas has room), the same lever lifting both budgets."""
    lin = load_all(lin_root)
    sim = load_all_sim(sim_root)
    ns = sorted({n for (n, _) in sim})
    fig, axes = plt.subplots(1, len(ns), figsize=(6.0 * len(ns), 4.7), squeeze=False)
    for ci, n in enumerate(ns):
        ax = axes[0][ci]
        lr = lin.get((n, "mixed")); sr = sim.get((n, "mixed"))
        if not sr:
            ax.set_visible(False); continue
        cs = sr[0][1]["clique_size_sweep"]["clique_sizes"]
        # similarity model
        mean_s, _, As = _mean_curve(sr, ["clique_size_sweep", "model_cross_acc"])
        for row in As:
            ax.plot(cs, row, color="tab:purple", alpha=0.18, lw=1)
        ax.plot(cs, mean_s, color="tab:purple", lw=2.6, marker="s", ms=4,
                label=f"model, SIMILARITY read-out ({len(sr)} seeds)")
        # linear model (same clique-size grid)
        if lr:
            cs_l = lr[0][1]["clique_size_sweep"]["clique_sizes"]
            mean_l, _, Al = _mean_curve(lr, ["clique_size_sweep", "model_cross_acc"])
            for row in Al:
                ax.plot(cs_l, row, color="tab:blue", alpha=0.18, lw=1)
            ax.plot(cs_l, mean_l, color="tab:blue", lw=2.6, marker="o", ms=4,
                    label=f"model, LINEAR read-out ({len(lr)} seeds)")
        mp, _, _ = _mean_curve(sr, ["clique_size_sweep", "oracle_mp_cross_acc"])
        ax.plot(cs, mp, color="tab:green", lw=2, ls="--",
                label="matrix-power oracle (distance-bounded)")
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
        ax.set_xlabel("clique size c (each clique has c nodes)")
        if ci == 0:
            ax.set_ylabel("cross-block accuracy on bridged cliques")
        ax.set_title(f"n={n}, mixed-trained", fontsize=10)
        if ci == 0:
            ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Does the similarity read-out move the node budget? Cross-block collapse vs clique size,\n"
                 "linear vs similarity read-out (mixed model). The knee moves right at n=40 "
                 "(canvas has room); at n=20 (c<=10) it is capped", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    f = out / "base_bridged_similarity_knee.png"
    fig.savefig(f, dpi=150); plt.close(fig); print("wrote", f)


TRAINED_TAG_RE = re.compile(r"n(\d+)_seed(\d+)")


def load_all_trained(root):
    """Loader for the TRAIN-ON-BRIDGED checkpoints (Section 5.6): bridged+split were
    ADDED to the mixed stream (random clique size), so the tags carry no train-set token
    (n{N}_seed{S}). Returned under the key (n, 'trained') to sit next to the held-out
    mixed baseline (n, 'mixed') from load_all."""
    groups = defaultdict(list)
    for jf in sorted(Path(root).glob("*/bridged_cliques.json")):
        m = TRAINED_TAG_RE.fullmatch(jf.parent.name)
        if not m:
            continue
        n, seed = int(m.group(1)), int(m.group(2))
        groups[(n, "trained")].append((seed, json.loads(jf.read_text())))
    return groups


def plot_trained_vs_heldout(lin_root, trained_root, out):
    """Report figure base_bridged_trained_knee.png (Section 5.6, the capacity-vs-data
    test): the MIXED model's cross-block accuracy vs clique size, HELD-OUT vs
    TRAINED-ON-BRIDGED overlaid, one panel per n. The held-out baseline (Section 5.2)
    never saw a clique+bridge and collapses to 0 once the near clique exceeds ~6-7 nodes.
    If the same L=2 architecture, with bridged cliques in its training stream, instead
    propagates the single bridge at EVERY clique size (cross stays at 1, flat like the
    matrix-power oracle), then the Section 5.2 collapse was a DATA gap, not a hard
    capacity wall -- the bridge is always <=3 hops, well within the 3^L=9 capacity."""
    held = load_all(lin_root)            # (n, 'mixed') held-out baseline
    trn = load_all_trained(trained_root)  # (n, 'trained')
    ns = sorted({n for (n, _) in trn})
    fig, axes = plt.subplots(1, len(ns), figsize=(6.0 * len(ns), 4.7), squeeze=False)
    for ci, n in enumerate(ns):
        ax = axes[0][ci]
        tr = trn.get((n, "trained")); hr = held.get((n, "mixed"))
        if not tr:
            ax.set_visible(False); continue
        cs = tr[0][1]["clique_size_sweep"]["clique_sizes"]
        # trained-on-bridged model
        mean_t, _, At = _mean_curve(tr, ["clique_size_sweep", "model_cross_acc"])
        for row in At:
            ax.plot(cs, row, color="tab:olive", alpha=0.20, lw=1)
        ax.plot(cs, mean_t, color="tab:olive", lw=2.6, marker="D", ms=4,
                label=f"trained on bridged ({len(tr)} seeds)")
        # held-out baseline (Section 5.2)
        if hr:
            cs_h = hr[0][1]["clique_size_sweep"]["clique_sizes"]
            mean_h, _, Ah = _mean_curve(hr, ["clique_size_sweep", "model_cross_acc"])
            for row in Ah:
                ax.plot(cs_h, row, color="tab:blue", alpha=0.18, lw=1)
            ax.plot(cs_h, mean_h, color="tab:blue", lw=2.6, marker="o", ms=4,
                    label=f"held out, baseline of 5.2 ({len(hr)} seeds)")
        mp, _, _ = _mean_curve(tr, ["clique_size_sweep", "oracle_mp_cross_acc"])
        ax.plot(cs, mp, color="tab:green", lw=2, ls="--",
                label="matrix-power oracle (distance-bounded)")
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
        ax.set_xlabel("clique size c (each clique has c nodes)")
        if ci == 0:
            ax.set_ylabel("cross-block accuracy on bridged cliques")
        ax.set_title(f"n={n}, mixed-trained, linear read-out", fontsize=10)
        if ci == 0:
            ax.legend(fontsize=8, loc="center left")
    fig.suptitle("Capacity or data gap? Cross-block collapse vs clique size, bridged cliques HELD OUT "
                 "(Section 5.2)\nvs ADDED to the training stream. With the structure in training the same "
                 "L=2 model carries the bridge at every c", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    f = out / "base_bridged_trained_knee.png"
    fig.savefig(f, dpi=150); plt.close(fig); print("wrote", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/report5/bridged_cliques")
    ap.add_argument("--output_dir", default="runs/report5/report5_figs")
    ap.add_argument("--similarity_root", default="runs/report5/bridged_similarity",
                    help="if present, also draw the linear-vs-similarity knee figure (Section 5.5)")
    ap.add_argument("--trained_root", default="runs/report5/bridged_cliques_trained",
                    help="if present, also draw the held-out-vs-trained knee figure (Section 5.6)")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    groups = load_all(args.root)
    if not groups:
        print(f"no JSONs found under {args.root}"); return
    print("conditions:", {f"n{n}_{s}": len(v) for (n, s), v in groups.items()})
    plot_clique_sweep(groups, out)
    plot_blocks(groups, out)
    plot_oracle_follow(groups, out)
    plot_combined_cross_sweep(groups, out)      # report figure (3 oracles)
    plot_combined_oracle_follow(groups, out)     # report figure
    if args.similarity_root and Path(args.similarity_root).exists():
        sim_groups = load_all_sim(args.similarity_root)
        if sim_groups:
            print("similarity conditions:",
                  {f"n{n}_{s}": len(v) for (n, s), v in sim_groups.items()})
            plot_similarity_vs_linear(args.root, args.similarity_root, out)  # report figure 5.5
    if args.trained_root and Path(args.trained_root).exists():
        trn_groups = load_all_trained(args.trained_root)
        if trn_groups:
            print("trained-on-bridged conditions:",
                  {f"n{n}_{s}": len(v) for (n, s), v in trn_groups.items()})
            plot_trained_vs_heldout(args.root, args.trained_root, out)  # report figure 5.6


if __name__ == "__main__":
    main()
