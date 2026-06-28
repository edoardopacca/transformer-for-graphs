"""Report VI, Thread C.1 -- aggregate and plot the chained-bridged-cliques probe (local, no GPU).

Reads runs/report6/clique_chain/<tag>/clique_chain.json (and the dense-ER-block variant in
runs/report6/clique_chain_er/), tag = n{N}_bridgedsplit_seed{S}, pools across seeds, and
produces, per n:
  * the COMPOSITION figure: cross-block accuracy vs the number of bridges crossed (the gap g),
    one line per chain length K, mean over seeds. Flat across g -> the learned bridge composes;
    a decay with g -> the hand-off degrades with each link. This is the headline of Thread C.1.
  * the LENGTH figure: whole-chain exact-match and the end-to-end cross accuracy vs the number
    of cliques K (for a representative clique size), mean over seeds.
  * the GUARD figure: the intact-vs-broken discrimination vs K -- a flat all-connected
    prediction would sit at ~0.5 here, so this confirms the model reads the bridges.
Also prints a compact per-cell table per n.

    python plot_clique_chain.py                  # complete cliques (clique_chain/)
    python plot_clique_chain.py --block er        # dense ER blocks (clique_chain_er/)
"""
import argparse, glob, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG_RE = re.compile(r"n(\d+)_bridgedsplit_seed(\d+)$")
FIGDIR = Path("runs/report6/report6_figs")


def load(root):
    """tag dirs -> {(n): {seed: {(K,c): cell}}}."""
    data = defaultdict(lambda: defaultdict(dict))
    for jf in sorted(glob.glob(f"{root}/*/clique_chain.json")):
        m = TAG_RE.search(Path(jf).parent.name)
        if not m:
            continue
        n, seed = int(m.group(1)), int(m.group(2))
        d = json.loads(Path(jf).read_text())
        for cell in d["cells"]:
            data[n][seed][(cell["n_cliques"], cell["clique_size"])] = cell
    return data


def mean_over_seeds(per_seed, key_fn):
    """per_seed: {seed: {(K,c): cell}} -> {(K,c): mean of key_fn(cell) over seeds (ignoring None)}."""
    acc = defaultdict(list)
    for seed, cells in per_seed.items():
        for kc, cell in cells.items():
            v = key_fn(cell)
            if v is not None:
                acc[kc].append(v)
    return {kc: float(np.mean(vs)) for kc, vs in acc.items() if vs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="clique", choices=["clique", "er"])
    ap.add_argument("--clique_size", type=int, default=4,
                    help="representative clique size for the length/guard figures")
    args = ap.parse_args()
    root = "runs/report6/clique_chain" if args.block == "clique" else "runs/report6/clique_chain_er"
    data = load(root)
    if not data:
        print(f"no json under {root}/*/clique_chain.json -- pull runs/report6 first."); return
    FIGDIR.mkdir(parents=True, exist_ok=True)
    tag = "" if args.block == "clique" else "_er"

    for n in sorted(data):
        per_seed = data[n]
        Ks = sorted({K for cells in per_seed.values() for (K, c) in cells})
        cs = sorted({c for cells in per_seed.values() for (K, c) in cells})
        print(f"\n=== n={n}  ({args.block} blocks, {len(per_seed)} seeds) ===")
        print(f"{'K':>2} {'c':>2} {'diam':>4} {'exact':>6} {'within':>6} {'e2e':>6} {'disc':>6}")
        for K in Ks:
            for c in cs:
                seeds = [s for s, cl in per_seed.items() if (K, c) in cl]
                if not seeds:
                    continue
                cells = [per_seed[s][(K, c)] for s in seeds]
                ex = np.mean([x["exact"] for x in cells])
                wi = np.mean([x["within_block_reach"] for x in cells if x["within_block_reach"] is not None])
                e2 = np.mean([x["cross_by_gap"].get(str(K - 1), x["cross_by_gap"].get(K - 1))
                              for x in cells if x["cross_by_gap"]])
                di = np.mean([x["broken"]["discrimination"] for x in cells if x["broken"]])
                print(f"{K:>2} {c:>2} {cells[0]['max_dist']:>4} {ex:>6.3f} {wi:>6.3f} {e2:>6.3f} {di:>6.3f}")

        # --- COMPOSITION figure: cross_by_gap vs gap, one line per K (rep clique size) ---
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        cmap = plt.cm.viridis(np.linspace(0.1, 0.85, max(1, len(Ks))))
        for K, col in zip(Ks, cmap):
            gaps = list(range(1, K))
            ys = []
            for g in gaps:
                vals = []
                for s, cl in per_seed.items():
                    cell = cl.get((K, args.clique_size))
                    if cell and cell["cross_by_gap"]:
                        v = cell["cross_by_gap"].get(str(g), cell["cross_by_gap"].get(g))
                        if v is not None:
                            vals.append(v)
                ys.append(np.mean(vals) if vals else np.nan)
            if np.isfinite(ys).any():
                ax.plot(gaps, ys, "o-", color=col, label=f"K={K}")
        ax.axhline(1.0, ls=":", c="gray", lw=0.8)
        ax.set_xlabel("bridges crossed (block gap g)"); ax.set_ylabel("cross-block accuracy")
        ax.set_ylim(0, 1.05); ax.set_xticks(range(1, max(Ks)))
        ax.set_title(f"n={n}, clique size c={args.clique_size}")
        ax.legend(fontsize=8); fig.tight_layout()
        fp = FIGDIR / f"clique_chain{tag}_composition_n{n}.png"
        fig.savefig(fp, dpi=150); plt.close(fig); print(f"  saved {fp}")

        # --- LENGTH + GUARD figure ---
        exK = mean_over_seeds(per_seed, lambda c: c["exact"])
        e2eK = mean_over_seeds(per_seed, lambda c: (c["cross_by_gap"].get(str(c["n_cliques"] - 1),
                               c["cross_by_gap"].get(c["n_cliques"] - 1)) if c["cross_by_gap"] else None))
        discK = mean_over_seeds(per_seed, lambda c: (c["broken"]["discrimination"] if c["broken"] else None))
        cc = args.clique_size
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        kk = [K for K in Ks if (K, cc) in exK]
        if kk:
            ax.plot(kk, [exK[(K, cc)] for K in kk], "s-", label="whole-chain exact")
            ax.plot(kk, [e2eK[(K, cc)] for K in kk], "o-", label="end-to-end cross acc")
            ax.plot(kk, [discK[(K, cc)] for K in kk], "^--", label="intact/broken discrim.")
            ax.axhline(0.5, ls=":", c="gray", lw=0.8)
            ax.set_xlabel("number of cliques in the chain K"); ax.set_ylabel("accuracy")
            ax.set_ylim(0, 1.05); ax.set_xticks(kk)
            ax.set_title(f"n={n}, clique size c={cc}")
            ax.legend(fontsize=8); fig.tight_layout()
            fp = FIGDIR / f"clique_chain{tag}_length_n{n}.png"
            fig.savefig(fp, dpi=150); plt.close(fig); print(f"  saved {fp}")


if __name__ == "__main__":
    main()
