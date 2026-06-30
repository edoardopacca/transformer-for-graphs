"""Report VI, Thread C.2 -- thicker bridges and different blocks (local, no GPU).

Reads runs/report6/thick_bridges/n{N}_{family}_bw{W}_seed{S}/bridged_cliques.json (the
eval_bridged_cliques.py clique-size sweep) and overlays, per n, the model cross-block
accuracy vs clique/block size for:
  * complete cliques joined by a bridge of width w in {1,2,3} (does a THICKER bridge --
    redundant hand-off edges, like Thread A's parallel routes -- change anything when the
    cross distance is already <= 3, inside capacity?), and
  * dense ER blocks joined by a single bridge (a HELD-OUT block type: does the learned
    clique-bridge skill transfer to a different dense region?).
Mean over seeds (thick), per seed (thin). The matrix-power oracle is flat at 1.0 (bridge
<= 3 hops) and drawn as a reference.

    python plot_thick_bridges.py
"""
import glob, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG_RE = re.compile(r"n(\d+)_(cliques|blocks)_bw(\d+)_seed(\d+)$")
FIGDIR = Path("runs/report6/report6_figs")


def load():
    """-> {n: {(family, bw): {seed: (clique_sizes, model_cross_acc)}}}."""
    data = defaultdict(lambda: defaultdict(dict))
    for jf in sorted(glob.glob("runs/report6/thick_bridges/*/bridged_cliques.json")):
        m = TAG_RE.search(Path(jf).parent.name)
        if not m:
            continue
        n, fam, bw, seed = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        d = json.loads(Path(jf).read_text())
        sw = d["clique_size_sweep"]
        data[n][(fam, bw)][seed] = (sw["clique_sizes"], sw["model_cross_acc"])
    return data


def main():
    data = load()
    if not data:
        print("no thick_bridges json -- pull runs/report6 first."); return
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # one colour per condition; cliques bw 1/2/3 in a blue ramp, ER blocks in red.
    style = {("cliques", 1): ("#1f4e8c", "single bridge, cliques"),
             ("cliques", 2): ("#3a8fd0", "double bridge, cliques"),
             ("cliques", 3): ("#9ecae1", "triple bridge, cliques"),
             ("blocks", 1):  ("#c0392b", "single bridge, dense ER blocks")}

    for n in sorted(data):
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        for key in [("cliques", 1), ("cliques", 2), ("cliques", 3), ("blocks", 1)]:
            if key not in data[n]:
                continue
            col, lab = style[key]
            seeds = data[n][key]
            cs = seeds[next(iter(seeds))][0]
            mat = np.array([acc for (_, acc) in seeds.values()])
            for row in mat:                                   # thin per-seed
                ax.plot(cs, row, color=col, lw=0.6, alpha=0.30)
            ax.plot(cs, mat.mean(0), "o-", color=col, lw=2.0, ms=4, label=lab)
        ax.axhline(1.0, ls=":", c="green", lw=1.0)            # matrix-power oracle
        ax.text(cs[1], 1.01, "matrix-power oracle", color="green", fontsize=8)
        ax.set_xlabel("block size $c$ (nodes per side)")
        ax.set_ylabel("cross-block accuracy (bridge propagated?)")
        ax.set_ylim(0, 1.08); ax.set_xticks(cs)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(f"$n={n}$: carrying one bridge across a dense region")
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fp = FIGDIR / f"thick_bridges_n{n}.png"
        fig.savefig(fp, dpi=150); plt.close(fig); print(f"  saved {fp}")


if __name__ == "__main__":
    main()
