"""Near-miss cut test: reach and cut are the same 3^L wall, seen from two sides.

Take a single chain on all n nodes, remove ONE edge so it splits into two
components, and look at the DISCONNECTED (cross-component) pairs. For each such pair
record how far apart the two nodes WOULD be if the edge were restored --- their
distance along the original chain --- the "near-miss distance". Then measure the cut
accuracy (fraction of disconnected pairs correctly called disconnected) as a
function of that distance.

This is the cut analogue of the reach-by-distance curve. The script also reports the
ordinary reach (within-component connected pairs) by distance on the SAME graphs, so
the two curves share one x-axis: a clean "two sides of the same wall" picture.

Prediction, if reach and cut are the same wall: the cut fails for near-miss
distances inside the reach radius (the model still feels the short path that almost
exists) and recovers beyond it --- the linear read-out turning over near 3^L and the
similarity read-out near 2*3^L.

Eval-only on an existing checkpoint (n auto-detected). No training.

    python eval_near_miss_cut.py --checkpoint runs/.../last.pt \
        --output_dir runs/.../near_miss_cut
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import add_self_loops, compute_connectivity_matrix
from eval_families import load_model, predict


def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def build_split_chains(n, n_graphs, rng):
    """Single chain 0..n-1 with one random edge removed -> two components.

    Returns, all in the PERMUTED node order the model sees:
      adjs : (G, n, n) adjacency without self-loops,
      pdist: (G, n, n) original-chain distance |pos_i - pos_j| (the near-miss
             distance for cross pairs, the true within-comp distance for same-comp
             pairs --- they coincide because each side is a contiguous sub-chain),
      same : (G, n, n) bool, True iff i,j in the same component (the target R).
    """
    adjs = np.zeros((n_graphs, n, n), np.float32)
    pdist = np.zeros((n_graphs, n, n), np.int64)
    same = np.zeros((n_graphs, n, n), bool)
    pos0 = np.arange(n)
    for g in range(n_graphs):
        k = int(rng.integers(0, n - 1))          # remove canonical edge (k, k+1)
        a = np.zeros((n, n), np.float32)
        idx = np.arange(n - 1)
        for i in idx:
            if i == k:
                continue
            a[i, i + 1] = a[i + 1, i] = 1.0
        comp = (pos0 > k).astype(np.int64)        # {0..k}=0, {k+1..n-1}=1
        p = rng.permutation(n)                     # node relabelling (no PE -> safe)
        adjs[g] = a[np.ix_(p, p)]
        cp = comp[p]; pp = pos0[p]
        pdist[g] = np.abs(pp[:, None] - pp[None, :])
        same[g] = cp[:, None] == cp[None, :]
    return adjs, pdist, same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")

    rng = np.random.default_rng(args.seed)
    adjs, pdist, same = build_split_chains(n, args.n_graphs, rng)
    xs = np.stack([add_self_loops(adjs[g]) for g in range(len(adjs))]).astype(np.float32)
    # sanity: target connectivity matches `same`
    ys = np.stack([compute_connectivity_matrix(adjs[g]) for g in range(len(adjs))]).astype(np.int8)
    assert (ys == same.astype(np.int8)).all(), "target/component bookkeeping mismatch"

    pred = predict(model, xs, dev)                # (G, n, n) in {0,1}

    offdiag = ~np.eye(n, dtype=bool)[None]
    cross = (~same) & offdiag                     # disconnected pairs (target 0)
    within = same & offdiag                       # connected pairs    (target 1)

    dmax = n - 1
    cut, reach = {}, {}
    for d in range(1, dmax + 1):
        at = pdist == d
        mc = cross & at
        mw = within & at
        if mc.sum() >= 50:
            cut[d] = {"cut_acc": float((pred[mc] == 0).mean()),
                      "false_connect": float((pred[mc] == 1).mean()),
                      "count": int(mc.sum())}
        if mw.sum() >= 50:
            reach[d] = {"reach_acc": float((pred[mw] == 1).mean()),
                        "count": int(mw.sum())}

    res = {"checkpoint": str(args.checkpoint), "arch": arch, "readout": readout,
           "n": n, "n_graphs": args.n_graphs,
           "cut_by_near_miss_distance": cut, "reach_by_distance": reach}
    (out / "near_miss_cut.json").write_text(json.dumps(res))
    print(f"  saved -> {out}/near_miss_cut.json")

    # quick per-checkpoint figure (the report figure pools seeds, built offline)
    fig, ax = plt.subplots(figsize=(8, 5))
    dr = sorted(reach); ax.plot(dr, [reach[d]["reach_acc"] for d in dr],
                                "-o", ms=3, color="C2", label="reach (connected pairs)")
    dc = sorted(cut); ax.plot(dc, [cut[d]["cut_acc"] for d in dc],
                              "-o", ms=3, color="C1", label="cut (disconnected, by near-miss dist)")
    for x, lab, col in [(9, "$3^L=9$", "gray"), (18, "$2\\cdot3^L=18$", "C0")]:
        if x <= dmax:
            ax.axvline(x, color=col, ls="--", lw=1)
    ax.set_xlabel("distance (reach) / near-miss distance (cut)")
    ax.set_ylabel("accuracy"); ax.set_ylim(-0.03, 1.05)
    ax.set_title(f"reach and cut vs distance  ({readout}, n={n})")
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(out / "near_miss_cut.png", dpi=140); plt.close(fig)
    print(f"  saved -> {out}/near_miss_cut.png")


if __name__ == "__main__":
    main()
