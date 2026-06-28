"""Report VI, Thread B --- asymmetric two-chains (eval-only).

The puzzle from Report IV: splitting n nodes into two paths of very UNEQUAL length
(e.g. 4+36 at n=40) looked EASIER than a balanced split (e.g. 17+23). This probe turns
the split into an explicit, controlled knob: for each split (a, n-a) we feed the base
model two-chain graphs (one path of a nodes, one of n-a, no isolated padding) and read,
BY SPLIT a, the whole picture the report needs -- never collapsed to one number:

  * exact            -- whole-graph exact match (the headline the puzzle is about).
  * reach_long / reach_short -- within-component pairwise accuracy on the long / short
                       path (target 1); the long path is where the 3^L=9 wall bites.
  * cut              -- across-component pairwise accuracy (target 0): kept separate?
  * long_block_exact / short_block_exact / cut_block_exact -- fraction of graphs whose
                       long / short within-block / cross block is ENTIRELY correct ->
                       WHERE the exact-match breaks as the split changes.
  * per_dist_long    -- within-long-component reach by shortest-path distance d (the
                       full capacity profile: perfect <=9, a valley past it, recovery on
                       the farthest pairs). This is the mechanism: a near-full-length
                       path may recover end-to-end (long_block_exact high) while two
                       medium paths both sit in the valley (long_block_exact low). We
                       READ the mechanism off these curves, we do not assume it.

Eval-only on existing checkpoints; no training. Works at any n; the split is swept over
a = 1..n//2 by default. Primary read = exact by split, beside the component decomposition.

    python eval_asym_chains.py --checkpoint runs/.../last.pt \
        --output_dir runs/report6/asym_chains/<tag>
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from data import (add_self_loops, compute_connectivity_matrix,
                  compute_all_pairs_shortest_paths, generate_split_chains_graph)
from eval_families import load_model, predict


def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def eval_split(model, dev, n, a, rng, n_graphs):
    """Build n_graphs permuted two-chain graphs split (a, n-a), predict, and return the
    metric dict for this split. The graph is fixed up to a node permutation, so we fix
    the connectivity/distance/component structure ONCE (unpermuted) and remap each
    prediction back to that base order -- all metrics are then vectorised over graphs."""
    base_adj = generate_split_chains_graph(n, a)
    base_y = compute_connectivity_matrix(base_adj).astype(np.int8)
    base_dist = compute_all_pairs_shortest_paths(base_adj)
    # long = the larger component, short = the smaller (unpermuted indices)
    seg0, seg1 = np.arange(0, a), np.arange(a, n)
    L, S = (seg1, seg0) if (n - a) >= a else (seg0, seg1)

    xs = np.empty((n_graphs, n, n), np.float32)
    invs = []
    for i in range(n_graphs):
        p = rng.permutation(n)
        xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
        invs.append(np.argsort(p))                       # position of each base node
    pred_perm = predict(model, xs, dev)
    pred = np.empty_like(pred_perm)
    for i, inv in enumerate(invs):
        pred[i] = pred_perm[i][np.ix_(inv, inv)]         # back to base node order
    eq = (pred == base_y[None])                          # (ng, n, n)
    ng = n_graphs

    exact = float(eq.reshape(ng, -1).all(1).mean())
    # long within-component (both directions via the off-diagonal mask)
    eqL = eq[:, L][:, :, L]; offL = ~np.eye(len(L), dtype=bool)
    reach_long = float(eqL[:, offL].mean())
    long_block_exact = float(eqL[:, offL].all(1).mean())
    # short within-component (a single-node short path has no within pairs)
    if len(S) > 1:
        eqS = eq[:, S][:, :, S]; offS = ~np.eye(len(S), dtype=bool)
        reach_short = float(eqS[:, offS].mean())
        short_block_exact = float(eqS[:, offS].all(1).mean())
    else:
        reach_short = None; short_block_exact = 1.0
    # cut: all cross pairs between the two components (both directions)
    memb = np.zeros(n, dtype=np.int8); memb[L] = 1
    cmask = (memb[:, None] != memb[None, :])
    eqC = eq[:, cmask]
    cut = float(eqC.mean()); cut_block_exact = float(eqC.all(1).mean())
    # reach within the long component, by shortest-path distance d (the capacity profile)
    dL = base_dist[np.ix_(L, L)]
    per_dist_long = {}
    for d in range(1, len(L)):
        m = (dL == d)
        cnt = int(m.sum())                               # ordered pairs at distance d (per graph)
        if cnt == 0:
            continue
        per_dist_long[d] = [round(float(eqL[:, m].mean()), 4), cnt]
    return {"split": int(a), "short_len": int(len(S)), "long_len": int(len(L)),
            "n_graphs": ng, "exact": round(exact, 4),
            "reach_long": round(reach_long, 4),
            "reach_short": (None if reach_short is None else round(reach_short, 4)),
            "cut": round(cut, 4),
            "long_block_exact": round(long_block_exact, 4),
            "short_block_exact": round(short_block_exact, 4),
            "cut_block_exact": round(cut_block_exact, 4),
            "per_dist_long": per_dist_long}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=300)
    ap.add_argument("--splits", type=int, nargs="+", default=None,
                    help="short-component sizes to sweep; default 1..n//2")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    splits = args.splits if args.splits is not None else list(range(1, n // 2 + 1))
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")
    rng = np.random.default_rng(args.seed)

    cells = []
    for a in splits:
        if not 1 <= a <= n - 1:
            continue
        c = eval_split(model, dev, n, a, rng, args.n_graphs)
        cells.append(c)
        print(f"  split=({c['short_len']:>2d},{c['long_len']:<2d}) exact={c['exact']:.3f} "
              f"reach_long={c['reach_long']:.3f} cut={c['cut']:.3f} "
              f"longblock={c['long_block_exact']:.3f}", flush=True)

    res = {"checkpoint": str(args.checkpoint), "arch": arch, "readout": readout, "n": n,
           "n_graphs": args.n_graphs, "cells": cells}
    (out / "asym_chains.json").write_text(json.dumps(res))
    print(f"  saved -> {out}/asym_chains.json")


if __name__ == "__main__":
    main()
