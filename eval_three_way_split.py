"""Report VII -- three-way split falsification test (eval-only).

Report VI, Thread B found that a two-chain split (a, n-a) is solved when a is
small (the model resolves the whole graph) but fails once a approaches capacity
(Report VI tab:basplit). One candidate mechanism (Report VII, hypothesis 2): the
model fully resolves whichever component is small enough to fit inside its
distance capacity, and DEFAULTS every other pair to "connected" -- a shortcut
that is right whenever exactly one component is small.

This script builds the decisive falsification test for that hypothesis: THREE
disjoint paths -- one small (within capacity, meant to be fully resolvable) and
two LARGE, individually-unresolvable, and mutually DISCONNECTED components. If
the "resolve-small-then-default" shortcut is what is happening, the model should
wrongly call the two large components connected to each other (both just read
as "not the small one"). If the model genuinely reasons about connectivity, the
two large components should be (correctly) called disconnected, or at least the
error pattern should differ.

    python eval_three_way_split.py --checkpoint runs/.../last.pt \
        --output_dir runs/report7/three_way/<tag>
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from data import (add_self_loops, compute_connectivity_matrix,
                  compute_all_pairs_shortest_paths, generate_three_way_split_graph)
from eval_families import load_model, predict


def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def _block_metrics(eq, idx):
    """within-component reach + fully-correct fraction for one component's node set."""
    if len(idx) <= 1:
        return None, 1.0
    e = eq[:, idx][:, :, idx]
    off = ~np.eye(len(idx), dtype=bool)
    return float(e[:, off].mean()), float(e[:, off].all(1).mean())


def _cross_metrics(eq, idx_a, idx_b):
    """across two disjoint node sets (both directions)."""
    e = eq[:, idx_a][:, :, idx_b]
    e2 = eq[:, idx_b][:, :, idx_a]
    both = np.concatenate([e.reshape(e.shape[0], -1), e2.reshape(e2.shape[0], -1)], axis=1)
    return float(both.mean()), float(both.all(1).mean())


def eval_split(model, dev, n, small_len, large_split, rng, n_graphs):
    base_adj = generate_three_way_split_graph(n, small_len, large_split)
    base_y = compute_connectivity_matrix(base_adj).astype(np.int8)
    base_dist = compute_all_pairs_shortest_paths(base_adj)
    bounds = (0, small_len, small_len + (large_split or (n - small_len) // 2), n)
    S = np.arange(bounds[0], bounds[1])
    L1 = np.arange(bounds[1], bounds[2])
    L2 = np.arange(bounds[2], bounds[3])

    xs = np.empty((n_graphs, n, n), np.float32)
    invs = []
    for i in range(n_graphs):
        p = rng.permutation(n)
        xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
        invs.append(np.argsort(p))
    pred_perm = predict(model, xs, dev)
    pred = np.empty_like(pred_perm)
    for i, inv in enumerate(invs):
        pred[i] = pred_perm[i][np.ix_(inv, inv)]
    eq = (pred == base_y[None])
    ng = n_graphs

    exact = float(eq.reshape(ng, -1).all(1).mean())
    reach_S, block_S = _block_metrics(eq, S)
    reach_L1, block_L1 = _block_metrics(eq, L1)
    reach_L2, block_L2 = _block_metrics(eq, L2)
    cut_S_L1, cutblock_S_L1 = _cross_metrics(eq, S, L1)
    cut_S_L2, cutblock_S_L2 = _cross_metrics(eq, S, L2)
    cut_L1_L2, cutblock_L1_L2 = _cross_metrics(eq, L1, L2)   # <-- the decisive pair

    dL1 = base_dist[np.ix_(L1, L1)]
    per_dist_L1 = {}
    eqL1 = eq[:, L1][:, :, L1]
    for d in range(1, len(L1)):
        m = (dL1 == d)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        per_dist_L1[d] = [round(float(eqL1[:, m].mean()), 4), cnt]

    return {"small_len": int(small_len), "large1_len": int(len(L1)), "large2_len": int(len(L2)),
            "n_graphs": ng, "exact": round(exact, 4),
            "reach_small": (None if reach_S is None else round(reach_S, 4)),
            "reach_large1": round(reach_L1, 4), "reach_large2": round(reach_L2, 4),
            "cut_small_large1": round(cut_S_L1, 4), "cut_small_large2": round(cut_S_L2, 4),
            "cut_large1_large2": round(cut_L1_L2, 4),
            "block_small": round(block_S, 4), "block_large1": round(block_L1, 4),
            "block_large2": round(block_L2, 4),
            "cutblock_small_large1": round(cutblock_S_L1, 4),
            "cutblock_small_large2": round(cutblock_S_L2, 4),
            "cutblock_large1_large2": round(cutblock_L1_L2, 4),
            "per_dist_large1": per_dist_L1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=300)
    ap.add_argument("--small_lens", type=int, nargs="+", default=None,
                    help="small-component sizes to sweep; default a spread within/near capacity")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    small_lens = args.small_lens if args.small_lens is not None else \
        sorted({s for s in (1, 2, 4, 7, 8, 10) if 1 <= s <= n - 2})
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")
    rng = np.random.default_rng(args.seed)

    cells = []
    for s in small_lens:
        c = eval_split(model, dev, n, s, None, rng, args.n_graphs)
        cells.append(c)
        print(f"  small={c['small_len']:>2d} large=({c['large1_len']},{c['large2_len']}) "
              f"exact={c['exact']:.3f} cut(L1,L2)={c['cut_large1_large2']:.3f} "
              f"cut(S,L1)={c['cut_small_large1']:.3f} cut(S,L2)={c['cut_small_large2']:.3f}",
              flush=True)

    res = {"checkpoint": str(args.checkpoint), "arch": arch, "readout": readout, "n": n,
           "n_graphs": args.n_graphs, "cells": cells}
    (out / "three_way_split.json").write_text(json.dumps(res))
    print(f"  saved -> {out}/three_way_split.json")


if __name__ == "__main__":
    main()
