"""Report IX, Thread A.4 -- K-component path unions beyond the training coverage (eval-only).

The online ``path_union`` training stream (Reports VI--VIII) draws its number of disjoint
path components uniformly from 1..4 -- so any test with 2, 3 or 4 total components is, by
construction, IN distribution in its *shape*, whatever its exact sizes. This script builds
the direct out-of-distribution generalisation test in the number of components: K=5, 6, 7
disjoint paths (never produced by the training stream), one LONG component plus K-1 SMALL
ones, with sizes chosen so the long component's internal diameter exceeds 18 (the doubled
capacity 2*3^L that Report VIII's endpoint-completion signal was shown to reach beyond, but
only ever in-distribution). If the endpoint trick generalises past the trained component
count, the long component should still resolve beyond distance 18 and the small components
should still be told apart from each other and from the long one; if it does not, reach should
fall back to the plain distance wall and/or the small-vs-small cut should degrade as K grows
past what training ever showed.

For a fixed K, every metric is reported PER (K, small_size) cell (never collapsed across
cells): whole-graph exact match; reach inside the long component, both an aggregate and split
by shortest-path-distance bucket (<=9 within capacity, 9<d<=18 within the doubled wall, >18
beyond it -- the decisive bucket); reach inside the small components (pooled, only if
small_size>1); cut between the long component and the small components (pooled, both
directions); and cut BETWEEN different small components (pooled over every pair) -- the direct
generalisation of the three-way falsification test's decisive "does the model conflate two
'other' components" column, now with more than two 'other' components than training ever
produced.

    python eval_multiway_split.py --checkpoint runs/.../last.pt \
        --output_dir runs/report9/multiway_split/<tag>
"""
import argparse, json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from data import (add_self_loops, compute_connectivity_matrix,
                  compute_all_pairs_shortest_paths, generate_multi_path_split_graph)
from eval_families import load_model, predict


def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def default_small_sizes(n, k, dist_cutoff, candidates=(2, 3, 4, 5, 6, 7, 8)):
    """Small-component sizes s (k-1 of them, plus one long component of size
    n-(k-1)*s) for which the long component's internal diameter (long_len-1)
    exceeds ``dist_cutoff`` -- the feasibility filter (mirrors the capacity-
    feasibility checks used throughout this project, e.g. the multipath probe
    of Report VI)."""
    out = []
    for s in candidates:
        long_len = n - (k - 1) * s
        if long_len >= 1 and (long_len - 1) > dist_cutoff:
            out.append(s)
    return out


def _pooled_within(eq, idx_groups):
    """Reach + block-exact pooled over within-group pairs of SEVERAL index groups
    (e.g. every small component's own internal pairs, treated as one pool)."""
    vals, blocks = [], []
    for idx in idx_groups:
        if len(idx) <= 1:
            continue
        e = eq[:, idx][:, :, idx]
        off = ~np.eye(len(idx), dtype=bool)
        vals.append(e[:, off].reshape(e.shape[0], -1))
        blocks.append(e[:, off].all(1))
    if not vals:
        return None, None
    allv = np.concatenate(vals, axis=1)
    allb = np.stack(blocks, axis=1).all(1)
    return float(allv.mean()), float(allb.mean())


def _pooled_cross(eq, pairs):
    """Cross reach (target 0) + block-exact pooled over SEVERAL (idx_a, idx_b) pairs
    of disjoint index groups, both directions."""
    vals, blocks = [], []
    for idx_a, idx_b in pairs:
        e = eq[:, idx_a][:, :, idx_b]
        e2 = eq[:, idx_b][:, :, idx_a]
        flat = np.concatenate([e.reshape(e.shape[0], -1), e2.reshape(e2.shape[0], -1)], axis=1)
        vals.append(flat)
        blocks.append(flat.all(1))
    if not vals:
        return None, None
    allv = np.concatenate(vals, axis=1)
    allb = np.stack(blocks, axis=1).all(1)
    return float(allv.mean()), float(allb.mean())


def eval_cell(model, dev, n, k, small_size, dist_cutoff, rng, n_graphs):
    long_len = n - (k - 1) * small_size
    sizes = (long_len,) + (small_size,) * (k - 1)
    base_adj = generate_multi_path_split_graph(n, sizes)
    base_y = compute_connectivity_matrix(base_adj).astype(np.int8)
    base_dist = compute_all_pairs_shortest_paths(base_adj)

    bounds = [0]
    for s in sizes:
        bounds.append(bounds[-1] + s)
    comps = [np.arange(bounds[i], bounds[i + 1]) for i in range(len(sizes))]
    long_idx, small_idxs = comps[0], comps[1:]

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
    reach_long, block_long = _pooled_within(eq, [long_idx])
    reach_small, block_small = _pooled_within(eq, small_idxs)
    cut_long_small, cutblock_long_small = _pooled_cross(eq, [(long_idx, s) for s in small_idxs])
    small_pairs = list(combinations(small_idxs, 2))
    cut_small_small, cutblock_small_small = _pooled_cross(eq, small_pairs) if small_pairs else (None, None)

    dL = base_dist[np.ix_(long_idx, long_idx)]
    eqL = eq[:, long_idx][:, :, long_idx]
    per_dist_long = {}
    for d in range(1, len(long_idx)):
        m = (dL == d)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        per_dist_long[d] = [round(float(eqL[:, m].mean()), 4), cnt]
    near = (dL > 0) & (dL <= 9)
    mid = (dL > 9) & (dL <= dist_cutoff)
    far = (dL > dist_cutoff)
    reach_long_near = float(eqL[:, near].mean()) if near.any() else None
    reach_long_mid = float(eqL[:, mid].mean()) if mid.any() else None
    reach_long_far = float(eqL[:, far].mean()) if far.any() else None

    return {"k": int(k), "small_size": int(small_size), "long_len": int(long_len),
            "n_small_pairs": len(small_pairs), "n_graphs": ng,
            "exact": round(exact, 4),
            "reach_long": round(reach_long, 4), "block_long": round(block_long, 4),
            "reach_long_near_d9": (None if reach_long_near is None else round(reach_long_near, 4)),
            "reach_long_mid_9to18": (None if reach_long_mid is None else round(reach_long_mid, 4)),
            "reach_long_far_gt18": (None if reach_long_far is None else round(reach_long_far, 4)),
            "reach_small": (None if reach_small is None else round(reach_small, 4)),
            "block_small": (None if block_small is None else round(block_small, 4)),
            "cut_long_small": round(cut_long_small, 4),
            "cutblock_long_small": round(cutblock_long_small, 4),
            "cut_small_small": (None if cut_small_small is None else round(cut_small_small, 4)),
            "cutblock_small_small": (None if cutblock_small_small is None else round(cutblock_small_small, 4)),
            "per_dist_long": per_dist_long}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=300)
    ap.add_argument("--n_components", type=int, nargs="+", default=[5, 6, 7],
                    help="total number of disjoint path components K (never <5: 1..4 is "
                         "already covered by the path_union training stream)")
    ap.add_argument("--small_sizes", type=int, nargs="+", default=None,
                    help="override the small-component-size sweep for EVERY K (default: "
                         "auto, filtered so the long component's diameter exceeds "
                         "--dist_cutoff)")
    ap.add_argument("--dist_cutoff", type=int, default=18,
                    help="the long component's diameter must exceed this (default 18 = "
                         "the doubled capacity 2*3^L for L=2)")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")
    rng = np.random.default_rng(args.seed)

    cells = []
    for k in args.n_components:
        sizes = args.small_sizes if args.small_sizes is not None else \
            default_small_sizes(n, k, args.dist_cutoff)
        if not sizes:
            print(f"  k={k}: no feasible small size gives long diameter > {args.dist_cutoff} "
                  f"at n={n} -- skipped")
            continue
        for s in sizes:
            if n - (k - 1) * s < 1:
                print(f"  k={k} small={s}: infeasible (long_len<1) -- skipped")
                continue
            c = eval_cell(model, dev, n, k, s, args.dist_cutoff, rng, args.n_graphs)
            cells.append(c)
            print(f"  k={c['k']} small={c['small_size']:>2d} long={c['long_len']:<2d} "
                  f"exact={c['exact']:.3f} reach_long={c['reach_long']:.3f} "
                  f"far(>{args.dist_cutoff})={c['reach_long_far_gt18']} "
                  f"cut(long,small)={c['cut_long_small']:.3f} "
                  f"cut(small,small)={c['cut_small_small']}", flush=True)

    res = {"checkpoint": str(args.checkpoint), "arch": arch, "readout": readout, "n": n,
           "n_graphs": args.n_graphs, "dist_cutoff": args.dist_cutoff, "cells": cells}
    (out / "multiway_split.json").write_text(json.dumps(res))
    print(f"  saved -> {out}/multiway_split.json")


if __name__ == "__main__":
    main()
