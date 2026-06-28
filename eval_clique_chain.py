"""Report VI, Thread C.1 --- chained bridged cliques (eval-only).

Report V taught the model a single-bridge "iff" decision: two cliques are one component
iff the bridge edge is there. This probe asks whether that local rule COMPOSES along a
chain clique--bridge--clique--bridge--... : the model must find, at each block, the node
that hands off to the next block, and carry the connection across every bridge in turn.
The bridged-cliques eval is wired for exactly two blocks; this one handles K>=2 blocks.

For each cell (n_cliques K, clique_size c) we build permuted K-block chains and read,
NEVER collapsed to one number:

  * exact / chain_block_exact -- whole-matrix exact match, and the fraction of graphs whose
                       entire (single-component) chain is pairwise correct.
  * within_block_reach -- within-block pairwise accuracy (target 1; the easy part, dist 1).
  * cross_by_gap[g]   -- pairwise accuracy on pairs whose blocks are g bridges apart
                       (g=1..K-1, target 1), pooled over block index. This is the heart:
                       does the connection propagate across g successive hand-offs, or
                       decay with the number of links? cross_by_gap[K-1] is end-to-end.
  * per_link[l]       -- accuracy across the single bridge between block l and l+1 (to spot
                       a specific weak hand-off, not just the pooled gap curve).
  * a BROKEN-chain control (one middle bridge dropped -> two components): cut accuracy
                       across the break (target 0) + a discrimination accuracy over
                       intact-vs-broken, so a flat all-connected prediction cannot pass.
  * max_dist / within_capacity -- the chain diameter (end-to-end distance grows ~2 hops per
                       added block); cells past the 3^L=9 capacity are FLAGGED, since a
                       failure there is the distance wall, not the unseen composition.

Eval-only on existing checkpoints; no training. Works at any n; cells with K*c > n or
(by default) diameter > 9 are skipped.

    python eval_clique_chain.py --checkpoint runs/.../last.pt \
        --output_dir runs/report6/clique_chain/<tag>
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from data import (add_self_loops, compute_connectivity_matrix,
                  compute_all_pairs_shortest_paths, generate_clique_chain_graph)
from eval_families import load_model, predict

CAP = 9  # 3^L at L=2


def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def _roles(n, c, K):
    """Block index per node: 0..K-1 for the chain blocks, K for isolated padding."""
    r = np.full(n, K, dtype=np.int64)
    for i in range(K):
        r[i * c:(i + 1) * c] = i
    return r


def _predict_remapped(model, dev, base_adj, rng, n_graphs):
    """Predict on n_graphs permuted copies and remap each prediction back to base order."""
    n = base_adj.shape[0]
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
    return pred


def eval_cell(model, dev, n, c, K, rng, n_graphs, bridge_width, block):
    """Metrics for one (n_cliques K, clique_size c) cell. The ER block kind is random per
    graph, so we rebuild a fresh base graph (and its true R / roles) for every sample;
    the complete-clique kind is fixed, so one base graph is reused."""
    roles = _roles(n, c, K)
    R = roles[:, None]; Rj = roles[None, :]
    blockpair = (R < K) & (Rj < K)
    within = blockpair & (R == Rj) & ~np.eye(n, dtype=bool)

    # --- intact chain (one component) ---
    if block == "clique":
        base = generate_clique_chain_graph(n, c, K, rng, bridge_width, block)
        base_y = compute_connectivity_matrix(base).astype(np.int8)
        dist = compute_all_pairs_shortest_paths(base)
        pred = _predict_remapped(model, dev, base, rng, n_graphs)
        eq = (pred == base_y[None])
        max_dist = int(dist[dist > 0].max()) if (dist > 0).any() else 0
    else:  # er blocks: structure varies per graph -> remap each against its own R
        eqs = []; prs = []; max_dist = 0
        for _ in range(n_graphs):
            b = generate_clique_chain_graph(n, c, K, rng, bridge_width, block)
            y = compute_connectivity_matrix(b).astype(np.int8)
            d = compute_all_pairs_shortest_paths(b)
            max_dist = max(max_dist, int(d[d > 0].max()) if (d > 0).any() else 0)
            pr = _predict_remapped(model, dev, b, rng, 1)[0]
            eqs.append(pr == y); prs.append(pr)
        eq = np.stack(eqs); pred = np.stack(prs)
    ng = eq.shape[0]

    exact = float(eq.reshape(ng, -1).all(1).mean())
    chain_block_exact = float(eq[:, blockpair].all(1).mean())
    within_reach = float(eq[:, within].mean()) if within.any() else None
    cross_by_gap = {}
    for g in range(1, K):
        m = blockpair & (np.abs(R - Rj) == g)
        cross_by_gap[g] = round(float(eq[:, m].mean()), 4) if m.any() else None
    per_link = []
    for l in range(K - 1):
        m = ((R == l) & (Rj == l + 1)) | ((R == l + 1) & (Rj == l))
        per_link.append(round(float(eq[:, m].mean()), 4))
    # the model's own "connected?" call end-to-end (mean over block0<->blockK-1 pairs)
    e2e = blockpair & (np.abs(R - Rj) == K - 1)

    # --- broken chain control: drop one middle bridge -> two components ---
    broken = None
    if K >= 2:
        l = (K - 1) // 2       # left comp = blocks 0..l, right = blocks l+1..K-1
        left = (roles <= l) & (roles < K); right = (roles > l) & (roles < K)
        cut = (left[:, None] & right[None, :]) | (right[:, None] & left[None, :])
        if block == "clique":
            bb = generate_clique_chain_graph(n, c, K, rng, bridge_width, block, broken_link=l)
            yb = compute_connectivity_matrix(bb).astype(np.int8)
            pb = _predict_remapped(model, dev, bb, rng, n_graphs)
            eqb = (pb == yb[None])
            predb = pb
        else:
            eqbs = []; predbs = []
            for _ in range(n_graphs):
                bb = generate_clique_chain_graph(n, c, K, rng, bridge_width, block, broken_link=l)
                yb = compute_connectivity_matrix(bb).astype(np.int8)
                pr = _predict_remapped(model, dev, bb, rng, 1)[0]
                eqbs.append(pr == yb); predbs.append(pr)
            eqb = np.stack(eqbs); predb = np.stack(predbs)
        ngb = eqb.shape[0]
        cut_acc = float(eqb[:, cut].mean()) if cut.any() else None
        # discrimination: intact should read connected end-to-end, broken should read cut
        intact_conn = np.array([pred[g][e2e].mean() > 0.5 for g in range(ng)]) if e2e.any() else np.ones(ng, bool)
        broken_cut = np.array([predb[g][cut].mean() > 0.5 for g in range(ngb)]) if cut.any() else np.zeros(ngb, bool)
        disc = float((intact_conn.sum() + (~broken_cut).sum()) / (ng + ngb))
        broken = {"broken_link": int(l), "cut_acc": round(cut_acc, 4) if cut_acc is not None else None,
                  "cut_pred_connected_rate": round(float(broken_cut.mean()), 4),
                  "intact_pred_connected_rate": round(float(intact_conn.mean()), 4),
                  "discrimination": round(disc, 4)}

    return {"n_cliques": K, "clique_size": c, "n_graphs": ng,
            "max_dist": max_dist, "within_capacity": bool(max_dist <= CAP),
            "exact": round(exact, 4), "chain_block_exact": round(chain_block_exact, 4),
            "within_block_reach": (None if within_reach is None else round(within_reach, 4)),
            "cross_by_gap": cross_by_gap, "per_link": per_link, "broken": broken}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=400)
    ap.add_argument("--n_cliques", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--clique_sizes", type=int, nargs="+", default=[3, 4, 5, 6, 8])
    ap.add_argument("--bridge_width", type=int, default=1)
    ap.add_argument("--block", default="clique", choices=["clique", "er"],
                    help="'clique' (complete blocks, C.1) or 'er' (dense ER blocks, C.2 chained)")
    ap.add_argument("--keep_beyond_capacity", action="store_true",
                    help="also evaluate cells whose chain diameter exceeds 9 (flagged, off by default)")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n; L = mcfg.n_layers
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} L={L} "
          f"block={args.block} bridge_width={args.bridge_width} device={dev}")
    rng = np.random.default_rng(args.seed)

    cells = []
    for K in sorted(set(args.n_cliques)):
        for c in sorted(set(args.clique_sizes)):
            if K < 2 or c < 2 or K * c > n:
                continue
            cell = eval_cell(model, dev, n, c, K, rng, args.n_graphs, args.bridge_width, args.block)
            if not cell["within_capacity"] and not args.keep_beyond_capacity:
                print(f"  -- skip K={K} c={c} (diameter {cell['max_dist']} > {CAP}, beyond capacity)")
                continue
            cells.append(cell)
            gap = cell["cross_by_gap"]
            e2e = gap.get(K - 1)
            disc = cell["broken"]["discrimination"] if cell["broken"] else None
            print(f"  K={K} c={c} d={cell['max_dist']:>2d} exact={cell['exact']:.3f} "
                  f"within={cell['within_block_reach']} e2e(gap{K-1})={e2e} "
                  f"disc={disc}", flush=True)

    res = {"checkpoint": str(args.checkpoint), "arch": arch, "readout": readout, "n": n,
           "n_layers": L, "block": args.block, "bridge_width": args.bridge_width,
           "n_graphs": args.n_graphs, "cells": cells}
    (out / "clique_chain.json").write_text(json.dumps(res))
    print(f"  saved -> {out}/clique_chain.json")


if __name__ == "__main__":
    main()
