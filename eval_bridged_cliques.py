"""Bridged-cliques vs. split-cliques: the core matrix-powering-vs-DFS probe (Report V).

Two labels on n nodes (left half / right half, each a clique of c = n//2 nodes):
  * BRIDGED  -- the two cliques joined by a SINGLE edge  -> one component, R all-ones.
  * SPLIT    -- the same two cliques, no bridge           -> two components, block R.
They differ by exactly one edge, and that edge makes a connection at distance <= 3.

What we measure on an existing checkpoint (eval-only, n auto-detected):

  1. DISCRIMINATION. Per graph we read whether the model thinks the two cliques are
     connected (mean predicted value over the cross block > 0.5) and compare to the
     true label, over a balanced bridged+split set. Matrix powering => ~perfect at any
     clique size (the bridge is <= 3 hops, inside the 9-hop capacity); a visit-bounded
     DFS => it confuses bridged with split, and worse as the cliques grow.

  2. PER-BLOCK exact-match (within-A, within-B, cross), the "where did it stop"
     read: a model that explores part of a component gets the within-clique pairs
     right and the cross pairs wrong (it never crossed the bridge), and may resolve
     one clique better than the other (asymmetry) -- a DFS signature, observable here
     because the RoBERTa linear read-out is NOT symmetrised.

  3. CLIQUE-SIZE SWEEP. Cross-block accuracy vs the clique size c, for the model and
     for the two reference oracles (matrix power: flat ~1; bounded DFS: falls with c).
     Matrix powering predicts a flat curve in c; DFS predicts a falling one.

  4. ORACLE AGREEMENT. On the entries where the matrix-power and bounded-DFS oracles
     DISAGREE, which one does the model follow?

  python eval_bridged_cliques.py --checkpoint runs/.../last.pt \
      --output_dir runs/.../bridged_cliques
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import (add_self_loops, compute_connectivity_matrix,
                  generate_bridged_cliques_graph, generate_split_cliques_graph)
from dfs_oracle import (matrix_power_connectivity, bounded_bfs_connectivity,
                        bounded_dfs_connectivity)
from eval_families import load_model, predict


def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def build(label, n, c, size, seed):
    """Return xs (with self-loops), ys (true R), roles (0=cliqueA,1=cliqueB,2=pad),
    and the raw no-loop adjacencies (for the oracles). label in {bridged, split}."""
    gen = generate_bridged_cliques_graph if label == "bridged" else generate_split_cliques_graph
    rng = np.random.default_rng(seed)
    role0 = np.array([0] * c + [1] * c + [2] * (n - 2 * c), dtype=np.int64)
    xs = np.empty((size, n, n), np.float32)
    ys = np.empty((size, n, n), np.int8)
    adj = np.empty((size, n, n), np.float32)
    roles = np.empty((size, n), np.int64)
    for i in range(size):
        a = gen(n, clique_size=c)
        p = rng.permutation(n)
        a = a[np.ix_(p, p)]
        adj[i] = a
        xs[i] = add_self_loops(a)
        ys[i] = compute_connectivity_matrix(a).astype(np.int8)
        roles[i] = role0[p]
    return xs, ys, roles, adj


def block_masks(roles, n):
    """Per-graph boolean masks for the within-A, within-B and cross (A<->B) blocks,
    off-diagonal only. roles: (G, n)."""
    G = roles.shape[0]
    A = roles == 0; B = roles == 1
    offdiag = ~np.eye(n, dtype=bool)[None].repeat(G, 0)
    within_A = A[:, :, None] & A[:, None, :] & offdiag
    within_B = B[:, :, None] & B[:, None, :] & offdiag
    cross = (A[:, :, None] & B[:, None, :]) | (B[:, :, None] & A[:, None, :])
    return within_A, within_B, cross, offdiag


def acc_on(mask, pred, ys):
    return float((pred[mask] == ys[mask]).mean()) if mask.any() else None


def discrimination(pred, roles, n, true_is_bridged):
    """Per graph: predicted_bridged = (cross block mostly predicted connected).
    Returns the per-graph boolean predictions for label = bridged."""
    A = roles == 0; B = roles == 1
    cross = (A[:, :, None] & B[:, None, :]) | (B[:, :, None] & A[:, None, :])
    G = pred.shape[0]
    pred_bridged = np.empty(G, bool)
    for g in range(G):
        m = cross[g]
        pred_bridged[g] = bool(pred[g][m].mean() > 0.5) if m.any() else False
    return pred_bridged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--sweep_step", type=int, default=1,
                    help="clique-size sweep granularity (c = 2,2+step,...,n//2)")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n; L = mcfg.n_layers
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} L={L} device={dev}")

    res = {"checkpoint": str(args.checkpoint), "arch": arch, "readout": readout,
           "n": n, "n_layers": L, "n_graphs": args.n_graphs}

    # ---- 1+2: main families at full size c=n//2 (no padding) -------------------
    c_full = n // 2
    main = {}
    preds_main = {}
    for label in ("bridged", "split"):
        xs, ys, roles, adj = build(label, n, c_full, args.n_graphs, args.seed)
        pred = predict(model, xs, dev)
        preds_main[label] = (pred, ys, roles, adj)
        wA, wB, cross, off = block_masks(roles, n)
        eq = (pred == ys)
        G = pred.shape[0]
        exact = float(eq.reshape(G, -1).all(1).mean())
        pairwise = float((eq & off).reshape(G, -1).sum(1).mean() / off.reshape(G, -1).sum(1).mean())
        # per-clique asymmetry on the cross block: rows-A->cols-B vs rows-B->cols-A
        A = roles == 0; B = roles == 1
        cross_AB = A[:, :, None] & B[:, None, :]
        cross_BA = B[:, :, None] & A[:, None, :]
        # directional disagreement on cross pairs (R_ij vs R_ji) -- the DFS asymmetry
        asym = float((pred[cross_AB] != pred.transpose(0, 2, 1)[cross_AB]).mean()) if cross_AB.any() else None
        main[label] = {
            "exact": exact, "pairwise": pairwise,
            "within_A_acc": acc_on(wA, pred, ys),
            "within_B_acc": acc_on(wB, pred, ys),
            "cross_acc": acc_on(cross, pred, ys),
            "cross_AB_acc": acc_on(cross_AB, pred, ys),
            "cross_BA_acc": acc_on(cross_BA, pred, ys),
            "cross_pred_connected_rate": float(pred[cross].mean()) if cross.any() else None,
            "cross_dir_disagree_rate": asym,
        }
        print(f"  [{label}] exact={exact:.3f} withinA={main[label]['within_A_acc']:.3f} "
              f"withinB={main[label]['within_B_acc']:.3f} cross={main[label]['cross_acc']:.3f} "
              f"crossPredConn={main[label]['cross_pred_connected_rate']:.3f}")
    res["main_c_full"] = {"clique_size": c_full, **main}

    # discrimination over the balanced bridged+split set
    pb_bridged = discrimination(preds_main["bridged"][0], preds_main["bridged"][2], n, True)
    pb_split = discrimination(preds_main["split"][0], preds_main["split"][2], n, False)
    disc_acc = float((pb_bridged.sum() + (~pb_split).sum()) / (len(pb_bridged) + len(pb_split)))
    res["discrimination_c_full"] = {
        "accuracy": disc_acc,
        "bridged_called_connected_rate": float(pb_bridged.mean()),
        "split_called_connected_rate": float(pb_split.mean()),
    }
    print(f"  discrimination acc={disc_acc:.3f} "
          f"(bridged->conn {pb_bridged.mean():.2f}, split->conn {pb_split.mean():.2f})")

    # ---- 3: clique-size sweep -------------------------------------------------
    cs = list(range(2, c_full + 1, max(1, args.sweep_step)))
    if cs[-1] != c_full:
        cs.append(c_full)
    sweep = {"clique_sizes": cs, "model_cross_acc": [], "model_disc_acc": [],
             "oracle_mp_cross_acc": [], "oracle_bfs_cross_acc": [],
             "oracle_dfs_cross_acc": [], "oracle_bounded_budget": []}
    size_s = max(400, args.n_graphs // 4)
    for c in cs:
        xb, yb, rb, ab = build("bridged", n, c, size_s, args.seed + c)
        xs2, ys2, rs2, as2 = build("split", n, c, size_s, args.seed + 1000 + c)
        pb = predict(model, xb, dev)
        ps = predict(model, xs2, dev)
        _, _, crossb, _ = block_masks(rb, n)
        model_cross = acc_on(crossb, pb, yb)            # bridged cross == reach the bridge
        # discrimination at this c
        d_b = discrimination(pb, rb, n, True); d_s = discrimination(ps, rs2, n, False)
        disc_c = float((d_b.sum() + (~d_s).sum()) / (len(d_b) + len(d_s)))
        # oracle cross accuracy on the SAME bridged graphs. The bounded-traversal
        # budget is the near-clique size c ("explore one clique, then stop").
        mp = np.stack([matrix_power_connectivity(ab[i], L) for i in range(len(ab))])
        bfs = np.stack([bounded_bfs_connectivity(ab[i], c) for i in range(len(ab))])
        dfs = np.stack([bounded_dfs_connectivity(ab[i], c) for i in range(len(ab))])
        sweep["model_cross_acc"].append(model_cross)
        sweep["model_disc_acc"].append(disc_c)
        sweep["oracle_mp_cross_acc"].append(acc_on(crossb, mp, yb))
        sweep["oracle_bfs_cross_acc"].append(acc_on(crossb, bfs, yb))
        sweep["oracle_dfs_cross_acc"].append(acc_on(crossb, dfs, yb))
        sweep["oracle_bounded_budget"].append(c)
        print(f"  c={c:2d}  model cross={model_cross:.3f} disc={disc_c:.3f} | "
              f"MP={sweep['oracle_mp_cross_acc'][-1]:.2f} "
              f"BFS(b=c)={sweep['oracle_bfs_cross_acc'][-1]:.2f} "
              f"DFS(b=c)={sweep['oracle_dfs_cross_acc'][-1]:.2f}")
    res["clique_size_sweep"] = sweep

    # ---- 4: oracle agreement on the discriminating entries (bridged, c=full) --
    pred_b, ys_b, roles_b, adj_b = preds_main["bridged"]
    cap = min(pred_b.shape[0], 500)                 # the budget scan is pure-Python BFS
    pred_b, adj_b = pred_b[:cap], adj_b[:cap]
    G = pred_b.shape[0]
    off = ~np.eye(n, dtype=bool)
    # scan the bounded-BFS budget; report agreement of the model with the matrix-power
    # oracle and with bounded-BFS(b), both overall and restricted to the entries where
    # the two oracles DISAGREE (the discriminating pairs -- the cross block).
    mp = np.stack([matrix_power_connectivity(adj_b[i], L) for i in range(G)])
    budgets = list(range(1, 2 * c_full + 1, max(1, args.sweep_step)))
    agree = {"budgets": budgets, "model_vs_mp_overall": None,
             "model_vs_bfs_overall": [], "disagree_frac": [],
             "model_follows_mp_on_disagree": [], "model_follows_bfs_on_disagree": []}
    om = off[None].repeat(G, 0)
    agree["model_vs_mp_overall"] = float((pred_b[om] == mp[om]).mean())
    for b in budgets:
        bfs = np.stack([bounded_bfs_connectivity(adj_b[i], b) for i in range(G)])
        agree["model_vs_bfs_overall"].append(float((pred_b[om] == bfs[om]).mean()))
        dis = om & (mp != bfs)
        agree["disagree_frac"].append(float(dis.mean()))
        if dis.any():
            agree["model_follows_mp_on_disagree"].append(float((pred_b[dis] == mp[dis]).mean()))
            agree["model_follows_bfs_on_disagree"].append(float((pred_b[dis] == bfs[dis]).mean()))
        else:
            agree["model_follows_mp_on_disagree"].append(None)
            agree["model_follows_bfs_on_disagree"].append(None)
    res["oracle_agreement_bridged"] = agree

    (out / "bridged_cliques.json").write_text(json.dumps(res, indent=2))
    print(f"  saved -> {out}/bridged_cliques.json")


if __name__ == "__main__":
    main()
