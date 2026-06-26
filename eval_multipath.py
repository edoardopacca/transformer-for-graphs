"""Report VI, Thread A1 --- the multipath probe (eval-only).

Generalises Report IV's confound-free parallel-paths probe. Two terminals s, t are
joined by k internally-disjoint paths of length ell (so dist(s,t)=ell is FIXED while
the number of routes k varies); terminals padded to a fixed degree, canvas filled
sparsely (no isolated-padding confound). We sweep (k, ell) and, for each cell, report
NOT just the terminal-pair accuracy but the full picture the report needs:

  * pair_acc        -- the headline: fraction with R-hat[s,t]=1 (the (a,b) pair).
  * matrix_exact    -- whole-matrix exact match (context: is the rest of R right?).
  * matrix_pairwise -- off-diagonal pairwise accuracy.
  * active_pairwise -- pairwise accuracy restricted to the active multipath component
                       (all those pairs are truly connected).
  * MECHANISM: per graph, how many of the k routes are "intact" (all their internal
    nodes are called connected to BOTH s and t). This tells whether the model declares
    (a,b) connected because it propagated along a SINGLE resolved route (n_intact>=1
    while others are wrong) or via the aggregate of several partial routes (pair right
    with n_intact=0). We dump the n_intact histogram and pair-correct counts per
    n_intact so the analysis can answer "single path vs multipath".

Eval-only on existing checkpoints; no training. Works at any n (cells that do not fit
are skipped). Primary read = pair_acc, always reported next to the matrix metrics.

    python eval_multipath.py --checkpoint runs/.../last.pt \
        --output_dir runs/report6/multipath/<tag>
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from data import (add_self_loops, compute_connectivity_matrix,
                  generate_multipath_graph, permute_with_meta)
from eval_families import load_model, predict


def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def eval_cell(model, dev, n, k, ell, rng, n_graphs, term_deg):
    """Build n_graphs clean multipath graphs with k full routes of length ell, predict,
    and return the metric dict for this (k, ell) cell (or None if it does not fit)."""
    xs, ys, metas = [], [], []
    for _ in range(n_graphs):
        r = generate_multipath_graph(n, k, ell, rng, n_trunc=0, term_deg=term_deg)
        if r is None:
            return None                       # infeasible at this n
        adj, meta = permute_with_meta(*r, rng)
        xs.append(add_self_loops(adj)); ys.append(compute_connectivity_matrix(adj))
        metas.append(meta)
    xs = np.stack(xs).astype(np.float32); ys = np.stack(ys).astype(np.int8)
    pred = predict(model, xs, dev)
    ng = len(xs)
    g = np.arange(ng)
    sp = np.array([m["s"] for m in metas]); tp = np.array([m["t"] for m in metas])

    pair_ok = (pred[g, sp, tp] == 1)
    pair_acc = float(pair_ok.mean())
    eq = (pred == ys)
    offdiag = ~np.eye(n, dtype=bool)                       # (n, n)
    matrix_exact = float(eq.reshape(ng, -1).all(1).mean())
    matrix_pw = float((eq & offdiag[None]).sum() / (ng * n * (n - 1)))

    # active-component pairwise (all those pairs are truly connected -> target 1)
    active_correct = 0; active_total = 0
    n_intact = np.zeros(ng, dtype=np.int64)
    reach_s_sum = 0.0; reach_t_sum = 0.0; reach_cnt = 0
    for i, m in enumerate(metas):
        active = [m["s"], m["t"]] + [x for pp in m["full_paths"] for x in pp] + m["leaves"]
        sub = pred[i][np.ix_(active, active)]
        active_correct += int((sub == 1).sum()); active_total += sub.size
        s, t = m["s"], m["t"]
        for pp in m["full_paths"]:
            if not pp:
                n_intact[i] += 1; continue
            to_s = pred[i][s, pp] == 1
            to_t = pred[i][t, pp] == 1
            reach_s_sum += float(to_s.mean()); reach_t_sum += float(to_t.mean()); reach_cnt += 1
            if to_s.all() and to_t.all():
                n_intact[i] += 1
    active_pw = float(active_correct / max(1, active_total))
    hist = [int((n_intact == j).sum()) for j in range(k + 1)]
    pair_ok_by_intact = [int(pair_ok[n_intact == j].sum()) for j in range(k + 1)]
    return {"k": k, "ell": ell, "n_graphs": ng,
            "pair_acc": round(pair_acc, 4),
            "matrix_exact": round(matrix_exact, 4),
            "matrix_pairwise": round(matrix_pw, 4),
            "active_pairwise": round(active_pw, 4),
            "mean_n_intact": round(float(n_intact.mean()), 4),
            "n_intact_hist": hist,                 # graphs with 0,1,..,k intact routes
            "pair_correct_by_n_intact": pair_ok_by_intact,
            "mean_reach_to_s": round(reach_s_sum / max(1, reach_cnt), 4),
            "mean_reach_to_t": round(reach_t_sum / max(1, reach_cnt), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=400)
    ap.add_argument("--term_deg", type=int, default=4)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--ells", type=int, nargs="+", default=[3, 5, 7, 9, 11, 13, 15])
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")
    rng = np.random.default_rng(args.seed)

    cells = []
    for ell in args.ells:
        for k in args.ks:
            c = eval_cell(model, dev, n, k, ell, rng, args.n_graphs, args.term_deg)
            if c is None:
                continue
            cells.append(c)
            print(f"  k={k} ell={ell:2d}: pair={c['pair_acc']:.3f} "
                  f"mat_exact={c['matrix_exact']:.3f} mat_pw={c['matrix_pairwise']:.3f} "
                  f"n_intact={c['mean_n_intact']:.2f}/{k} (hist {c['n_intact_hist']})",
                  flush=True)

    res = {"checkpoint": str(args.checkpoint), "arch": arch, "readout": readout, "n": n,
           "term_deg": args.term_deg, "n_graphs": args.n_graphs, "cells": cells}
    (out / "multipath.json").write_text(json.dumps(res))
    print(f"  saved -> {out}/multipath.json")


if __name__ == "__main__":
    main()
