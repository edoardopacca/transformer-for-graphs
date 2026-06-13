"""Confound-free parallel-paths probe (Report IV, Thread 2 follow-up).

The earlier parallel-paths probe (on the n=64 reach checkpoints) was inconclusive because
the construction was out-of-distribution: a few active nodes in a canvas of isolated
padding, with terminals whose degree grew with the number of paths. This version removes
those confounds and runs on the n=40 connectivity checkpoints:

  - DISTANCE fixed: the two terminals s, t are joined by k internally-disjoint paths, each
    of length ell, so dist(s,t) = ell for every k.
  - TERMINAL DEGREE fixed: each terminal is padded with leaf nodes to a constant degree
    (default 4), so it does not become a higher-degree hub as k grows.
  - DENSITY ~fixed and CANVAS FILLED: the remaining nodes are wired into a sparse path
    (no isolated padding), so the mean degree stays ~2 (like the path-union training) for
    every k.

Only k -- the number of parallel routes, i.e. the effective resistance ell/k between the
terminals -- varies. We measure the model's accuracy on the (s, t) pair (which is always
connected) as a function of k at fixed ell:
  * if difficulty is purely DISTANCE, the accuracy is flat in k;
  * if a BOTTLENECK / resistance adds difficulty, the accuracy rises with k.

Eval-only on existing checkpoints. No training.

    python eval_parallel_paths_clean.py --checkpoint runs/.../last.pt \
        --output_dir runs/.../parallel_paths_clean
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from data import add_self_loops
from eval_families import load_model, predict


def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def gen_pp_clean(n, k, ell, rng, term_deg=4):
    """Two terminals joined by k disjoint length-ell paths; terminals padded to a fixed
    degree with leaves; the rest of the canvas filled by a sparse path. Returns
    (adjacency without self-loops, s position, t position) after a random permutation,
    or None if it does not fit in n nodes."""
    need = 2 + k * (ell - 1) + 2 * max(0, term_deg - k)
    if need > n:
        return None
    a = np.zeros((n, n), np.float32)
    s, t = 0, 1
    cur = 2
    for _ in range(k):
        prev = s
        for _ in range(ell - 1):
            a[prev, cur] = a[cur, prev] = 1.0
            prev = cur; cur += 1
        a[prev, t] = a[t, prev] = 1.0
    for term in (s, t):
        for _ in range(max(0, term_deg - k)):
            a[term, cur] = a[cur, term] = 1.0; cur += 1
    rest = list(range(cur, n))
    for i in range(len(rest) - 1):
        a[rest[i], rest[i + 1]] = a[rest[i + 1], rest[i]] = 1.0
    p = rng.permutation(n); inv = np.argsort(p)
    return a[np.ix_(p, p)], int(inv[s]), int(inv[t])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=400)
    ap.add_argument("--term_deg", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")
    rng = np.random.default_rng(args.seed)

    KS = [1, 2, 3, 4]
    ELLS = [5, 7, 9, 11, 13, 15]
    cells = []
    for ell in ELLS:
        for k in KS:
            adjs, sp, tp = [], [], []
            for _ in range(args.n_graphs):
                r = gen_pp_clean(n, k, ell, rng, args.term_deg)
                if r is None:
                    break
                a, si, ti = r
                adjs.append(add_self_loops(a)); sp.append(si); tp.append(ti)
            if not adjs:
                continue  # infeasible at this n
            xs = np.stack(adjs).astype(np.float32)
            pred = predict(model, xs, dev)
            g = np.arange(len(adjs))
            acc = float((pred[g, np.array(sp), np.array(tp)] == 1).mean())
            cells.append({"k": k, "ell": ell, "term_acc": round(acc, 4),
                          "n_graphs": len(adjs)})
            print(f"  k={k} ell={ell}: term-pair acc={acc:.3f}  (n={len(adjs)})")

    res = {"checkpoint": str(args.checkpoint), "arch": arch, "readout": readout, "n": n,
           "term_deg": args.term_deg, "cells": cells}
    (out / "parallel_paths_clean.json").write_text(json.dumps(res))
    print(f"  saved -> {out}/parallel_paths_clean.json")


if __name__ == "__main__":
    main()
