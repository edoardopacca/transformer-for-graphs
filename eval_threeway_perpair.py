"""Report X -- 2026-08-31, Edoardo: investigate WHY cut on the plain disjoint (20,20,20)
split is 1.000 for some K=3 n=60 seeds but only 0.85-0.92 for others (eval_threeway_
splitchains.py only gives the pooled cut_12/13/23 numbers, no per-pair breakdown -- this
script fills that gap). Ground truth is exact and needs no BFS: every within-component pair
is connected (it's a path), every cross-component pair is disconnected, so per-pair accuracy
is just the real per-pair predicted-connected rate (correct for same-component pairs,
1-rate for cross-component pairs), averaged over many real relabelled test graphs -- same
mechanism as eval_n60_multipath_v2.py's per_pair_acc, applied to the UNMODIFIED disjoint
split instead of a multipath construction.

    python eval_threeway_perpair.py --checkpoint runs/.../last.pt \\
        --output_dir runs/report10/threeway_perpair/<tag> --n_graphs 300
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from data import add_self_loops, generate_multi_path_split_graph
from eval_families import load_model
from stagewise_diagnostics import run_with_stages, _device, _selftest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--sizes", type=int, nargs=3, default=[20, 20, 20])
    ap.add_argument("--n_graphs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")
    _selftest(model, dev, n)

    base_adj = generate_multi_path_split_graph(n, tuple(args.sizes))
    rng = np.random.default_rng(args.seed)

    pred_sum = np.zeros((n, n), dtype=np.float64)
    pred_count = 0
    B = 256
    for i in range(0, args.n_graphs, B):
        g = min(B, args.n_graphs - i)
        xs = np.empty((g, n, n), np.float32)
        invs = []
        for j in range(g):
            p = rng.permutation(n)
            xs[j] = add_self_loops(base_adj[np.ix_(p, p)])
            invs.append(np.argsort(p))
        xb = torch.from_numpy(xs).to(dev, torch.float32)
        with torch.no_grad():
            _, _, logits = run_with_stages(model, xb)
        logits_base = np.stack([logits[j][np.ix_(inv, inv)] for j, inv in enumerate(invs)])
        pred = logits_base > 0
        pred_sum += pred.sum(axis=0)
        pred_count += g
        print(f"  {i+g}/{args.n_graphs} graphs done", flush=True)

    per_pair_pred_connected = pred_sum / pred_count

    bounds = np.cumsum([0] + list(args.sizes))
    comp_of = np.zeros(n, dtype=int)
    for ci in range(3):
        comp_of[bounds[ci]:bounds[ci + 1]] = ci

    np.savez(out / "perpair.npz", pred_connected=per_pair_pred_connected,
             sizes=args.sizes, comp_of=comp_of)
    print(f"saved -> {out}/perpair.npz")


if __name__ == "__main__":
    main()
