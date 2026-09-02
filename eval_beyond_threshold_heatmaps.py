"""Report X (personal working doc) -- per-split H^(2) similarity-geometry heatmaps for the
two-path beyond-threshold checkpoints of paper_draft.tex Sec 4.1 (full split-size distribution
and odd/grid training), for every failing split a=12..23. Answers "which node pairs does the
model get wrong" directly: plots Z_ij = scale*cos(h_i,h_j)+bias, the model's own decision
quantity (connected iff Z>0), so a wrongly-disconnected pair inside a component or a
wrongly-connected pair across components is visible as a colour anomaly against the expected
block-diagonal pattern (all-red within each path, all-blue across the two).

Same style/scripts this project already uses for exactly this quantity (Report X's n=60
geometry heatmaps): eval_n60_geometry_heatmap.py computes Z on GPU from a checkpoint and saves
one .npz per case; plot_n60_geometry_heatmaps_v2.py re-renders locally from the .npz, no GPU
needed. This script is the eval half, generalised to a whole split sweep instead of one fixed
split. Canonical node order: short path first (positions 0..a-1), then long path
(a..n-1) -- matching Figure 2 (fig:attn-heatmap) in the paper.

Eval-only (forward passes only, run_with_stages from stagewise_diagnostics.py).

    python eval_beyond_threshold_heatmaps.py --checkpoint runs/report9/n46_train/<dir>/last.pt \\
        --output_dir runs/report10/beyond_threshold_heatmaps/<tag> --n_graphs 64
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from data import add_self_loops, generate_split_chains_graph
from eval_families import load_model
from stagewise_diagnostics import run_with_stages, _cosine_batch, _device, _selftest

SPLITS = list(range(12, 24))  # a = 12..23, the failing half of the sweep


def _unperm_embed(arr, invs):
    """[G,n,d] -> unpermute the node axis only (correct for embeddings, NOT for a pairwise
    [n,n] matrix -- see istruzioni.md error #34)."""
    return np.stack([arr[i][inv] for i, inv in enumerate(invs)])


def h2_Z_matrix(model, dev, n, base_adj, rng, n_graphs):
    xs = np.empty((n_graphs, n, n), np.float32)
    invs = []
    for i in range(n_graphs):
        p = rng.permutation(n)
        xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
        invs.append(np.argsort(p))
    xb = torch.from_numpy(xs).to(dev, torch.float32)
    stages, _, _ = run_with_stages(model, xb)
    h2_base = _unperm_embed(stages["H2"], invs)          # [G,n,d], canonical order
    G = _cosine_batch(h2_base).mean(0)                    # [n,n] mean cosine, canonical order
    scale = float(model.sim_scale.detach().cpu())
    bias = float(model.sim_bias.detach().cpu())
    return scale * G + bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tag", default="")
    ap.add_argument("--splits", default="",
                    help="comma-separated list of a values to compute (default: the module "
                         "constant SPLITS, a=12..23); e.g. --splits 3,5,7,9,11 for the "
                         "professor-requested a<12 panels (2026-09-02, Edoardo)")
    args = ap.parse_args()
    splits = [int(x) for x in args.splits.split(",") if x.strip()] if args.splits else SPLITS

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    if readout != "similarity":
        raise NotImplementedError("this script reads model.sim_scale/sim_bias directly")
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")
    _selftest(model, dev, n)
    rng = np.random.default_rng(args.seed)

    n_done = 0
    for a in splits:
        base_adj = generate_split_chains_graph(n, a)
        Z = h2_Z_matrix(model, dev, n, base_adj, rng, args.n_graphs)
        boundary = a - 0.5
        out_f = out / f"a{a}_Z.npz"
        np.savez(out_f, Z=Z, boundary=boundary, a=a, n=n, tag=args.tag)
        print(f"  a={a}: wrote {out_f}")
        n_done += 1

    print(f"DONE: {n_done}/{len(splits)} splits written to {out}")
    if n_done != len(splits):
        raise RuntimeError(f"only {n_done}/{len(splits)} splits completed -- check the log above")


if __name__ == "__main__":
    main()
