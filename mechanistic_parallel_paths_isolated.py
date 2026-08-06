"""Report IX -- clean parallel-paths rescue test on the n=46 split_chains-trained
checkpoints: two terminals s,t joined by k in {1,2,3,4} internally-disjoint paths of
``path_len`` edges each (default 11, chosen so 2+4*(11-1)=42<=46 -- k up to 4 fits),
with every OTHER node completely ISOLATED (degree 0, no leaf padding, no filler
component) -- data.py::generate_parallel_paths_graph, the plain construction (as
opposed to mechanistic_multipath_heatmaps.py's generate_multipath_graph, which pads
terminal degree with leaves and fills the rest with a sparse background chain).

For each k this reports, over many independent random relabellings of the SAME
canonical graph:
  * term_connect   : predicted-connected accuracy on the (s,t) pair (target 1) --
                      the headline "does k parallel routes rescue the connection"
                      number, directly comparable across k at fixed distance.
  * reach_route     : predicted-connected accuracy pooled over every within-route
                       pair (both endpoints on the same one of the k paths,
                       including s/t), target 1.
  * isolation_acc   : predicted-DISconnected accuracy pooled over every pair
                       touching an isolated node (isolated-isolated or
                       isolated-anything), target 0 -- checks the model does not
                       spuriously connect the isolated filler to the active
                       component now that there is no leaf padding to absorb it.
  * pred_positive_rate : fraction of ALL pairs in the graph predicted connected.
and the real attention score/alpha matrices (layer 0/1), averaged over the same
relabellings and mapped back to canonical node order (0=s, 1=t, then each route's
internal nodes in turn, then the isolated nodes), exactly like every other
attention-heatmap script in this project.

Eval-only, CPU-friendly (forward passes only).

  python mechanistic_parallel_paths_isolated.py --checkpoint <ckpt.pt> \\
      --output_dir runs/report9/parallel_paths_isolated/<tag> --ks 1 2 3 4 --path_len 11
"""
import argparse, math
from pathlib import Path

import numpy as np
import torch

from data import add_self_loops, generate_parallel_paths_graph
from eval_families import load_model
from mechanistic_asym_chains import run_with_cache, _device


def _route_bounds(k, path_len):
    bounds = [0, 2]
    for _ in range(k):
        bounds.append(bounds[-1] + (path_len - 1))
    return bounds


def _pair_categories(n, k, path_len):
    """-> dict name -> boolean [n,n] mask (symmetric, diagonal excluded)."""
    bounds = _route_bounds(k, path_len)
    n_active = bounds[-1]
    isolated = np.zeros(n, dtype=bool)
    isolated[n_active:] = True

    within_route = np.zeros((n, n), dtype=bool)
    for ri in range(k):
        lo, hi = bounds[ri + 1], bounds[ri + 2]
        members = np.array([0, 1] + list(range(lo, hi)))
        within_route[np.ix_(members, members)] = True
    np.fill_diagonal(within_route, False)

    touches_isolated = np.zeros((n, n), dtype=bool)
    touches_isolated[isolated, :] = True
    touches_isolated[:, isolated] = True
    np.fill_diagonal(touches_isolated, False)

    term = np.zeros((n, n), dtype=bool)
    term[0, 1] = term[1, 0] = True

    return {"within_route": within_route, "touches_isolated": touches_isolated,
            "term": term, "n_active": n_active}


def parallel_paths_probe(model, dev, n, ks, path_len, rng, n_graphs):
    out = {}
    for k in ks:
        need = 2 + k * (path_len - 1)
        if need > n:
            print(f"  k={k} path_len={path_len}: needs {need} > n={n}, skipped")
            continue
        base_adj = generate_parallel_paths_graph(n, k, path_len)
        cats = _pair_categories(n, k, path_len)
        xs = np.empty((n_graphs, n, n), np.float32)
        invs = []
        for i in range(n_graphs):
            p = rng.permutation(n)
            xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
            invs.append(np.argsort(p))
        xb = torch.from_numpy(xs).to(dev, torch.float32)
        cache, h_final, logits = run_with_cache(model, xb)

        def unperm(mat_bnn):
            o = np.empty_like(mat_bnn)
            for i, inv in enumerate(invs):
                o[i] = mat_bnn[i][np.ix_(inv, inv)]
            return o

        pred_conn = unperm(logits) > 0   # [B,n,n], canonical node order

        term_connect = pred_conn[:, cats["term"]].mean()
        reach_route = pred_conn[:, cats["within_route"]].mean()
        isolation_acc = (~pred_conn[:, cats["touches_isolated"]]).mean()
        off_diag = ~np.eye(n, dtype=bool)
        pred_positive_rate = pred_conn[:, off_diag].mean()

        d = {"term_connect": float(term_connect), "reach_route": float(reach_route),
             "isolation_acc": float(isolation_acc),
             "pred_positive_rate": float(pred_positive_rate),
             "n_active": int(cats["n_active"])}

        def unperm_pair(mat):
            o = np.empty_like(mat)
            for i, inv in enumerate(invs):
                o[i] = mat[i][np.ix_(inv, inv)]
            return o.mean(0)

        for li in range(len(model.blocks)):
            q, kk = cache[f"layer{li}_q"], cache[f"layer{li}_k"]
            alpha = cache[f"layer{li}_alpha"]
            head_dim = q.shape[-1]
            scores = np.einsum("gid,gjd->gij", q, kk) / math.sqrt(head_dim)
            d[f"scores{li}"] = unperm_pair(scores)
            d[f"alpha{li}"] = unperm_pair(alpha)
        d["route_bounds"] = np.array(_route_bounds(k, path_len))
        out[f"k{k}"] = d
        print(f"  k={k} path_len={path_len}: term_connect={term_connect:.3f} "
              f"reach_route={reach_route:.3f} isolation_acc={isolation_acc:.3f} "
              f"pred_pos={pred_positive_rate:.3f} (n_active={cats['n_active']}/{n})",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--path_len", type=int, default=11,
                    help="edges per route; beyond the 3^L=9 capacity by default")
    ap.add_argument("--n_graphs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} "
          f"attn_kind={mcfg.attn_kind} device={dev}")
    if arch != "roberta":
        raise NotImplementedError("run_with_cache is written for RobertaGraphTransformer only")

    rng = np.random.default_rng(args.seed)
    result = parallel_paths_probe(model, dev, n, args.ks, args.path_len, rng, args.n_graphs)

    summary = {"checkpoint": str(args.checkpoint), "arch": arch, "readout": readout,
               "n": n, "path_len": args.path_len,
               "cells": [{"k": int(kkey[1:]), "term_connect": d["term_connect"],
                          "reach_route": d["reach_route"], "isolation_acc": d["isolation_acc"],
                          "pred_positive_rate": d["pred_positive_rate"], "n_active": d["n_active"]}
                         for kkey, d in result.items()]}
    (out / "parallel_paths_isolated.json").write_text(__import__("json").dumps(summary, indent=2))

    flat = {}
    for kkey, d in result.items():
        for name, arr in d.items():
            if name in ("term_connect", "reach_route", "isolation_acc",
                        "pred_positive_rate", "n_active"):
                continue
            flat[f"{kkey}__{name}"] = arr
    flat["ks_present"] = np.array([int(kk[1:]) for kk in result.keys()])
    np.savez(out / "parallel_paths_isolated_attn.npz", **flat)
    print(f"  saved -> {out}/parallel_paths_isolated.json (+ _attn.npz)")


if __name__ == "__main__":
    main()
