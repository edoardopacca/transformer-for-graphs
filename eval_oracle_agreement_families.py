"""Capstone (Report V): does the base connectivity model look like matrix powering
or like a bounded traversal, ACROSS ALL the families we have studied?

We already showed (eval_bridged_cliques.py) that on two dense cliques joined by one
bridge the base model follows a visit-bounded BFS, not the distance-bounded matrix
power. This script asks the same question on the *natural* families of Reports III--IV
(the training stream + the held-out structured graphs), for every base checkpoint
(n20/n40, ER/mixed). It is eval-only (n auto-detected from the checkpoint).

For each family we build a pool, run the model, and compare its connectivity matrix
$\\hat R$ to THREE reference algorithms (dfs_oracle.py):
  * matrix-power oracle  : 1[(A+I)^{3^L} > 0]   (distance-bounded; = true connectivity
                           within capacity)
  * bounded-BFS(b) oracle: visit at most b nearest nodes per start (node-bounded, stalls)
  * bounded-DFS(b) oracle: depth-first, at most b nodes per start (node-bounded, dives)
sweeping the budget b. We report, per family and pooled:
  - model_vs_mp / model_vs_bfs(b) / model_vs_dfs(b) / model_vs_truth agreement (off-diag),
  - three disagreement masses (MP-vs-BFS, MP-vs-DFS, BFS-vs-DFS) -- how discriminating b is,
  - on each pair of oracles' DISAGREEING pairs only, the fraction the model follows each.
  The BFS-vs-DFS contrast is the direct separator (sec 5.2: DFS rises, BFS falls, so they
  split exactly on the contested pairs -- we re-run the capstone WITH the DFS oracle to
  check whether the model resembles a depth- or a breadth-first traversal on the natural
  families, the barbell in particular, rather than provisionally dropping DFS).

HONESTY NOTE built into the output: on sparse, low-degree families distance ~= nodes
traversed, so MP and BFS(b~=9) barely disagree -- the comparison there is vacuous and
the `disagree_frac` field flags it. The discriminating power lives in the dense /
large families. Read the on-disagree numbers weighted by disagree_frac.

  python eval_oracle_agreement_families.py --checkpoint runs/.../last.pt \\
      --output_dir runs/.../oracle_families
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dfs_oracle import (matrix_power_connectivity, bounded_bfs_connectivity,
                        bounded_dfs_connectivity)
from eval_families import build_family, load_model, predict

NATURAL = ["er", "er_blocks", "clique_blocks", "path_union", "2chains", "2cliques",
           "1cycle", "2cycle", "1chain", "barbell", "expander", "chain_plus"]


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def agreement_block(pred, ys, adj, n, L, budgets, off):
    """pred/ys: (G,n,n) int; adj: (G,n,n) no-loop. Returns the per-family summary."""
    G = pred.shape[0]
    om = off[None].repeat(G, 0)
    mp = np.stack([matrix_power_connectivity(adj[i], L) for i in range(G)])
    out = {
        "n_graphs": G,
        "model_vs_truth": float((pred[om] == ys[om]).mean()),
        "model_vs_mp": float((pred[om] == mp[om]).mean()),
        "mp_vs_truth": float((mp[om] == ys[om]).mean()),
        "budgets": list(budgets),
        "model_vs_bfs": [],
        "model_vs_dfs": [],
        "disagree_frac": [],            # MP vs BFS(b): how discriminating this b is
        "model_follows_mp_on_disagree": [],
        "model_follows_bfs_on_disagree": [],
        # MP vs DFS(b): the same read but against the depth-first reference.
        "disagree_frac_mp_dfs": [],
        "model_follows_mp_on_mpdfs": [],
        "model_follows_dfs_on_mpdfs": [],
        # BFS(b) vs DFS(b): the cleanest separator -- on the bridged cliques DFS RISES
        # (it dives across the bridge) while BFS FALLS (it stalls in the near clique),
        # so the two visit-bounded oracles disagree exactly on the contested pairs.
        # Which one the model follows there decides between a depth- and a breadth-
        # first traversal directly.
        "disagree_frac_bfs_dfs": [],
        "model_follows_bfs_on_bfsdfs": [],
        "model_follows_dfs_on_bfsdfs": [],
    }
    for b in budgets:
        bfs = np.stack([bounded_bfs_connectivity(adj[i], b) for i in range(G)])
        dfs = np.stack([bounded_dfs_connectivity(adj[i], b) for i in range(G)])
        out["model_vs_bfs"].append(float((pred[om] == bfs[om]).mean()))
        out["model_vs_dfs"].append(float((pred[om] == dfs[om]).mean()))
        # MP vs BFS
        dis = om & (mp != bfs)
        out["disagree_frac"].append(float(dis.mean()))
        if dis.any():
            out["model_follows_mp_on_disagree"].append(float((pred[dis] == mp[dis]).mean()))
            out["model_follows_bfs_on_disagree"].append(float((pred[dis] == bfs[dis]).mean()))
        else:
            out["model_follows_mp_on_disagree"].append(None)
            out["model_follows_bfs_on_disagree"].append(None)
        # MP vs DFS
        disd = om & (mp != dfs)
        out["disagree_frac_mp_dfs"].append(float(disd.mean()))
        if disd.any():
            out["model_follows_mp_on_mpdfs"].append(float((pred[disd] == mp[disd]).mean()))
            out["model_follows_dfs_on_mpdfs"].append(float((pred[disd] == dfs[disd]).mean()))
        else:
            out["model_follows_mp_on_mpdfs"].append(None)
            out["model_follows_dfs_on_mpdfs"].append(None)
        # BFS vs DFS (the visit-bounded separator)
        disbd = om & (bfs != dfs)
        out["disagree_frac_bfs_dfs"].append(float(disbd.mean()))
        if disbd.any():
            out["model_follows_bfs_on_bfsdfs"].append(float((pred[disbd] == bfs[disbd]).mean()))
            out["model_follows_dfs_on_bfsdfs"].append(float((pred[disbd] == dfs[disbd]).mean()))
        else:
            out["model_follows_bfs_on_bfsdfs"].append(None)
            out["model_follows_dfs_on_bfsdfs"].append(None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--families", nargs="+", default=NATURAL)
    ap.add_argument("--n_graphs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n, L = mcfg.n, mcfg.n_layers
    budgets = list(range(2, n + 1, 1 if n <= 20 else 2))
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} L={L} device={dev}")

    off = ~np.eye(n, dtype=bool)
    res = {"checkpoint": str(args.checkpoint), "arch": arch, "readout": readout,
           "n": n, "n_layers": L, "n_graphs": args.n_graphs, "budgets": budgets,
           "families": {}}

    # accumulate a pooled-over-families view as well
    pooled_pred, pooled_ys, pooled_adj = [], [], []
    for fam in args.families:
        xs, ys, dist, diam, gap = build_family(fam, n, args.n_graphs, args.seed)
        pred = predict(model, xs, dev)
        adj = xs.copy()
        for i in range(adj.shape[0]):
            np.fill_diagonal(adj[i], 0)
        res["families"][fam] = agreement_block(pred, ys, adj, n, L, budgets, off)
        b9 = budgets.index(9) if 9 in budgets else len(budgets) // 2
        fm = res["families"][fam]
        print(f"  {fam:13s} vs_truth={fm['model_vs_truth']:.3f} vs_mp={fm['model_vs_mp']:.3f} "
              f"| b=9: vs_bfs={fm['model_vs_bfs'][b9]:.3f} vs_dfs={fm['model_vs_dfs'][b9]:.3f} "
              f"| BFSvsDFS disagree={fm['disagree_frac_bfs_dfs'][b9]:.3f} "
              f"follows_bfs={fm['model_follows_bfs_on_bfsdfs'][b9]} "
              f"follows_dfs={fm['model_follows_dfs_on_bfsdfs'][b9]}")
        pooled_pred.append(pred); pooled_ys.append(ys); pooled_adj.append(adj)

    res["pooled"] = agreement_block(np.concatenate(pooled_pred), np.concatenate(pooled_ys),
                                    np.concatenate(pooled_adj), n, L, budgets, off)
    (out / "oracle_families.json").write_text(json.dumps(res, indent=2))
    print(f"  saved -> {out}/oracle_families.json")


if __name__ == "__main__":
    main()
