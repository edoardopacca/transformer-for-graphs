"""Difficulty map: per-graph accuracy vs (diameter, spectral gap, density).

The professor's first question: how does the standard model's accuracy vary with
the DIAMETER and with the SPECTRAL GAP, and which one is the operative variable?

To answer it we need points that SPREAD across the (diameter, gap) plane, not the
clustered family points of eval_families.py, and we need the JOINT per-graph data
(diam, gap, accuracy together), which eval_families throws away. This script
samples a pool of single-component graphs spanning the plane and dumps, per graph:
diameter, spectral gap, mean degree (a density control), #components, exact-match
and pairwise accuracy. From that JSON we build, offline, the 2-D accuracy map and
the partial-dependence curves (accuracy vs diameter at fixed gap, and vice versa).

Eval-only; runs on an existing LINEAR checkpoint. No training.

Pool (all connected, one component, so exact-match == "reached every pair"):
  - ER(n, p) over a p-sweep        -> the main arc through the plane;
  - random-regular, degree sweep   -> sweeps the gap at ~fixed diameter band
                                      (degree 2 = cycle: big diam/tiny gap;
                                       high degree: tiny diam/big gap);
  - barbell, clique-size sweep     -> the small-gap / bottleneck corner;
  - 1chain, 1cycle                 -> the large-diameter corner.
Disconnected samples are kept and flagged (ncomp>1); filter them out for the
clean reach map.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from data import (add_self_loops, compute_connectivity_matrix,
                  compute_all_pairs_shortest_paths, compute_spectral_gap,
                  generate_er_graph, generate_one_chain_graph,
                  generate_one_cycle_graph, generate_barbell_graph,
                  generate_random_regular_graph)
from eval_families import load_model, predict


def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def build_pool(n, size_per, seed):
    """Return (list of adjacency matrices without self-loops, list of labels)."""
    rng = np.random.default_rng(seed)
    adjs, labels = [], []

    def add(a, lab):
        p = rng.permutation(n)               # random node relabelling
        adjs.append(a[np.ix_(p, p)]); labels.append(lab)

    # ER p-sweep: dense (small diam / big gap) -> sparse (big diam / small gap).
    ps = np.round(np.linspace(0.05, 0.50, 12), 3) if n >= 40 \
         else np.round(np.linspace(0.08, 0.60, 12), 3)
    for p in ps:
        for _ in range(size_per):
            add(generate_er_graph(n, float(p), rng), f"er_p{p}")

    # random-regular degree sweep: degree 2 = C_n, high degree -> near-complete.
    degs = sorted(set(int(d) for d in np.linspace(2, n // 2, 10).round()))
    for deg in degs:
        for _ in range(size_per):
            try:
                add(generate_random_regular_graph(n, rng, degree=deg), f"rr_d{deg}")
            except Exception:
                pass

    # barbell clique-size sweep: the bottleneck / small-gap corner.
    cls = sorted(set(int(c) for c in np.linspace(2, n // 2, 8).round()))
    for cs in cls:
        for _ in range(max(1, size_per // 2)):
            try:
                add(generate_barbell_graph(n, rng, clique_size=cs), f"barbell_c{cs}")
            except Exception:
                pass

    # explicit large-diameter corner
    for _ in range(size_per):
        add(generate_one_chain_graph(n), "1chain")
    for _ in range(size_per):
        add(generate_one_cycle_graph(n), "1cycle")

    return adjs, labels


def graph_stats(a):
    """diameter (max finite shortest path), spectral gap, mean degree, #components."""
    d = compute_all_pairs_shortest_paths(a)
    fin = d[d >= 0]
    diam = int(fin.max()) if fin.size else 0
    gap = float(compute_spectral_gap(a))
    mean_deg = float(a.sum(1).mean())
    ncomp = int(connected_components(csr_matrix(a), directed=False)[0])
    return diam, gap, mean_deg, ncomp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_nodes", type=int, required=True)
    ap.add_argument("--size_per", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = args.n_nodes
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")

    adjs, labels = build_pool(n, args.size_per, args.seed)
    ng = len(adjs)
    print(f"  pool: {ng} graphs")

    xs = np.empty((ng, n, n), np.float32)
    ys = np.empty((ng, n, n), np.int8)
    diam = np.empty(ng, np.int64); gap = np.empty(ng, np.float64)
    mdeg = np.empty(ng, np.float64); ncomp = np.empty(ng, np.int64)
    for i, a in enumerate(adjs):
        xs[i] = add_self_loops(a)
        ys[i] = compute_connectivity_matrix(a).astype(np.int8)
        diam[i], gap[i], mdeg[i], ncomp[i] = graph_stats(a)

    pred = predict(model, xs, dev)
    eq = (pred == ys)
    offdiag = ~np.eye(n, dtype=bool)[None]
    exact = eq.reshape(ng, -1).all(1).astype(np.float64)
    pw = (eq & offdiag).reshape(ng, -1).sum(1) / offdiag.reshape(ng, -1).sum(1)

    res = {
        "checkpoint": str(args.checkpoint), "arch": arch, "readout": readout,
        "n": n, "size_per": args.size_per, "n_graphs": ng,
        "per_graph": {
            "label": labels,
            "diameter": diam.tolist(), "gap": gap.tolist(),
            "mean_degree": mdeg.tolist(), "ncomp": ncomp.tolist(),
            "exact": exact.tolist(), "pairwise": pw.round(4).tolist(),
        },
    }
    (out / "difficulty_map.json").write_text(json.dumps(res))
    print(f"  saved -> {out}/difficulty_map.json")

    # quick sanity scatter (connected only): diameter vs gap, coloured by exact.
    cmask = ncomp == 1
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sc = ax[0].scatter(diam[cmask], gap[cmask], c=exact[cmask], cmap="viridis",
                       s=10, vmin=0, vmax=1)
    ax[0].set_xlabel("diameter"); ax[0].set_ylabel("spectral gap")
    ax[0].set_yscale("log"); ax[0].set_title(f"exact-match map (connected) {readout} n={n}")
    fig.colorbar(sc, ax=ax[0], label="exact match")
    sc2 = ax[1].scatter(diam[cmask], gap[cmask], c=pw[cmask], cmap="viridis",
                        s=10, vmin=0.5, vmax=1)
    ax[1].set_xlabel("diameter"); ax[1].set_ylabel("spectral gap")
    ax[1].set_yscale("log"); ax[1].set_title("pairwise accuracy map")
    fig.colorbar(sc2, ax=ax[1], label="pairwise")
    fig.tight_layout(); fig.savefig(out / "difficulty_map.png", dpi=150); plt.close(fig)
    print(f"  saved -> {out}/difficulty_map.png")


if __name__ == "__main__":
    main()
