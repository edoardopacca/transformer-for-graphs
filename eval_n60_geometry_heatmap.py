"""Report X -- full n x n final-layer (H^(2)) similarity-geometry heatmap for the two n=60
K=3 (path_union_k3) graphs discussed in report/10 sec:n60-grid/sec:n60-multipath: the disjoint
balanced 3-way split (20,20,20) and the multipath graph with 3 routes of 19,19,20 edges.
Same style as the Report VIII/IX stagewise-cosine figures (stagewise_diagnostics.py /
plot_stagewise_diagnostics.py): plots Z_ij = scale*cos(h_i,h_j)+bias (not raw cosine), so the
model's own decision boundary sits at exactly 0 on a diverging colormap, RdBu_r, centered at 0.
Averaged over many random relabellings, mapped back to a fixed CANONICAL node order (never a
single arbitrary permutation):
  * disjoint: component 1 (nodes along the path in order), then component 2, then component 3.
  * multipath: s, t, then each route's internal nodes in turn, then leaves, then filler --
    the same canonical order already used by mechanistic_multipath_heatmaps.py's attention
    heatmaps in this project.

Eval-only (forward passes only, run_with_stages from stagewise_diagnostics.py).

    python eval_n60_geometry_heatmap.py --checkpoint runs/.../last.pt \\
        --output_dir runs/report10/geometry_heatmap/<tag> --n_graphs 64
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import add_self_loops, generate_multi_path_split_graph, generate_multipath_graph
from eval_families import load_model
from stagewise_diagnostics import run_with_stages, _cosine_batch, _device, _selftest


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


def plot_heatmap(Z, boundaries, title, out_png):
    vmax = np.abs(Z).max()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for b in boundaries:
        ax.axvline(b, color="k", ls="--", lw=1)
        ax.axhline(b, color="k", ls="--", lw=1)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=r"$Z_{ij}=\mathrm{scale}\cdot\cos(h_i,h_j)+\mathrm{bias}$ "
                       r"(connected $>0$, disconnected $<0$)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=64)
    ap.add_argument("--disjoint_sizes", type=int, nargs=3, default=[20, 20, 20])
    ap.add_argument("--multipath_lens", type=int, nargs=3, default=[19, 19, 20])
    ap.add_argument("--term_deg", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    if readout != "similarity":
        raise NotImplementedError("this script reads model.sim_scale/sim_bias directly")
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")
    _selftest(model, dev, n)
    rng = np.random.default_rng(args.seed)

    # --- disjoint (20,20,20): canonical order is already component1,component2,component3 ---
    sizes = tuple(args.disjoint_sizes)
    base_adj_d = generate_multi_path_split_graph(n, sizes)
    Z_d = h2_Z_matrix(model, dev, n, base_adj_d, rng, args.n_graphs)
    bounds_d = [sizes[0] - 0.5, sizes[0] + sizes[1] - 0.5]
    np.savez(out / "disjoint_Z.npz", Z=Z_d, boundaries=bounds_d, sizes=sizes)
    plot_heatmap(Z_d, bounds_d,
                 f"Disjoint 3-way split {sizes}, $n$={n}, H$^{{(2)}}$ geometry ({args.tag})",
                 out / "disjoint_h2_heatmap.png")

    # --- multipath (19,19,20): canonical order s,t,route1,route2,route3,leaves,filler ---
    lens = list(args.multipath_lens)
    built = generate_multipath_graph(n, len(lens), lens, rng, term_deg=args.term_deg)
    if built is None:
        raise ValueError(f"multipath_lens={lens} term_deg={args.term_deg} does not fit n={n}")
    base_adj_m, meta = built
    Z_m = h2_Z_matrix(model, dev, n, base_adj_m, rng, args.n_graphs)
    b = 2  # after s,t
    bounds_m = [1.5]
    for route in meta["full_paths"]:
        b += len(route)
        bounds_m.append(b - 0.5)
    bounds_m.append(b + len(meta["leaves"]) - 0.5)
    np.savez(out / "multipath_Z.npz", Z=Z_m, boundaries=bounds_m, lens=lens)
    plot_heatmap(Z_m, bounds_m,
                 f"Multipath routes {lens}, $n$={n}, H$^{{(2)}}$ geometry ({args.tag})\n"
                 f"order: $s,t$ | route 1 | route 2 | route 3 | leaves | filler",
                 out / "multipath_h2_heatmap.png")


if __name__ == "__main__":
    main()
