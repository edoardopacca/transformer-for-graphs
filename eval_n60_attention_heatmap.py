"""Report X -- real attention-weight heatmaps (layer 1 and layer 2, alpha = model's actual
normalized-ReLU attention matrix) for the three n=60 K=3 graphs already reported: the disjoint
(20,20,20) split (Figure 16) and the two leaf-free multipath constructions A/B (Figure 17).
Same canonical 1-indexed node order + labelled blocks as
plot_n60_geometry_heatmaps_v2.py/eval_n60_multipath_v2.py, magma colormap (never jet, per
istruzioni.md figure-style rules) since attention weights are non-negative (no "connected vs
disconnected" sign to encode, unlike the H^(2) similarity heatmaps).

Averaged over many random relabellings, alpha mapped back to canonical order via the two-axis
np.ix_ unpermute (alpha[i,j] = attention FROM query i TO key j, a genuine pairwise [n,n]
matrix -- see istruzioni.md error #34).

    python eval_n60_attention_heatmap.py --checkpoint runs/.../last.pt \\
        --output_dir runs/report10/attention_heatmap/<tag> --n_graphs 64
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import add_self_loops, generate_multi_path_split_graph
from eval_families import load_model
from eval_n60_multipath_v2 import build_construction_a, build_construction_b
from stagewise_diagnostics import run_with_stages, _device, _selftest

MIN_LABEL_FRAC = 0.05


def _unperm_pair(arr, invs):
    return np.stack([arr[i][np.ix_(inv, inv)] for i, inv in enumerate(invs)])


def mean_alpha(model, dev, n, base_adj, rng, n_graphs):
    xs = np.empty((n_graphs, n, n), np.float32)
    invs = []
    for i in range(n_graphs):
        p = rng.permutation(n)
        xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
        invs.append(np.argsort(p))
    xb = torch.from_numpy(xs).to(dev, torch.float32)
    _, attn_cache, _ = run_with_stages(model, xb)
    n_layers = sum(1 for k in attn_cache if k.endswith("_alpha"))
    return [_unperm_pair(attn_cache[f"layer{li}_alpha"], invs).mean(0) for li in range(n_layers)]


def plot_attn(alpha, blocks, title, out_png, markers=None):
    n = alpha.shape[0]
    vmax = float(alpha.max())
    fig, ax = plt.subplots(figsize=(9.6, 8.6 if markers else 7.8))
    top = 0.66 if markers else 0.76
    fig.subplots_adjust(left=0.26, right=0.78, top=top, bottom=0.09)
    im = ax.imshow(alpha, cmap="magma", vmin=0, vmax=vmax)

    for lo, hi, _ in blocks:
        if lo > 0:
            ax.axvline(lo - 0.5, color="w", ls="--", lw=1)
            ax.axhline(lo - 0.5, color="w", ls="--", lw=1)

    ticks = sorted({0} | {b[0] for b in blocks} | {n})
    ax.set_xticks([t - 0.5 for t in ticks]); ax.set_xticklabels([t + 1 for t in ticks], fontsize=9)
    ax.set_yticks([t - 0.5 for t in ticks]); ax.set_yticklabels([t + 1 for t in ticks], fontsize=9)
    ax.set_xlabel("key node $j$ (1-indexed)"); ax.set_ylabel("query node $i$ (1-indexed)")

    row_toggle = 0
    for lo, hi, label in blocks:
        if (hi - lo) / n < MIN_LABEL_FRAC:
            continue
        y = 1.035 + 0.075 * row_toggle; row_toggle = 1 - row_toggle
        ax.annotate(label, xy=((lo + (hi - lo) / 2) / n, y), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=9.5)
    row_toggle = 0
    for lo, hi, label in blocks:
        if (hi - lo) / n < MIN_LABEL_FRAC:
            continue
        mid_frac = 1 - (lo + (hi - lo) / 2) / n
        x = -0.05 - 0.20 * row_toggle; row_toggle = 1 - row_toggle
        ax.annotate(label, xy=(x, mid_frac), xycoords="axes fraction", ha="right", va="center", fontsize=9.5)
    if markers:
        for k, (pos, label) in enumerate(markers):
            frac = pos / n
            y0 = 1.005 + 0.11 * (k % 2)
            ax.annotate("$\\blacktriangledown$", xy=(frac, 1.0), xycoords="axes fraction",
                        ha="center", va="bottom", fontsize=11, color="black")
            ax.annotate(label, xy=(frac, y0 + 0.02), xycoords="axes fraction",
                        ha="center", va="bottom", fontsize=8.5, style="italic")

    fig.suptitle(title, fontsize=11, y=0.985)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"attention weight $\alpha_{ij}$ (query $i$ $\to$ key $j$)", fontsize=9)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=64)
    ap.add_argument("--route_lens_a", type=int, nargs=3, default=[20, 21, 20])
    ap.add_argument("--term_deg_a", type=int, default=3)
    ap.add_argument("--disjoint_sizes", type=int, nargs=3, default=[20, 20, 20])
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")
    _selftest(model, dev, n)

    sizes = tuple(args.disjoint_sizes)
    adj_disjoint = generate_multi_path_split_graph(n, sizes)
    blocks_disjoint = [(0, sizes[0], f"Component 1\n(nodes 1-{sizes[0]})"),
                        (sizes[0], sizes[0] + sizes[1], f"Component 2\n(nodes {sizes[0]+1}-{sizes[0]+sizes[1]})"),
                        (sizes[0] + sizes[1], n, f"Component 3\n(nodes {sizes[0]+sizes[1]+1}-{n})")]

    adj_a, pair_a, _, bounds_a = build_construction_a(n, args.route_lens_a, args.term_deg_a)
    b0a, b1a = int(bounds_a[0] + 0.5), int(bounds_a[1] + 0.5)
    blocks_a = [(0, 1, ""), (1, 2, ""),
                (2, b0a, f"Route 1\n(nodes 3-{b0a})"),
                (b0a, b1a, f"Route 2\n(nodes {b0a+1}-{b1a})"),
                (b1a, n, f"Route 3\n(nodes {b1a+1}-{n})")]

    adj_b, pair_b, _, _ = build_construction_b(n, sizes)
    hub1, hub2 = pair_b
    blocks_b = [(0, sizes[0], f"Component 1\n(nodes 1-{sizes[0]})\n[loop via hub 1]"),
                (sizes[0], sizes[0] + sizes[1], f"Component 2\n(nodes {sizes[0]+1}-{sizes[0]+sizes[1]})\n[direct route]"),
                (sizes[0] + sizes[1], n, f"Component 3\n(nodes {sizes[0]+sizes[1]+1}-{n})\n[loop via hub 2]")]

    configs = [
        ("disjoint", adj_disjoint, blocks_disjoint, None,
         f"Disjoint 3-way split {sizes}, $n$={n}"),
        ("A_clean", adj_a, blocks_a, [(0.5, "$s$\n(node 1)"), (1.5, "$t$\n(node 2)")],
         f"Construction A (clean, no leaves), $n$={n}"),
        ("B_stitched", adj_b, blocks_b, [(hub1, f"hub 1\n(node {hub1+1})"), (hub2 + 0.5, f"hub 2\n(node {hub2+1})")],
         f"Construction B (stitched theta), $n$={n}"),
    ]

    for name, adj, blocks, markers, label in configs:
        rng = np.random.default_rng(args.seed)
        alphas = mean_alpha(model, dev, n, adj, rng, args.n_graphs)
        for li, alpha in enumerate(alphas):
            np.savez(out / f"{name}_layer{li+1}_alpha.npz", alpha=alpha)
            plot_attn(alpha, blocks, f"{label}, attention layer {li+1} ({args.tag})",
                      out / f"{name}_layer{li+1}_attention.png", markers=markers)


if __name__ == "__main__":
    main()
