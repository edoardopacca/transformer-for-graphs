"""Report X -- REPLACES the earlier multipath test (mechanistic_multipath_geometry.py /
eval_n60_geometry_heatmap.py's multipath half), which used generate_multipath_graph's
leaf-padded construction: Edoardo's objection is that the leaves (extra pendant nodes added
only to keep terminal degree constant across k) actively help the transformer recognise the
terminals, so they are not a clean test of "does having several parallel routes help." This
script tests two LEAF-FREE constructions instead, both using every one of the n=60 nodes as
a real node ON one of the three routes -- nothing added, nothing padded:

CONSTRUCTION A ("clean"): three parallel routes directly between two terminals s,t, built with
generate_multipath_graph's now-generalised per-route path_len (data.py, extended earlier this
session) and term_deg=3 (= n_full, so leaves_s=leaves_t=0 by construction: 2 + 19 + 20 + 19 =
60 nodes exactly, no leaves, no filler). Route edge-lengths [20,21,20] (route 2 one edge
longer, matching the shape Edoardo specified: two 19-internal-node routes + one 20-internal-
node route). Target pair: s,t themselves (node 0,1).

CONSTRUCTION B ("stitched theta"): take the disjoint 3-way BALANCED split (20,20,20) --
already reported in sec:n60-grid -- unchanged, and add 4 edges Edoardo specified (1-indexed):
(1,21), (20,40), (21,41), (40,60) -- i.e. (0-indexed) (0,20), (19,39), (20,40), (39,59). This
turns component B's own two endpoints (0-indexed 20 and 39) into degree-3 hubs, connected by
THREE routes: directly through B (19 edges), and two "there-and-back" loops through A and
through C (21 edges each, using the stitching edges). Target pair: the two hubs, (20,39).
Node order for construction B is the SAME as the disjoint (20,20,20) graph already reported
(component A, then B, then C, in path order) -- it IS that graph, plus 4 edges -- so its
geometry heatmap is directly comparable to Figure fig:n60-disjoint-heatmap panel-for-panel.

Both constructions are evaluated identically: target-pair accuracy (5 independent repeats x 50
graphs, distinct seed per checkpoint), a pooled "reach_route"-style accuracy over every
within-route pair (excluding the target pair itself), and a full n x n H^(2) similarity-
geometry heatmap (Z = scale*cos+bias, 64 relabellings averaged, canonical node order).

    python eval_n60_multipath_v2.py --checkpoint runs/.../last.pt \\
        --output_dir runs/report10/multipath_v2/<tag> --seed 1000
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import add_self_loops, generate_multi_path_split_graph, generate_multipath_graph
from eval_families import load_model
from stagewise_diagnostics import run_with_stages, _cosine_batch, _device, _selftest


def _unperm_embed(arr, invs):
    return np.stack([arr[i][inv] for i, inv in enumerate(invs)])


def _unperm_pair(arr, invs):
    return np.stack([arr[i][np.ix_(inv, inv)] for i, inv in enumerate(invs)])


def _forward(model, dev, n, base_adj, rng, n_graphs):
    """-> (logits_base [G,n,n], h2_base [G,n,d]), both in canonical node order."""
    xs = np.empty((n_graphs, n, n), np.float32)
    invs = []
    for i in range(n_graphs):
        p = rng.permutation(n)
        xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
        invs.append(np.argsort(p))
    xb = torch.from_numpy(xs).to(dev, torch.float32)
    stages, _, logits = run_with_stages(model, xb)
    logits_base = _unperm_pair(logits, invs)
    h2_base = _unperm_embed(stages["H2"], invs)
    return logits_base, h2_base


def eval_behaviour(model, dev, n, base_adj, target_pair, route_masks, rng, n_graphs, n_repeats):
    tc_list, rr_list, pp_list = [], [], []
    for rep in range(n_repeats):
        logits_base, _ = _forward(model, dev, n, base_adj, rng, n_graphs)
        pred = logits_base > 0
        i, j = target_pair
        tc_list.append(float(pred[:, i, j].mean()))
        rr_list.append(float(pred[:, route_masks].mean()))
        pp_list.append(float(pred.mean()))
    def agg(v): return float(np.mean(v)), float(np.std(v))
    tc_m, tc_s = agg(tc_list); rr_m, rr_s = agg(rr_list); pp_m, pp_s = agg(pp_list)
    return {"target_pair_connected": round(tc_m, 4), "target_pair_connected_std": round(tc_s, 4),
            "reach_route": round(rr_m, 4), "reach_route_std": round(rr_s, 4),
            "pred_positive_rate": round(pp_m, 4), "pred_positive_rate_std": round(pp_s, 4),
            "per_repeat_target_pair": [round(v, 4) for v in tc_list]}


def h2_Z_matrix(model, dev, n, base_adj, rng, n_graphs):
    _, h2_base = _forward(model, dev, n, base_adj, rng, n_graphs)
    G = _cosine_batch(h2_base).mean(0)
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
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    print("wrote", out_png)


def build_construction_a(n, route_lens, term_deg):
    built = generate_multipath_graph(n, len(route_lens), route_lens, np.random.default_rng(0),
                                      term_deg=term_deg)
    if built is None:
        raise ValueError(f"route_lens={route_lens} term_deg={term_deg} does not fit n={n}")
    adj, meta = built
    assert len(meta["leaves"]) == 0 and len(meta["filler"]) == 0, \
        f"expected no leaves/filler, got {len(meta['leaves'])} leaves, {len(meta['filler'])} filler"
    s, t = meta["s"], meta["t"]
    within = np.zeros((n, n), dtype=bool)
    for route in meta["full_paths"]:
        idx = np.array([s, t] + route, dtype=int)
        within[np.ix_(idx, idx)] = True
    np.fill_diagonal(within, False)
    within[s, t] = within[t, s] = False   # target pair excluded from the pooled route metric
    bounds = [1.5]
    cur = 2
    for route in meta["full_paths"][:-1]:
        cur += len(route)
        bounds.append(cur - 0.5)
    return adj, (s, t), within, bounds


def build_construction_b(n, sizes=(20, 20, 20)):
    adj = generate_multi_path_split_graph(n, sizes).copy()
    a0, a1 = 0, sizes[0] - 1
    b0, b1 = sizes[0], sizes[0] + sizes[1] - 1
    c0, c1 = sizes[0] + sizes[1], n - 1
    stitches = [(a0, b0), (a1, b1), (b0, c0), (b1, c1)]
    for u, v in stitches:
        adj[u, v] = adj[v, u] = 1.0
    hub1, hub2 = b0, b1
    within = np.zeros((n, n), dtype=bool)
    for lo, hi in [(a0, a1), (b0, b1), (c0, c1)]:
        idx = list(range(lo, hi + 1)) + [hub1, hub2]
        idx = sorted(set(idx))
        within[np.ix_(idx, idx)] = True
    np.fill_diagonal(within, False)
    within[hub1, hub2] = within[hub2, hub1] = False
    bounds = [a1 + 0.5, b1 + 0.5]
    return adj, (hub1, hub2), within, bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs_behaviour", type=int, default=50)
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--n_graphs_heatmap", type=int, default=64)
    ap.add_argument("--route_lens_a", type=int, nargs=3, default=[20, 21, 20])
    ap.add_argument("--term_deg_a", type=int, default=3)
    ap.add_argument("--disjoint_sizes_b", type=int, nargs=3, default=[20, 20, 20])
    ap.add_argument("--seed", type=int, default=12345,
                    help="MUST differ across checkpoints -- see istruzioni.md error #34/report/10 "
                         "sec:n60-multipath history")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    if readout != "similarity":
        raise NotImplementedError("this script reads model.sim_scale/sim_bias directly")
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev} "
          f"seed_base={args.seed}")
    _selftest(model, dev, n)

    results = {}
    for name, (adj, pair, within, bounds) in [
        ("A_clean", build_construction_a(n, args.route_lens_a, args.term_deg_a)),
        ("B_stitched", build_construction_b(n, tuple(args.disjoint_sizes_b))),
    ]:
        rng = np.random.default_rng(args.seed)
        beh = eval_behaviour(model, dev, n, adj, pair, within, rng,
                              args.n_graphs_behaviour, args.n_repeats)
        beh["target_pair"] = pair
        print(f"  [{name}] target_pair={pair}: "
              f"connected={beh['target_pair_connected']:.3f}+-{beh['target_pair_connected_std']:.3f} "
              f"reach_route={beh['reach_route']:.3f}+-{beh['reach_route_std']:.3f} "
              f"pred_pos={beh['pred_positive_rate']:.3f}", flush=True)
        rng2 = np.random.default_rng(args.seed + 1000)
        Z = h2_Z_matrix(model, dev, n, adj, rng2, args.n_graphs_heatmap)
        np.savez(out / f"{name}_Z.npz", Z=Z, boundaries=bounds, target_pair=pair)
        plot_heatmap(Z, bounds, f"Construction {name}, $n$={n}, H$^{{(2)}}$ geometry ({args.tag})",
                     out / f"{name}_h2_heatmap.png")
        results[name] = beh

    (out / "multipath_v2.json").write_text(json.dumps(results, indent=2))
    print(f"saved -> {out}/multipath_v2.json")


if __name__ == "__main__":
    main()
