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
    tc_list, rr_list, pp_list, ec_list = [], [], [], []
    offdiag = ~np.eye(n, dtype=bool)
    pred_sum = np.zeros((n, n), dtype=np.float64)
    pred_count = 0
    for rep in range(n_repeats):
        logits_base, _ = _forward(model, dev, n, base_adj, rng, n_graphs)
        pred = logits_base > 0
        i, j = target_pair
        tc_list.append(float(pred[:, i, j].mean()))
        rr_list.append(float(pred[:, route_masks].mean()))
        pp_list.append(float(pred.mean()))
        # whole-graph exact match: this graph is one connected component (every route joins at
        # the target pair / hubs), so ground truth is "every off-diagonal pair connected" --
        # exact match iff pred is True on every off-diagonal entry (Edoardo, 2026-08-27: asked
        # for this number directly, was never computed by the original eval_behaviour, which
        # only pooled target-pair/route/global-positive-rate accuracy).
        ec_list.append(float(pred[:, offdiag].all(axis=1).mean()))
        # per-pair predicted-connected rate over every test graph and repeat -- ground truth is
        # "connected" everywhere, so this rate IS the per-pair accuracy; lets us see WHICH pairs
        # drive the near-zero exact match above, not just that some pair is wrong somewhere
        # (Edoardo, 2026-08-27: "sarebbe per capire quali coppie sbaglia").
        pred_sum += pred.sum(axis=0)
        pred_count += pred.shape[0]
    def agg(v): return float(np.mean(v)), float(np.std(v))
    tc_m, tc_s = agg(tc_list); rr_m, rr_s = agg(rr_list); pp_m, pp_s = agg(pp_list)
    ec_m, ec_s = agg(ec_list)
    per_pair_acc = pred_sum / pred_count
    return {"target_pair_connected": round(tc_m, 4), "target_pair_connected_std": round(tc_s, 4),
            "reach_route": round(rr_m, 4), "reach_route_std": round(rr_s, 4),
            "pred_positive_rate": round(pp_m, 4), "pred_positive_rate_std": round(pp_s, 4),
            "exact": round(ec_m, 4), "exact_std": round(ec_s, 4),
            "per_repeat_target_pair": [round(v, 4) for v in tc_list],
            "per_repeat_exact": [round(v, 4) for v in ec_list],
            "per_pair_acc": per_pair_acc}


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


def build_construction_c(n, route_lens, term_deg, cut_routes=(0, 1)):
    """Ablation requested by Edoardo (2026-08-27): same s,t hub structure as Construction A
    (term_deg=3, so s and t keep degree 3, i.e. they still look locally like hubs, not simple
    path endpoints) but sever two of the three routes by deleting one INTERNAL edge from each
    (an edge strictly between two interior nodes, never the edge touching s or t itself, so
    s/t's own degree is untouched). Only one route -- the one NOT in cut_routes -- still
    actually connects s to t. Tests whether the model's success on Construction A tracks real
    path redundancy, or just the superficial degree>1 cue at s/t: if accuracy on the target
    pair collapses back toward the plain single-path failure once only one route is real, the
    redundancy interpretation is confirmed; if it stays near 1.0, the model was keying off
    degree/hub-ness rather than genuine multi-route connectivity.

    Default cut_routes=(0,1) leaves route index 2 (the second length-route_lens[2] route) as
    the sole real path; with route_lens=[20,21,20] that cuts the 21-edge route and the first
    20-edge route, leaving the second 20-edge route intact.
    """
    built = generate_multipath_graph(n, len(route_lens), route_lens, np.random.default_rng(0),
                                      term_deg=term_deg)
    if built is None:
        raise ValueError(f"route_lens={route_lens} term_deg={term_deg} does not fit n={n}")
    adj, meta = built
    assert len(meta["leaves"]) == 0 and len(meta["filler"]) == 0
    s, t = meta["s"], meta["t"]
    within = np.zeros((n, n), dtype=bool)
    for route in meta["full_paths"]:
        idx = np.array([s, t] + route, dtype=int)
        within[np.ix_(idx, idx)] = True
    np.fill_diagonal(within, False)
    within[s, t] = within[t, s] = False
    bounds = [1.5]
    cur = 2
    for route in meta["full_paths"][:-1]:
        cur += len(route)
        bounds.append(cur - 0.5)

    cut_edges = []
    for r in cut_routes:
        nodes = meta["full_paths"][r]
        assert len(nodes) >= 2, f"route {r} too short to cut an internal edge ({len(nodes)} nodes)"
        mid = len(nodes) // 2
        u, v = nodes[mid - 1], nodes[mid]
        assert adj[u, v] == 1.0, f"expected an edge at ({u},{v})"
        adj[u, v] = adj[v, u] = 0.0
        cut_edges.append((int(u), int(v)))
    assert adj[s].sum() == term_deg and adj[t].sum() == term_deg, \
        "s/t degree changed -- cut edge must be strictly internal, not touching s or t"
    return adj, (s, t), within, bounds, cut_edges


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


def build_construction_d(n, route_lens, term_deg, core_n):
    """Degree-controlled follow-up requested by Edoardo (2026-08-27), NOT to be written up
    anywhere -- purely investigative. Construction A's s,t are the ONLY degree-3 nodes in the
    whole graph (every other node has degree <=2, since the K=3 path-union training
    distribution never contains a degree-3 node), which is a plausible confound: the model
    may be keying off s/t's rare degree signature rather than genuine path redundancy.

    This construction moves the tested pair one hop further out, to two new nodes with
    perfectly ORDINARY degree 2 -- no different from any other interior path node -- so the
    degree confound is gone:

        A --- B --- [hub s] === (3 parallel routes, route_lens) === [hub t] --- M --- N

    A/N: new degree-1 endpoints (the actual open ends of the graph now).
    B/M: new degree-2 nodes -- one edge to the true endpoint (A/N), one edge to the hub
         (s/t). THESE are the tested pair, not the hubs.
    s/t: unchanged from Construction A structurally (term_deg=3 route hub), but now ALSO
         carry the extra edge to B/M, so their own degree is 4 -- irrelevant here since s/t
         are not the tested pair anymore.

    route_lens should sum so that 2 (hubs) + sum(route_lens)-3 (route interior nodes) + 4
    (A,B,M,N) <= n; any leftover canvas is auto-filled by generate_multipath_graph into a
    small inert separate component (harmless, not connected to anything tested).
    """
    built = generate_multipath_graph(core_n, len(route_lens), route_lens,
                                      np.random.default_rng(0), term_deg=term_deg)
    if built is None:
        raise ValueError(f"route_lens={route_lens} term_deg={term_deg} does not fit core_n={core_n}")
    core_adj, meta = built
    s, t = meta["s"], meta["t"]

    adj = np.zeros((n, n), dtype=np.float32)
    adj[:core_n, :core_n] = core_adj
    b, a, m, nn = core_n, core_n + 1, core_n + 2, core_n + 3
    assert nn == n - 1, f"expected exactly 4 extra nodes to reach n={n}, got core_n={core_n}"
    adj[s, b] = adj[b, s] = 1.0
    adj[b, a] = adj[a, b] = 1.0
    adj[t, m] = adj[m, t] = 1.0
    adj[m, nn] = adj[nn, m] = 1.0

    within = np.zeros((n, n), dtype=bool)
    for route in meta["full_paths"]:
        idx = np.array([s, t] + route, dtype=int)
        within[np.ix_(idx, idx)] = True
    np.fill_diagonal(within, False)
    within[b, m] = within[m, b] = False   # tested pair excluded from the pooled route metric

    bounds = [core_n - 0.5]  # separates the core (hub+routes) from the new A,B,M,N tail nodes
    return adj, (b, m), within, bounds, dict(s=int(s), t=int(t), a=a, b=b, m=m, n_node=nn,
                                              filler=meta["filler"])


def build_construction_f(n, route_lens, term_deg):
    """Requested by Edoardo (2026-08-29): s,t joined by len(route_lens) independent routes
    (route_lens edges each -- default [20,20], both routes the same length this time, unlike
    Construction A's [20,21,20]), s/t at ordinary degree = len(route_lens) (2 here). The
    leftover canvas is left as ISOLATED nodes (no edges at all, not even a filler chain --
    "nodi liberi sparsi") rather than auto-filled into a chain. Target pair: s,t themselves.
    """
    built = generate_multipath_graph(n, len(route_lens), route_lens, np.random.default_rng(0),
                                      term_deg=term_deg, fill=False)
    if built is None:
        raise ValueError(f"route_lens={route_lens} term_deg={term_deg} does not fit n={n}")
    adj, meta = built
    assert len(meta["leaves"]) == 0, f"expected no leaves, got {len(meta['leaves'])}"
    s, t = meta["s"], meta["t"]
    within = np.zeros((n, n), dtype=bool)
    for route in meta["full_paths"]:
        idx = np.array([s, t] + route, dtype=int)
        within[np.ix_(idx, idx)] = True
    np.fill_diagonal(within, False)
    within[s, t] = within[t, s] = False
    bounds = [1.5]
    cur = 2
    for route in meta["full_paths"][:-1]:
        cur += len(route)
        bounds.append(cur - 0.5)
    return adj, (s, t), within, bounds, meta["filler"]


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
    ap.add_argument("--cut_routes_c", type=int, nargs="+", default=[0, 1],
                    help="Construction C (ablation): route indices to sever internally, "
                         "leaving s/t degree unchanged -- default cuts the 21-edge route "
                         "and the first 20-edge route, leaving the second 20-edge route as "
                         "the sole real s-t path")
    ap.add_argument("--route_lens_d", type=int, nargs=3, default=[18, 18, 19])
    ap.add_argument("--term_deg_d", type=int, default=3)
    ap.add_argument("--core_n_d", type=int, default=56,
                    help="Construction D: canvas for the hub+routes core; the remaining "
                         "n-core_n_d nodes are the new A,B,M,N tail (4) plus any leftover "
                         "auto-filler from generate_multipath_graph")
    ap.add_argument("--route_lens_f", type=int, nargs=2, default=[20, 20])
    ap.add_argument("--term_deg_f", type=int, default=2)
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

    adj_c, pair_c, within_c, bounds_c, cut_edges_c = build_construction_c(
        n, args.route_lens_a, args.term_deg_a, tuple(args.cut_routes_c))
    print(f"  [C_ablated] cut internal edges: {cut_edges_c} "
          f"(s,t degree preserved at {args.term_deg_a})", flush=True)

    adj_d, pair_d, within_d, bounds_d, info_d = build_construction_d(
        n, args.route_lens_d, args.term_deg_d, args.core_n_d)
    print(f"  [D_degreecontrol] tested pair (B,M)={pair_d}, degree "
          f"{int(adj_d[pair_d[0]].sum())}/{int(adj_d[pair_d[1]].sum())}; hubs s,t={info_d['s']},{info_d['t']} "
          f"degree {int(adj_d[info_d['s']].sum())}/{int(adj_d[info_d['t']].sum())}; "
          f"filler={info_d['filler']}", flush=True)

    adj_f, pair_f, within_f, bounds_f, filler_f = build_construction_f(
        n, args.route_lens_f, args.term_deg_f)
    print(f"  [F_isolated] target_pair={pair_f}, degree "
          f"{int(adj_f[pair_f[0]].sum())}/{int(adj_f[pair_f[1]].sum())} (ordinary); "
          f"isolated nodes={filler_f}", flush=True)

    results = {}
    for name, (adj, pair, within, bounds) in [
        ("A_clean", build_construction_a(n, args.route_lens_a, args.term_deg_a)),
        ("B_stitched", build_construction_b(n, tuple(args.disjoint_sizes_b))),
        ("C_ablated", (adj_c, pair_c, within_c, bounds_c)),
        ("D_degreecontrol", (adj_d, pair_d, within_d, bounds_d)),
        ("F_isolated", (adj_f, pair_f, within_f, bounds_f)),
    ]:
        rng = np.random.default_rng(args.seed)
        beh = eval_behaviour(model, dev, n, adj, pair, within, rng,
                              args.n_graphs_behaviour, args.n_repeats)
        beh["target_pair"] = pair
        if name == "C_ablated":
            beh["cut_edges"] = cut_edges_c
        if name == "F_isolated":
            beh["isolated_nodes"] = [int(x) for x in filler_f]
        if name == "D_degreecontrol":
            beh["hub_s_t"] = [info_d["s"], info_d["t"]]
            beh["tested_pair_degree"] = [int(adj[pair[0]].sum()), int(adj[pair[1]].sum())]
        print(f"  [{name}] target_pair={pair}: "
              f"connected={beh['target_pair_connected']:.3f}+-{beh['target_pair_connected_std']:.3f} "
              f"reach_route={beh['reach_route']:.3f}+-{beh['reach_route_std']:.3f} "
              f"pred_pos={beh['pred_positive_rate']:.3f}", flush=True)
        rng2 = np.random.default_rng(args.seed + 1000)
        Z = h2_Z_matrix(model, dev, n, adj, rng2, args.n_graphs_heatmap)
        np.savez(out / f"{name}_Z.npz", Z=Z, boundaries=bounds, target_pair=pair)
        plot_heatmap(Z, bounds, f"Construction {name}, $n$={n}, H$^{{(2)}}$ geometry ({args.tag})",
                     out / f"{name}_h2_heatmap.png")

        per_pair_acc = beh.pop("per_pair_acc")
        np.savez(out / f"{name}_per_pair_acc.npz", acc=per_pair_acc, boundaries=bounds,
                 target_pair=pair)
        results[name] = beh

    (out / "multipath_v2.json").write_text(json.dumps(results, indent=2))
    print(f"saved -> {out}/multipath_v2.json")


if __name__ == "__main__":
    main()
