"""Report X -- does the n=60 path_union_k3 checkpoint (trained ONLY on 3 disjoint path
components, never on parallel/shared-endpoint routes) benefit from extra parallel routes
between two terminals the way every path_union-trained checkpoint since Report VI has, and
what does its final-layer geometry look like when it does?

Two terminals s,t are joined by k in {1,2,3} internally-disjoint paths of the SAME length
(default 19, chosen so terminal degree stays fixed at ``term_deg`` for every k -- k is the
only thing that varies, data.py::generate_multipath_graph, the leaf-padded + sparse-filler
construction Report VI/IX already used for this exact question on path_union-trained
checkpoints). This is a narrower, geometry-focused companion to eval_multipath.py (which
already sweeps many (k, ell) cells but only reports behavioural accuracy, no embeddings) --
run that too for the broader accuracy sweep; this script's only reason to exist is the
final-layer similarity geometry eval_multipath.py does not compute.

For each k, over ``n_graphs`` independent random relabellings of the same canonical graph:
  * term_connect        -- predicted-connected accuracy on the (s,t) pair (target 1), the
                            headline "do extra routes rescue the connection" number.
  * reach_route         -- predicted-connected accuracy pooled over within-route pairs
                            (both endpoints on the same route, including s/t), target 1.
  * pred_positive_rate  -- fraction of ALL pairs in the graph predicted connected.
  * H2 (final-stage, after the last layer's MLP) cosine-similarity geometry, block-averaged
    by node-pair category: s-t itself, within-route (other pairs on the same route),
    route-to-route (internal nodes of two DIFFERENT routes -- never queried directly by
    term_connect, but genuinely connected via s/t), terminal-to-own-leaf, leaf-to-leaf,
    active-to-filler (should read as disconnected: filler is a separate component by
    construction), within-filler (should read as connected). Every category's mean cosine
    is also converted through the checkpoint's own scale/bias into an implied logit and
    predicted-connected call, so the geometry table can be read directly against the
    checkpoint's actual decision rule.

Eval-only, CPU-friendly (forward passes only, run_with_stages from stagewise_diagnostics.py).

    python mechanistic_multipath_geometry.py --checkpoint runs/.../last.pt \\
        --output_dir runs/report10/multipath_geometry/<tag> --ks 1 2 3 --path_len 19
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import add_self_loops, generate_multipath_graph
from eval_families import load_model
from stagewise_diagnostics import run_with_stages, _cosine_batch, _device, _selftest

BLUE = "#0072B2"
KCOLORS = {1: "#999999", 2: "#0072B2", 3: "#D55E00"}
CATEGORIES = ["s_t", "within_route", "route_to_route", "terminal_leaf", "leaf_leaf",
              "active_filler", "within_filler"]
CAT_LABEL = {"s_t": "$s$–$t$ (headline pair)", "within_route": "within a route",
             "route_to_route": "route $i$ vs route $j$", "terminal_leaf": "terminal–own leaf",
             "leaf_leaf": "leaf–leaf", "active_filler": "active–filler (should cut)",
             "within_filler": "within filler (should reach)"}


def _category_masks(n, meta):
    s, t = meta["s"], meta["t"]
    routes = meta["full_paths"]                       # list of k lists of internal nodes
    leaves = np.array(meta["leaves"], dtype=int)
    filler = np.array(meta["filler"], dtype=int)
    active = np.array([s, t] + [x for r in routes for x in r] + list(leaves), dtype=int)

    def pairmask(idx_a, idx_b, exclude_diag=False):
        m = np.zeros((n, n), dtype=bool)
        m[np.ix_(idx_a, idx_b)] = True
        m[np.ix_(idx_b, idx_a)] = True
        if exclude_diag:
            np.fill_diagonal(m, False)
        return m

    m_st = np.zeros((n, n), dtype=bool); m_st[s, t] = m_st[t, s] = True

    m_within = np.zeros((n, n), dtype=bool)
    for r in routes:
        idx = np.array([s, t] + r, dtype=int)
        m_within |= pairmask(idx, idx, exclude_diag=True)
    m_within &= ~m_st  # keep s-t as its own category only

    m_cross_route = np.zeros((n, n), dtype=bool)
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            m_cross_route |= pairmask(np.array(routes[i]), np.array(routes[j]))

    m_term_leaf = pairmask(np.array([s, t]), leaves) if len(leaves) else np.zeros((n, n), bool)
    m_leaf_leaf = pairmask(leaves, leaves, exclude_diag=True) if len(leaves) else np.zeros((n, n), bool)
    m_active_filler = pairmask(active, filler) if len(filler) else np.zeros((n, n), bool)
    m_within_filler = pairmask(filler, filler, exclude_diag=True) if len(filler) > 1 else np.zeros((n, n), bool)

    return {"s_t": m_st, "within_route": m_within, "route_to_route": m_cross_route,
            "terminal_leaf": m_term_leaf, "leaf_leaf": m_leaf_leaf,
            "active_filler": m_active_filler, "within_filler": m_within_filler}


def eval_k(model, dev, n, k, path_len, term_deg, rng, n_graphs):
    built = generate_multipath_graph(n, k, path_len, rng, term_deg=term_deg)
    if built is None:
        raise ValueError(f"k={k}, path_len={path_len}, term_deg={term_deg} does not fit n={n}")
    base_adj, meta = built
    s, t = meta["s"], meta["t"]
    routes = meta["full_paths"]
    masks = _category_masks(n, meta)

    xs = np.empty((n_graphs, n, n), np.float32)
    invs = []
    for i in range(n_graphs):
        p = rng.permutation(n)
        xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
        invs.append(np.argsort(p))
    xb = torch.from_numpy(xs).to(dev, torch.float32)
    stages, _, logits = run_with_stages(model, xb)

    def unperm(arr):
        return np.stack([arr[i][inv] for i, inv in enumerate(invs)])

    logits_base = unperm(logits)                       # [G,n,n]
    pred = logits_base > 0
    term_connect = float(pred[:, s, t].mean())
    within_route_pairs = np.zeros((n, n), dtype=bool)
    for r in routes:
        idx = np.array([s, t] + r, dtype=int)
        within_route_pairs[np.ix_(idx, idx)] = True
    np.fill_diagonal(within_route_pairs, False)
    reach_route = float(pred[:, within_route_pairs].mean())
    pred_positive_rate = float(pred.mean())

    h2_base = unperm(stages["H2"])                      # [G,n,d]
    G2_per_graph = _cosine_batch(h2_base)               # [G,n,n], NOT averaged over graphs
    G2 = G2_per_graph.mean(0)                           # [n,n] mean cosine, base order
    scale = float(model.sim_scale.detach().cpu())
    bias = float(model.sim_bias.detach().cpu())

    geometry = {}
    for cat, m in masks.items():
        if not m.any():
            geometry[cat] = None
            continue
        mu_cos = float(G2[m].mean())
        logit = scale * mu_cos + bias
        geometry[cat] = {"mean_cosine_H2": round(mu_cos, 4), "implied_logit": round(logit, 4),
                          "implied_connected": bool(logit > 0)}

    # s-t specifically: per-graph (not pre-averaged) cosine, to check whether the mean is
    # representative of a typical graph or pulled up by a minority (the mean-vs-term_connect
    # paradox flagged in report/10 sec:n60-multipath).
    st_cos_per_graph = G2_per_graph[:, s, t]            # [G]
    st_logit_per_graph = scale * st_cos_per_graph + bias
    geometry["s_t"]["std_cosine_H2"] = round(float(st_cos_per_graph.std()), 4)
    geometry["s_t"]["frac_positive_logit"] = round(float((st_logit_per_graph > 0).mean()), 4)

    return {"k": k, "path_len": path_len, "term_deg": term_deg, "n_graphs": n_graphs,
            "term_connect": round(term_connect, 4), "reach_route": round(reach_route, 4),
            "pred_positive_rate": round(pred_positive_rate, 4), "geometry_H2": geometry}


def make_figures(results, out_dir, term_connect_err=None):
    ks = [r["k"] for r in results]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    yerr = term_connect_err if term_connect_err is not None else None
    ax.errorbar(ks, [r["term_connect"] for r in results], yerr=yerr,
                fmt="-o", color=BLUE, ms=8, lw=2, capsize=4)
    ax.set_xlabel("number of parallel routes $k$"); ax.set_ylabel("term_connect (s-t predicted connected)")
    ax.set_ylim(-0.03, 1.05); ax.set_xticks(ks); ax.grid(alpha=0.3)
    if yerr is not None:
        ax.set_title("error bars: std across independent seed repeats", fontsize=10, color="#555555")
    fig.tight_layout(); fig.savefig(out_dir / "term_connect_vs_k.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(results)
    x = np.arange(len(CATEGORIES))
    for i, r in enumerate(results):
        vals = [r["geometry_H2"][c]["mean_cosine_H2"] if r["geometry_H2"][c] else np.nan for c in CATEGORIES]
        ax.bar(x + i * width, vals, width=width, color=KCOLORS.get(r["k"], "gray"), label=f"k={r['k']}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels([CAT_LABEL[c] for c in CATEGORIES], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("mean cosine similarity, $H^{(2)}$ (final stage)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "h2_geometry_by_category.png", dpi=150); plt.close(fig)
    print("wrote", out_dir / "term_connect_vs_k.png", "and h2_geometry_by_category.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=50)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--path_len", type=int, default=19)
    ap.add_argument("--term_deg", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345,
                    help="base seed; each of --n_repeats repeats uses base+r, r=0..n_repeats-1, "
                         "so DIFFERENT checkpoints must be given different --seed to draw "
                         "independent graphs (report/10 sec:n60-multipath's methodological "
                         "caveat: without this, two checkpoints evaluated with the same "
                         "default --seed see IDENTICAL relabelled graphs, since "
                         "generate_multipath_graph's base graph is itself deterministic)")
    ap.add_argument("--n_repeats", type=int, default=1,
                    help="independent repeats per k, each with a fresh rng seeded base+r; "
                         "report/10 asked for repeats specifically to get a seed-repeat error "
                         "bar on term_connect instead of a single point estimate")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    if readout != "similarity":
        raise NotImplementedError("this script reads model.sim_scale/sim_bias directly")
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev} "
          f"seed_base={args.seed} n_repeats={args.n_repeats}")
    _selftest(model, dev, n)

    per_repeat = {k: [] for k in args.ks}   # k -> list of per-repeat result dicts
    for rep in range(args.n_repeats):
        rng = np.random.default_rng(args.seed + rep)
        for k in args.ks:
            r = eval_k(model, dev, n, k, args.path_len, args.term_deg, rng, args.n_graphs)
            per_repeat[k].append(r)
            print(f"  repeat={rep} k={k}: term_connect={r['term_connect']:.3f} "
                  f"reach_route={r['reach_route']:.3f} pred_pos={r['pred_positive_rate']:.3f} "
                  f"s_t_mean_cos={r['geometry_H2']['s_t']['mean_cosine_H2']:+.3f} "
                  f"s_t_std_cos={r['geometry_H2']['s_t']['std_cosine_H2']:.3f}", flush=True)

    def agg(vals):
        arr = np.array(vals, dtype=float)
        return float(arr.mean()), float(arr.std())

    results = []
    for k in args.ks:
        reps = per_repeat[k]
        tc_mean, tc_std = agg([r["term_connect"] for r in reps])
        rr_mean, rr_std = agg([r["reach_route"] for r in reps])
        pp_mean, pp_std = agg([r["pred_positive_rate"] for r in reps])
        geometry_agg = {}
        for cat in CATEGORIES:
            vals = [r["geometry_H2"][cat] for r in reps if r["geometry_H2"][cat] is not None]
            if not vals:
                geometry_agg[cat] = None
                continue
            mu_mean, mu_std = agg([v["mean_cosine_H2"] for v in vals])
            entry = {"mean_cosine_H2": round(mu_mean, 4), "mean_cosine_H2_std_across_repeats": round(mu_std, 4),
                      "implied_logit": round(float(model.sim_scale.detach().cpu()) * mu_mean
                                              + float(model.sim_bias.detach().cpu()), 4)}
            if cat == "s_t":
                entry["std_cosine_H2_within_repeat_mean"] = round(
                    float(np.mean([v["std_cosine_H2"] for v in vals])), 4)
                entry["frac_positive_logit_mean"] = round(
                    float(np.mean([v["frac_positive_logit"] for v in vals])), 4)
            geometry_agg[cat] = entry
        results.append({"k": k, "n_repeats": args.n_repeats,
                         "term_connect": round(tc_mean, 4), "term_connect_std": round(tc_std, 4),
                         "reach_route": round(rr_mean, 4), "reach_route_std": round(rr_std, 4),
                         "pred_positive_rate": round(pp_mean, 4), "pred_positive_rate_std": round(pp_std, 4),
                         "geometry_H2": geometry_agg, "per_repeat": reps})
        print(f"  AGGREGATE k={k}: term_connect={tc_mean:.3f}+-{tc_std:.3f} "
              f"reach_route={rr_mean:.3f}+-{rr_std:.3f}", flush=True)

    (out / "multipath_geometry.json").write_text(json.dumps(results, indent=2))
    print(f"  saved -> {out}/multipath_geometry.json")
    term_connect_err = [r["term_connect_std"] for r in results] if args.n_repeats > 1 else None
    make_figures(results, out, term_connect_err=term_connect_err)


if __name__ == "__main__":
    main()
