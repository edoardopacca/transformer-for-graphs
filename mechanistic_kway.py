"""Report IX, Thread A.4 (mechanistic extension) -- behavioural sweep + raw similarity logit
for K-way (K=5,6,7) disjoint-path unions, the direct analogue of mechanistic_asym_chains.py's
Table-1/Figure-1 battery (Report VIII sec:battery-sweep) but for graphs with MORE components
than the path_union training stream ever produces (which draws only 1..4).

Reuses, unmodified, the generic (topology-independent) building blocks already validated in
mechanistic_asym_chains.py (run_with_cache, exact_contribution, the self-tests) and in
eval_multiway_split.py (the feasible small-size search, generate_multi_path_split_graph) --
only the graph construction and the long/short index split are new, since a K-way graph needs
one LONG component plus K-1 SHORT ones pooled into a single "S" side, rather than the single
two-way split (a, n-a) every other mechanistic script in this project sweeps over.

    python mechanistic_kway.py --checkpoint runs/.../last.pt \\
        --output_dir runs/report9/mechanistic_kway/<tag>
"""
import argparse, csv, json
from pathlib import Path

import numpy as np
import torch

from data import (add_self_loops, compute_connectivity_matrix,
                  compute_all_pairs_shortest_paths, generate_multi_path_split_graph)
from eval_families import load_model
from eval_multiway_split import default_small_sizes
from mechanistic_asym_chains import (_device, _selftest, weights_geometry_similarity)


def _build_kway_graph(n, k, small_size):
    """One long component + (k-1) short ones (each of size small_size), pooled into a single
    long-index array L and a single (union) short-index array S -- the natural K-way
    generalisation of the (a, n-a) two-way split used everywhere else in this project."""
    long_len = n - (k - 1) * small_size
    sizes = (long_len,) + (small_size,) * (k - 1)
    adj = generate_multi_path_split_graph(n, sizes)
    L = np.arange(0, long_len)
    S = np.arange(long_len, n)
    return adj, L, S


def behavioural_sweep_kway(model, dev, n, cells, rng, n_graphs):
    rows = []
    for k, s in cells:
        base_adj, L, S = _build_kway_graph(n, k, s)
        base_y = compute_connectivity_matrix(base_adj).astype(np.int8)
        base_dist = compute_all_pairs_shortest_paths(base_adj)

        xs = np.empty((n_graphs, n, n), np.float32)
        invs = []
        for i in range(n_graphs):
            p = rng.permutation(n)
            xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
            invs.append(np.argsort(p))
        xb = torch.from_numpy(xs).to(dev, torch.float32)
        preds = np.empty((n_graphs, n, n), np.int8)
        for st in range(0, n_graphs, 128):
            e = min(st + 128, n_graphs)
            logits = model.forward(xb[st:e])
            preds[st:e] = (logits > 0).detach().cpu().numpy().astype(np.int8)
        pred = np.empty_like(preds)
        for i, inv in enumerate(invs):
            pred[i] = preds[i][np.ix_(inv, inv)]
        eq = (pred == base_y[None])

        exact = float(eq.reshape(n_graphs, -1).all(1).mean())
        eqL = eq[:, L][:, :, L]; offL = ~np.eye(len(L), dtype=bool)
        reach_long = float(eqL[:, offL].mean())
        eqS = eq[:, S][:, :, S]; offS = ~np.eye(len(S), dtype=bool)
        reach_short = float(eqS[:, offS].mean()) if len(S) > 1 else float("nan")
        memb = np.zeros(n, dtype=np.int8); memb[L] = 1
        cmask = memb[:, None] != memb[None, :]
        cut = float(eq[:, cmask].mean())
        pos_rate = float(pred.mean())

        dL = base_dist[np.ix_(L, L)]
        cell_tag = f"k{k}_s{s}"
        for d_ in sorted(set(int(v) for v in dL[dL > 0])):
            m = dL == d_
            rows.append({"cell": cell_tag, "k": k, "small_size": s, "metric": "reach_long_by_dist",
                         "distance": d_, "value": round(float(eqL[:, m].mean()), 4), "n_pairs": int(m.sum())})
        for metric, value in [("exact", exact), ("reach_long", reach_long),
                              ("reach_short", reach_short), ("cut", cut),
                              ("pred_positive_rate", pos_rate)]:
            rows.append({"cell": cell_tag, "k": k, "small_size": s, "metric": metric,
                         "distance": None, "value": round(value, 4) if value == value else value,
                         "n_pairs": None})
        print(f"  k={k} s={s:>2d} exact={exact:.3f} reach_long={reach_long:.3f} "
              f"reach_short={reach_short:.3f} cut={cut:.3f}", flush=True)
    return rows


def readout_decomposition_kway_similarity(model, dev, n, cells, rng, n_graphs):
    rows = []
    for k, s in cells:
        base_adj, L, S = _build_kway_graph(n, k, s)
        base_dist = compute_all_pairs_shortest_paths(base_adj)

        xs = np.empty((n_graphs, n, n), np.float32)
        invs = []
        for i in range(n_graphs):
            p = rng.permutation(n)
            xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
            invs.append(np.argsort(p))
        xb = torch.from_numpy(xs).to(dev, torch.float32)
        h_all = np.empty((n_graphs, n, model.config.d_model), np.float32)
        for st in range(0, n_graphs, 128):
            e = min(st + 128, n_graphs)
            h = model.embeddings(xb[st:e])
            h_all[st:e] = h.detach().cpu().numpy()
        hn = h_all / (np.linalg.norm(h_all, axis=-1, keepdims=True) + 1e-9)
        cos_perm = np.einsum("gid,gjd->gij", hn, hn)
        scale = float(model.sim_scale.detach().cpu()); bias = float(model.sim_bias.detach().cpu())
        cos = np.empty_like(cos_perm)
        for i, inv in enumerate(invs):
            cos[i] = cos_perm[i][np.ix_(inv, inv)]
        z = scale * cos + bias

        cell_tag = f"k{k}_s{s}"
        for i_set, j_set, label in ((L, L, "within_long"), (S, S, "within_short"),
                                     (L, S, "cut"), (S, L, "cut")):
            if len(i_set) == 0 or len(j_set) == 0 or (label != "cut" and len(i_set) <= 1):
                continue
            sub = cos[:, i_set][:, :, j_set]
            subz = z[:, i_set][:, :, j_set]
            if label == "cut":
                vals, valsz = sub.reshape(sub.shape[0], -1), subz.reshape(subz.shape[0], -1)
            else:
                off = ~np.eye(len(i_set), dtype=bool)
                vals, valsz = sub[:, off], subz[:, off]
            rows.append({"cell": cell_tag, "k": k, "small_size": s, "pair_type": label,
                         "mean_cos": round(float(vals.mean()), 4),
                         "frac_positive": round(float((valsz > 0).mean()), 4),
                         "n_pairs": int(vals.size)})
        dL = base_dist[np.ix_(L, L)]
        cosL = cos[:, L][:, :, L]; zL = z[:, L][:, :, L]
        for d_ in sorted(set(int(v) for v in dL[dL > 0])):
            m = dL == d_
            rows.append({"cell": cell_tag, "k": k, "small_size": s, "pair_type": f"within_long_d{d_}",
                         "mean_cos": round(float(cosL[:, m].mean()), 4),
                         "frac_positive": round(float((zL[:, m] > 0).mean()), 4),
                         "n_pairs": int(m.sum())})
        print(f"  readout(similarity) k={k} s={s:>2d} done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=200)
    ap.add_argument("--n_components", type=int, nargs="+", default=[5, 6, 7])
    ap.add_argument("--small_sizes", type=int, nargs="+", default=None,
                    help="override the small-component-size sweep for EVERY K (default: "
                         "auto, filtered so the long component's diameter exceeds --dist_cutoff)")
    ap.add_argument("--dist_cutoff", type=int, default=18)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--skip_selftest", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")
    if arch != "roberta":
        raise NotImplementedError("this script targets the RobertaGraphTransformer only")
    if readout != "similarity":
        raise NotImplementedError("this script is specific to the similarity read-out "
                                   "(Report IX standardises on it, like Report VIII)")
    if not args.skip_selftest:
        _selftest(model, dev, n)

    cells = []
    for k in args.n_components:
        sizes = args.small_sizes if args.small_sizes is not None else \
            default_small_sizes(n, k, args.dist_cutoff)
        for s in sizes:
            if n - (k - 1) * s >= 1:
                cells.append((k, s))
    if not cells:
        raise ValueError("no feasible (k, small_size) cells at this n/dist_cutoff")
    print(f"  cells: {cells}")

    rng = np.random.default_rng(args.seed)
    print("\n== behavioural sweep (Table 1 / Figure 1, top panel) ==")
    rows = behavioural_sweep_kway(model, dev, n, cells, rng, args.n_graphs)
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "k", "small_size", "metric", "distance", "value", "n_pairs"])
        w.writeheader(); w.writerows(rows)
    print(f"  saved -> {out}/metrics.csv")

    print("\n== readout decomposition cos(h_i,h_j) (Figure 1, bottom panel) ==")
    rrows = readout_decomposition_kway_similarity(model, dev, n, cells, rng, args.n_graphs)
    with (out / "readout.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "k", "small_size", "pair_type", "mean_cos", "frac_positive", "n_pairs"])
        w.writeheader(); w.writerows(rrows)
    print(f"  saved -> {out}/readout.csv")

    print("\n== W_in geometry (similarity read-out has no W_out) ==")
    wg = weights_geometry_similarity(model)
    wg["readout_kind"] = readout
    (out / "weights_summary.json").write_text(json.dumps(wg))
    print(f"  saved -> {out}/weights_summary.json")


if __name__ == "__main__":
    main()
