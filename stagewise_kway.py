"""Report IX, Thread A.4 (mechanistic extension) -- layerwise similarity geometry (the
stagewise "where is information combined" probe, Report VIII sec:stagewise) for K-way
(K=5,6,7) disjoint-path unions. Companion to mechanistic_kway.py/mechanistic_kway_heatmaps.py;
reuses run_with_stages/_cosine_batch (via stagewise_diagnostics) unmodified -- only the graph
construction is new. Saves scale/bias in the npz from the start (Report IX rescale fix,
istruzioni.md), so the cosine-geometry heatmap plots Z=scale*cos+bias, not raw cosine.

    python stagewise_kway.py --checkpoint runs/.../last.pt \\
        --output_dir runs/report9/stagewise_kway/<tag> --cells 5,4 6,3 7,2
"""
import argparse, csv
from pathlib import Path

import numpy as np
import torch

from data import compute_connectivity_matrix, compute_all_pairs_shortest_paths, add_self_loops
from eval_families import load_model
from mechanistic_asym_chains import _device
from mechanistic_kway import _build_kway_graph
from stagewise_diagnostics import run_with_stages, _cosine_batch, MAIN_STAGES, SUBBLOCKS, _selftest


def stagewise_probe_kway(model, dev, n, cells, rng, n_graphs, cap=9):
    if model.readout_kind != "similarity":
        raise NotImplementedError("specific to the similarity read-out (Report VIII/IX standard)")
    scale = float(model.sim_scale.detach().cpu())
    bias = float(model.sim_bias.detach().cpu())
    out = {}
    for k, s in cells:
        base_adj, L, S = _build_kway_graph(n, k, s)
        base_y = compute_connectivity_matrix(base_adj).astype(np.int8)
        base_dist = compute_all_pairs_shortest_paths(base_adj)
        offL = ~np.eye(len(L), dtype=bool)
        offS = ~np.eye(len(S), dtype=bool) if len(S) > 1 else None
        dL = base_dist[np.ix_(L, L)]
        near_mask = offL & (dL <= cap)
        far_mask = offL & (dL > cap)
        memb = np.zeros(n, dtype=np.int8); memb[L] = 1
        cmask = memb[:, None] != memb[None, :]

        xs = np.empty((n_graphs, n, n), np.float32)
        invs = []
        for i in range(n_graphs):
            p = rng.permutation(n)
            xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
            invs.append(np.argsort(p))
        xb = torch.from_numpy(xs).to(dev, torch.float32)
        stages, _, _ = run_with_stages(model, xb)

        def unperm(arr):
            return np.stack([arr[i][inv] for i, inv in enumerate(invs)])

        G = {}
        for X in MAIN_STAGES:
            arr_base = unperm(stages[X])
            G_all = _cosine_batch(arr_base)
            G[X] = G_all.mean(0)

        metrics_rows, margin_rows = [], []
        cell_tag = f"k{k}_s{s}"
        for X in MAIN_STAGES:
            arr_base = unperm(stages[X])
            G_all = _cosine_batch(arr_base)
            Z_all = scale * G_all + bias
            Rhat = (Z_all > 0)
            eq = (Rhat == base_y[None])
            exact = float(eq.reshape(n_graphs, -1).all(1).mean())
            reach_long = float(eq[:, L][:, :, L][:, offL].mean())
            reach_long_near = float(eq[:, L][:, :, L][:, near_mask].mean()) if near_mask.any() else float("nan")
            reach_long_far = float(eq[:, L][:, :, L][:, far_mask].mean()) if far_mask.any() else float("nan")
            reach_short = float(eq[:, S][:, :, S][:, offS].mean()) if offS is not None else float("nan")
            cut = float(eq[:, cmask].mean())
            pos_rate = float(Rhat.mean())
            for metric, value in [("exact", exact), ("reach_long", reach_long),
                                  ("reach_long_near", reach_long_near),
                                  ("reach_long_far", reach_long_far),
                                  ("reach_short", reach_short), ("cut", cut),
                                  ("pred_positive_rate", pos_rate)]:
                metrics_rows.append({"cell": cell_tag, "k": k, "small_size": s, "stage": X,
                                     "metric": metric, "value": round(value, 5) if value == value else value})

            Gx = G[X]
            mu_short = float(Gx[np.ix_(S, S)][offS].mean()) if offS is not None else float("nan")
            mu_long_near = float(Gx[np.ix_(L, L)][near_mask].mean()) if near_mask.any() else float("nan")
            mu_long_far = float(Gx[np.ix_(L, L)][far_mask].mean()) if far_mask.any() else float("nan")
            mu_cross = float(Gx[np.ix_(L, S)].mean())
            m_far = mu_long_far - mu_cross if mu_long_far == mu_long_far else float("nan")
            for quantity, value in [("mu_short", mu_short), ("mu_long_near", mu_long_near),
                                    ("mu_long_far", mu_long_far), ("mu_cross", mu_cross),
                                    ("M_far", m_far)]:
                margin_rows.append({"cell": cell_tag, "k": k, "small_size": s, "stage": X,
                                    "quantity": quantity, "value": round(value, 5) if value == value else value})

        Z = {X: scale * G[X] + bias for X in MAIN_STAGES}
        dZ = {name: Z[to] - Z[frm] for name, to, frm in SUBBLOCKS}
        deltaz_rows = []
        for branch, mat in dZ.items():
            cats = [("within_short", mat[np.ix_(S, S)][offS].mean() if offS is not None else float("nan")),
                    ("within_long_near", mat[np.ix_(L, L)][near_mask].mean() if near_mask.any() else float("nan")),
                    ("within_long_far", mat[np.ix_(L, L)][far_mask].mean() if far_mask.any() else float("nan")),
                    ("cross", mat[np.ix_(L, S)].mean())]
            for cat, value in cats:
                deltaz_rows.append({"cell": cell_tag, "k": k, "small_size": s, "branch": branch,
                                    "category": cat, "value": round(float(value), 5) if value == value else value})

        out[cell_tag] = {"G": G, "dZ": dZ, "long_idx": L, "short_idx": S,
                        "metrics_rows": metrics_rows, "margin_rows": margin_rows,
                        "deltaz_rows": deltaz_rows}
        print(f"  stagewise k={k} s={s:>2d} done "
              f"(M_far H0={[r['value'] for r in margin_rows if r['stage']=='H0' and r['quantity']=='M_far'][0]:.3f}, "
              f"H2={[r['value'] for r in margin_rows if r['stage']=='H2' and r['quantity']=='M_far'][0]:.3f})",
              flush=True)
    return out, scale, bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=64)
    ap.add_argument("--cells", type=str, nargs="+", required=True,
                    help="K,small_size pairs, e.g. --cells 5,4 6,3 7,2")
    ap.add_argument("--cap", type=int, default=9)
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
    if not args.skip_selftest:
        _selftest(model, dev, n)

    cells = [tuple(int(x) for x in c.split(",")) for c in args.cells]
    rng = np.random.default_rng(args.seed)
    probe, scale, bias = stagewise_probe_kway(model, dev, n, cells, rng, args.n_graphs, cap=args.cap)

    npz_dict = {"scale": np.float64(scale), "bias": np.float64(bias)}
    metrics_rows, margin_rows, deltaz_rows = [], [], []
    for cell_tag, d in probe.items():
        for X, mat in d["G"].items():
            npz_dict[f"{cell_tag}__G_{X}"] = mat
        for branch, mat in d["dZ"].items():
            npz_dict[f"{cell_tag}__{branch}"] = mat
        npz_dict[f"{cell_tag}__long_idx"] = d["long_idx"]
        npz_dict[f"{cell_tag}__short_idx"] = d["short_idx"]
        metrics_rows += d["metrics_rows"]
        margin_rows += d["margin_rows"]
        deltaz_rows += d["deltaz_rows"]
    np.savez_compressed(out / "stagewise_geometry.npz", **npz_dict)
    with (out / "stagewise_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "k", "small_size", "stage", "metric", "value"])
        w.writeheader(); w.writerows(metrics_rows)
    with (out / "stagewise_margins.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "k", "small_size", "stage", "quantity", "value"])
        w.writeheader(); w.writerows(margin_rows)
    with (out / "stagewise_deltaz.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "k", "small_size", "branch", "category", "value"])
        w.writeheader(); w.writerows(deltaz_rows)
    print(f"  saved -> {out}/stagewise_{{geometry.npz,metrics.csv,margins.csv,deltaz.csv}}")


if __name__ == "__main__":
    main()
