"""Report IX, Thread A.3 (mechanistic extension) -- layerwise similarity geometry (the
stagewise "where is information combined" probe) for specific three-way split-size
combinations, ARBITRARY (s1,s2,s3) (three separate components, not the K-way pooled
"long"/"short" shape of stagewise_kway.py). Companion to mechanistic_threeway_heatmaps.py and
eval_threeway_splitchains.py's behavioural sweep. Reuses run_with_stages/_cosine_batch
(stagewise_diagnostics.py) unmodified. Saves scale/bias in the npz (Report IX rescale
convention), so the cosine-geometry heatmap plots Z=scale*cos+bias, not raw cosine.

    python stagewise_threeway.py --checkpoint runs/.../last.pt \\
        --output_dir runs/report9/stagewise_threeway/<tag> --cells 1,22,23 4,15,27 2,10,34 15,15,16
"""
import argparse, csv
from pathlib import Path

import numpy as np
import torch

from data import compute_connectivity_matrix, add_self_loops, generate_multi_path_split_graph
from eval_families import load_model
from mechanistic_asym_chains import _device
from stagewise_diagnostics import run_with_stages, _cosine_batch, MAIN_STAGES, SUBBLOCKS, _selftest

PAIRS = [((0, 1), "12"), ((0, 2), "13"), ((1, 2), "23")]


def stagewise_probe_threeway(model, dev, n, cells, rng, n_graphs):
    if model.readout_kind != "similarity":
        raise NotImplementedError("specific to the similarity read-out (Report VIII/IX standard)")
    scale = float(model.sim_scale.detach().cpu())
    bias = float(model.sim_bias.detach().cpu())
    out = {}
    for sizes in cells:
        base_adj = generate_multi_path_split_graph(n, sizes)
        base_y = compute_connectivity_matrix(base_adj).astype(np.int8)
        bounds = [0]
        for s in sizes:
            bounds.append(bounds[-1] + s)
        comps = [np.arange(bounds[i], bounds[i + 1]) for i in range(3)]
        offs = [~np.eye(len(c), dtype=bool) if len(c) > 1 else None for c in comps]

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

        cell_tag = f"s{sizes[0]}_{sizes[1]}_{sizes[2]}"
        metrics_rows, margin_rows = [], []
        for X in MAIN_STAGES:
            arr_base = unperm(stages[X])
            G_all = _cosine_batch(arr_base)
            Z_all = scale * G_all + bias
            Rhat = (Z_all > 0)
            eq = (Rhat == base_y[None])
            exact = float(eq.reshape(n_graphs, -1).all(1).mean())
            reach = []
            for c, off in zip(comps, offs):
                r = float(eq[:, c][:, :, c][:, off].mean()) if off is not None else float("nan")
                reach.append(r)
            cuts = {}
            for (a, b), name in PAIRS:
                cuts[name] = float(eq[:, comps[a]][:, :, comps[b]].mean())
            pos_rate = float(Rhat.mean())
            metrics_rows.append({"cell": cell_tag, "s1": sizes[0], "s2": sizes[1], "s3": sizes[2],
                                 "stage": X, "exact": round(exact, 5),
                                 "reach_1": round(reach[0], 5) if reach[0] == reach[0] else reach[0],
                                 "reach_2": round(reach[1], 5) if reach[1] == reach[1] else reach[1],
                                 "reach_3": round(reach[2], 5) if reach[2] == reach[2] else reach[2],
                                 "cut_12": round(cuts["12"], 5), "cut_13": round(cuts["13"], 5),
                                 "cut_23": round(cuts["23"], 5),
                                 "pred_positive_rate": round(pos_rate, 5)})

            Gx = G[X]
            mu = []
            for c, off in zip(comps, offs):
                mu.append(float(Gx[np.ix_(c, c)][off].mean()) if off is not None else float("nan"))
            mu_cross = {}
            for (a, b), name in PAIRS:
                mu_cross[name] = float(Gx[np.ix_(comps[a], comps[b])].mean())
            margin_rows.append({"cell": cell_tag, "s1": sizes[0], "s2": sizes[1], "s3": sizes[2],
                                "stage": X,
                                "mu_1": round(mu[0], 5) if mu[0] == mu[0] else mu[0],
                                "mu_2": round(mu[1], 5) if mu[1] == mu[1] else mu[1],
                                "mu_3": round(mu[2], 5) if mu[2] == mu[2] else mu[2],
                                "mu_12": round(mu_cross["12"], 5), "mu_13": round(mu_cross["13"], 5),
                                "mu_23": round(mu_cross["23"], 5)})

        Z = {X: scale * G[X] + bias for X in MAIN_STAGES}
        dZ = {name: Z[to] - Z[frm] for name, to, frm in SUBBLOCKS}

        out[cell_tag] = {"G": G, "dZ": dZ, "comps": comps,
                         "metrics_rows": metrics_rows, "margin_rows": margin_rows}
        print(f"  stagewise sizes={sizes} done "
              f"(exact H0={metrics_rows[0]['exact']:.3f}, H2={metrics_rows[-1]['exact']:.3f}, "
              f"cut12/13/23 @H2={metrics_rows[-1]['cut_12']:.3f}/"
              f"{metrics_rows[-1]['cut_13']:.3f}/{metrics_rows[-1]['cut_23']:.3f})", flush=True)
    return out, scale, bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=64)
    ap.add_argument("--cells", type=str, nargs="+", required=True,
                    help="s1,s2,s3 triples summing to n, e.g. --cells 1,22,23 4,15,27")
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
    for c in cells:
        if sum(c) != n:
            raise ValueError(f"cell {c} does not sum to n={n}")
    rng = np.random.default_rng(args.seed)
    probe, scale, bias = stagewise_probe_threeway(model, dev, n, cells, rng, args.n_graphs)

    npz_dict = {"scale": np.float64(scale), "bias": np.float64(bias)}
    metrics_rows, margin_rows = [], []
    for cell_tag, d in probe.items():
        for X, mat in d["G"].items():
            npz_dict[f"{cell_tag}__G_{X}"] = mat
        for branch, mat in d["dZ"].items():
            npz_dict[f"{cell_tag}__{branch}"] = mat
        for ci, c in enumerate(d["comps"]):
            npz_dict[f"{cell_tag}__comp{ci + 1}_idx"] = c
        metrics_rows += d["metrics_rows"]
        margin_rows += d["margin_rows"]
    np.savez_compressed(out / "stagewise_geometry.npz", **npz_dict)
    with (out / "stagewise_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        w.writeheader(); w.writerows(metrics_rows)
    with (out / "stagewise_margins.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(margin_rows[0].keys()))
        w.writeheader(); w.writerows(margin_rows)
    print(f"  saved -> {out}/stagewise_{{geometry.npz,metrics.csv,margins.csv}}")


if __name__ == "__main__":
    main()
