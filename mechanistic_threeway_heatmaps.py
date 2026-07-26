"""Report IX, Thread A.3 (mechanistic extension) -- real attention-score/alpha heatmaps for
specific three-way split-size combinations, ARBITRARY (s1,s2,s3) (not the K-way "one long +
(K-1) equal short" shape of mechanistic_kway_heatmaps.py). Companion to
eval_threeway_splitchains.py's behavioural sweep: run this on whichever cells from that sweep's
CSV are worth a closer look. Reuses run_with_cache/exact_contribution
(mechanistic_asym_chains.py) and raw_weights (mechanistic_heatmaps.py) unmodified -- only the
graph construction (three separate, non-pooled components) is new.

    python mechanistic_threeway_heatmaps.py --checkpoint runs/.../last.pt \\
        --output_dir runs/report9/heatmaps_threeway/<tag> --cells 1,22,23 4,15,27 2,10,34 15,15,16
"""
import argparse, math
from pathlib import Path

import numpy as np
import torch

from eval_families import load_model
from mechanistic_asym_chains import _device, run_with_cache, exact_contribution
from mechanistic_heatmaps import raw_weights
from data import add_self_loops, generate_multi_path_split_graph


def heatmap_probe_threeway(model, dev, n, cells, rng, n_graphs, contrib_n_graphs=8):
    out = {}
    for sizes in cells:
        base_adj = generate_multi_path_split_graph(n, sizes)
        bounds = [0]
        for s in sizes:
            bounds.append(bounds[-1] + s)
        comps = [np.arange(bounds[i], bounds[i + 1]) for i in range(3)]

        xs = np.empty((n_graphs, n, n), np.float32)
        invs = []
        for i in range(n_graphs):
            p = rng.permutation(n)
            xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
            invs.append(np.argsort(p))
        xb = torch.from_numpy(xs).to(dev, torch.float32)
        cache, h_final, logits = run_with_cache(model, xb)

        def unperm_pair(mat):
            o = np.empty_like(mat)
            for i, inv in enumerate(invs):
                o[i] = mat[i][np.ix_(inv, inv)]
            return o.mean(0)

        def unperm_node(mat):
            o = np.empty_like(mat)
            for i, inv in enumerate(invs):
                o[i] = mat[i][inv]
            return o.mean(0)

        d = {}
        for li in (0, 1):
            q, kk, v = cache[f"layer{li}_q"], cache[f"layer{li}_k"], cache[f"layer{li}_v"]
            alpha = cache[f"layer{li}_alpha"]
            wo_v = cache[f"layer{li}_wo_v"]
            head_dim = q.shape[-1]
            scores = np.einsum("gid,gjd->gij", q, kk) / math.sqrt(head_dim)
            contrib = alpha * np.linalg.norm(wo_v, axis=-1)[:, None, :]
            d[f"scores{li}"] = unperm_pair(scores)
            d[f"alpha{li}"] = unperm_pair(alpha)
            d[f"contrib{li}"] = unperm_pair(contrib)
            d[f"row_mass{li}"] = unperm_pair(alpha).sum(-1)
            d[f"q{li}"] = unperm_node(q)
            d[f"k{li}"] = unperm_node(kk)
            d[f"v{li}"] = unperm_node(v)
        h0_all = cache["h0"]
        g_idx = np.arange(min(contrib_n_graphs, n_graphs))
        contrib_exact_list = []
        for gi in g_idx:
            h0_row = torch.from_numpy(h0_all[gi]).to(dev, torch.float32)
            C = exact_contribution(model, h0_row)
            contrib_exact_list.append(C[np.ix_(invs[gi], invs[gi])])
        d["contrib_exact"] = np.stack(contrib_exact_list).mean(0)
        d["comp1_idx"], d["comp2_idx"], d["comp3_idx"] = comps
        cell_key = f"s{sizes[0]}_{sizes[1]}_{sizes[2]}"
        out[cell_key] = d
        print(f"  heatmap probe sizes={sizes} done", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=80)
    ap.add_argument("--contrib_n_graphs", type=int, default=8)
    ap.add_argument("--cells", type=str, nargs="+", required=True,
                    help="s1,s2,s3 triples summing to n, e.g. --cells 1,22,23 4,15,27")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")
    if arch != "roberta":
        raise NotImplementedError("this script targets the RobertaGraphTransformer only")
    cells = [tuple(int(x) for x in c.split(",")) for c in args.cells]
    for c in cells:
        if sum(c) != n:
            raise ValueError(f"cell {c} does not sum to n={n}")
    rng = np.random.default_rng(args.seed)

    print("== attention/Q/K/V heatmap probe (three-way, arbitrary sizes) ==")
    probe = heatmap_probe_threeway(model, dev, n, cells, rng, args.n_graphs,
                                    contrib_n_graphs=args.contrib_n_graphs)
    flat = {}
    for cell_key, d in probe.items():
        for kk, v in d.items():
            flat[f"{cell_key}__{kk}"] = np.asarray(v)
    np.savez_compressed(out / "heatmap_data.npz", **flat)
    print(f"  saved -> {out}/heatmap_data.npz")

    print("== raw weight matrices ==")
    w = raw_weights(model)
    np.savez_compressed(out / "raw_weights.npz", **w)
    print(f"  saved -> {out}/raw_weights.npz")


if __name__ == "__main__":
    main()
