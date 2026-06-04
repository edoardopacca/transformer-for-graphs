"""Experiment A: embedding geometry of a trained connectivity transformer.

No training change --- we load a checkpoint and look at the hidden states
H^(ell). If the model represents connected components, within-component node
pairs should have CLOSE embeddings and between-component pairs FAR ones:
    within = E_{i,j: R_ij=1} ||h_i - h_j||^2  <<  between = E_{i,j: R_ij=0} ||...||^2 .

The sharper, capacity-linked question is how within-distance grows with the
shortest-path distance d: if the model is matrix-powering with reach 3^L, the
embeddings of within-component pairs at d > 3^L should be FAR (the model does not
"know" they are connected), i.e. the separation collapses past the wall.

Runs on HPC (GPU + checkpoint). Outputs a json + figure next to the checkpoint.

    python analyze_embeddings.py --checkpoint <path/best.pt> --kind twochains
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import (add_self_loops, compute_connectivity_matrix,
                  compute_all_pairs_shortest_paths, generate_er_graph,
                  generate_two_chains_graph, generate_path_union_graph)
from model import GraphConnectivityTransformer, ModelConfig


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    c = ck["model_config"]
    mcfg = ModelConfig(n=c["n"], d_model=c["d_model"], n_heads=c["n_heads"],
                       d_ff=c["d_ff"], n_layers=c["n_layers"],
                       attn_kind=c.get("attn_kind", "normalized_relu"),
                       readout=c.get("readout", "linear"))
    m = GraphConnectivityTransformer(mcfg).to(device); m.load_state_dict(ck["model_state_dict"])
    m.eval(); return m, mcfg


def build(kind, n, size, seed):
    rng = np.random.default_rng(seed)
    xs = np.empty((size, n, n), np.uint8); ys = np.empty((size, n, n), np.uint8)
    ds = np.empty((size, n, n), np.int16)
    for i in range(size):
        if kind == "er":
            a = generate_er_graph(n, 0.05, rng)
        elif kind == "twochains":
            a = generate_two_chains_graph(n, n // 2)
        elif kind == "pathunion":
            a = generate_path_union_graph(n, rng, 4)
        else:
            raise ValueError(kind)
        p = rng.permutation(n); a = a[np.ix_(p, p)]
        xs[i] = add_self_loops(a).astype(np.uint8)
        ys[i] = compute_connectivity_matrix(a).astype(np.uint8)
        ds[i] = compute_all_pairs_shortest_paths(a).astype(np.int16)
    return xs, ys, ds


@torch.no_grad()
def analyze(model, xs, ys, ds, device, batch=128):
    ng, n, _ = xs.shape
    # accumulators per layer
    L = None
    wsum = bsum = None; wcnt = bcnt = None
    dist_sum = None; dist_cnt = None
    dmax = int(ds.max())
    eye = ~np.eye(n, dtype=bool)
    for s in range(0, ng, batch):
        e = min(s + batch, ng)
        xb = torch.from_numpy(xs[s:e]).to(device, torch.float32)
        states = model.hidden_states(xb)            # list of [b,n,d]
        if L is None:
            L = len(states)
            wsum = np.zeros(L); bsum = np.zeros(L); wcnt = np.zeros(L); bcnt = np.zeros(L)
            dist_sum = np.zeros((L, dmax + 1)); dist_cnt = np.zeros((L, dmax + 1))
        R = ys[s:e].astype(bool); D = ds[s:e]
        for ell, H in enumerate(states):
            D2 = torch.cdist(H.float(), H.float()).pow(2).cpu().numpy()   # [b,n,n]
            for g in range(e - s):
                off = eye
                within = R[g] & off; between = (~R[g]) & off
                wsum[ell] += D2[g][within].sum(); wcnt[ell] += within.sum()
                bsum[ell] += D2[g][between].sum(); bcnt[ell] += between.sum()
                for d in range(1, dmax + 1):
                    m = within & (D[g] == d); c = int(m.sum())
                    if c:
                        dist_sum[ell, d] += D2[g][m].sum(); dist_cnt[ell, d] += c
    within = wsum / np.maximum(wcnt, 1)
    between = bsum / np.maximum(bcnt, 1)
    per_dist = np.divide(dist_sum, np.maximum(dist_cnt, 1))
    return {"within": within, "between": between, "per_dist": per_dist,
            "dist_counts": dist_cnt, "n_layers_states": L, "dmax": dmax}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--kind", default="twochains", choices=["er", "twochains", "pathunion"])
    ap.add_argument("--size", type=int, default=500)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model, mcfg = load_model(args.checkpoint, device)
    n = mcfg.n; print(f"device={device} n={n} L={mcfg.n_layers} kind={args.kind}")
    xs, ys, ds = build(args.kind, n, args.size, seed=7)
    r = analyze(model, xs, ys, ds, device)

    labels = ["read-in"] + [f"layer {i}" for i in range(1, r["n_layers_states"] - 1)] + ["final"]
    print("\nlayer        within     between    ratio b/w")
    for ell in range(r["n_layers_states"]):
        w, b = r["within"][ell], r["between"][ell]
        print(f"  {labels[ell]:>10}  {w:9.3f}  {b:9.3f}  {b/max(w,1e-9):7.2f}")

    out_dir = Path(args.checkpoint).parent
    tag = args.kind
    # figure: within-distance vs shortest-path distance, per layer, with between line
    fig, ax = plt.subplots(figsize=(8, 5))
    ds_axis = list(range(1, r["dmax"] + 1))
    cmap = plt.cm.viridis(np.linspace(0, 1, r["n_layers_states"]))
    for ell in range(r["n_layers_states"]):
        ys_ = [r["per_dist"][ell, d] if r["dist_counts"][ell, d] > 0 else np.nan for d in ds_axis]
        ax.plot(ds_axis, ys_, marker="o", ms=3, color=cmap[ell], label=labels[ell])
        ax.axhline(r["between"][ell], color=cmap[ell], ls=":", lw=1, alpha=0.7)
    cap = 3 ** mcfg.n_layers
    if cap <= r["dmax"]:
        ax.axvline(cap, color="red", ls="--", lw=1.2, label=f"$3^L={cap}$")
    ax.set_xlabel("shortest-path distance $d$ (within component)")
    ax.set_ylabel("mean $\\|h_i-h_j\\|^2$")
    ax.set_title(f"Embedding within-distance vs $d$ ({tag}, n={n}, L={mcfg.n_layers})\n"
                 "dotted = between-component distance (per layer)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    out_png = out_dir / f"embed_geometry_{tag}.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close(fig)
    json.dump({"within": r["within"].tolist(), "between": r["between"].tolist(),
               "per_dist": r["per_dist"].tolist(),
               "labels": labels, "n": n, "L": mcfg.n_layers, "kind": tag},
              (out_dir / f"embed_geometry_{tag}.json").open("w"), indent=2)
    print(f"\nsaved: {out_png}")


if __name__ == "__main__":
    main()
