"""The controlled experiment that separates DISTANCE from NUMBER-OF-PATHS.

Two terminal nodes s,t are joined by k internally-disjoint paths, each of length
ell edges. The shortest-path distance s--t is ell for every k, but the effective
resistance is ell/k (falls with k). We vary k at fixed ell and ask whether the
model's error on the (s,t) pair depends on k:

  * if difficulty == distance (the paper's claim): error is flat in k;
  * if difficulty == resistance / bottleneck (our claim): error falls with k.

For each (ell, k) cell we report, on the terminal pair:
  - accuracy        : does the model predict s,t connected?
  - 1 - cos(h_s,h_t): how close it made their embeddings (similarity read-out)
  - influence       : ||d h_t / d x_s||, the REAL end-to-end mixing of the trained
                      model (true weights, includes attention+residual+FFN)
and the structural quantities R_eff(s,t), P^(3^L)(s,t), distance.

The contrast is sharpest when ell exceeds the L=2 capacity (3^L=9), which needs
room: run it on the n=64 reach checkpoints (and n=40), not only n=20.

    python analyze_parallel_paths.py --checkpoint runs/reach_depth/reach_n64_L2_.../last.pt \
        --output_dir runs/reach_depth/parallel_paths --path_lens 7 9 11 13 --max_paths 4
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import (add_self_loops, compute_all_pairs_shortest_paths,
                  effective_resistance, diffusion_reach, generate_parallel_paths_graph)
from eval_families import load_model


def build_cell(n, n_paths, path_len, size, seed):
    """`size` permuted copies of the parallel-paths graph; returns adj-with-loops and
    the permuted positions of the two terminals (originally 0 and 1)."""
    rng = np.random.default_rng(seed)
    base = generate_parallel_paths_graph(n, n_paths, path_len)
    xs = np.empty((size, n, n), np.float32)
    st = np.empty((size, 2), np.int64)
    for i in range(size):
        perm = rng.permutation(n)
        a = base[np.ix_(perm, perm)]
        xs[i] = add_self_loops(a)
        inv = np.argsort(perm)            # inv[original] = new position
        st[i] = (inv[0], inv[1])
    return xs, st, base


@torch.no_grad()
def model_pair(model, xs, st, device, batch=256):
    """Per graph: predicted-connected on (s,t) and 1-cos(h_s,h_t)."""
    ng = xs.shape[0]
    acc = np.empty(ng); oneminuscos = np.empty(ng)
    cuda = device.type == "cuda"
    for b in range(0, ng, batch):
        e = min(b + batch, ng)
        x = torch.from_numpy(xs[b:e]).to(device, torch.float32)
        if cuda:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, H = model.forward_and_embeddings(x)
        else:
            logits, H = model.forward_and_embeddings(x)
        logits = logits.float(); H = H.float()
        for j in range(e - b):
            s, t = st[b + j]
            acc[b + j] = float(logits[j, s, t] > 0)
            hs, ht = H[j, s], H[j, t]
            cos = torch.dot(hs, ht) / (hs.norm() * ht.norm() + 1e-9)
            oneminuscos[b + j] = float(1.0 - cos)
    return float(acc.mean()), float(oneminuscos.mean())


def influence(model, x_single, s, t, device):
    """||d h_t / d x_s|| : how much the final embedding of t depends on the input row
    of s, under the trained model (true weights, full forward). End-to-end mixing."""
    x = torch.from_numpy(x_single[None]).to(device, torch.float32).requires_grad_(True)
    H = model.embeddings(x).float()           # [1, n, d]
    g, = torch.autograd.grad(H[0, t].sum(), x, retain_graph=False)
    return float(g[0, s].abs().sum().cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--path_lens", type=int, nargs="+", default=[5, 7, 9])
    ap.add_argument("--max_paths", type=int, default=4)
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--n_influence", type=int, default=24)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model, mcfg, arch, readout = load_model(args.checkpoint, device)
    n, L = mcfg.n, mcfg.n_layers
    steps = 3 ** L                      # the model's real reach budget, not L
    print(f"{arch}/{readout} n={n} L={L} -> diffusion steps 3^L={steps}")

    res = {"checkpoint": str(args.checkpoint), "arch": arch, "readout": readout,
           "n": n, "n_layers": L, "diffusion_steps": steps, "cells": []}
    for pl in args.path_lens:
        for k in range(1, args.max_paths + 1):
            if 2 + k * (pl - 1) > n:
                continue
            xs, st, base = build_cell(n, k, pl, args.size, args.seed)
            acc, omc = model_pair(model, xs, st, device)
            R = effective_resistance(base)[0, 1]
            P = diffusion_reach(base, steps)[0, 1]
            d = int(compute_all_pairs_shortest_paths(base)[0, 1])
            infl = float(np.mean([influence(model, xs[i], st[i, 0], st[i, 1], device)
                                  for i in range(min(args.n_influence, len(xs)))]))
            cell = {"path_len": pl, "n_paths": k, "distance": d,
                    "R_eff": float(R), "P_3L": float(P), "accuracy": acc,
                    "one_minus_cos": omc, "influence": infl}
            res["cells"].append(cell)
            print(f" ell={pl:2d} k={k} d={d:2d} R={R:5.2f} P^{steps}={P:.4f} "
                  f"acc={acc:.3f} 1-cos={omc:.3f} infl={infl:.2e}", flush=True)

    (out / "parallel_paths.json").write_text(json.dumps(res, indent=2))
    _plot(res, out)
    print(f"saved -> {out}/parallel_paths.json (+ png)")


def _plot(res, out):
    cells = res["cells"]
    lens = sorted({c["path_len"] for c in cells})
    metrics = [("accuracy", "accuracy on (s,t)", False),
               ("one_minus_cos", "1 - cos(h_s,h_t)", False),
               ("influence", "influence ||dh_t/dx_s||", True)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(lens)))
    for ax, (key, lab, logy) in zip(axes, metrics):
        for c, pl in zip(cmap, lens):
            pts = sorted([(cc["n_paths"], cc[key]) for cc in cells if cc["path_len"] == pl])
            if pts:
                ks, vs = zip(*pts)
                ax.plot(ks, vs, "o-", color=c, label=f"ell={pl}")
        ax.set_xlabel("number of parallel paths k (distance fixed per curve)")
        ax.set_ylabel(lab); ax.grid(alpha=0.3)
        if logy: ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.suptitle(f"Distance fixed, paths varied ({res['arch']}/{res['readout']}, "
                 f"n={res['n']}, L={res['n_layers']}): does difficulty follow k?")
    fig.tight_layout(); fig.savefig(out / "parallel_paths.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
