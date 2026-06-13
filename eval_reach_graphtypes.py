"""Evaluate the depth-sweep reach models (L=1..4, trained on path-union, n=64)
on several graph FAMILIES, to see accuracy per depth L per graph type.

Families (all at n=64): path-union (in-distribution), 2chains (two paths of 32),
2cliques (two 32-cliques), 1cycle (C_64), 2cycle (2 x C_32). For each (model,
family) we report exact-match and pairwise accuracy, and produce a grouped-bar
figure (exact match) plus a json.

Run on HPC (GPU + checkpoints).  python eval_reach_graphtypes.py
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

from data import (add_self_loops, compute_connectivity_matrix, generate_er_graph,
                  generate_two_chains_graph, generate_two_cliques_graph,
                  generate_one_cycle_graph, generate_two_cycles_graph,
                  generate_path_union_graph)
from model import GraphConnectivityTransformer, ModelConfig

FAMILIES = ["pathunion", "2chains", "2cliques", "1cycle", "2cycle"]


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
    for i in range(size):
        if kind == "pathunion":  a = generate_path_union_graph(n, rng, 4)
        elif kind == "2chains":  a = generate_two_chains_graph(n, n // 2)
        elif kind == "2cliques": a = generate_two_cliques_graph(n, n // 2)
        elif kind == "1cycle":   a = generate_one_cycle_graph(n)
        elif kind == "2cycle":   a = generate_two_cycles_graph(n, n // 2)
        elif kind == "er":       a = generate_er_graph(n, 0.05, rng)
        else: raise ValueError(kind)
        p = rng.permutation(n); a = a[np.ix_(p, p)]
        xs[i] = add_self_loops(a).astype(np.uint8); ys[i] = compute_connectivity_matrix(a).astype(np.uint8)
    return xs, ys


@torch.no_grad()
def evaluate(model, xs, ys, device, batch=128):
    ng, n, _ = xs.shape; pred = np.empty((ng, n, n), np.int8)
    for s in range(0, ng, batch):
        e = min(s + batch, ng)
        xb = torch.from_numpy(xs[s:e]).to(device, torch.float32)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xb)
        pred[s:e] = (logits > 0).cpu().numpy().astype(np.int8)
    eq = pred == ys.astype(np.int8)
    return float(eq.reshape(ng, -1).all(1).mean()), float(eq.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs/report3/reach_depth")
    ap.add_argument("--size", type=int, default=5000)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    runs = sorted(p for p in Path(args.runs_dir).iterdir()
                  if p.is_dir() and (p / "last.pt").exists()
                  and "reach_n" in p.name and "_L" in p.name)
    results = {}; n = None
    tests = {}
    for d in runs:
        model, mcfg = load_model(str(d / "last.pt"), device); n = mcfg.n; L = mcfg.n_layers
        for fam in FAMILIES:
            if fam not in tests:
                tests[fam] = build(fam, n, args.size, seed=11)
            ex, pw = evaluate(model, *tests[fam], device)
            results.setdefault(L, {})[fam] = {"exact": ex, "pairwise": pw}
            print(f"L={L} {fam:10s} exact={ex:.3f} pairwise={pw:.3f}")
        del model

    Ls = sorted(results)
    json.dump(results, (Path(args.runs_dir) / "reach_graphtypes.json").open("w"), indent=2)
    # grouped bars: x = family, bars = L (exact match)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(FAMILIES)); w = 0.8 / len(Ls)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(Ls)))
    for i, L in enumerate(Ls):
        vals = [results[L][f]["exact"] for f in FAMILIES]
        ax.bar(x + (i - (len(Ls)-1)/2)*w, vals, w, color=cmap[i], label=f"L={L}")
    ax.set_xticks(x); ax.set_xticklabels(FAMILIES)
    ax.set_ylabel("exact-match accuracy"); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Depth-sweep reach models (trained on path-union, n={n}): "
                 "exact match per graph family")
    ax.legend(title="depth")
    fig.tight_layout()
    out = Path(args.runs_dir) / "reach_graphtypes.png"
    fig.savefig(out, dpi=170, bbox_inches="tight"); print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
