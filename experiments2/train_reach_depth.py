"""
Depth -> reach law: does the standard transformer's reachability wall sit at
3^L (3, 9, 27, 81 for L = 1, 2, 3, 4)?

We train a connectivity transformer (big config: d_model=512, 4 heads,
normalised-ReLU attention) on a distribution that *exercises long-range
reachability*: disjoint unions of 1..4 random paths over n nodes (default
n = 64). With k = 1 the graph is a single path of n nodes, giving shortest-path
distances up to n-1; with k > 1 there are genuine cross-component disconnections
so the model cannot trivially predict "all connected".

During training we evaluate on a held-out path-union test set and record, for
each shortest-path distance d, the fraction of within-component (target = 1)
pairs predicted connected ("reach at distance d"), and the maximum distance
d* at which reach >= 0.99. The prediction of the 3^L capacity theory is
d* ~= 3^L.

Vary only --n_layers across runs (everything else fixed) to read the law off d*.
"""
from __future__ import annotations

import sys
import time
import math as pymath
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from data import (
    add_self_loops,
    compute_connectivity_matrix,
    compute_all_pairs_shortest_paths,
    generate_path_union_graph,
)
from model import GraphConnectivityTransformer, ModelConfig
from utils import ensure_dir, get_device, save_json, set_seed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEST_SIZE = 10_000
MAX_PATHS = 4
REACH_THR = 0.99
MIN_PAIRS = 50          # ignore distance bins with too few pairs for d*


# ── Online path-union stream ─────────────────────────────────────────────────

def _sample_one(rng, n):
    adj_no = generate_path_union_graph(n, rng, MAX_PATHS)
    perm = rng.permutation(n)
    adj_no = adj_no[np.ix_(perm, perm)]
    adj = add_self_loops(adj_no).astype(np.uint8)
    target = compute_connectivity_matrix(adj_no).astype(np.uint8)
    return adj, target


class PathUnionStream(IterableDataset):
    def __init__(self, n, seed):
        self.n = n; self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info is not None else 0
        rng = np.random.default_rng((self.seed * 100003 + wid * 31337) & 0x7FFFFFFF)
        while True:
            yield _sample_one(rng, self.n)


def _collate(batch):
    xs = np.stack([b[0] for b in batch]); ys = np.stack([b[1] for b in batch])
    return torch.from_numpy(xs), torch.from_numpy(ys)


# ── Test set (parallel) ──────────────────────────────────────────────────────

def _gen_chunk(args):
    start, size, n, seed = args
    rng = np.random.default_rng(seed)
    xs = np.empty((size, n, n), dtype=np.uint8)
    ys = np.empty((size, n, n), dtype=np.uint8)
    ds = np.empty((size, n, n), dtype=np.int16)
    for i in range(size):
        adj_no = generate_path_union_graph(n, rng, MAX_PATHS)
        perm = rng.permutation(n)
        adj_no = adj_no[np.ix_(perm, perm)]
        xs[i] = add_self_loops(adj_no).astype(np.uint8)
        ys[i] = compute_connectivity_matrix(adj_no).astype(np.uint8)
        ds[i] = compute_all_pairs_shortest_paths(adj_no).astype(np.int16)
    return start, xs, ys, ds


def build_test(n, num_workers, seed=0):
    print(f"Generating {TEST_SIZE:,} path-union test graphs (n={n}) …")
    t0 = time.perf_counter()
    CHUNK = 1000
    specs = [(s, min(CHUNK, TEST_SIZE - s), n, 4242 + seed * 1000 + s)
             for s in range(0, TEST_SIZE, CHUNK)]
    xs = np.empty((TEST_SIZE, n, n), np.uint8)
    ys = np.empty((TEST_SIZE, n, n), np.uint8)
    ds = np.empty((TEST_SIZE, n, n), np.int16)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for s, x, y, d in ex.map(_gen_chunk, specs):
            e = s + len(x); xs[s:e] = x; ys[s:e] = y; ds[s:e] = d
    print(f"  done in {time.perf_counter()-t0:.1f}s")
    return xs, ys, ds


# ── Evaluation: per-distance reach + d* ──────────────────────────────────────

@torch.no_grad()
def evaluate(model, tx, ty, td, device, batch=256):
    model.eval()
    ng, n, _ = tx.shape
    pred = np.empty((ng, n, n), np.int8)
    for s in range(0, ng, batch):
        e = min(s + batch, ng)
        xb = torch.from_numpy(tx[s:e]).to(device, torch.float32)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xb)
        pred[s:e] = (logits > 0).cpu().numpy().astype(np.int8)
    tgt = ty.astype(np.int8); eq = pred == tgt
    exact = float(eq.reshape(ng, -1).all(1).mean())
    pairwise = float(eq.mean())
    eye = ~np.eye(n, dtype=bool)
    off = np.broadcast_to(eye[None], (ng, n, n))
    conn = off & (tgt == 1)
    disc = off & (tgt == 0)
    disc_acc = float(eq[disc].mean()) if disc.any() else float("nan")
    per_dist = {}
    dmax = int(td.max())
    for d in range(1, dmax + 1):
        m = conn & (td == d)
        c = int(m.sum())
        if c > 0:
            per_dist[d] = (float(eq[m].mean()), c)
    # d* = largest d with reach >= THR over a contiguous run from d=1
    d_star = 0
    for d in range(1, dmax + 1):
        if d in per_dist and per_dist[d][1] >= MIN_PAIRS and per_dist[d][0] >= REACH_THR:
            d_star = d
        elif d in per_dist and per_dist[d][1] >= MIN_PAIRS:
            break
    model.train()
    return {"exact": exact, "pairwise": pairwise, "disc_acc": disc_acc,
            "per_dist": per_dist, "d_star": d_star}


def lr_at(step, warm, total, peak):
    if step < warm:
        return peak * (step + 1) / max(1, warm)
    prog = (step - warm) / max(1, total - warm)
    return peak * 0.5 * (1 + pymath.cos(pymath.pi * min(1.0, prog)))


def _plot(h, out, title, L):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(h["steps"], h["d_star"], lw=2, marker="o", ms=3)
    ax[0].axhline(3 ** L, color="red", ls="--", label=f"$3^{L}={3**L}$")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("d* (reach $\\geq$ 0.99)")
    ax[0].set_title("Max exact-reach distance vs step"); ax[0].grid(alpha=0.3); ax[0].legend()
    if h["final_per_dist"]:
        ds = sorted(h["final_per_dist"]); vals = [h["final_per_dist"][d] for d in ds]
        ax[1].bar([str(d) for d in ds], vals, color="#1f77b4")
        ax[1].axhline(0.99, color="green", ls=":")
        ax[1].set_xlabel("within-component distance d"); ax[1].set_ylabel("reach")
        ax[1].set_title("Final reach by distance"); ax[1].set_ylim(0, 1.05)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def train(out_dir, loader, tx, ty, td, cfg, seed, L):
    set_seed(seed); device = get_device("auto")
    mcfg = ModelConfig(n=cfg["n"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                       d_ff=cfg["d_ff"], n_layers=L, attn_kind=cfg["attn_kind"],
                       readout=cfg.get("readout", "linear"))
    model = GraphConnectivityTransformer(mcfg).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"  device={device} L={L} params={nparam:,}")
    opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    crit = nn.BCEWithLogitsLoss()
    total = cfg["train_steps"]; warm = cfg["warmup_steps"]; ev = cfg["eval_every"]
    hist = {"steps": [], "train_loss": [], "d_star": [], "exact": [],
            "pairwise": [], "disc_acc": [], "final_per_dist": {},
            "n_parameters": nparam, "n_layers": L, "capacity_3L": 3 ** L}
    lw: List[float] = []; t0 = time.perf_counter(); model.train(); step = 0
    for xb, yb in loader:
        step += 1
        if step > total: break
        for g in opt.param_groups: g["lr"] = lr_at(step, warm, total, cfg["lr"])
        xb = xb.to(device, torch.float32); yb = yb.to(device, torch.float32)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = crit(model(xb), yb)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        lw.append(float(loss.item()))
        if len(lw) > ev: lw.pop(0)
        if step % ev == 0:
            m = evaluate(model, tx, ty, td, device)
            hist["steps"].append(step); hist["train_loss"].append(sum(lw)/len(lw))
            hist["d_star"].append(m["d_star"]); hist["exact"].append(m["exact"])
            hist["pairwise"].append(m["pairwise"]); hist["disc_acc"].append(m["disc_acc"])
            hist["final_per_dist"] = {d: v[0] for d, v in m["per_dist"].items()}
            print(f"  step {step:>7d} | loss={sum(lw)/len(lw):.5f} | d*={m['d_star']:>2d} "
                  f"(3^{L}={3**L}) | exact={m['exact']:.3f} pair={m['pairwise']:.3f} "
                  f"disc={m['disc_acc']:.3f} | {time.perf_counter()-t0:.0f}s", flush=True)
            torch.save({"model_state_dict": model.state_dict(),
                        "model_config": mcfg.__dict__, "step": step}, out_dir / "last.pt")
            t0 = time.perf_counter()
    hist["best_d_star"] = max(hist["d_star"]) if hist["d_star"] else 0
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--n_nodes", type=int, default=64)
    ap.add_argument("--n_layers", type=int, required=True)
    ap.add_argument("--attn_kind", default="normalized_relu",
                    choices=["normalized_relu", "softmax"])
    ap.add_argument("--readout", default="linear",
                    choices=["linear", "similarity"])
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--train_steps", type=int, default=1_000_000)
    ap.add_argument("--batch_size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    # the readout tag keeps similarity runs from overwriting the linear ones;
    # linear keeps the original (untagged) name for backward compatibility.
    rtag = "" if args.readout == "linear" else f"_{args.readout}"
    run = f"reach_n{args.n_nodes}_L{args.n_layers}_{args.attn_kind}{rtag}_seed{args.seed}"
    out_dir = Path(args.output_root) / run; ensure_dir(out_dir)
    cfg = {"n": args.n_nodes, "d_model": 512, "n_heads": 4, "d_ff": 2048,
           "attn_kind": args.attn_kind, "readout": args.readout,
           "lr": 1e-4, "weight_decay": 1e-4,
           "train_steps": args.train_steps, "warmup_steps": 1000, "eval_every": 5000,
           "batch_size": args.batch_size}
    print(f"\n{'='*64}\n  Depth->reach: n={args.n_nodes} L={args.n_layers} "
          f"attn={args.attn_kind} seed={args.seed}\n  capacity 3^L = {3**args.n_layers}\n{'='*64}\n")

    tx, ty, td = build_test(args.n_nodes, args.num_workers, seed=args.seed)
    loader = DataLoader(PathUnionStream(args.n_nodes, args.seed + 7),
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=_collate, pin_memory=True, prefetch_factor=4,
                        persistent_workers=True)
    hist = train(out_dir, loader, tx, ty, td, cfg, args.seed, args.n_layers)
    _plot(hist, out_dir / f"{run}.png",
          f"n={args.n_nodes} L={args.n_layers} ({args.attn_kind})", args.n_layers)
    save_json(out_dir / "history.json", hist)
    print(f"\nDone. best d* = {hist['best_d_star']} (capacity 3^{args.n_layers} = {3**args.n_layers})")


if __name__ == "__main__":
    main()
