"""Experiment B: does a spectral / Laplacian inductive bias move the 3^L wall?

Same setup as the depth-sweep reach experiment (path-union at n=64), but we add
two optional ingredients that push the model toward the SPECTRAL solution of
connectivity (embeddings ~ constant per connected component = null space of the
graph Laplacian) instead of the LOCAL matrix-powering solution (which is what
caps reach at 3^L):

  * --readout similarity : R_ij = scale * <h_i_norm, h_j_norm> + bias, so
    "connected == similar embedding" is built into the architecture;
  * --lambda_lap LAMBDA  : an auxiliary loss  LAMBDA * Tr(H^T L H)
    = LAMBDA * sum_{(i,j) in E} ||h_i - h_j||^2, the Laplacian Dirichlet energy,
    pushing adjacent nodes to similar embeddings.

  loss = BCE(R_hat, R) + lambda_lap * Tr(H^T L H).

We measure reach by shortest-path distance (does d* exceed 3^L = 9 for L = 2?).
"""
from __future__ import annotations

import sys, time, math as pymath
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model import GraphConnectivityTransformer, ModelConfig, laplacian_smoothness
from utils import ensure_dir, get_device, save_json, set_seed
from experiments2.train_reach_depth import (PathUnionStream, _collate, build_test,
                                            evaluate, lr_at)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def train(out_dir, loader, tx, ty, td, cfg, seed):
    set_seed(seed); device = get_device("auto")
    mcfg = ModelConfig(n=cfg["n"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                       d_ff=cfg["d_ff"], n_layers=cfg["n_layers"],
                       attn_kind=cfg["attn_kind"], readout=cfg["readout"])
    model = GraphConnectivityTransformer(mcfg).to(device)
    print(f"  device={device} L={cfg['n_layers']} readout={cfg['readout']} "
          f"lambda_lap={cfg['lambda_lap']} params={sum(p.numel() for p in model.parameters()):,}")
    opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    crit = nn.BCEWithLogitsLoss()
    eye = torch.eye(cfg["n"], device=device)
    total = cfg["train_steps"]; warm = cfg["warmup_steps"]; ev = cfg["eval_every"]
    lam = cfg["lambda_lap"]
    hist = {"steps": [], "train_loss": [], "bce": [], "lap": [], "d_star": [],
            "exact": [], "pairwise": [], "disc_acc": [], "final_per_dist": {},
            "n_layers": cfg["n_layers"], "readout": cfg["readout"], "lambda_lap": lam,
            "capacity_3L": 3 ** cfg["n_layers"]}
    bw: List[float] = []; t0 = time.perf_counter(); model.train(); step = 0
    for xb, yb in loader:
        step += 1
        if step > total: break
        for g in opt.param_groups: g["lr"] = lr_at(step, warm, total, cfg["lr"])
        xb = xb.to(device, torch.float32); yb = yb.to(device, torch.float32)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, H = model.forward_and_embeddings(xb)
            bce = crit(logits, yb)
            if lam > 0:
                adj_noloop = xb * (1.0 - eye)            # strip self-loops -> A
                lap = laplacian_smoothness(H.float(), adj_noloop)
                loss = bce + lam * lap
            else:
                lap = torch.zeros((), device=device); loss = bce
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        bw.append(float(bce.item()))
        if len(bw) > ev: bw.pop(0)
        if step % ev == 0:
            m = evaluate(model, tx, ty, td, device)
            hist["steps"].append(step); hist["bce"].append(sum(bw)/len(bw))
            hist["lap"].append(float(lap.item())); hist["train_loss"].append(float(loss.item()))
            hist["d_star"].append(m["d_star"]); hist["exact"].append(m["exact"])
            hist["pairwise"].append(m["pairwise"]); hist["disc_acc"].append(m["disc_acc"])
            hist["final_per_dist"] = {d: v[0] for d, v in m["per_dist"].items()}
            print(f"  step {step:>7d} | bce={sum(bw)/len(bw):.4f} lap={float(lap.item()):.4f} "
                  f"| d*={m['d_star']:>2d} (3^L={3**cfg['n_layers']}) exact={m['exact']:.3f} "
                  f"pair={m['pairwise']:.3f} | {time.perf_counter()-t0:.0f}s", flush=True)
            torch.save({"model_state_dict": model.state_dict(),
                        "model_config": mcfg.__dict__, "step": step}, out_dir / "last.pt")
            t0 = time.perf_counter()
    hist["best_d_star"] = max(hist["d_star"]) if hist["d_star"] else 0
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--n_nodes", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--readout", default="similarity", choices=["linear", "similarity"])
    ap.add_argument("--lambda_lap", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--train_steps", type=int, default=500_000)
    ap.add_argument("--batch_size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    run = (f"lap_n{args.n_nodes}_L{args.n_layers}_{args.readout}"
           f"_lam{args.lambda_lap:g}_seed{args.seed}")
    out_dir = Path(args.output_root) / run; ensure_dir(out_dir)
    cfg = {"n": args.n_nodes, "d_model": 512, "n_heads": 4, "d_ff": 2048,
           "attn_kind": "normalized_relu", "readout": args.readout,
           "lambda_lap": args.lambda_lap, "n_layers": args.n_layers,
           "lr": 1e-4, "weight_decay": 1e-4, "train_steps": args.train_steps,
           "warmup_steps": 1000, "eval_every": 5000, "batch_size": args.batch_size}
    print(f"\n{'='*64}\n  Laplacian exp: n={args.n_nodes} L={args.n_layers} "
          f"readout={args.readout} lambda={args.lambda_lap} seed={args.seed}\n{'='*64}\n")

    tx, ty, td = build_test(args.n_nodes, args.num_workers, seed=args.seed)
    loader = DataLoader(PathUnionStream(args.n_nodes, args.seed + 7),
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=_collate, pin_memory=True, prefetch_factor=4,
                        persistent_workers=True)
    hist = train(out_dir, loader, tx, ty, td, cfg, args.seed)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hist["steps"], hist["d_star"], marker="o", ms=3, label="d* (reach)")
    ax.axhline(3 ** args.n_layers, color="red", ls="--", label=f"$3^L={3**args.n_layers}$")
    ax.set_xlabel("step"); ax.set_ylabel("d*"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"Laplacian (lambda={args.lambda_lap}, {args.readout}) L={args.n_layers}: reach vs step")
    fig.tight_layout(); fig.savefig(out_dir / f"{run}.png", dpi=160); plt.close(fig)
    save_json(out_dir / "history.json", hist)
    print(f"\nDone. best d* = {hist['best_d_star']} (3^{args.n_layers}={3**args.n_layers}); "
          f"final exact = {hist['exact'][-1]:.3f}")


if __name__ == "__main__":
    main()
