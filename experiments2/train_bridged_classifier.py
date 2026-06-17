"""Report V: a DEDICATED binary classifier for bridged-cliques vs split-cliques.

The connectivity-matrix probe (eval_bridged_cliques.py) asks what the *trained base
model* already does. This script asks the complementary question the advisor posed
directly: can a standard transformer be *trained* to tell the two labels apart, and
does that get harder as the cliques grow?

  * label 1 = BRIDGED : two cliques of size c joined by a SINGLE edge -> 1 component
  * label 0 = SPLIT   : the same two cliques, no bridge               -> 2 components
They differ by one edge that makes a connection at distance <= 3.

The DFS test, now in a trainable setting: we train on a 50/50 stream with the clique
size c drawn at RANDOM per sample (c in [2, n//2]) and evaluate accuracy PER clique
size. If the model learns it at every c (flat), the bridge is detectable at any size
-- matrix powering. If accuracy falls at large c, the model cannot cross a large
clique to feel the bridge -- a bounded-traversal (DFS-like) limit. Passing a fixed
--clique_size instead measures convergence speed at one density (the density-in-
optimisation angle: is a denser clique slower to learn?).

Trunk = the base standard transformer (minimal/A.1-style, single head,
d_model=512, normalized-ReLU, 2 layers, GraphBinaryClassifier mean-pool + 1 logit),
matching the scale of the connectivity base model. Online data, bf16, AdamW + cosine.

  python experiments2/train_bridged_classifier.py --output_root runs/report5/bridged_clf \
      --n_nodes 20 --clique_size -1 --train_steps 200000 --seed 1000
"""
from __future__ import annotations

import argparse
import math as pymath
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from data import (add_self_loops, generate_bridged_cliques_graph,
                  generate_split_cliques_graph)
from model import GraphBinaryClassifier, ModelConfig
from utils import ensure_dir, get_device, save_json, set_seed

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sample(rng, n, clique_size):
    """(adj_with_loops uint8, label, c). label 1 = bridged, 0 = split."""
    c = clique_size if clique_size and clique_size > 0 else int(rng.integers(2, n // 2 + 1))
    label = int(rng.integers(0, 2))
    gen = generate_bridged_cliques_graph if label == 1 else generate_split_cliques_graph
    a = gen(n, clique_size=c)
    perm = rng.permutation(n)
    a = a[np.ix_(perm, perm)]
    return add_self_loops(a).astype(np.uint8), label, c


class BridgedStream(IterableDataset):
    def __init__(self, n, seed, clique_size):
        self.n = n; self.seed = seed; self.clique_size = clique_size

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info is not None else 0
        rng = np.random.default_rng((self.seed * 100003 + wid * 31337) & 0x7FFFFFFF)
        while True:
            x, label, _ = _sample(rng, self.n, self.clique_size)
            yield x, label


def _collate(batch):
    xs = np.stack([b[0] for b in batch])
    ys = np.array([b[1] for b in batch], dtype=np.float32)
    return torch.from_numpy(xs), torch.from_numpy(ys)


def build_test(n, per_cell, seed, clique_size):
    """Balanced test set. If clique_size<=0, sweep c over [2..n//2] with per_cell
    graphs of each label per c; else all graphs at the fixed c. Returns xs, ys, cs."""
    rng = np.random.default_rng(seed)
    cs_list = list(range(2, n // 2 + 1)) if (not clique_size or clique_size <= 0) else [clique_size]
    xs, ys, cs = [], [], []
    for c in cs_list:
        for label in (0, 1):
            gen = generate_bridged_cliques_graph if label == 1 else generate_split_cliques_graph
            for _ in range(per_cell):
                a = gen(n, clique_size=c)
                perm = rng.permutation(n)
                a = a[np.ix_(perm, perm)]
                xs.append(add_self_loops(a).astype(np.uint8)); ys.append(float(label)); cs.append(c)
    return np.stack(xs), np.array(ys, np.float32), np.array(cs, np.int64)


@torch.no_grad()
def evaluate(model, tx, ty, tc, device, batch=512):
    model.eval()
    ng = tx.shape[0]; preds = np.empty(ng, np.int8)
    use_cuda = device.type == "cuda"
    for s in range(0, ng, batch):
        e = min(s + batch, ng)
        xb = torch.from_numpy(tx[s:e]).to(device, torch.float32)
        if use_cuda:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
        else:
            logits = model(xb)
        preds[s:e] = (logits > 0).cpu().numpy().astype(np.int8)
    y = ty.astype(np.int8)
    acc = float((preds == y).mean())
    acc1 = float((preds[y == 1] == 1).mean()) if (y == 1).any() else float("nan")
    acc0 = float((preds[y == 0] == 0).mean()) if (y == 0).any() else float("nan")
    by_c = {}
    for c in np.unique(tc):
        m = tc == c
        by_c[int(c)] = float((preds[m] == y[m]).mean())
    model.train()
    return {"acc": acc, "acc_bridged": acc1, "acc_split": acc0, "by_c": by_c}


def lr_at(step, warmup, total, peak):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + pymath.cos(pymath.pi * min(1.0, prog)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--n_nodes", type=int, default=20)
    ap.add_argument("--clique_size", type=int, default=-1,
                    help="fixed clique size c; -1 = random per sample (sweep, default)")
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--train_steps", type=int, default=200_000)
    ap.add_argument("--batch_size", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    n = args.n_nodes

    ctag = "rand" if args.clique_size <= 0 else f"c{args.clique_size}"
    run_name = f"bridged_clf_n{n}_{ctag}_seed{args.seed}"
    out_dir = Path(args.output_root) / run_name; ensure_dir(out_dir)

    set_seed(args.seed)
    device = get_device("auto")
    mcfg = ModelConfig(n=n, d_model=512, n_heads=1, d_ff=2048, n_layers=2,
                       dropout=0.0, attn_kind="normalized_relu")
    model = GraphBinaryClassifier(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"== {run_name} == device={device} params={n_params:,}", flush=True)

    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    tx, ty, tc = build_test(n, per_cell=200, seed=999, clique_size=args.clique_size)
    print(f" test set {tx.shape[0]} graphs over clique sizes {sorted(set(tc.tolist()))}", flush=True)

    stream = BridgedStream(n, args.seed + 7, args.clique_size)
    loader = DataLoader(stream, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=_collate, pin_memory=True, prefetch_factor=4,
                        persistent_workers=True)

    hist: Dict[str, Any] = {"steps": [], "train_loss": [], "val_acc": [],
                            "val_acc_bridged": [], "val_acc_split": [], "val_acc_by_c": [],
                            "config": {"n": n, "clique_size": args.clique_size,
                                       "seed": args.seed, "n_parameters": n_params}}
    loss_win: List[float] = []; best = -1.0; total = args.train_steps
    use_cuda = device.type == "cuda"; t0 = time.perf_counter(); model.train(); step = 0

    for xb_cpu, yb_cpu in loader:
        step += 1
        if step > total:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, 1000, total, 1e-4)
        xb = xb_cpu.to(device, torch.float32, non_blocking=True)
        yb = yb_cpu.to(device, torch.float32, non_blocking=True)
        if use_cuda:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(xb), yb)
        else:
            loss = criterion(model(xb), yb)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        loss_win.append(float(loss.item()))
        if len(loss_win) > args.eval_every:
            loss_win.pop(0)

        if step % args.eval_every == 0:
            m = evaluate(model, tx, ty, tc, device)
            hist["steps"].append(step)
            hist["train_loss"].append(sum(loss_win) / len(loss_win))
            hist["val_acc"].append(m["acc"]); hist["val_acc_bridged"].append(m["acc_bridged"])
            hist["val_acc_split"].append(m["acc_split"]); hist["val_acc_by_c"].append(m["by_c"])
            dt = time.perf_counter() - t0
            print(f" step {step:>7d} loss={hist['train_loss'][-1]:.5f} acc={m['acc']:.4f} "
                  f"(bridged={m['acc_bridged']:.3f} split={m['acc_split']:.3f}) {dt:.0f}s", flush=True)
            t0 = time.perf_counter()
            ck = {"model_state_dict": model.state_dict(), "model_config": mcfg.__dict__,
                  "step": step, "task": "bridged_vs_split"}
            torch.save(ck, out_dir / "last.pt")
            if m["acc"] > best:
                best = m["acc"]; torch.save(ck, out_dir / "best.pt")

    hist["best_acc"] = best
    save_json(out_dir / "history.json", hist)

    # curves: overall/per-class over steps, and final accuracy-by-clique-size
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.plot(hist["steps"], hist["val_acc"], lw=2, label="accuracy")
    a1.plot(hist["steps"], hist["val_acc_bridged"], lw=1.3, ls="--", label="bridged (1)")
    a1.plot(hist["steps"], hist["val_acc_split"], lw=1.3, ls="--", label="split (0)")
    a1.axhline(0.5, color="gray", ls=":", lw=1)
    a1.set_xlabel("step"); a1.set_ylabel("accuracy"); a1.set_ylim(0.4, 1.02)
    a1.grid(alpha=0.3); a1.legend(fontsize=8); a1.set_title("convergence")
    if hist["val_acc_by_c"]:
        final = hist["val_acc_by_c"][-1]
        cs = sorted(final)
        a2.plot(cs, [final[c] for c in cs], marker="o")
        a2.axhline(0.5, color="gray", ls=":", lw=1)
        a2.set_xlabel("clique size c"); a2.set_ylabel("accuracy (final)")
        a2.set_ylim(0.4, 1.02); a2.grid(alpha=0.3)
        a2.set_title("accuracy by clique size\n(flat=matrix-power, falling=DFS-like)")
    fig.suptitle(run_name); fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_curves.png", dpi=150); plt.close(fig)
    print(f"done. best acc={best:.4f} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
