"""Report VI, Threads A2 (sample efficiency) and A3 (truncated routes).

Trains the base connectivity transformer (RoBERTa-faithful, linear read-out, the model
of Reports III--V) on a SINGLE, explicitly-named distribution: the multipath family
(two terminals s,t joined by k full routes of length ell). No opaque mixed stream.

  * A2 (--trunc_frac 0): pure multipath. We log, over training, BOTH the whole-matrix
    exact match AND the accuracy on the single (s,t) pair on a held-out clean multipath
    val set, so "how many samples to learn the multipath connectivity" can be read for
    the pair and for the whole matrix (steps x batch = samples).
  * A3 (--trunc_frac f>0): a fraction f of the training graphs are TRUNCATED -- some of
    the k routes are dead ends that stop one hop short of t (n_full ~ U{0..k-1} full
    routes + the rest truncated, so s-t is connected iff n_full>=1, with correct labels).
    The number of routes leaving each terminal stays k and t's degree stays term_deg, so
    the label is not given away by degree. We evaluate on the SAME clean multipath val
    set as A2: does training with dead ends degrade/slow the clean-multipath connectivity?

  python experiments2/train_multipath.py --output_root runs/report6/multipath_train \
      --n_nodes 40 --n_full 2 --path_len 11 --trunc_frac 0 \
      --train_steps 300000 --eval_every 2000 --batch_size 1000 --seed 1000
"""
from __future__ import annotations

import argparse
import math as pymath
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from data import (add_self_loops, compute_connectivity_matrix,
                  generate_multipath_graph, permute_with_meta)
from model import RobertaGraphTransformer, GraphConnectivityTransformer, ModelConfig
from utils import ensure_dir, get_device, save_json, set_seed

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sample_multipath(n, k, ell, term_deg, trunc_frac, rng):
    """One training graph. With prob trunc_frac a TRUNCATED graph (n_full in {0..k-1}
    full routes + the rest dead ends of length ell-1); else a clean k-route multipath.
    Returns (adj_no_loops_permuted, meta_permuted)."""
    if trunc_frac > 0.0 and rng.random() < trunc_frac:
        n_full = int(rng.integers(0, k))                  # 0..k-1
        n_trunc = k - n_full
        r = generate_multipath_graph(n, n_full, ell, rng, n_trunc=n_trunc,
                                     term_deg=term_deg, trunc_len=max(1, ell - 1))
    else:
        r = generate_multipath_graph(n, k, ell, rng, n_trunc=0, term_deg=term_deg)
    if r is None:
        raise ValueError(f"multipath n={n} k={k} ell={ell} term_deg={term_deg} does not fit")
    return permute_with_meta(*r, rng)


class MultipathStream(IterableDataset):
    def __init__(self, n, k, ell, term_deg, trunc_frac, seed):
        self.n, self.k, self.ell = n, k, ell
        self.term_deg, self.trunc_frac, self.seed = term_deg, trunc_frac, seed

    def __iter__(self):
        info = get_worker_info(); wid = info.id if info is not None else 0
        rng = np.random.default_rng((self.seed * 100003 + wid * 31337) & 0x7FFFFFFF)
        while True:
            adj, _ = sample_multipath(self.n, self.k, self.ell, self.term_deg,
                                      self.trunc_frac, rng)
            x = add_self_loops(adj).astype(np.float32)
            y = compute_connectivity_matrix(adj).astype(np.float32)
            yield x, y


def _collate(batch):
    xs = np.stack([b[0] for b in batch]); ys = np.stack([b[1] for b in batch])
    return torch.from_numpy(xs), torch.from_numpy(ys)


def build_clean_val(n, k, ell, term_deg, size, seed):
    """Clean multipath val set (k full routes, always connected) + terminal positions."""
    rng = np.random.default_rng(seed)
    xs = np.empty((size, n, n), np.float32); ys = np.empty((size, n, n), np.int8)
    sp = np.empty(size, np.int64); tp = np.empty(size, np.int64)
    for i in range(size):
        r = generate_multipath_graph(n, k, ell, rng, n_trunc=0, term_deg=term_deg)
        if r is None:
            raise ValueError(f"clean val multipath n={n} k={k} ell={ell} does not fit")
        adj, meta = permute_with_meta(*r, rng)
        xs[i] = add_self_loops(adj); ys[i] = compute_connectivity_matrix(adj).astype(np.int8)
        sp[i] = meta["s"]; tp[i] = meta["t"]
    return xs, ys, sp, tp


@torch.no_grad()
def evaluate(model, tx, ty, sp, tp, device, batch=512):
    model.eval(); ng, n, _ = tx.shape
    pred = np.empty((ng, n, n), np.int8); use_cuda = device.type == "cuda"
    for s in range(0, ng, batch):
        e = min(s + batch, ng)
        xb = torch.from_numpy(tx[s:e]).to(device, torch.float32)
        if use_cuda:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
        else:
            logits = model(xb)
        pred[s:e] = (logits > 0).cpu().numpy().astype(np.int8)
    eq = pred == ty.astype(np.int8)
    exact = float(eq.reshape(ng, -1).all(1).mean())
    offdiag = ~np.eye(n, dtype=bool)
    pairwise = float((eq & offdiag[None]).sum() / (ng * n * (n - 1)))
    pair_acc = float((pred[np.arange(ng), sp, tp] == 1).mean())
    model.train()
    return exact, pairwise, pair_acc


def lr_at(step, warmup, total, peak):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + pymath.cos(pymath.pi * min(1.0, prog)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--n_nodes", type=int, default=40)
    ap.add_argument("--n_full", type=int, default=2, help="k = number of full routes")
    ap.add_argument("--path_len", type=int, default=11, help="ell = route length (edges)")
    ap.add_argument("--term_deg", type=int, default=4)
    ap.add_argument("--trunc_frac", type=float, default=0.0,
                    help="fraction of TRUNCATED training graphs (Thread A3); 0 = clean (A2)")
    ap.add_argument("--arch", choices=["roberta", "minimal"], default="roberta")
    ap.add_argument("--readout", choices=["linear", "similarity"], default="linear")
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--train_steps", type=int, default=300_000)
    ap.add_argument("--batch_size", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--val_size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    n, k, ell = args.n_nodes, args.n_full, args.path_len

    tr_tag = "clean" if args.trunc_frac <= 0 else f"trunc{args.trunc_frac:g}"
    L_tag = "" if args.n_layers == 2 else f"_L{args.n_layers}"
    run_name = f"n{n}_k{k}_ell{ell}_{tr_tag}_{args.arch}_{args.readout}{L_tag}_seed{args.seed}"
    out_dir = Path(args.output_root) / run_name
    ensure_dir(out_dir)

    set_seed(args.seed)
    device = get_device("auto")
    is_roberta = args.arch == "roberta"
    mcfg = ModelConfig(n=n, d_model=512, n_heads=1, d_ff=2048, n_layers=args.n_layers,
                       dropout=0.1 if is_roberta else 0.0, attn_kind="normalized_relu",
                       norm_style="post" if is_roberta else "pre",
                       layer_norm_eps=1e-5, init_std=0.02, readout=args.readout)
    Cls = RobertaGraphTransformer if is_roberta else GraphConnectivityTransformer
    model = Cls(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"== {run_name} ==\n device={device} arch={args.arch} readout={args.readout} "
          f"n={n} k={k} ell={ell} trunc_frac={args.trunc_frac} params={n_params:,}", flush=True)

    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    val_x, val_y, val_s, val_t = build_clean_val(n, k, ell, args.term_deg, args.val_size, 777)
    stream = MultipathStream(n, k, ell, args.term_deg, args.trunc_frac, args.seed + 7)
    dl_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                 collate_fn=_collate, pin_memory=True)
    if args.num_workers > 0:                       # prefetch/persistent need workers (err. 37)
        dl_kw.update(prefetch_factor=4, persistent_workers=True)
    loader = DataLoader(stream, **dl_kw)

    hist: Dict[str, Any] = {"steps": [], "train_loss": [], "val_exact": [],
                            "val_pairwise": [], "val_pair_acc": [],
                            "run_stats": {"n_parameters": n_params},
                            "config": {"n": n, "k": k, "ell": ell,
                                       "trunc_frac": args.trunc_frac, "arch": args.arch,
                                       "readout": args.readout, "n_layers": args.n_layers,
                                       "term_deg": args.term_deg, "seed": args.seed}}
    loss_win: List[float] = []; best = -1.0
    total = args.train_steps; use_cuda = device.type == "cuda"
    t0 = time.perf_counter(); model.train(); step = 0

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
            ve, vp, vpair = evaluate(model, val_x, val_y, val_s, val_t, device)
            hist["steps"].append(step)
            hist["train_loss"].append(sum(loss_win) / len(loss_win))
            hist["val_exact"].append(ve); hist["val_pairwise"].append(vp)
            hist["val_pair_acc"].append(vpair)
            dt = time.perf_counter() - t0
            print(f" step {step:>7d} loss={hist['train_loss'][-1]:.5f} "
                  f"val_exact={ve:.4f} val_pw={vp:.4f} val_pair={vpair:.4f} {dt:.0f}s",
                  flush=True)
            t0 = time.perf_counter()
            ck = {"model_state_dict": model.state_dict(), "model_config": mcfg.__dict__,
                  "step": step}
            torch.save(ck, out_dir / "last.pt")
            if ve > best:
                best = ve; torch.save(ck, out_dir / "best.pt")

    # steps-to-threshold (samples = steps * batch_size), for the pair and the matrix
    def steps_to(metric_key, thr):
        for st, v in zip(hist["steps"], hist[metric_key]):
            if v >= thr:
                return st
        return None
    hist["best_val_exact"] = best
    hist["steps_to"] = {f"pair_{t}": steps_to("val_pair_acc", t) for t in (0.9, 0.99, 0.999)}
    hist["steps_to"].update({f"exact_{t}": steps_to("val_exact", t) for t in (0.9, 0.99, 0.999)})
    save_json(out_dir / "history.json", hist)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hist["steps"], hist["val_pair_acc"], label="val (s,t)-pair acc")
    ax.plot(hist["steps"], hist["val_exact"], label="val whole-matrix exact")
    ax.set_xscale("log"); ax.set_xlabel("step"); ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.01); ax.grid(alpha=0.3); ax.legend(); ax.set_title(run_name)
    fig.tight_layout(); fig.savefig(out_dir / f"{run_name}_curves.png", dpi=150); plt.close(fig)
    print(f"done. steps_to={hist['steps_to']}  best exact={best:.4f}  -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
