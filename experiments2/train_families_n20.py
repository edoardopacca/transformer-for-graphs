"""Train a standard (RoBERTa-faithful) connectivity transformer at n=20 either on
ER only (Set A) or on a MIX of graph families (Set B), with a choice of read-out
(linear / similarity) and an optional Laplacian-smoothness auxiliary loss with a
warm-up on lambda. Rich per-family / per-diameter / per-spectral-gap evaluation is
done separately by ``eval_families.py`` on the saved checkpoint.

Why the lambda warm-up: the Laplacian term  lambda * mean_edges ||h_i - h_j||^2
is ~O(d_model) at init (~400 for d_model=512) while the BCE is ~O(1); applied
from step 0 it dominates and collapses embeddings to the trivial "all connected"
optimum (observed with lambda=1 in the n=64 study). We therefore (i) use small
lambda (~1e-3) and (ii) keep it at 0 for the first ``lap_warmup_start`` steps,
then ramp linearly over ``lap_warmup_ramp`` steps, so the local solution forms
before smoothness is applied as a regulariser.

  python experiments2/train_families_n20.py --output_root runs/report4/families \
      --families mixed --arch roberta --readout similarity --lambda_lap 1e-3 \
      --train_steps 1000000 --batch_size 1000 --seed 1000
"""
from __future__ import annotations

import argparse
import math as pymath
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from data import (add_self_loops, compute_connectivity_matrix,
                  generate_er_graph, generate_two_chains_graph,
                  generate_two_cliques_graph, generate_one_cycle_graph,
                  generate_two_cycles_graph, generate_one_chain_graph,
                  generate_path_union_graph, generate_blocks_graph)
from model import (GraphConnectivityTransformer, RobertaGraphTransformer,
                   ModelConfig, laplacian_smoothness)
from utils import ensure_dir, get_device, save_json, set_seed

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Families used to build the MIXED training stream (barbell/expander/chain_plus are
# held out as OOD, so the spectral-gap and wall tests are on unseen structure).
MIXED_FAMILIES = ["er", "er_blocks", "clique_blocks", "path_union",
                  "2chains", "2cliques", "1cycle", "2cycle", "1chain"]


def sample_family(kind: str, n: int, rng: np.random.Generator, p: float) -> np.ndarray:
    if kind == "er":            a = generate_er_graph(n, p, rng)
    elif kind == "er_blocks":   a = generate_blocks_graph(n, rng, "er")
    elif kind == "clique_blocks": a = generate_blocks_graph(n, rng, "clique")
    elif kind == "path_union":  a = generate_path_union_graph(n, rng, 4)
    elif kind == "2chains":     a = generate_two_chains_graph(n, n // 2)
    elif kind == "2cliques":    a = generate_two_cliques_graph(n, n // 2)
    elif kind == "1cycle":      a = generate_one_cycle_graph(n)
    elif kind == "2cycle":      a = generate_two_cycles_graph(n, n // 2)
    elif kind == "1chain":      a = generate_one_chain_graph(n)
    else: raise ValueError(kind)
    perm = rng.permutation(n)
    return a[np.ix_(perm, perm)]


class OnlineFamilyStream(IterableDataset):
    """Infinite stream. If ``families`` is ["er"] it is plain ER(n,p); otherwise a
    family is drawn uniformly per sample. Each worker gets a disjoint seed."""

    def __init__(self, families: List[str], n: int, seed: int, p: float):
        self.families = families; self.n = n; self.seed = seed; self.p = p

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info is not None else 0
        rng = np.random.default_rng((self.seed * 100003 + wid * 31337) & 0x7FFFFFFF)
        n = self.n
        while True:
            kind = self.families[int(rng.integers(len(self.families)))]
            a = sample_family(kind, n, rng, self.p)
            x = add_self_loops(a).astype(np.float32)
            y = compute_connectivity_matrix(a).astype(np.float32)
            yield x, y


def _collate(batch):
    xs = np.stack([b[0] for b in batch]); ys = np.stack([b[1] for b in batch])
    return torch.from_numpy(xs), torch.from_numpy(ys)


def build_fixed_test(families: List[str], n: int, size: int, seed: int, p: float):
    rng = np.random.default_rng(seed)
    xs = np.empty((size, n, n), np.float32); ys = np.empty((size, n, n), np.int8)
    for i in range(size):
        kind = families[int(rng.integers(len(families)))]
        a = sample_family(kind, n, rng, p)
        xs[i] = add_self_loops(a); ys[i] = compute_connectivity_matrix(a).astype(np.int8)
    return xs, ys


@torch.no_grad()
def evaluate_exact(model, tx, ty, device, batch=512):
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
    model.train()
    return float(eq.reshape(ng, -1).all(1).mean()), float(eq.mean())


def lr_at(step, warmup, total, peak):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + pymath.cos(pymath.pi * min(1.0, prog)))


def lam_at(step, lam, start, ramp):
    if lam <= 0.0:
        return 0.0
    return lam * float(np.clip((step - start) / max(1, ramp), 0.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--n_nodes", type=int, default=20)
    ap.add_argument("--p", type=float, default=0.08, help="ER edge prob (0.08 at n=20, 0.05 at n=40)")
    ap.add_argument("--families", default="mixed", help="'er' (Set A) or 'mixed' (Set B)")
    ap.add_argument("--arch", choices=["roberta", "minimal"], default="roberta")
    ap.add_argument("--readout", choices=["linear", "similarity"], default="linear")
    ap.add_argument("--lambda_lap", type=float, default=0.0)
    ap.add_argument("--lap_warmup_start", type=int, default=50_000)
    ap.add_argument("--lap_warmup_ramp", type=int, default=50_000)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--train_steps", type=int, default=1_000_000)
    ap.add_argument("--batch_size", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    n = args.n_nodes; p = args.p

    fam_tag = "mixed" if args.families == "mixed" else "er"
    families = MIXED_FAMILIES if args.families == "mixed" else ["er"]
    lam_tag = f"lam{args.lambda_lap:g}" if args.lambda_lap > 0 else "lam0"
    run_name = f"n{n}_{fam_tag}_{args.arch}_{args.readout}_{lam_tag}_seed{args.seed}"
    out_dir = Path(args.output_root) / run_name
    ensure_dir(out_dir)

    set_seed(args.seed)
    device = get_device("auto")
    is_roberta = args.arch == "roberta"
    mcfg = ModelConfig(n=n, d_model=512, n_heads=1, d_ff=2048, n_layers=2,
                       dropout=0.1 if is_roberta else 0.0, attn_kind="normalized_relu",
                       norm_style="post" if is_roberta else "pre",
                       layer_norm_eps=1e-5, init_std=0.02, readout=args.readout)
    Cls = RobertaGraphTransformer if is_roberta else GraphConnectivityTransformer
    model = Cls(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"== {run_name} ==\n device={device} arch={args.arch} readout={args.readout} "
          f"lambda={args.lambda_lap} params={n_params:,}", flush=True)

    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # In-distribution test (matches training stream) + fixed OOD references.
    val_x, val_y = build_fixed_test(families, n, 5000, 777, p)
    ch_x, ch_y = build_fixed_test(["2chains"], n, 5000, 12345, p)
    cq_x, cq_y = build_fixed_test(["2cliques"], n, 5000, 23456, p)

    stream = OnlineFamilyStream(families, n, args.seed + 7, p)
    loader = DataLoader(stream, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=_collate, pin_memory=True, prefetch_factor=4,
                        persistent_workers=True)

    eye = torch.eye(n, device=device)
    hist: Dict[str, Any] = {"steps": [], "train_loss": [], "bce": [], "lap": [],
                            "lam_eff": [], "val_exact": [], "val_pairwise": [],
                            "val_2chain_exact": [], "val_2clique_exact": [],
                            "run_stats": {"n_parameters": n_params},
                            "config": {"families": fam_tag, "arch": args.arch,
                                       "readout": args.readout, "lambda_lap": args.lambda_lap,
                                       "lap_warmup_start": args.lap_warmup_start,
                                       "lap_warmup_ramp": args.lap_warmup_ramp,
                                       "seed": args.seed}}
    loss_win: List[float] = []; bce_win: List[float] = []; lap_win: List[float] = []
    best = -1.0; total = args.train_steps; use_cuda = device.type == "cuda"
    t0 = time.perf_counter(); model.train(); step = 0

    for xb_cpu, yb_cpu in loader:
        step += 1
        if step > total:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, 1000, total, 1e-4)
        lam_eff = lam_at(step, args.lambda_lap, args.lap_warmup_start, args.lap_warmup_ramp)
        xb = xb_cpu.to(device, torch.float32, non_blocking=True)
        yb = yb_cpu.to(device, torch.float32, non_blocking=True)

        def _step_body():
            if lam_eff > 0.0:
                logits, H = model.forward_and_embeddings(xb)
                bce = criterion(logits, yb)
                adj_nl = (xb * (1.0 - eye)).clamp(0.0, 1.0)
                lap = laplacian_smoothness(H.float(), adj_nl)
                return bce, lap
            else:
                logits = model(xb)
                return criterion(logits, yb), None

        if use_cuda:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                bce, lap = _step_body()
        else:
            bce, lap = _step_body()
        loss = bce if lap is None else bce + lam_eff * lap

        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        loss_win.append(float(loss.item())); bce_win.append(float(bce.item()))
        lap_win.append(float(lap.item()) if lap is not None else 0.0)
        for w in (loss_win, bce_win, lap_win):
            if len(w) > args.eval_every:
                w.pop(0)

        if step % args.eval_every == 0:
            ve, vp = evaluate_exact(model, val_x, val_y, device)
            che, _ = evaluate_exact(model, ch_x, ch_y, device)
            cqe, _ = evaluate_exact(model, cq_x, cq_y, device)
            hist["steps"].append(step)
            hist["train_loss"].append(sum(loss_win) / len(loss_win))
            hist["bce"].append(sum(bce_win) / len(bce_win))
            hist["lap"].append(sum(lap_win) / len(lap_win))
            hist["lam_eff"].append(lam_eff)
            hist["val_exact"].append(ve); hist["val_pairwise"].append(vp)
            hist["val_2chain_exact"].append(che); hist["val_2clique_exact"].append(cqe)
            dt = time.perf_counter() - t0
            print(f" step {step:>7d} loss={hist['train_loss'][-1]:.5f} bce={hist['bce'][-1]:.5f} "
                  f"lap={hist['lap'][-1]:.2f} lam={lam_eff:.1e} val={ve:.4f} "
                  f"2ch={che:.3f} 2cl={cqe:.3f} {dt:.0f}s", flush=True)
            t0 = time.perf_counter()
            ck = {"model_state_dict": model.state_dict(), "model_config": mcfg.__dict__,
                  "step": step}
            torch.save(ck, out_dir / "last.pt")
            if ve > best:
                best = ve; torch.save(ck, out_dir / "best.pt")

    hist["best_val_exact"] = best
    save_json(out_dir / "history.json", hist)
    # quick training-curve plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, lab in [("val_exact", "in-dist exact"), ("val_2chain_exact", "2chain exact"),
                   ("val_2clique_exact", "2clique exact")]:
        if hist[k]:
            ax.plot(hist["steps"], hist[k], label=lab)
    ax.set_xlabel("step"); ax.set_ylabel("exact match"); ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.3); ax.legend(); ax.set_title(run_name)
    fig.tight_layout(); fig.savefig(out_dir / f"{run_name}_curves.png", dpi=150); plt.close(fig)
    print(f"done. best in-dist exact={best:.4f}  -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
