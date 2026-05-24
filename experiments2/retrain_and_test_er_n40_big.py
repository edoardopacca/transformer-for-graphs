"""
Train + test the "big" graph-connectivity Transformer on ER(n=40, p=0.05).

Differs from retrain_and_test_er_n40.py in four key ways:
  * Larger model:   d_model=512, d_ff=2048, n_heads=4 (~6.3M params)
  * Attention:      normalized ReLU instead of softmax (matches Ye et al. 2026)
  * Online data:    each batch is generated on-the-fly, no epochs over a fixed
                    set; the model sees ~1 B unique graphs over 1 M training
                    steps with batch size 1000.
  * bf16 + cosine:  bfloat16 autocast on H100/H200, cosine LR decay with linear
                    warmup, weight_decay = 1e-4.

Experiment modes (via --max_diameter):
  (omit)         no diameter filter
  7 / 9 / 11     rejection sampling with diameter ≤ D
"""
from __future__ import annotations

import sys
import time
import math as pymath
import resource
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
    generate_er_graph,
)
from model import GraphConnectivityTransformer, ModelConfig
from utils import ensure_dir, get_device, save_json, set_seed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Constants ────────────────────────────────────────────────────────────────
N_NODES         = 40
P               = 0.05
TEST_SIZE       = 10_000
MAX_DIST_LOG    = N_NODES - 1


def _get_ram_gb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / (1024.0 ** 2)


# ── Online graph generator (IterableDataset) ─────────────────────────────────

def _sample_one(rng: np.random.Generator, n: int, p: float,
                max_diameter: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    while True:
        adj_no = generate_er_graph(n, p, rng)
        if max_diameter is not None:
            dist = compute_all_pairs_shortest_paths(adj_no)
            finite = dist[dist >= 0]
            diam = int(finite.max()) if len(finite) > 0 else -1
            if diam > max_diameter:
                continue
        adj = add_self_loops(adj_no).astype(np.uint8)
        target = compute_connectivity_matrix(adj_no).astype(np.uint8)
        return adj, target


class OnlineERStream(IterableDataset):
    """Infinite stream of ER(n, p) graphs (optionally diameter-filtered).
    Each worker gets a disjoint seed, so different workers produce different
    graphs. There is no epoch — the model sees each graph at most once."""

    def __init__(self, n: int, p: float, max_diameter: Optional[int], seed: int):
        self.n = n
        self.p = p
        self.max_diameter = max_diameter
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        seed = (self.seed * 100003 + worker_id * 31337) & 0x7FFFFFFF
        rng = np.random.default_rng(seed)
        while True:
            x, y = _sample_one(rng, self.n, self.p, self.max_diameter)
            yield x, y


def _stream_collate(batch):
    """Stack lists of (x, y) into torch tensors. Returns int8 tensors to be
    converted to float on the GPU later."""
    xs = np.stack([b[0] for b in batch])
    ys = np.stack([b[1] for b in batch])
    return torch.from_numpy(xs), torch.from_numpy(ys)


# ── Test set generation (parallel, one-shot) ─────────────────────────────────

def _gen_test_chunk(args):
    start_idx, size, p, n, max_diameter, seed = args
    rng = np.random.default_rng(seed)
    xs = np.empty((size, n, n), dtype=np.uint8)
    ys = np.empty((size, n, n), dtype=np.uint8)
    ds = np.empty((size, n, n), dtype=np.int16)
    for i in range(size):
        while True:
            adj_no = generate_er_graph(n, p, rng)
            dist = compute_all_pairs_shortest_paths(adj_no)
            finite = dist[dist >= 0]
            diam = int(finite.max()) if len(finite) > 0 else -1
            if max_diameter is None or diam <= max_diameter:
                break
        xs[i] = add_self_loops(adj_no).astype(np.uint8)
        ys[i] = compute_connectivity_matrix(adj_no).astype(np.uint8)
        ds[i] = dist.astype(np.int16)
    return start_idx, xs, ys, ds


def build_test_dataset(n: int, p: float, max_diameter: Optional[int],
                       num_workers: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"Generating {TEST_SIZE:,} test graphs "
          f"({'D≤'+str(max_diameter) if max_diameter is not None else 'unfiltered'}) "
          f"with {num_workers} workers …")
    t0 = time.perf_counter()
    CHUNK = 1000
    seed_offset = seed * 100003
    specs = [(s, min(CHUNK, TEST_SIZE - s), p, n, max_diameter, 54321 + seed_offset + s)
             for s in range(0, TEST_SIZE, CHUNK)]
    test_x = np.empty((TEST_SIZE, n, n), dtype=np.uint8)
    test_y = np.empty((TEST_SIZE, n, n), dtype=np.uint8)
    test_d = np.empty((TEST_SIZE, n, n), dtype=np.int16)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for s, xs, ys, ds in ex.map(_gen_test_chunk, specs):
            e = s + len(xs)
            test_x[s:e] = xs
            test_y[s:e] = ys
            test_d[s:e] = ds
    print(f"  done in {time.perf_counter()-t0:.1f}s")
    return test_x, test_y, test_d


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module,
             test_x: np.ndarray, test_y: np.ndarray, test_d: np.ndarray,
             device: torch.device,
             batch_size: int = 256) -> Dict[str, Any]:
    model.eval()
    n_graphs, n, _ = test_x.shape
    all_pred = np.empty((n_graphs, n, n), dtype=np.int8)
    for start in range(0, n_graphs, batch_size):
        end = min(start + batch_size, n_graphs)
        xb = torch.from_numpy(test_x[start:end]).to(device, dtype=torch.float32,
                                                     non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xb)
        all_pred[start:end] = (logits > 0).cpu().numpy().astype(np.int8)
    targets = test_y.astype(np.int8)
    eq = all_pred == targets

    exact_per_graph = eq.reshape(n_graphs, -1).all(axis=1)
    exact_match = float(exact_per_graph.mean())
    pairwise_acc = float(eq.mean())

    eye_mask = ~np.eye(n, dtype=bool)
    offdiag = np.broadcast_to(eye_mask[None, :, :], (n_graphs, n, n))
    per_dist: Dict[str, float] = {}
    dist_counts: Dict[str, int] = {}
    for dv in range(1, MAX_DIST_LOG + 1):
        mask = offdiag & (test_d == dv)
        cnt = int(mask.sum())
        if cnt > 0:
            per_dist[str(dv)] = float(eq[mask].mean())
            dist_counts[str(dv)] = cnt
    unreach = offdiag & (test_d == -1)
    if unreach.any():
        per_dist["disconnected"] = float(eq[unreach].mean())
        dist_counts["disconnected"] = int(unreach.sum())

    # per-diameter-bucket (max-finite-distance per graph)
    d_clipped = np.where(test_d < 0, 0, test_d)
    per_graph_diam = d_clipped.reshape(n_graphs, -1).max(axis=1)
    per_diam: Dict[str, Any] = {}
    for thr in [7, 9, 11]:
        mask = per_graph_diam <= thr
        if mask.any():
            per_diam[f"exact_le{thr}"] = float(exact_per_graph[mask].mean())
            per_diam[f"n_graphs_le{thr}"] = int(mask.sum())
    mask_gt = per_graph_diam > 11
    if mask_gt.any():
        per_diam["exact_gt11"] = float(exact_per_graph[mask_gt].mean())
        per_diam["n_graphs_gt11"] = int(mask_gt.sum())

    model.train()
    return {
        "exact_match": exact_match,
        "pairwise_acc": pairwise_acc,
        "per_dist_acc": per_dist,
        "dist_counts": dist_counts,
        "per_diam_bucket": per_diam,
    }


# ── LR schedule: linear warmup → cosine decay to 0 ───────────────────────────

def lr_at_step(step: int, warmup: int, total: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + pymath.cos(pymath.pi * min(1.0, progress)))


# ── Plots ────────────────────────────────────────────────────────────────────

def _plot_accuracy(h: Dict, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h["steps"], h["val_exact_match"], lw=2, label="Exact Match")
    ax.plot(h["steps"], h["val_pairwise_acc"], lw=2, label="Pairwise Acc")
    ax.set_title(title); ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.01); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def _plot_loss(h: Dict, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h["steps"], h["train_loss"], lw=2, color="#1f77b4")
    ax.axhline(1e-2, color="orange", ls=":", lw=1, label="1e-2")
    ax.axhline(1e-3, color="red", ls=":", lw=1, label="1e-3")
    ax.set_yscale("log")
    ax.set_title(title); ax.set_xlabel("Step"); ax.set_ylabel("Training loss (smoothed)")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def _plot_per_dist_sm(h: Dict, out: Path, title: str) -> None:
    steps = h["steps"]
    per_dist_list = h["val_per_dist_acc"]
    dist_counts = h["test_dist_counts"]
    numeric = sorted(int(k) for k in dist_counts
                     if k != "disconnected" and dist_counts[k] > 0)
    panels = numeric + (["disconnected"] if dist_counts.get("disconnected", 0) > 0 else [])
    if not panels:
        return
    ncols = 3
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 2.7 * nrows), sharex=True)
    axes_flat = np.array(axes).flatten().tolist()
    for ax, key in zip(axes_flat, panels):
        k = str(key)
        vals = [e.get(k) for e in per_dist_list]
        xs, ys = zip(*[(s, v) for s, v in zip(steps, vals) if v is not None]) \
                 if any(v is not None for v in vals) else ([], [])
        ax.plot(xs, ys, lw=1.0, color="#1f77b4")
        non_none = [v for v in vals if v is not None]
        if non_none:
            vmin, vmax = min(non_none), max(non_none)
            if vmin >= 0.99:
                ax.set_ylim(0.985, 1.001)
            else:
                pad = max(0.005, (vmax - vmin) * 0.10)
                ax.set_ylim(max(0.0, vmin - pad), min(1.005, vmax + pad / 2))
        label = "disconnected" if key == "disconnected" else f"d = {key}"
        ax.set_title(f"{label}  (n={dist_counts[k]:,})", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Step"); ax.set_ylabel("Pairwise acc")
    for ax in axes_flat[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=180); plt.close(fig)


def _plot_per_diam_bucket(h: Dict, out: Path, title: str) -> None:
    steps = h["steps"]
    buckets = h["val_per_diam_bucket"]
    keys = ["exact_le7", "exact_le9", "exact_le11", "exact_gt11"]
    labels = ["D≤7", "D≤9", "D≤11", "D>11"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label in zip(keys, labels):
        vals = [e.get(key) for e in buckets]
        if any(v is not None for v in vals):
            clean = [v if v is not None else float("nan") for v in vals]
            ax.plot(steps, clean, lw=2, label=label)
    ax.set_title(title); ax.set_xlabel("Step"); ax.set_ylabel("Exact-match accuracy")
    ax.set_ylim(0, 1.01); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


# ── Training loop ────────────────────────────────────────────────────────────

def train(out_dir: Path, dataset_iter, test_x, test_y, test_d,
          config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    set_seed(seed)
    device = get_device("auto")
    print(f"  device: {device}")

    model_cfg = ModelConfig(
        n=config["n"], d_model=config["d_model"], n_heads=config["n_heads"],
        d_ff=config["d_ff"], n_layers=config["n_layers"],
        dropout=config.get("dropout", 0.0),
        attn_kind=config["attn_kind"],
    )
    model = GraphConnectivityTransformer(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    opt = AdamW(model.parameters(), lr=config["lr"],
                weight_decay=config.get("weight_decay", 1e-4))
    criterion = nn.BCEWithLogitsLoss()

    total_steps = config["train_steps"]
    warmup = config.get("warmup_steps", 1000)
    eval_every = config.get("eval_every", 5000)
    grad_clip = config.get("grad_clip_norm", 1.0)
    peak_lr = config["lr"]

    history: Dict[str, Any] = {
        "steps": [], "train_loss": [],
        "val_exact_match": [], "val_pairwise_acc": [],
        "val_per_dist_acc": [], "val_per_diam_bucket": [],
        "timing_stats": {"time_per_5000_steps_sec": []},
        "run_stats": {
            "n_parameters": n_params,
            "ram_before_training_gb": _get_ram_gb(),
        },
        "test_dist_counts": {},
    }

    loss_window: List[float] = []
    best_exact = -1.0
    t_block = time.perf_counter()
    model.train()

    print(f"  Starting training: {total_steps:,} steps × batch "
          f"{config['batch_size']} = {total_steps * config['batch_size']:,} samples")

    step = 0
    for xb_cpu, yb_cpu in dataset_iter:
        step += 1
        if step > total_steps:
            break

        # set LR for this step
        lr = lr_at_step(step, warmup, total_steps, peak_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        xb = xb_cpu.to(device, dtype=torch.float32, non_blocking=True)
        yb = yb_cpu.to(device, dtype=torch.float32, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xb)
            loss = criterion(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        loss_window.append(float(loss.item()))
        if len(loss_window) > eval_every:
            loss_window.pop(0)

        if step % eval_every == 0:
            elapsed = time.perf_counter() - t_block
            history["timing_stats"]["time_per_5000_steps_sec"].append(
                {"step": step, "seconds": elapsed}
            )
            metrics = evaluate(model, test_x, test_y, test_d, device)
            avg_loss = sum(loss_window) / max(1, len(loss_window))

            history["steps"].append(step)
            history["train_loss"].append(avg_loss)
            history["val_exact_match"].append(metrics["exact_match"])
            history["val_pairwise_acc"].append(metrics["pairwise_acc"])
            history["val_per_dist_acc"].append(metrics["per_dist_acc"])
            history["val_per_diam_bucket"].append(metrics["per_diam_bucket"])
            history["test_dist_counts"] = metrics["dist_counts"]

            print(f"  step {step:>7d} | lr={lr:.2e} | loss={avg_loss:.6f} | "
                  f"exact={metrics['exact_match']:.4f} | pair={metrics['pairwise_acc']:.4f} "
                  f"| {elapsed:.1f}s for {eval_every} steps",
                  flush=True)

            if metrics["exact_match"] > best_exact:
                best_exact = metrics["exact_match"]
                if config.get("save_best", True):
                    torch.save({"model_state_dict": model.state_dict(),
                                "model_config": model_cfg.__dict__,
                                "step": step}, out_dir / "best.pt")
            # checkpoint every eval (so we can resume if the job is killed)
            torch.save({"model_state_dict": model.state_dict(),
                        "model_config": model_cfg.__dict__,
                        "step": step}, out_dir / "last.pt")
            t_block = time.perf_counter()

    history["run_stats"]["ram_after_training_gb"] = _get_ram_gb()
    history["best_exact_match"] = best_exact
    return history


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--max_diameter", type=int, default=None,
                        help="rejection sampling cutoff (omit = no filter)")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="workers for the online dataloader AND test-set generation")
    parser.add_argument("--train_steps", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    diam_tag = f"diam{args.max_diameter}" if args.max_diameter is not None else "unfiltered"
    run_name = f"n{N_NODES}_p{int(P*100):03d}_{diam_tag}_big"
    out_root = Path(args.output_root)
    out_dir = out_root / run_name
    ensure_dir(out_dir)

    config = {
        "n":             N_NODES,
        "p":             P,
        "d_model":       512,
        "n_heads":       4,
        "d_ff":          2048,
        "n_layers":      2,
        "dropout":       0.0,
        "attn_kind":     "normalized_relu",
        "batch_size":    args.batch_size,
        "lr":            1e-4,
        "weight_decay":  1e-4,
        "train_steps":   args.train_steps,
        "warmup_steps":  1000,
        "eval_every":    5000,
        "grad_clip_norm": 1.0,
        "save_best":     True,
        "save_last":     True,
        "max_diameter":  args.max_diameter,
        "test_size":     TEST_SIZE,
        "num_workers":   args.num_workers,
    }

    print(f"\n{'='*72}")
    print(f"  Big ER training: ER(n={N_NODES}, p={P}), filter={diam_tag}")
    print(f"  d_model=512, normalized-ReLU, online data, bf16")
    print(f"  Output: {out_dir}")
    print(f"{'='*72}\n")

    test_x, test_y, test_d = build_test_dataset(N_NODES, P, args.max_diameter,
                                                  num_workers=args.num_workers,
                                                  seed=args.seed)

    stream = OnlineERStream(N_NODES, P, args.max_diameter, seed=args.seed + 7)
    loader = DataLoader(
        stream,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_stream_collate,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    history = train(out_dir, loader, test_x, test_y, test_d, config, args.seed)

    # ── Plots ──
    prefix = f"er_n40_{diam_tag}_big"
    _plot_accuracy       (history, out_dir / f"{prefix}_accuracy.png",
                           f"Big ER(n={N_NODES}, p={P}) {diam_tag}: Accuracy")
    _plot_loss           (history, out_dir / f"{prefix}_loss.png",
                           f"Big ER(n={N_NODES}, p={P}) {diam_tag}: Training Loss")
    _plot_per_diam_bucket(history, out_dir / f"{prefix}_per_diam_bucket.png",
                           f"Big ER(n={N_NODES}, p={P}) {diam_tag}: Exact Match per diameter bucket")
    _plot_per_dist_sm    (history, out_dir / f"{prefix}_per_dist_sm.png",
                           f"Big ER(n={N_NODES}, p={P}) {diam_tag}: Pairwise acc by distance")

    summary = {"config": config, "best_exact_match": history["best_exact_match"]}
    save_json(out_root / f"summary_{run_name}.json", summary)
    save_json(out_dir / "history.json", history)
    print(f"\nDone. Best exact match: {history['best_exact_match']:.4f}")
    print(f"Results in: {out_dir}")


if __name__ == "__main__":
    main()
