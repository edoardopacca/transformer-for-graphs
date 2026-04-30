from __future__ import annotations

import sys
import time
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
N_NODES      = 40
P            = 0.05
TRAIN_SIZE   = 6_000_000
TEST_SIZE    = 10_000
CHUNK_SIZE   = 25_000
MAX_DIST_LOG = N_NODES - 1   # maximum possible shortest-path distance


def _get_ram_gb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / (1024.0 ** 2)


# ── Data generation ──────────────────────────────────────────────────────────

def _sample_one(rng: np.random.Generator, n: int, p: float,
                max_diameter: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw one ER(n,p) graph, optionally with rejection for diameter ≤ max_diameter."""
    while True:
        adj_no = generate_er_graph(n, p, rng)
        dist   = compute_all_pairs_shortest_paths(adj_no)
        if max_diameter is not None:
            finite = dist[dist >= 0]
            diam   = int(finite.max()) if len(finite) > 0 else -1
            if diam > max_diameter:
                continue
        adj    = add_self_loops(adj_no)
        target = compute_connectivity_matrix(adj_no)
        return adj.astype(np.uint8), target.astype(np.uint8), dist.astype(np.int16)


def _generate_train_chunk(
    args: Tuple[int, int, float, int, Optional[int], int],
) -> Tuple[int, np.ndarray, np.ndarray]:
    start_idx, size, p, n, max_diameter, seed = args
    rng = np.random.default_rng(seed)
    xs = np.empty((size, n, n), dtype=np.uint8)
    ys = np.empty((size, n, n), dtype=np.uint8)
    for i in range(size):
        adj, target, _ = _sample_one(rng, n, p, max_diameter)
        xs[i] = adj
        ys[i] = target
    return start_idx, xs, ys


def _generate_test_chunk(
    args: Tuple[int, int, float, int, Optional[int], int],
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    start_idx, size, p, n, max_diameter, seed = args
    rng = np.random.default_rng(seed)
    xs = np.empty((size, n, n), dtype=np.uint8)
    ys = np.empty((size, n, n), dtype=np.uint8)
    ds = np.empty((size, n, n), dtype=np.int16)
    for i in range(size):
        adj, target, dist = _sample_one(rng, n, p, max_diameter)
        xs[i] = adj
        ys[i] = target
        ds[i] = dist
    return start_idx, xs, ys, ds


def _chunk_specs(total: int, p: float, n: int, max_diam: Optional[int],
                 seed_base: int) -> List[Tuple]:
    specs = []
    for start in range(0, total, CHUNK_SIZE):
        size = min(CHUNK_SIZE, total - start)
        specs.append((start, size, p, n, max_diam, seed_base + start))
    return specs


def generate_dataset(
    n: int, p: float, train_size: int, test_size: int,
    max_diameter: Optional[int], num_workers: int,
) -> Dict[str, Any]:
    diam_tag = f"D≤{max_diameter}" if max_diameter is not None else "unfiltered"

    # ── Train ──
    print(f"Generating {train_size:,} train samples ({diam_tag}) "
          f"with {num_workers} workers …")
    t0 = time.perf_counter()
    train_x = np.empty((train_size, n, n), dtype=np.uint8)
    train_y = np.empty((train_size, n, n), dtype=np.uint8)
    specs = _chunk_specs(train_size, p, n, max_diameter, seed_base=12345)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for s, xs, ys in ex.map(_generate_train_chunk, specs):
            e = s + len(xs)
            train_x[s:e] = xs
            train_y[s:e] = ys
            print(f"  train {e:>9,}/{train_size:,}", flush=True)
    train_gen_sec = time.perf_counter() - t0
    print(f"  train done in {train_gen_sec:.1f}s ({train_gen_sec/60:.1f} min)")

    # ── Test ──
    print(f"Generating {test_size:,} test samples ({diam_tag}) …")
    t0 = time.perf_counter()
    test_x = np.empty((test_size, n, n), dtype=np.uint8)
    test_y = np.empty((test_size, n, n), dtype=np.uint8)
    test_d = np.empty((test_size, n, n), dtype=np.int16)
    specs = _chunk_specs(test_size, p, n, max_diameter, seed_base=54321)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for s, xs, ys, ds in ex.map(_generate_test_chunk, specs):
            e = s + len(xs)
            test_x[s:e] = xs
            test_y[s:e] = ys
            test_d[s:e] = ds
    test_gen_sec = time.perf_counter() - t0
    print(f"  test  done in {test_gen_sec:.1f}s")

    # Distance distribution in test set
    unique_d, counts_d = np.unique(test_d.ravel(), return_counts=True)
    dist_counts = {int(k): int(v) for k, v in zip(unique_d, counts_d) if k >= 0}

    return {
        "train_x": train_x, "train_y": train_y,
        "test_x":  test_x,  "test_y":  test_y,  "test_d": test_d,
        "dataset_stats": {
            "n": n, "p": p, "max_diameter": max_diameter,
            "train_size": train_size, "test_size": test_size,
            "train_gen_sec": train_gen_sec,
            "test_gen_sec":  test_gen_sec,
            "test_dist_counts": dist_counts,
        },
    }


# ── Evaluation ───────────────────────────────────────────────────────────────

def _to_device(arr: np.ndarray, device: torch.device,
               dtype: torch.dtype) -> torch.Tensor:
    t = torch.from_numpy(arr)
    if device.type == "cuda":
        return t.pin_memory().to(device=device, dtype=dtype, non_blocking=True)
    return t.to(device=device, dtype=dtype)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_x: np.ndarray, test_y: np.ndarray, test_d: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    n_graphs, n, _ = test_x.shape

    x_dev = _to_device(test_x, device, torch.float32)
    y_dev = _to_device(test_y, device, torch.float32)
    d_dev = _to_device(test_d, device, torch.int64)

    logits = model(x_dev)
    preds  = (logits > 0).to(torch.int64)
    y_int  = y_dev.to(torch.int64)
    eq     = preds == y_int                          # (B, n, n)

    # Exact match (all entries of a graph correct)
    exact_per_graph = eq.view(n_graphs, -1).all(dim=1)   # (B,)
    exact_match     = float(exact_per_graph.float().mean().item())

    # Pairwise accuracy
    pairwise_acc = float(eq.float().mean().item())

    # Per-distance pairwise accuracy (off-diagonal only)
    eye    = torch.eye(n, dtype=torch.bool, device=device).unsqueeze(0)
    offdiag = ~eye.expand(n_graphs, n, n)
    per_dist: Dict[str, float] = {}
    dist_counts: Dict[str, int] = {}
    for dv in range(1, MAX_DIST_LOG + 1):
        mask = offdiag & (d_dev == dv)
        cnt  = int(mask.sum().item())
        if cnt > 0:
            per_dist[str(dv)]   = float(eq[mask].float().mean().item())
            dist_counts[str(dv)] = cnt

    # Per-diameter-bucket exact match (using per-graph max finite distance)
    # Compute max finite distance per graph: replace -1 with 0, take max
    d_clipped = d_dev.clamp(min=0)
    per_graph_diam = d_clipped.view(n_graphs, -1).max(dim=1).values.cpu().numpy()
    exact_np = exact_per_graph.cpu().numpy()
    per_diam_bucket: Dict[str, Any] = {}
    for thr in [7, 9, 11]:
        mask = per_graph_diam <= thr
        if mask.any():
            per_diam_bucket[f"exact_le{thr}"] = float(exact_np[mask].mean())
            per_diam_bucket[f"n_graphs_le{thr}"] = int(mask.sum())
    mask_gt11 = per_graph_diam > 11
    if mask_gt11.any():
        per_diam_bucket["exact_gt11"] = float(exact_np[mask_gt11].mean())
        per_diam_bucket["n_graphs_gt11"] = int(mask_gt11.sum())

    model.train()
    return {
        "exact_match":              exact_match,
        "pairwise_acc":             pairwise_acc,
        "per_dist_acc":             per_dist,
        "dist_counts":              dist_counts,
        "per_diam_bucket":          per_diam_bucket,
    }


# ── Plots ────────────────────────────────────────────────────────────────────

def _plot_accuracy_curves(history: Dict, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["steps"], history["val_exact_match"],  lw=2, label="Exact Match")
    ax.plot(history["steps"], history["val_pairwise_acc"], lw=2, label="Pairwise Acc")
    ax.set_title(title, fontsize=13); ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.01); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def _plot_loss_curve(history: Dict, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["steps"], history["train_loss"], lw=2)
    ax.set_title(title, fontsize=13); ax.set_xlabel("Step"); ax.set_ylabel("BCE Loss")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def _plot_per_dist_curves(history: Dict, out: Path, title: str) -> None:
    steps = history["steps"]
    per_dist_list = history["val_per_dist_acc"]
    all_keys = sorted({int(k) for entry in per_dist_list for k in entry},
                      key=lambda x: x)
    fig, ax = plt.subplots(figsize=(12, 7))
    for d in all_keys:
        vals = [entry.get(str(d), 0.0) for entry in per_dist_list]
        if any(v > 0 for v in vals):
            ax.plot(steps, vals, lw=2, label=f"d={d}")
    ax.set_title(title, fontsize=13); ax.set_xlabel("Step"); ax.set_ylabel("Pairwise Acc")
    ax.set_ylim(0, 1.01); ax.grid(alpha=0.3); ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def _plot_per_dist_small_multiples(history: Dict, out: Path, title: str) -> None:
    steps = history["steps"]
    per_dist_list = history["val_per_dist_acc"]
    dist_counts = history["test_dist_counts"]
    active = sorted([int(k) for k, v in dist_counts.items() if v > 0])
    if not active:
        return
    ncols = 3
    nrows = (len(active) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows),
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten()
    for ax, d in zip(axes, active):
        vals = [entry.get(str(d), 0.0) for entry in per_dist_list]
        ax.plot(steps, vals, lw=2)
        ax.set_title(f"d={d}  (n={dist_counts.get(str(d), 0):,})", fontsize=9)
        ax.set_ylim(0.9, 1.001); ax.grid(alpha=0.3)
        ax.set_xlabel("Step"); ax.set_ylabel("Acc")
    for ax in axes[len(active):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=200); plt.close(fig)


def _plot_diam_bucket_curves(history: Dict, out: Path, title: str) -> None:
    steps   = history["steps"]
    buckets = history["val_per_diam_bucket"]
    keys = ["exact_le7", "exact_le9", "exact_le11", "exact_gt11"]
    labels = ["D≤7", "D≤9", "D≤11", "D>11"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label in zip(keys, labels):
        vals = [entry.get(key, None) for entry in buckets]
        if any(v is not None for v in vals):
            clean = [v if v is not None else float("nan") for v in vals]
            ax.plot(steps, clean, lw=2, label=label)
    ax.set_title(title, fontsize=13); ax.set_xlabel("Step")
    ax.set_ylabel("Exact Match Acc")
    ax.set_ylim(0, 1.01); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


# ── Training loop ────────────────────────────────────────────────────────────

def train_and_evaluate(
    out_dir: Path,
    dataset: Dict[str, Any],
    config: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    set_seed(seed)
    device = get_device(config.get("device", "auto"))
    print(f"  device: {device}")

    n = config["n"]
    model_cfg = ModelConfig(
        n=n, d_model=config["d_model"], n_heads=config["n_heads"],
        d_ff=config["d_ff"], n_layers=config["n_layers"],
        dropout=config.get("dropout", 0.0),
    )
    model = GraphConnectivityTransformer(model_cfg).to(device)
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = AdamW(model.parameters(), lr=config["lr"],
                weight_decay=config.get("weight_decay", 0.01))
    criterion = nn.BCEWithLogitsLoss()

    train_x, train_y = dataset["train_x"], dataset["train_y"]
    test_x, test_y, test_d = dataset["test_x"], dataset["test_y"], dataset["test_d"]
    train_size = len(train_x)

    batch_size   = config["batch_size"]
    total_steps  = config["train_steps"]
    eval_every   = config.get("eval_every", 1000)
    grad_clip    = config.get("grad_clip_norm", 1.0)

    rng        = np.random.default_rng(seed + 99)
    perm       = rng.permutation(train_size)
    cursor     = 0
    loss_window: List[float] = []

    history: Dict[str, Any] = {
        "steps": [], "train_loss": [],
        "val_exact_match": [], "val_pairwise_acc": [],
        "val_per_dist_acc": [], "val_per_diam_bucket": [],
        "timing_stats": {"time_per_1000_steps_sec": []},
        "run_stats": {
            "train_size": train_size,
            "test_size":  len(test_x),
            "ram_before_training_gb": _get_ram_gb(),
        },
        "test_dist_counts": {},
    }

    best_exact = -1.0
    t_block    = time.perf_counter()
    model.train()

    for step in range(1, total_steps + 1):
        # ── batch indexing (wraps epoch cleanly) ──
        if cursor + batch_size > train_size:
            remaining  = train_size - cursor
            first_part = perm[cursor:]
            perm       = rng.permutation(train_size)
            second_part = perm[: batch_size - remaining]
            batch_idx  = np.concatenate([first_part, second_part])
            cursor     = batch_size - remaining
        else:
            batch_idx = perm[cursor: cursor + batch_size]
            cursor   += batch_size

        x = _to_device(train_x[batch_idx], device, torch.float32)
        y = _to_device(train_y[batch_idx], device, torch.float32)

        logits = model(x)
        loss   = criterion(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        loss_window.append(float(loss.item()))
        if len(loss_window) > eval_every:
            loss_window.pop(0)

        # ── timing (every 1000 steps) ──
        if step % 1000 == 0:
            now = time.perf_counter()
            history["timing_stats"]["time_per_1000_steps_sec"].append(
                {"step": step, "seconds": now - t_block}
            )
            t_block = now

        # ── evaluation ──
        if step % eval_every == 0:
            metrics = evaluate(model, test_x, test_y, test_d, device)
            avg_loss = sum(loss_window) / max(1, len(loss_window))

            history["steps"].append(step)
            history["train_loss"].append(avg_loss)
            history["val_exact_match"].append(metrics["exact_match"])
            history["val_pairwise_acc"].append(metrics["pairwise_acc"])
            history["val_per_dist_acc"].append(metrics["per_dist_acc"])
            history["val_per_diam_bucket"].append(metrics["per_diam_bucket"])
            history["test_dist_counts"] = metrics["dist_counts"]

            print(f"  step {step:6d}  loss={avg_loss:.5f}  "
                  f"exact={metrics['exact_match']:.4f}  "
                  f"pairwise={metrics['pairwise_acc']:.4f}",
                  flush=True)

            if metrics["exact_match"] > best_exact:
                best_exact = metrics["exact_match"]
                if config.get("save_best", True):
                    torch.save({"model_state_dict": model.state_dict(),
                                "model_config": model_cfg.__dict__,
                                "step": step}, out_dir / "best.pt")

    if config.get("save_last", True):
        torch.save({"model_state_dict": model.state_dict(),
                    "model_config": model_cfg.__dict__,
                    "step": total_steps}, out_dir / "last.pt")

    history["run_stats"]["ram_after_training_gb"] = _get_ram_gb()
    history["best_exact_match"] = best_exact
    return history


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, required=True,
                        help="Directory to write results into")
    parser.add_argument("--max_diameter", type=int, default=None,
                        help="Diameter filter for rejection sampling (omit = no filter)")
    parser.add_argument("--num_workers", type=int, default=60,
                        help="Worker processes for data generation")
    args = parser.parse_args()

    diam_tag = f"diam{args.max_diameter}" if args.max_diameter is not None else "unfiltered"
    run_name = f"n{N_NODES}_p{int(P*100):03d}_{diam_tag}"

    out_root = Path(args.output_root)
    out_dir  = out_root / run_name
    ensure_dir(out_dir)

    config = {
        "n":             N_NODES,
        "p":             P,
        "d_model":       64,
        "n_heads":       1,
        "d_ff":          128,
        "n_layers":      2,
        "dropout":       0.0,
        "batch_size":    256,
        "lr":            1e-3,
        "weight_decay":  0.01,
        "train_steps":   500_000,
        "eval_every":    1000,
        "grad_clip_norm": 1.0,
        "save_best":     True,
        "save_last":     True,
        "device":        "auto",
        "max_diameter":  args.max_diameter,
        "train_size":    TRAIN_SIZE,
        "test_size":     TEST_SIZE,
        "num_workers":   args.num_workers,
    }

    print(f"\n{'='*60}")
    print(f"Experiment: ER(n={N_NODES}, p={P})  filter={diam_tag}")
    print(f"Output:     {out_dir}")
    print(f"{'='*60}\n")

    dataset = generate_dataset(
        n=N_NODES, p=P,
        train_size=TRAIN_SIZE, test_size=TEST_SIZE,
        max_diameter=args.max_diameter,
        num_workers=args.num_workers,
    )

    ram_after_gen = _get_ram_gb()
    print(f"RAM after generation: {ram_after_gen:.1f} GB\n")

    history = train_and_evaluate(
        out_dir=out_dir, dataset=dataset, config=config, seed=1000,
    )

    # ── Plots ──
    prefix = f"er_n{N_NODES}_{diam_tag}"
    _plot_accuracy_curves(
        history, out_dir / f"{prefix}_accuracy.png",
        f"ER(n={N_NODES}, p={P}) {diam_tag}: Accuracy vs Step",
    )
    _plot_loss_curve(
        history, out_dir / f"{prefix}_loss.png",
        f"ER(n={N_NODES}, p={P}) {diam_tag}: Training Loss vs Step",
    )
    _plot_per_dist_curves(
        history, out_dir / f"{prefix}_per_dist.png",
        f"ER(n={N_NODES}, p={P}) {diam_tag}: Pairwise Acc by Distance",
    )
    _plot_per_dist_small_multiples(
        history, out_dir / f"{prefix}_per_dist_sm.png",
        f"ER(n={N_NODES}, p={P}) {diam_tag}: Pairwise Acc by Distance",
    )
    _plot_diam_bucket_curves(
        history, out_dir / f"{prefix}_per_diam_bucket.png",
        f"ER(n={N_NODES}, p={P}) {diam_tag}: Exact Match by Diameter Bucket",
    )

    # ── Summary JSON ──
    summary = {
        "config":        config,
        "best_exact_match": history["best_exact_match"],
        "dataset_stats": dataset["dataset_stats"],
        "ram_after_gen_gb": ram_after_gen,
    }
    save_json(out_root / "summary.json", summary)
    save_json(out_dir  / "history.json", history)

    print(f"\nDone. Best exact match: {history['best_exact_match']:.4f}")
    print(f"Results in: {out_dir}")


if __name__ == "__main__":
    main()
