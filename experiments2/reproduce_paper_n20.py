"""
Reproduce the *standard Transformer* experiments of Ye et al. 2026
("Transformers Provably Learn Algorithmic Solutions for Graph Connectivity,
But Only with the Right Data") — specifically the data needed for:

  * Figure 1   — 2-layer Transformer trained on ER(n=20, p=0.08) (unrestricted),
                 exact-match accuracy on ER (in-dist), 2Chain, 2Clique vs step.
  * Figure 7   — unrestricted vs restricted (diam ≤ 9 = 3^2) on OOD 2Chain.
  * Figure 11  — same two runs evaluated on OOD 2Clique.

All three figures are produced from just TWO training configs (unrestricted and
restricted diam≤9), because each run is evaluated on all three test sets
(ER in-dist, 2Chain, 2Clique) throughout training.

Configuration is kept faithful to the paper (Appendix D.1, "Standard
Transformers"):
  * n = 20 nodes, ER edge probability p = 0.08
  * 2 layers, single-head self-attention, hidden dim d = 512
  * normalized-ReLU attention  α = (1/n)·ReLU(QK^T/√d_h)   (Definition A.1)
  * pre-norm LayerNorm + GeLU feed-forward
  * batch size 1000, 1,000,000 steps (online data, ~1B unique graphs)
  * AdamW, lr = 1e-4, weight_decay = 1e-4, cosine LR decay (+ short warmup)

Infrastructure (online IterableDataset, bf16 autocast, parallel test-set
generation, cosine schedule) is reused from the n=40 BIG experiments; only the
data scale (n=20, p=0.08), the architecture (single head, per the paper) and the
OOD evaluation on 2Chain/2Clique are new.

Modes (via --max_diameter):
  (omit)   unrestricted ER(n=20, p=0.08)                 -> Fig 1 + "Unrestricted"
  9        rejection-sample diam(G) ≤ 9  (= 3^L, L=2)     -> "Restrict Diameter (<=9)"
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
    generate_two_chains_graph,
    generate_two_cliques_graph,
)
from model import GraphConnectivityTransformer, RobertaGraphTransformer, ModelConfig
from utils import ensure_dir, get_device, save_json, set_seed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Constants (paper preliminary-study / Fig 1,7,11 setup) ────────────────────
N_NODES   = 20
P         = 0.08
K         = 10          # 2Chain / 2Clique block size (n == 2*k)
TEST_SIZE = 10_000
OOD_SIZE  = 10_000
MAX_DIST_LOG = N_NODES - 1


def _get_ram_gb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / (1024.0 ** 2)


# ── Online ER generator (IterableDataset) ─────────────────────────────────────

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
    Each worker gets a disjoint seed -> different graphs; no epoch."""

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
    xs = np.stack([b[0] for b in batch])
    ys = np.stack([b[1] for b in batch])
    return torch.from_numpy(xs), torch.from_numpy(ys)


# ── ER test set (parallel, one-shot, optionally diameter-filtered) ────────────

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


def build_er_test(n: int, p: float, max_diameter: Optional[int],
                  num_workers: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tag = "D≤" + str(max_diameter) if max_diameter is not None else "unfiltered"
    print(f"Generating {TEST_SIZE:,} ER test graphs ({tag}) with {num_workers} workers …")
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


# ── OOD test sets (2Chain / 2Clique, random node permutations) ────────────────

def build_structured_test(mode: str, n: int, k: int, size: int,
                          seed: int) -> Tuple[np.ndarray, np.ndarray]:
    assert mode in ("two_chains", "two_cliques")
    rng = np.random.default_rng(seed)
    base = (generate_two_chains_graph(n, k) if mode == "two_chains"
            else generate_two_cliques_graph(n, k))
    xs = np.empty((size, n, n), dtype=np.uint8)
    ys = np.empty((size, n, n), dtype=np.uint8)
    print(f"Generating {size:,} {mode} test graphs …")
    t0 = time.perf_counter()
    for i in range(size):
        perm = rng.permutation(n)
        adj_no = base[np.ix_(perm, perm)]
        xs[i] = add_self_loops(adj_no).astype(np.uint8)
        ys[i] = compute_connectivity_matrix(adj_no).astype(np.uint8)
    print(f"  done in {time.perf_counter()-t0:.1f}s")
    return xs, ys


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_exact(model: nn.Module, test_x: np.ndarray, test_y: np.ndarray,
                   device: torch.device, batch_size: int = 512) -> Tuple[float, float]:
    """Return (exact_match_accuracy, pairwise_accuracy)."""
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
    eq = all_pred == test_y.astype(np.int8)
    exact_match = float(eq.reshape(n_graphs, -1).all(axis=1).mean())
    pairwise = float(eq.mean())
    model.train()
    return exact_match, pairwise


# ── LR schedule: linear warmup → cosine decay ────────────────────────────────

def lr_at_step(step: int, warmup: int, total: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + pymath.cos(pymath.pi * min(1.0, progress)))


# ── Per-run plot (3 curves) ──────────────────────────────────────────────────

def _plot_run(h: Dict, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h["steps"], h["val_er_exact"], lw=2, label="Erdős-Rényi (in-dist)")
    ax.plot(h["steps"], h["val_2chain_exact"], lw=2, label="Two Chains")
    ax.plot(h["steps"], h["val_2clique_exact"], lw=2, label="Two Cliques")
    ax.set_title(title); ax.set_xlabel("Training Step"); ax.set_ylabel("Exact Match Accuracy")
    ax.set_ylim(0, 1.01); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(out_dir: Path, dataset_iter, eval_sets: Dict[str, Tuple[np.ndarray, np.ndarray]],
          config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    set_seed(seed)
    device = get_device("auto")
    print(f"  device: {device}")

    model_cfg = ModelConfig(
        n=config["n"], d_model=config["d_model"], n_heads=config["n_heads"],
        d_ff=config["d_ff"], n_layers=config["n_layers"],
        dropout=config.get("dropout", 0.0), attn_kind=config["attn_kind"],
        norm_style=config.get("norm_style", "pre"),
        layer_norm_eps=config.get("layer_norm_eps", 1e-5),
        init_std=config.get("init_std", 0.02),
    )
    if config.get("arch", "minimal") == "roberta":
        model = RobertaGraphTransformer(model_cfg).to(device)
    else:
        model = GraphConnectivityTransformer(model_cfg).to(device)
    print(f"  arch: {config.get('arch', 'minimal')}")
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
        "val_er_exact": [], "val_er_pairwise": [],
        "val_2chain_exact": [], "val_2chain_pairwise": [],
        "val_2clique_exact": [], "val_2clique_pairwise": [],
        "timing_stats": {"time_per_eval_block_sec": []},
        "run_stats": {"n_parameters": n_params,
                      "ram_before_training_gb": _get_ram_gb()},
    }

    loss_window: List[float] = []
    best_er_exact = -1.0
    t_block = time.perf_counter()
    model.train()

    print(f"  Starting training: {total_steps:,} steps × batch "
          f"{config['batch_size']} = {total_steps * config['batch_size']:,} samples")

    step = 0
    for xb_cpu, yb_cpu in dataset_iter:
        step += 1
        if step > total_steps:
            break

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
            history["timing_stats"]["time_per_eval_block_sec"].append(
                {"step": step, "seconds": elapsed})

            er_x, er_y = eval_sets["er"]
            c_x, c_y = eval_sets["two_chains"]
            q_x, q_y = eval_sets["two_cliques"]
            er_e, er_p = evaluate_exact(model, er_x, er_y, device)
            ch_e, ch_p = evaluate_exact(model, c_x, c_y, device)
            cq_e, cq_p = evaluate_exact(model, q_x, q_y, device)
            avg_loss = sum(loss_window) / max(1, len(loss_window))

            history["steps"].append(step)
            history["train_loss"].append(avg_loss)
            history["val_er_exact"].append(er_e)
            history["val_er_pairwise"].append(er_p)
            history["val_2chain_exact"].append(ch_e)
            history["val_2chain_pairwise"].append(ch_p)
            history["val_2clique_exact"].append(cq_e)
            history["val_2clique_pairwise"].append(cq_p)

            print(f"  step {step:>7d} | lr={lr:.2e} | loss={avg_loss:.6f} | "
                  f"ER={er_e:.4f} | 2chain={ch_e:.4f} | 2clique={cq_e:.4f} "
                  f"| {elapsed:.1f}s/{eval_every}", flush=True)

            if er_e > best_er_exact:
                best_er_exact = er_e
                torch.save({"model_state_dict": model.state_dict(),
                            "model_config": model_cfg.__dict__, "step": step},
                           out_dir / "best.pt")
            torch.save({"model_state_dict": model.state_dict(),
                        "model_config": model_cfg.__dict__, "step": step},
                       out_dir / "last.pt")
            t_block = time.perf_counter()

    history["run_stats"]["ram_after_training_gb"] = _get_ram_gb()
    history["best_er_exact"] = best_er_exact
    return history


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--max_diameter", type=int, default=None,
                        help="rejection-sampling cutoff for ER training (omit = unrestricted)")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--train_steps", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--arch", choices=["minimal", "roberta"], default="minimal",
                        help="'roberta' = RoBERTa-faithful (App. D.1); 'minimal' = clean A.1-like")
    args = parser.parse_args()

    diam_tag = f"diam{args.max_diameter}" if args.max_diameter is not None else "unrestricted"
    run_name = f"n{N_NODES}_p{int(P*100):03d}_{diam_tag}_seed{args.seed}"
    out_root = Path(args.output_root)
    out_dir = out_root / run_name
    ensure_dir(out_dir)

    # RoBERTa-faithful (App. D.1) brings post-LayerNorm, dropout 0.1, init 0.02;
    # the minimal A.1-like variant is pre-norm, dropout 0, default init.
    is_roberta = (args.arch == "roberta")
    config = {
        "arch":          args.arch,
        "n":             N_NODES,
        "p":             P,
        "k":             K,
        "d_model":       512,
        "n_heads":       1,          # paper: single-head self-attention (App. D.1)
        "d_ff":          2048,       # RoBERTa intermediate = 4 x d_model
        "n_layers":      2,
        "dropout":       0.1 if is_roberta else 0.0,
        "attn_kind":     "normalized_relu",
        "norm_style":    "post" if is_roberta else "pre",
        "layer_norm_eps": 1e-5,
        "init_std":      0.02,
        "batch_size":    args.batch_size,
        "lr":            1e-4,
        "weight_decay":  1e-4,
        "train_steps":   args.train_steps,
        "warmup_steps":  1000,
        "eval_every":    5000,
        "grad_clip_norm": 1.0,
        "max_diameter":  args.max_diameter,
        "test_size":     TEST_SIZE,
        "ood_size":      OOD_SIZE,
        "num_workers":   args.num_workers,
        "seed":          args.seed,
    }

    print(f"\n{'='*72}")
    print(f"  Paper reproduction (standard Transformer): ER(n={N_NODES}, p={P}), "
          f"filter={diam_tag}, seed={args.seed}")
    print(f"  arch={args.arch}, d_model=512, single-head, normalized-ReLU, 2 layers, online, bf16")
    print(f"  Output: {out_dir}")
    print(f"{'='*72}\n")

    # Test sets: ER matches the training filter (in-distribution held-out);
    # 2Chain / 2Clique are fixed OOD sets (seed kept constant across runs).
    er_x, er_y, _ = build_er_test(N_NODES, P, args.max_diameter,
                                  num_workers=args.num_workers, seed=args.seed)
    ch_x, ch_y = build_structured_test("two_chains", N_NODES, K, OOD_SIZE, seed=12345)
    cq_x, cq_y = build_structured_test("two_cliques", N_NODES, K, OOD_SIZE, seed=23456)
    eval_sets = {"er": (er_x, er_y), "two_chains": (ch_x, ch_y),
                 "two_cliques": (cq_x, cq_y)}

    stream = OnlineERStream(N_NODES, P, args.max_diameter, seed=args.seed + 7)
    loader = DataLoader(
        stream, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=_stream_collate, pin_memory=True,
        prefetch_factor=4, persistent_workers=True,
    )

    history = train(out_dir, loader, eval_sets, config, args.seed)

    _plot_run(history, out_dir / f"{run_name}_curves.png",
              f"ER(n={N_NODES}, p={P}) {diam_tag} seed{args.seed}: Exact Match")

    save_json(out_dir / "history.json", history)
    save_json(out_root / f"summary_{run_name}.json",
              {"config": config, "best_er_exact": history["best_er_exact"],
               "final_2chain_exact": history["val_2chain_exact"][-1] if history["val_2chain_exact"] else None,
               "final_2clique_exact": history["val_2clique_exact"][-1] if history["val_2clique_exact"] else None})
    print(f"\nDone. Best ER exact={history['best_er_exact']:.4f} | "
          f"final 2chain={history['val_2chain_exact'][-1]:.4f} | "
          f"final 2clique={history['val_2clique_exact'][-1]:.4f}")
    print(f"Results in: {out_dir}")


if __name__ == "__main__":
    main()
