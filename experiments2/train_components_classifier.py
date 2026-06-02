"""
Exp 3 — new task: learn to tell ONE chain from TWO chains (= 1 vs 2 connected
components) at n = 40.

Each training sample is, with probability 1/2 each:
  * 1chain : a single path over all 40 nodes  -> 1 connected component -> label 0
  * 2chain : two disjoint paths of 20 nodes    -> 2 connected components -> label 1
(node order randomly permuted every time). The "+1 / -1" framing of the advisor
maps to BCE targets {1, 0}; +1 = 2chain, -1/0 = 1chain.

Note the two classes are almost indistinguishable by cheap global cues:
  - both use all 40 nodes (no isolated padding),
  - edge counts are 39 (1chain) vs 38 (2chain),
  - degree histogram differs only by 2 vs 4 degree-1 endpoints.
So the model must essentially decide whether the graph has one or two components.

Architecture is identical to the n=40 BIG connectivity models (d_model=512,
n_heads=4, d_ff=2048, 2 layers, normalized-ReLU attention), with the n x n
read-out replaced by a mean-pool + single-logit head (GraphBinaryClassifier).
Online data, bf16 autocast, AdamW + cosine, batch 1000.
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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from data import add_self_loops, generate_one_chain_graph, generate_two_chains_graph
from model import GraphBinaryClassifier, ModelConfig
from utils import ensure_dir, get_device, save_json, set_seed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


N_NODES   = 40
TEST_SIZE = 10_000


def _get_ram_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def _make_sample(rng: np.random.Generator, n: int) -> Tuple[np.ndarray, int]:
    """Return (adj_with_self_loops uint8, label). label 1 = 2chain, 0 = 1chain."""
    label = int(rng.integers(0, 2))
    if label == 1:                       # two chains of n/2
        adj_no = generate_two_chains_graph(n, n // 2)
    else:                                # one chain over all n
        adj_no = generate_one_chain_graph(n)
    perm = rng.permutation(n)
    adj_no = adj_no[np.ix_(perm, perm)]
    return add_self_loops(adj_no).astype(np.uint8), label


class ChainCountStream(IterableDataset):
    """Infinite 50/50 stream of 1chain / 2chain graphs (n nodes), permuted."""

    def __init__(self, n: int, seed: int):
        self.n = n
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = np.random.default_rng((self.seed * 100003 + worker_id * 31337) & 0x7FFFFFFF)
        while True:
            yield _make_sample(rng, self.n)


def _collate(batch):
    xs = np.stack([b[0] for b in batch])
    ys = np.array([b[1] for b in batch], dtype=np.float32)
    return torch.from_numpy(xs), torch.from_numpy(ys)


def build_test_set(n: int, size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Balanced test set: exactly half 1chain, half 2chain (permuted)."""
    rng = np.random.default_rng(seed)
    xs = np.empty((size, n, n), dtype=np.uint8)
    ys = np.empty(size, dtype=np.float32)
    labels = np.array([0] * (size // 2) + [1] * (size - size // 2))
    rng.shuffle(labels)
    one = generate_one_chain_graph(n)
    two = generate_two_chains_graph(n, n // 2)
    for i, lab in enumerate(labels):
        base = two if lab == 1 else one
        perm = rng.permutation(n)
        xs[i] = add_self_loops(base[np.ix_(perm, perm)]).astype(np.uint8)
        ys[i] = float(lab)
    return xs, ys


@torch.no_grad()
def evaluate(model, test_x, test_y, device, batch_size=512) -> Dict[str, float]:
    model.eval()
    n_graphs = test_x.shape[0]
    preds = np.empty(n_graphs, dtype=np.int8)
    probs = np.empty(n_graphs, dtype=np.float32)
    for s in range(0, n_graphs, batch_size):
        e = min(s + batch_size, n_graphs)
        xb = torch.from_numpy(test_x[s:e]).to(device, dtype=torch.float32)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xb)
        probs[s:e] = torch.sigmoid(logits.float()).cpu().numpy()
        preds[s:e] = (logits > 0).cpu().numpy().astype(np.int8)
    y = test_y.astype(np.int8)
    acc = float((preds == y).mean())
    # per-class accuracy
    acc1 = float((preds[y == 1] == 1).mean()) if (y == 1).any() else float("nan")
    acc0 = float((preds[y == 0] == 0).mean()) if (y == 0).any() else float("nan")
    model.train()
    return {"acc": acc, "acc_2chain": acc1, "acc_1chain": acc0}


def lr_at_step(step, warmup, total, peak):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + pymath.cos(pymath.pi * min(1.0, progress)))


def _plot(h, out, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h["steps"], h["val_acc"], lw=2, label="accuracy")
    ax.plot(h["steps"], h["val_acc_1chain"], lw=1.5, ls="--", label="1chain (label 0)")
    ax.plot(h["steps"], h["val_acc_2chain"], lw=1.5, ls="--", label="2chain (label 1)")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
    ax.set_title(title); ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, 1.02); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def train(out_dir, loader, test_x, test_y, config, seed) -> Dict[str, Any]:
    set_seed(seed)
    device = get_device("auto")
    print(f"  device: {device}")

    mcfg = ModelConfig(n=config["n"], d_model=config["d_model"], n_heads=config["n_heads"],
                       d_ff=config["d_ff"], n_layers=config["n_layers"],
                       dropout=config.get("dropout", 0.0), attn_kind=config["attn_kind"])
    model = GraphBinaryClassifier(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    opt = AdamW(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 1e-4))
    criterion = nn.BCEWithLogitsLoss()

    total = config["train_steps"]; warmup = config.get("warmup_steps", 1000)
    eval_every = config.get("eval_every", 2000); grad_clip = config.get("grad_clip_norm", 1.0)
    peak_lr = config["lr"]

    history: Dict[str, Any] = {
        "steps": [], "train_loss": [], "val_acc": [],
        "val_acc_1chain": [], "val_acc_2chain": [],
        "run_stats": {"n_parameters": n_params, "ram_before_gb": _get_ram_gb()},
    }
    loss_window: List[float] = []
    best_acc = -1.0
    t_block = time.perf_counter()
    model.train()
    print(f"  Training {total:,} steps × batch {config['batch_size']}")

    step = 0
    for xb_cpu, yb_cpu in loader:
        step += 1
        if step > total:
            break
        lr = lr_at_step(step, warmup, total, peak_lr)
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
            m = evaluate(model, test_x, test_y, device)
            avg_loss = sum(loss_window) / max(1, len(loss_window))
            history["steps"].append(step)
            history["train_loss"].append(avg_loss)
            history["val_acc"].append(m["acc"])
            history["val_acc_1chain"].append(m["acc_1chain"])
            history["val_acc_2chain"].append(m["acc_2chain"])
            print(f"  step {step:>7d} | lr={lr:.2e} | loss={avg_loss:.5f} | "
                  f"acc={m['acc']:.4f} (1chain={m['acc_1chain']:.3f} "
                  f"2chain={m['acc_2chain']:.3f}) | {elapsed:.1f}s/{eval_every}",
                  flush=True)
            if m["acc"] > best_acc:
                best_acc = m["acc"]
                torch.save({"model_state_dict": model.state_dict(),
                            "model_config": mcfg.__dict__, "step": step,
                            "task": "chain_count"}, out_dir / "best.pt")
            torch.save({"model_state_dict": model.state_dict(),
                        "model_config": mcfg.__dict__, "step": step,
                        "task": "chain_count"}, out_dir / "last.pt")
            t_block = time.perf_counter()

    history["run_stats"]["ram_after_gb"] = _get_ram_gb()
    history["best_acc"] = best_acc
    return history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--train_steps", type=int, default=200_000)
    ap.add_argument("--batch_size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    run_name = f"chaincount_n{N_NODES}_seed{args.seed}"
    out_root = Path(args.output_root); out_dir = out_root / run_name
    ensure_dir(out_dir)

    config = {
        "n": N_NODES, "task": "1chain_vs_2chain",
        "d_model": 512, "n_heads": 4, "d_ff": 2048, "n_layers": 2, "dropout": 0.0,
        "attn_kind": "normalized_relu",
        "batch_size": args.batch_size, "lr": 1e-4, "weight_decay": 1e-4,
        "train_steps": args.train_steps, "warmup_steps": 1000, "eval_every": 2000,
        "grad_clip_norm": 1.0, "test_size": TEST_SIZE,
        "num_workers": args.num_workers, "seed": args.seed,
    }

    print(f"\n{'='*72}")
    print(f"  Exp 3: 1chain vs 2chain classifier, n={N_NODES}, seed={args.seed}")
    print(f"  arch = n40big (d_model=512, n_heads=4, normalized-ReLU, 2 layers)")
    print(f"  Output: {out_dir}")
    print(f"{'='*72}\n")

    test_x, test_y = build_test_set(N_NODES, TEST_SIZE, seed=999)
    print(f"balanced test set: {int((test_y==1).sum())} 2chain / "
          f"{int((test_y==0).sum())} 1chain")

    stream = ChainCountStream(N_NODES, seed=args.seed + 7)
    loader = DataLoader(stream, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=_collate, pin_memory=True, prefetch_factor=4,
                        persistent_workers=True)

    history = train(out_dir, loader, test_x, test_y, config, args.seed)
    _plot(history, out_dir / f"{run_name}_accuracy.png",
          f"1chain vs 2chain (n={N_NODES}) seed{args.seed}: accuracy")
    save_json(out_dir / "history.json", history)
    save_json(out_root / f"summary_{run_name}.json",
              {"config": config, "best_acc": history["best_acc"]})
    print(f"\nDone. Best accuracy: {history['best_acc']:.4f}\nResults in: {out_dir}")


if __name__ == "__main__":
    main()
