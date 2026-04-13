from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Tuple

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


TRAIN_SET_SIZE = 6_000_000
TEST_SET_SIZE = 10_000
P_VALUES = [0.2, 0.5]
NUM_WORKERS = 16
CHUNK_SIZE = 100_000


def _generate_train_chunk(args: Tuple[int, int, float, int]) -> Tuple[int, np.ndarray, np.ndarray]:
    start_idx, size, p, n = args
    rng = np.random.default_rng(12345 + int(p * 1000) * 1_000_000 + start_idx)

    train_x = np.empty((size, n, n), dtype=np.uint8)
    train_y = np.empty((size, n, n), dtype=np.uint8)

    for i in range(size):
        adj_no_loops = generate_er_graph(n, p, rng)
        adj = add_self_loops(adj_no_loops)
        target = compute_connectivity_matrix(adj_no_loops)
        train_x[i] = adj.astype(np.uint8)
        train_y[i] = target.astype(np.uint8)

    return start_idx, train_x, train_y


def _generate_test_chunk(args: Tuple[int, int, float, int]) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    start_idx, size, p, n = args
    rng = np.random.default_rng(54321 + int(p * 1000) * 1_000_000 + start_idx)

    test_x = np.empty((size, n, n), dtype=np.uint8)
    test_y = np.empty((size, n, n), dtype=np.uint8)
    test_d = np.empty((size, n, n), dtype=np.int16)

    for i in range(size):
        adj_no_loops = generate_er_graph(n, p, rng)
        adj = add_self_loops(adj_no_loops)
        target = compute_connectivity_matrix(adj_no_loops)
        dist = compute_all_pairs_shortest_paths(adj_no_loops)

        test_x[i] = adj.astype(np.uint8)
        test_y[i] = target.astype(np.uint8)
        test_d[i] = dist.astype(np.int16)

    return start_idx, test_x, test_y, test_d


def _build_chunk_specs(total_size: int, p: float, n: int) -> List[Tuple[int, int, float, int]]:
    specs: List[Tuple[int, int, float, int]] = []
    for start_idx in range(0, total_size, CHUNK_SIZE):
        size = min(CHUNK_SIZE, total_size - start_idx)
        specs.append((start_idx, size, p, n))
    return specs


def generate_fixed_dataset(
    n: int,
    p: float,
    train_size: int,
    test_size: int,
) -> Dict[str, np.ndarray]:
    train_x = np.empty((train_size, n, n), dtype=np.uint8)
    train_y = np.empty((train_size, n, n), dtype=np.uint8)

    test_x = np.empty((test_size, n, n), dtype=np.uint8)
    test_y = np.empty((test_size, n, n), dtype=np.uint8)
    test_d = np.empty((test_size, n, n), dtype=np.int16)

    train_specs = _build_chunk_specs(train_size, p, n)
    test_specs = _build_chunk_specs(test_size, p, n)

    print(f"Generating fixed TRAIN dataset for p={p} with {NUM_WORKERS} processes...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for start_idx, chunk_x, chunk_y in executor.map(_generate_train_chunk, train_specs):
            end_idx = start_idx + chunk_x.shape[0]
            train_x[start_idx:end_idx] = chunk_x
            train_y[start_idx:end_idx] = chunk_y
            print(f"  Train p={p}: generated {end_idx}/{train_size}")

    print(f"Generating fixed TEST dataset for p={p} with {NUM_WORKERS} processes...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for start_idx, chunk_x, chunk_y, chunk_d in executor.map(_generate_test_chunk, test_specs):
            end_idx = start_idx + chunk_x.shape[0]
            test_x[start_idx:end_idx] = chunk_x
            test_y[start_idx:end_idx] = chunk_y
            test_d[start_idx:end_idx] = chunk_d
            print(f"  Test p={p}: generated {end_idx}/{test_size}")

    return {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "test_d": test_d,
    }


def build_fixed_datasets(n: int) -> Dict[float, Dict[str, np.ndarray]]:
    datasets: Dict[float, Dict[str, np.ndarray]] = {}
    for p in P_VALUES:
        datasets[p] = generate_fixed_dataset(
            n=n,
            p=p,
            train_size=TRAIN_SET_SIZE,
            test_size=TEST_SET_SIZE,
        )
    return datasets


def _to_device_batch(arr: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.from_numpy(arr)
    if device.type == "cuda":
        return tensor.pin_memory().to(device=device, dtype=dtype, non_blocking=True)
    return tensor.to(device=device, dtype=dtype)


def compute_capacity_and_distance_counts(
    model: nn.Module,
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_d: np.ndarray,
    device: torch.device,
    threshold: float = 0.0,
) -> Dict[str, Any]:
    model.eval()

    n = test_x.shape[1]
    num_samples = test_x.shape[0]

    correct_per_dist = {k: 0 for k in range(1, 11)}
    count_per_dist = {k: 0 for k in range(1, 11)}

    eye = torch.eye(n, dtype=torch.bool, device=device).unsqueeze(0)

    with torch.no_grad():
        x = _to_device_batch(test_x, device=device, dtype=torch.float32)
        y = _to_device_batch(test_y, device=device, dtype=torch.float32)
        d = _to_device_batch(test_d, device=device, dtype=torch.int64)

        logits = model(x)
        preds = (logits > threshold).to(torch.int64)
        y_int = y.to(torch.int64)

        eq = preds == y_int

        exact_match_correct = int(eq.view(num_samples, -1).all(dim=1).sum().item())
        pairwise_correct = int(eq.sum().item())
        pairwise_total = int(eq.numel())

        offdiag = ~eye.expand(num_samples, n, n)
        for dist_val in range(1, 11):
            mask = offdiag & (d == dist_val)
            if mask.any():
                count_per_dist[dist_val] += int(mask.sum().item())
                correct_per_dist[dist_val] += int(eq[mask].sum().item())

    per_dist_acc = {
        str(k): (correct_per_dist[k] / count_per_dist[k] if count_per_dist[k] > 0 else 0.0)
        for k in range(1, 11)
    }

    distance_counts = {str(k): count_per_dist[k] for k in range(1, 11)}

    exact_match = exact_match_correct / max(1, num_samples)
    pairwise_acc = pairwise_correct / max(1, pairwise_total)

    return {
        "exact_match": exact_match,
        "pairwise_accuracy": pairwise_acc,
        "per_distance_pairwise_accuracy": per_dist_acc,
        "distance_counts_1_10": distance_counts,
    }


def train_and_evaluate_single_run(
    out_dir: Path,
    n: int,
    p: float,
    seed: int,
    config: Dict[str, Any],
    dataset: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    set_seed(seed)
    device = get_device(config.get("device", "auto"))

    model_cfg = ModelConfig(
        n=n,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
    )

    model = GraphConnectivityTransformer(model_cfg).to(device)
    opt = AdamW(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0.0))
    criterion = nn.BCEWithLogitsLoss()

    train_x = dataset["train_x"]
    train_y = dataset["train_y"]
    test_x = dataset["test_x"]
    test_y = dataset["test_y"]
    test_d = dataset["test_d"]

    train_size = train_x.shape[0]
    batch_size = config["batch_size"]
    steps = config["train_steps"]
    eval_every = config["eval_every"]

    rng = np.random.default_rng(seed + 123)
    permutation = rng.permutation(train_size)
    cursor = 0

    best_exact = -1.0
    best_ckpt = out_dir / "best.pt"
    last_ckpt = out_dir / "last.pt"

    history = {
        "steps": [],
        "train_loss": [],
        "val_exact_match": [],
        "val_pairwise_acc": [],
        "test_set_distance_counts": {str(k): 0 for k in range(1, 11)},
        "val_per_distance_pairwise_acc": [],
    }

    loss_window: List[float] = []

    for step in range(1, steps + 1):
        model.train()

        if cursor + batch_size > train_size:
            remaining = train_size - cursor
            first_part = permutation[cursor:]
            permutation = rng.permutation(train_size)
            second_needed = batch_size - remaining
            second_part = permutation[:second_needed]
            batch_idx = np.concatenate([first_part, second_part], axis=0)
            cursor = second_needed
        else:
            batch_idx = permutation[cursor: cursor + batch_size]
            cursor += batch_size

        x = _to_device_batch(train_x[batch_idx], device=device, dtype=torch.float32)
        y = _to_device_batch(train_y[batch_idx], device=device, dtype=torch.float32)

        logits = model(x)
        loss = criterion(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip_norm", 1.0))
        opt.step()

        loss_window.append(float(loss.item()))
        if len(loss_window) > eval_every:
            loss_window.pop(0)

        if step % eval_every == 0:
            metrics = compute_capacity_and_distance_counts(
                model=model,
                test_x=test_x,
                test_y=test_y,
                test_d=test_d,
                device=device,
            )

            history["steps"].append(step)
            history["train_loss"].append(sum(loss_window) / max(1, len(loss_window)))
            history["val_exact_match"].append(metrics["exact_match"])
            history["val_pairwise_acc"].append(metrics["pairwise_accuracy"])
            history["test_set_distance_counts"] = metrics["distance_counts_1_10"]
            history["val_per_distance_pairwise_acc"].append(metrics["per_distance_pairwise_accuracy"])

            if metrics["exact_match"] > best_exact:
                best_exact = float(metrics["exact_match"])
                if config.get("save_best", True):
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "model_config": model_cfg.__dict__,
                            "step": step,
                        },
                        best_ckpt,
                    )

    if config.get("save_last", True):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model_cfg.__dict__,
                "step": steps,
            },
            last_ckpt,
        )

    save_json(out_dir / "history.json", history)

    return {
        "out_dir": str(out_dir),
        "best_exact_match": best_exact,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="runs/retrain_er")
    args = parser.parse_args()

    base_config = {
        "n": 10,
        "d_model": 64,
        "n_heads": 1,
        "d_ff": 128,
        "n_layers": 2,
        "dropout": 0.0,
        "batch_size": 256,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "train_steps": 500_000,
        "eval_every": 1000,
        "save_best": True,
        "save_last": True,
        "grad_clip_norm": 1.0,
        "device": "auto",
    }

    out_root = Path(args.output_root)
    ensure_dir(out_root)

    fixed_datasets = build_fixed_datasets(base_config["n"])

    summary = {}
    for pval in P_VALUES:
        seed = 1000
        run_name = f"n{base_config['n']}_p{pval}_rep0"
        out_dir = out_root / run_name
        ensure_dir(out_dir)

        print(f"Starting run: p={pval} -> {out_dir}")
        res = train_and_evaluate_single_run(
            out_dir=out_dir,
            n=base_config["n"],
            p=pval,
            seed=seed,
            config=base_config,
            dataset=fixed_datasets[pval],
        )
        summary[f"p{pval}"] = res

    save_json(out_root / "summary.json", summary)
    print("All runs finished. Summaries written to", str(out_root))


if __name__ == "__main__":
    main()