from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import DatasetConfig, GraphMatrixDataset
from eval import (
    evaluate_distance_conditioned_accuracy,
    evaluate_model,
    evaluate_ood_suites,
    load_checkpoint,
)
from plots import plot_distance_accuracy, plot_training_history
from train import TrainConfig, train_model
from utils import get_device, save_json


def main() -> None:
    out_dir = PROJECT_ROOT / "runs" / "baseline"
    cfg = TrainConfig(
        output_dir=str(out_dir),
        n=20,
        p=0.08,
        train_size=20000,
        val_size=2000,
        batch_size=64,
        epochs=20,
        lr=1e-3,
        weight_decay=1e-2,
        d_model=128,
        n_heads=4,
        d_ff=256,
        n_layers=2,
        dropout=0.0,
        train_mode="er",
        val_mode="er",
        max_diameter_train=None,
        seed=42,
        device="auto",
        num_workers=0,
    )
    train_info = train_model(cfg)
    plot_training_history(out_dir / "history.json", out_dir / "training_history.png")

    device = get_device(cfg.device)
    loaded = load_checkpoint(train_info["best_checkpoint"], device)
    model = loaded.model

    val_ds = GraphMatrixDataset(
        DatasetConfig(mode="er", n=cfg.n, p=cfg.p, size=cfg.val_size, seed=cfg.seed + 999)
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    val_metrics = evaluate_model(model, val_loader, device, "er_val")
    ood_metrics = evaluate_ood_suites(
        model=model,
        n=cfg.n,
        k=cfg.n // 2,
        size=1000,
        batch_size=cfg.batch_size,
        device=device,
    )
    dist_metrics = evaluate_distance_conditioned_accuracy(
        model, val_loader, device=device, reliable_threshold=0.99
    )
    save_json(out_dir / "er_val_metrics.json", val_metrics)
    save_json(out_dir / "ood_metrics.json", ood_metrics)
    save_json(out_dir / "distance_metrics.json", dist_metrics)
    plot_distance_accuracy(out_dir / "distance_metrics.json", out_dir / "distance_accuracy.png")

    summary = {
        "train_info": train_info,
        "val_metrics": val_metrics,
        "ood_metrics": ood_metrics,
        "distance_metrics": dist_metrics,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["ood_metrics"], indent=2))


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    main()
