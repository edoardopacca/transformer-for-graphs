from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import DatasetConfig, GraphMatrixDataset
from eval import (
    evaluate_distance_conditioned_accuracy,
    load_checkpoint,
)
from train import TrainConfig, train_model
from utils import get_device, save_json, load_json, ensure_dir


def make_run_id(c: TrainConfig) -> str:
    parts = [
        f"n{c.n}",
        f"d{c.d_model}",
        f"layers{c.n_layers}",
        f"heads{c.n_heads}",
        f"mode{c.train_mode}",
        f"ep{c.epochs}",
        f"seed{c.seed}",
    ]
    return "_".join(parts)


def extract_epoch_from_filename(p: Path) -> int | None:
    m = re.search(r"epoch_(\d+)\.pt$", p.name)
    if m:
        return int(m.group(1))
    return None


def compute_max_reliable_length(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> int:
    metrics = evaluate_distance_conditioned_accuracy(model, loader, device, reliable_threshold=0.99)
    val = metrics.get("max_reliable_path_length_exact")
    return int(val) if (val is not None) else 0


def plot_capacity(xs: list[int], ys: list[int], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, marker="o", color="#ff7f0e", linewidth=2, label="Max Perfect Path Length (Acc≥0.99)")
    ax.fill_between(xs, ys, color="#ff7f0e", alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("Training Step (epoch)")
    ax.set_ylabel("Maximum Perfect Path Length")
    ax.set_ylim(0, max(ys + [1]) + 1)
    ax.grid(True, alpha=0.25)
    # horizontal dashed blue line at 3^2
    yline = 3 ** 2
    ax.hlines(yline, xs[0], xs[-1], colors="#1f77b4", linestyles="--", label="3^2 (capacity)")
    ax.legend(loc="lower right")
    fig.suptitle("Progression of Perfect Path Length Prediction")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = TrainConfig(
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

    run_id = make_run_id(cfg)
    out_dir = PROJECT_ROOT / "trainings" / run_id
    cfg.output_dir = str(out_dir)

    history_path = out_dir / "history.json"
    best_ckpt = out_dir / "best.pt"
    last_ckpt = out_dir / "last.pt"

    if history_path.exists() and (best_ckpt.exists() or last_ckpt.exists()):
        print(f"Found existing run in {out_dir} — skipping training.")
        chosen_ckpt = best_ckpt if best_ckpt.exists() else last_ckpt
        train_info = {
            "best_checkpoint": str(chosen_ckpt),
            "last_checkpoint": str(last_ckpt) if last_ckpt.exists() else str(chosen_ckpt),
            "output_dir": str(out_dir),
            "run_id": run_id,
        }
    else:
        train_info = train_model(cfg)

    device = get_device(cfg.device)

    # Build test loader on ER with n=30
    test_ds = GraphMatrixDataset(DatasetConfig(mode="er", n=30, p=cfg.p, size=1000, seed=cfg.seed + 999))
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # Try to build per-epoch maxima from saved epoch checkpoints if available
    epoch_ckpts = sorted(out_dir.glob("epoch_*.pt"))
    epoch_values: list[tuple[int, int]] = []
    if epoch_ckpts:
        print("Building per-epoch capacity metrics from epoch checkpoints...")
        for ckpt in epoch_ckpts:
            epoch = extract_epoch_from_filename(ckpt)
            if epoch is None:
                continue
            loaded = load_checkpoint(ckpt, device)
            model = loaded.model
            max_rel = compute_max_reliable_length(model, test_loader, device)
            epoch_values.append((epoch, max_rel))
    else:
        # Fallback: evaluate best checkpoint only
        print("No epoch checkpoints found; evaluating best checkpoint only.")
        loaded = load_checkpoint(train_info["best_checkpoint"], device)
        model = loaded.model
        max_rel = compute_max_reliable_length(model, test_loader, device)
        epoch_values.append((cfg.epochs, max_rel))

    if not epoch_values:
        raise RuntimeError("No checkpoints found to evaluate.")

    epoch_values.sort()
    xs = [e for e, _ in epoch_values]
    ys = [v for _, v in epoch_values]

    out_png = out_dir / "capacity.png"
    plot_capacity(xs, ys, out_png)

    # copy into runs/capacity_test for quick inspection
    try:
        runs_dir = PROJECT_ROOT / "runs" / "capacity_test"
        runs_dir.mkdir(parents=True, exist_ok=True)
        if out_png.exists():
            shutil.copy2(out_png, runs_dir / "capacity.png")
    except Exception:
        pass

    summary = {"run_info": train_info, "epoch_capacity": [{"epoch": e, "max_reliable_path_length": v} for e, v in epoch_values]}
    save_json(out_dir / "capacity_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    main()
