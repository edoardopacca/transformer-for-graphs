from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any
import argparse

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
from utils import (
    get_device,
    save_json,
    ensure_dir,
    canonical_run_id,
    get_training_dir,
)


def extract_epoch_from_filename(p: Path) -> int | None:
    m = re.search(r"epoch_(\d+)\.pt$", p.name)
    if m:
        return int(m.group(1))
    return None


def compute_max_reliable_length(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> int:
    metrics = evaluate_distance_conditioned_accuracy(model, loader, device, reliable_threshold=0.99)
    val = metrics.get("max_reliable_path_length_exact")
    return int(val) if (val is not None) else 0


def plot_capacity(xs: list[int], ys: list[int], out_png: Path, *, threshold: float = 0.99, L: int = 2) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    label_curve = f"Max reliable path length (accuracy >= {threshold})"
    ax.plot(xs, ys, marker="o", color="#ff7f0e", linewidth=2, label=label_curve)

    if len(xs) > 1:
        # multiple epochs: linear x-axis labeled as Epoch
        ax.set_xlabel("Epoch")
        # ensure xs are sorted
        ax.plot(xs, ys, marker="o", color="#ff7f0e", linewidth=2)
    else:
        # single point: no log scale and no fill
        ax.set_xlabel("Epoch")

    if len(xs) > 1:
        try:
            ax.fill_between(xs, ys, color="#ff7f0e", alpha=0.12)
        except Exception:
            pass

    ax.set_ylabel("Maximum Reliable Path Length")
    ax.set_ylim(0, max(ys + [1]) + 1)
    ax.grid(True, alpha=0.25)
    # theoretical capacity line using L from config
    yline = 3 ** L
    ax.hlines(yline, min(xs), max(xs), colors="#1f77b4", linestyles="--", label=f"Theoretical capacity: 3^{L}")
    ax.legend(loc="lower right")
    fig.suptitle("Maximum reliable path length across training")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_recompute", action="store_true", help="Force recompute even if outputs exist")
    parser.add_argument("--eval_n", type=int, default=None, help="Override evaluation graph size (must match model n)")
    parser.add_argument("--eval_size", type=int, default=1000)
    parser.add_argument("--reliable_threshold", type=float, default=0.99)
    args = parser.parse_args()

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

    # Resolve canonical run id and training directory
    run_id = canonical_run_id(cfg)
    out_dir = get_training_dir(cfg, PROJECT_ROOT)
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

    # Print resolved paths and checkpoint presence (HPC-friendly logging)
    print(f"Canonical run id: {run_id}")
    print(f"Resolved training directory: {out_dir.resolve()}" )
    print(f"history.json exists: {history_path.exists()}")
    print(f"best.pt exists: {best_ckpt.exists()}")

    # find epoch checkpoints
    epoch_ckpts = sorted(out_dir.glob("epoch_*.pt"))
    print(f"Found {len(epoch_ckpts)} epoch_*.pt checkpoints in {out_dir}")

    # Decision: if outputs exist and not forcing recompute, exit
    out_png = out_dir / "capacity.png"
    summary_json = out_dir / "capacity_summary.json"
    if out_png.exists() and summary_json.exists() and not args.force_recompute:
        print(f"Found existing outputs: {out_png} and {summary_json}; use --force_recompute to overwrite.")
        return

    # Evaluate checkpoints: prefer epoch_*.pt if present
    epoch_values: list[tuple[int, int]] = []
    if epoch_ckpts:
        print("Building per-epoch capacity metrics from epoch checkpoints...")
        # derive eval n from first checkpoint's model config
        first_loaded = load_checkpoint(epoch_ckpts[0], device)
        model_n = first_loaded.model_config.n
        if args.eval_n is not None and args.eval_n != model_n:
            print(f"Warning: --eval_n {args.eval_n} does not match model n {model_n}; using model n={model_n} for evaluation.")
        eval_n = model_n
        test_ds = GraphMatrixDataset(DatasetConfig(mode="er", n=eval_n, p=cfg.p, size=args.eval_size, seed=cfg.seed + 999))
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
        for ckpt in epoch_ckpts:
            epoch = extract_epoch_from_filename(ckpt)
            if epoch is None:
                continue
            loaded = load_checkpoint(ckpt, device)
            max_rel = compute_max_reliable_length(loaded.model, test_loader, device)
            epoch_values.append((epoch, max_rel))
    else:
        # No epoch ckpts: fallback to best.pt if present
        if best_ckpt.exists():
            print("No epoch checkpoints found; evaluating best checkpoint only.")
            loaded = load_checkpoint(best_ckpt, device)
            model_n = loaded.model_config.n
            if args.eval_n is not None and args.eval_n != model_n:
                print(f"Warning: --eval_n {args.eval_n} does not match model n {model_n}; using model n={model_n} for evaluation.")
            eval_n = model_n
            test_ds = GraphMatrixDataset(DatasetConfig(mode="er", n=eval_n, p=cfg.p, size=args.eval_size, seed=cfg.seed + 999))
            test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
            max_rel = compute_max_reliable_length(loaded.model, test_loader, device)
            epoch_values.append((cfg.epochs, max_rel))
        else:
            raise RuntimeError(f"No epoch_* checkpoints found and no best.pt at expected path: {best_ckpt}")

    if not epoch_values:
        raise RuntimeError("No checkpoints found to evaluate.")

    epoch_values.sort()
    xs = [e for e, _ in epoch_values]
    ys = [v for _, v in epoch_values]

    out_png = out_dir / "capacity.png"
    # When possible, obtain model config L for theoretical capacity line
    # prefer best or first epoch loaded above
    loaded_for_meta = None
    if epoch_ckpts:
        loaded_for_meta = load_checkpoint(epoch_ckpts[0], device)
    elif best_ckpt.exists():
        loaded_for_meta = load_checkpoint(best_ckpt, device)
    L = getattr(loaded_for_meta.model_config, "n_layers", 2) if loaded_for_meta is not None else 2

    plot_capacity(xs, ys, out_png, threshold=args.reliable_threshold, L=L)
    print(f"Saved capacity plot to: {out_png.resolve()}")

    # copy into runs/capacity_test for quick inspection
    runs_dir = PROJECT_ROOT / "runs" / "capacity_test"
    try:
        ensure_dir(runs_dir)
        shutil.copy2(out_png, runs_dir / "capacity.png")
        print(f"Copied capacity plot to: {runs_dir / 'capacity.png'}")
    except Exception as e:
        print(f"Error copying capacity plot to {runs_dir}: {e}")

    summary = {
        "run_info": train_info,
        "epoch_capacity": [{"epoch": e, "max_reliable_path_length": v} for e, v in epoch_values],
        "threshold": args.reliable_threshold,
        "theoretical_capacity": 3 ** L,
    }
    save_json(out_dir / "capacity_summary.json", summary)
    print(f"Saved capacity summary to: {(out_dir / 'capacity_summary.json').resolve()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    main()
