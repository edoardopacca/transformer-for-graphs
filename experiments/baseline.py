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
from utils import get_device, save_json, load_json, canonical_run_id, get_training_dir
import shutil


def main() -> None:
    # Build TrainConfig here (caller selects hyperparameters so run is fully identified)
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

    # Canonical run id + training directory
    run_id = canonical_run_id(cfg)
    out_dir = get_training_dir(cfg, PROJECT_ROOT)
    cfg.output_dir = str(out_dir)

    # If a previous run exists with saved history and checkpoints under trainings/<run_id>, skip retraining.
    # If you want to retrain from scratch, delete the folder `trainings/<run_id>` (see note below).
    history_path = out_dir / "history.json"
    best_ckpt = out_dir / "best.pt"
    last_ckpt = out_dir / "last.pt"

    if history_path.exists() and (best_ckpt.exists() or last_ckpt.exists()):
        print(f"Found existing run in {out_dir} — skipping training.")
        # Prefer the best checkpoint if present.
        chosen_ckpt = best_ckpt if best_ckpt.exists() else last_ckpt
        train_info = {
            "best_checkpoint": str(chosen_ckpt),
            "last_checkpoint": str(last_ckpt) if last_ckpt.exists() else str(chosen_ckpt),
            "output_dir": str(out_dir),
            "run_id": run_id,
        }
    else:
        # No existing run: perform training (this will also save per-epoch checkpoints and OOD history).
        # NOTE: If you previously changed training hyperparameters, consider deleting `trainings/<run_id>` first.
        train_info = train_model(cfg)

    # Create a descriptive title for figures including model and dataset configuration
    fig_title = f"Transformer d_model={cfg.d_model}, layers={cfg.n_layers}, heads={cfg.n_heads}, dataset={cfg.train_mode}"
    # Plot training history (uses per-epoch OOD if available, otherwise falls back to final OOD values)
    plot_training_history(out_dir / "history.json", out_dir / "training_history.png", title=fig_title)

    # Copy human-facing plots into the legacy `runs/baseline` folder so quick viewers
    # can find PNGs there while `trainings/<run_id>` keeps the full dataset/checkpoints.
    try:
        runs_dir = PROJECT_ROOT / "runs" / "baseline"
        runs_dir.mkdir(parents=True, exist_ok=True)
        src_hist = out_dir / "training_history.png"
        if src_hist.exists():
            shutil.copy2(src_hist, runs_dir / "training_history.png")
    except Exception:
        pass

    device = get_device(cfg.device)
    loaded = load_checkpoint(train_info["best_checkpoint"], device)
    model = loaded.model

    val_ds = GraphMatrixDataset(
        DatasetConfig(mode="er", n=cfg.n, p=cfg.p, size=cfg.val_size, seed=cfg.seed + 999)
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    val_metrics = evaluate_model(model, val_loader, device, "er_val")

    # If history contains per-epoch OOD we can use it; otherwise compute final OOD metrics now.
    ood_metrics = None
    if history_path.exists():
        try:
            history = load_json(history_path)
            # If per-epoch OOD keys are present, we already have OOD evolution recorded.
            if not (
                "ood_two_chains_exact" in history
                and "ood_two_cliques_exact" in history
            ):
                # Try to build per-epoch OOD metrics from saved epoch checkpoints if available.
                epoch_ckpts = sorted(out_dir.glob("epoch_*.pt"))
                if epoch_ckpts:
                    print("Building per-epoch OOD metrics from epoch checkpoints...")
                    chains_exact = []
                    chains_pair = []
                    cliques_exact = []
                    cliques_pair = []
                    for ckpt in epoch_ckpts:
                        ld = load_checkpoint(ckpt, device)
                        m = ld.model
                        ood = evaluate_ood_suites(
                            model=m,
                            n=cfg.n,
                            k=cfg.n // 2,
                            size=1000,
                            batch_size=cfg.batch_size,
                            device=device,
                        )
                        chains = ood.get("two_chains", {})
                        cliques = ood.get("two_cliques", {})
                        chains_exact.append(float(chains.get("exact_match_acc", float("nan"))))
                        chains_pair.append(float(chains.get("pairwise_acc", float("nan"))))
                        cliques_exact.append(float(cliques.get("exact_match_acc", float("nan"))))
                        cliques_pair.append(float(cliques.get("pairwise_acc", float("nan"))))
                    history["ood_two_chains_exact"] = chains_exact
                    history["ood_two_chains_pairwise"] = chains_pair
                    history["ood_two_cliques_exact"] = cliques_exact
                    history["ood_two_cliques_pairwise"] = cliques_pair
                    save_json(history_path, history)
        except Exception:
            pass

    # If no per-epoch OOD recorded, compute final OOD metrics now and save them.
    if not (out_dir / "ood_metrics.json").exists():
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
    if ood_metrics is not None:
        save_json(out_dir / "ood_metrics.json", ood_metrics)
    save_json(out_dir / "distance_metrics.json", dist_metrics)
    plot_distance_accuracy(out_dir / "distance_metrics.json", out_dir / "distance_accuracy.png")

    # Also copy distance plot into `runs/baseline` for quick inspection
    try:
        runs_dir = PROJECT_ROOT / "runs" / "baseline"
        runs_dir.mkdir(parents=True, exist_ok=True)
        src_dist = out_dir / "distance_accuracy.png"
        if src_dist.exists():
            shutil.copy2(src_dist, runs_dir / "distance_accuracy.png")
    except Exception:
        pass

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
