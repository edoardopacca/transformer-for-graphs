from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import TrainConfig, train_model
from data import DatasetConfig, GraphMatrixDataset
from eval import load_checkpoint, evaluate_model
from utils import canonical_run_id, get_training_dir, get_device, save_json, ensure_dir
from plots import plot_restrict_diameter_sweep


def run_point(cfg: TrainConfig, force: bool) -> dict[str, Any]:
    run_id = canonical_run_id(cfg)
    out_dir = get_training_dir(cfg, PROJECT_ROOT)
    cfg.output_dir = str(out_dir)
    history_path = out_dir / "history.json"
    best_ckpt = out_dir / "best.pt"

    if history_path.exists() and (best_ckpt.exists()) and not force:
        print(f"Reusing existing run {out_dir}")
    else:
        print(f"Training run for {run_id} -> {out_dir}")
        train_model(cfg)

    if not best_ckpt.exists():
        raise RuntimeError(f"Expected checkpoint not found at {best_ckpt}")

    device = get_device(cfg.device)
    loaded = load_checkpoint(best_ckpt, device)
    model = loaded.model

    results = {}
    for k in (2, 3):
        ds = GraphMatrixDataset(DatasetConfig(mode="two_chains", n=cfg.n, k=k, size=2000, seed=cfg.seed + 100))
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
        m = evaluate_model(model, dl, device, split_name=f"two_chains_k{k}")
        results[f"two_chains_k{k}_exact"] = m["exact_match_acc"]
        results[f"two_chains_k{k}_pairwise"] = m["pairwise_acc"]

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--p_min", type=float, default=0.05)
    parser.add_argument("--p_max", type=float, default=0.55)
    parser.add_argument("--p_steps", type=int, default=11)
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--val_size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    ps = [round(args.p_min + i * (args.p_max - args.p_min) / (args.p_steps - 1), 4) for i in range(args.p_steps)]
    diameters = [None, 2, 3, 4]

    aggregated: dict[str, Any] = {}
    out_runs = PROJECT_ROOT / "runs" / "restrict_diameter_sweep"
    ensure_dir(out_runs)

    for p in ps:
        aggregated[str(p)] = {}
        for diam in diameters:
            cfg = TrainConfig(
                n=8,
                p=p,
                train_size=args.train_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=1e-3,
                weight_decay=1e-2,
                d_model=128,
                n_heads=1,
                d_ff=256,
                n_layers=1,
                dropout=0.0,
                train_mode="er",
                val_mode="er",
                max_diameter_train=diam,
                seed=42,
                device="auto",
                num_workers=0,
            )
            try:
                res = run_point(cfg, force=args.force_recompute)
            except Exception as e:
                print(f"Point p={p} diam={diam} failed: {e}")
                res = {"error": str(e)}
            aggregated[str(p)][str(diam)] = res

    results_json = out_runs / "restrict_diameter_sweep_results.json"
    save_json(results_json, aggregated)
    png = out_runs / "restrict_diameter_sweep.png"
    plot_restrict_diameter_sweep(results_json, png)
    print(f"Saved sweep plot to {png}")


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    main()
