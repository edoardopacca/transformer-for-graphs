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
from plots import plot_restrict_diameter_dynamics


def prepare_extra_loaders(n: int, batch_size: int, eval_size: int) -> dict[str, DataLoader]:
    loaders = {}
    for k in (2, 3):
        ds = GraphMatrixDataset(DatasetConfig(mode="two_chains", n=n, k=k, size=eval_size, seed=12345))
        loaders[f"two_chains_k{k}"] = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loaders


def run_dynamics(cfg: TrainConfig, eval_every_steps: int, eval_size: int, force: bool) -> dict[str, Any]:
    run_id = canonical_run_id(cfg)
    out_dir = get_training_dir(cfg, PROJECT_ROOT)
    cfg.output_dir = str(out_dir)
    history_path = out_dir / "history.json"
    best_ckpt = out_dir / "best.pt"

    extra_loaders = prepare_extra_loaders(cfg.n, cfg.batch_size, eval_size)

    if history_path.exists() and best_ckpt.exists() and not force:
        print(f"Reusing existing run {out_dir}")
    else:
        print(f"Training dynamics run for {run_id} -> {out_dir}")
        # set eval_every_steps in config and call train_model with extra loaders
        cfg.eval_every_steps = eval_every_steps
        cfg.save_every_steps = max(0, eval_every_steps * 5)
        train_model(cfg, extra_eval_loaders=extra_loaders)

    if not history_path.exists():
        raise RuntimeError(f"Expected history.json in {out_dir}")

    history = json.loads(history_path.read_text(encoding="utf-8"))
    steps = history.get("global_step", [])
    k2 = history.get("extra_two_chains_k2_exact", [])
    k3 = history.get("extra_two_chains_k3_exact", [])

    return {"steps": steps, "k2": k2, "k3": k3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--p", type=float, default=0.2)
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_every_steps", type=int, default=200)
    parser.add_argument("--eval_size", type=int, default=1000)
    args = parser.parse_args()

    diameters = [None, 2, 3, 4]
    aggregated = {}
    out_runs = PROJECT_ROOT / "runs" / "restrict_diameter_dynamics"
    ensure_dir(out_runs)

    for diam in diameters:
        cfg = TrainConfig(
            n=8,
            p=args.p,
            train_size=args.train_size,
            val_size=2000,
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
            res = run_dynamics(cfg, eval_every_steps=args.eval_every_steps, eval_size=args.eval_size, force=args.force_recompute)
        except Exception as e:
            print(f"Dynamics run for diam={diam} failed: {e}")
            res = {"error": str(e)}
        aggregated[str(diam)] = res

    agg_json = out_runs / "restrict_diameter_dynamics_aggregated.json"
    save_json(agg_json, aggregated)
    png = out_runs / "restrict_diameter_dynamics.png"
    plot_restrict_diameter_dynamics(agg_json, png)
    print(f"Saved dynamics plot to {png}")


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    main()
