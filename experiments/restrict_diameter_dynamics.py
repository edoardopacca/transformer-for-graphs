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


def prepare_extra_loaders(n: int, batch_size: int, eval_size: int, max_attempts: int) -> dict[str, DataLoader]:
    loaders = {}
    k = n // 2
    ds_ch = GraphMatrixDataset(
        DatasetConfig(mode="two_chains", n=n, k=k, size=eval_size, seed=12345, max_attempts=max_attempts)
    )
    ds_cl = GraphMatrixDataset(
        DatasetConfig(mode="two_cliques", n=n, k=k, size=eval_size, seed=12346, max_attempts=max_attempts)
    )
    loaders[f"two_chains_k{k}"] = DataLoader(ds_ch, batch_size=batch_size, shuffle=False)
    loaders[f"two_cliques_k{k}"] = DataLoader(ds_cl, batch_size=batch_size, shuffle=False)
    return loaders


def run_dynamics(cfg: TrainConfig, eval_every_steps: int, eval_size: int, force: bool) -> dict[str, Any]:
    run_id = canonical_run_id(cfg)
    out_dir = get_training_dir(cfg, PROJECT_ROOT)
    cfg.output_dir = str(out_dir)
    history_path = out_dir / "history.json"
    best_ckpt = out_dir / "best.pt"

    extra_loaders = prepare_extra_loaders(cfg.n, cfg.batch_size, eval_size, cfg.max_attempts)

    if history_path.exists() and best_ckpt.exists() and not force:
        print(f"Reusing existing run {out_dir}")
    else:
        print(f"Training dynamics run for {run_id} -> {out_dir}")
        # set eval_every_steps in config and call train_model with extra loaders
        cfg.eval_every_steps = eval_every_steps
        cfg.save_every_steps = max(0, eval_every_steps * 5)
        try:
            train_model(cfg, extra_eval_loaders=extra_loaders)
        except Exception as e:
            raise RuntimeError(
                f"Training failed for p={cfg.p}, max_diameter_train={cfg.max_diameter_train}, train_size={cfg.train_size}: {e}"
            )

    if not history_path.exists():
        raise RuntimeError(f"Expected history.json in {out_dir}")

    history = json.loads(history_path.read_text(encoding="utf-8"))
    steps = history.get("global_step", [])
    k = cfg.n // 2
    key_ch = f"extra_two_chains_k{k}_exact"
    key_cl = f"extra_two_cliques_k{k}_exact"
    ch_vals = history.get(key_ch, [])
    cl_vals = history.get(key_cl, [])
    return {"steps": steps, "two_chains": ch_vals, "two_cliques": cl_vals}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--p", type=float, default=0.08)
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--val_size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_every_steps", type=int, default=200)
    parser.add_argument("--eval_size", type=int, default=1000)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_attempts", type=int, default=200000)
    args = parser.parse_args()

    diameters = [None, 7, 9, 11]
    aggregated = {}
    out_runs = PROJECT_ROOT / "runs" / "restrict_diameter_dynamics"
    ensure_dir(out_runs)

    for diam in diameters:
        cfg = TrainConfig(
            n=20,
            p=args.p,
            train_size=args.train_size,
            val_size=args.val_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            n_layers=args.n_layers,
            dropout=0.0,
            train_mode="er",
            val_mode="er",
            max_diameter_train=diam,
            seed=42,
            device="auto",
            num_workers=0,
            max_attempts=args.max_attempts,
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
