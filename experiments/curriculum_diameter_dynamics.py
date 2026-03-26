from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import itertools
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import ModelConfig, GraphConnectivityTransformer
from data import DatasetConfig, GraphMatrixDataset
from eval import evaluate_model
from utils import canonical_run_id, get_training_dir, get_device, save_json, ensure_dir, debug_torch_device_info
from plots import plot_curriculum_diameter_dynamics
from train import TrainConfig
import math
import re
from utils import load_json, is_compatible_train_config


def make_ood_loaders(n: int, batch_size: int, eval_size: int, max_attempts: int):
    k = n // 2
    ds_ch = GraphMatrixDataset(DatasetConfig(mode="two_chains", n=n, k=k, size=eval_size, seed=1234, max_attempts=max_attempts))
    ds_cl = GraphMatrixDataset(DatasetConfig(mode="two_cliques", n=n, k=k, size=eval_size, seed=5678, max_attempts=max_attempts))
    return (DataLoader(ds_ch, batch_size=batch_size, shuffle=False), DataLoader(ds_cl, batch_size=batch_size, shuffle=False))


def train_mode_run(mode: str, cfg: TrainConfig, total_steps: int, eval_every_steps: int, t1: int, eval_size: int, force: bool) -> dict[str, Any]:
    # build explicit training directory that encodes this experiment and mode
    base_dir = get_training_dir(cfg, PROJECT_ROOT)
    out_dir = ensure_dir(base_dir / f"curriculum_{mode}_T1{t1}_S{total_steps}_seed{cfg.seed}")
    print(f"Mode={mode} -> output dir: {out_dir}")

    history_path = out_dir / "history.json"
    last_ckpt = out_dir / "last.pt"
    best_ckpt = out_dir / "best.pt"

    # Exact-match reuse for this mode's dedicated dir
    if history_path.exists() and not force:
        print(f"Reusing existing history for mode {mode} at {out_dir}")
        return json.loads(history_path.read_text(encoding="utf-8"))

    # For 'standard' and 'restrict_only' modes, attempt to reuse compatible baseline runs
    if mode in {"standard", "restrict_only"} and not force:
        # build compatible baseline config
        baseline_cfg = TrainConfig(
            n=cfg.n,
            p=cfg.p,
            train_size=cfg.train_size,
            val_size=cfg.val_size,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            train_mode="er",
            val_mode="er",
            max_diameter_train=(9 if mode == "restrict_only" else None),
            seed=cfg.seed,
            device=cfg.device,
            num_workers=cfg.num_workers,
            max_attempts=cfg.max_attempts,
        )
        base_dir = get_training_dir(baseline_cfg, PROJECT_ROOT)
        print(f"Mode {mode} will try to reuse baseline run at {base_dir}")

        # 1) Try full history reuse
        base_history = base_dir / "history.json"
        key_ch = f"extra_two_chains_k{cfg.n//2}_exact"
        key_cl = f"extra_two_cliques_k{cfg.n//2}_exact"
        if base_history.exists():
            try:
                bh = load_json(base_history)
                if ("global_step" in bh) and (key_ch in bh) and (key_cl in bh):
                    print(f"Mode {mode} reused baseline dynamics from {base_history}")
                    return {"global_step": bh["global_step"], "two_chains_exact": bh[key_ch], "two_cliques_exact": bh[key_cl]}
            except Exception:
                print(f"Warning: could not read base history at {base_history}")

        # 2) Try post-hoc file
        posthoc = base_dir / "posthoc_dynamics.json"
        if posthoc.exists():
            try:
                pj = load_json(posthoc)
                if pj.get("source") == "posthoc_from_epoch_checkpoints":
                    print(f"Mode {mode} reused post-hoc dynamics from {posthoc}")
                    return {"global_step": pj.get("global_step", []), "two_chains_exact": pj.get("two_chains_exact", []), "two_cliques_exact": pj.get("two_cliques_exact", [])}
            except Exception:
                print(f"Warning: could not read {posthoc}")

        # 3) Try reconstructing from epoch_*.pt
        epoch_files = []
        if base_dir.exists():
            for pth in base_dir.glob("epoch_*.pt"):
                m = re.search(r"epoch_(\d+)\.pt$", pth.name)
                if m:
                    epoch_files.append((int(m.group(1)), pth))
            epoch_files.sort()
        if epoch_files:
            print(f"Mode {mode} reconstructing dynamics from epoch checkpoints at {base_dir}")
            steps = []
            ch_vals: list[float] = []
            cl_vals: list[float] = []
            steps_per_epoch = math.ceil(cfg.train_size / cfg.batch_size) if cfg.batch_size > 0 else 1
            device = get_device(cfg.device)
            for (epoch_idx, path) in epoch_files:
                payload = load_checkpoint(path, device)
                model = payload.model
                ch_loader, cl_loader = make_ood_loaders(cfg.n, cfg.batch_size, eval_size, cfg.max_attempts)
                m1 = evaluate_model(model, ch_loader, device, split_name=f"two_chains_epoch{epoch_idx}")
                m2 = evaluate_model(model, cl_loader, device, split_name=f"two_cliques_epoch{epoch_idx}")
                ch_vals.append(m1["exact_match_acc"])
                cl_vals.append(m2["exact_match_acc"])
                steps.append(epoch_idx * steps_per_epoch)

            post = {
                "global_step": steps,
                "two_chains_exact": ch_vals,
                "two_cliques_exact": cl_vals,
                "source": "posthoc_from_epoch_checkpoints",
                "checkpoint_dir": str(base_dir),
            }
            # Save into the baseline dir so future runs can reuse
            save_json(base_dir / "posthoc_dynamics.json", post)
            print(f"Saved reconstructed post-hoc dynamics to {base_dir / 'posthoc_dynamics.json'}")
            return post

        print(f"Mode {mode} did not find compatible baseline at {base_dir}, will train fresh in {out_dir}")

    device = get_device(cfg.device)
    print(f"Using device for mode {mode}: {device}")
    model_cfg = ModelConfig(n=cfg.n, d_model=cfg.d_model, n_heads=cfg.n_heads, d_ff=cfg.d_ff, n_layers=cfg.n_layers, dropout=cfg.dropout)
    model = GraphConnectivityTransformer(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    ood_ch_loader, ood_cl_loader = make_ood_loaders(cfg.n, cfg.batch_size, eval_size, cfg.max_attempts)

    history = {"global_step": [], "two_chains_exact": [], "two_cliques_exact": []}
    global_step = 0
    best_metric = -1.0

    steps_per_chunk = eval_every_steps
    # dataset size per chunk: ensure enough samples to produce steps_per_chunk updates
    dataset_size = max(steps_per_chunk * cfg.batch_size, cfg.batch_size)

    while global_step < total_steps:
        # determine current phase's max_diameter
        if mode == "standard":
            phase_max = None
        elif mode == "restrict_only":
            phase_max = 9
        elif mode == "curriculum_1":
            phase_max = 9 if global_step < t1 else None
        elif mode == "curriculum_2":
            phase_max = 7 if global_step < t1 else 9
        else:
            raise ValueError(f"Unknown mode: {mode}")

        print(f"Mode={mode} global_step={global_step} phase_max_diameter={phase_max}")

        # create training dataset for this chunk
        seed_offset = cfg.seed + global_step
        train_ds = GraphMatrixDataset(DatasetConfig(mode="er", n=cfg.n, p=cfg.p, size=dataset_size, seed=seed_offset, max_diameter=phase_max, max_attempts=cfg.max_attempts))
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        it = iter(train_loader)

        steps_this_chunk = min(steps_per_chunk, total_steps - global_step)
        done = 0
        while done < steps_this_chunk:
            try:
                batch = next(it)
            except StopIteration:
                # restart iterator if exhausted
                it = iter(train_loader)
                batch = next(it)

            x = batch["adj"].to(device)
            y = batch["target"].to(device)
            model.train()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            global_step += 1
            done += 1

        # end of chunk: evaluate
        model.eval()
        ch_metrics = evaluate_model(model, ood_ch_loader, device, split_name="two_chains")
        cl_metrics = evaluate_model(model, ood_cl_loader, device, split_name="two_cliques")
        ch_exact = float(ch_metrics.get("exact_match_acc", float("nan")))
        cl_exact = float(cl_metrics.get("exact_match_acc", float("nan")))
        history["global_step"].append(global_step)
        history["two_chains_exact"].append(ch_exact)
        history["two_cliques_exact"].append(cl_exact)

        # save last checkpoint and history
        payload = {
            "model_state_dict": model.state_dict(),
            "model_config": model_cfg.__dict__,
            "train_config": cfg.__dict__,
            "global_step": global_step,
        }
        torch.save(payload, last_ckpt)
        save_json(history_path, history)
        print(f"Mode={mode} eval at step={global_step}: two_chains_exact={ch_exact:.4f} two_cliques_exact={cl_exact:.4f}")

        avg_metric = (ch_exact + cl_exact) / 2.0
        if avg_metric > best_metric:
            best_metric = avg_metric
            torch.save(payload, best_ckpt)

    # final save
    save_json(history_path, history)
    print(f"Finished mode={mode}, saved history to {history_path}, last checkpoint to {last_ckpt}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--p", type=float, default=0.08)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--total_steps", type=int, default=5000)
    parser.add_argument("--eval_every_steps", type=int, default=200)
    parser.add_argument("--t1", type=int, default=2000)
    parser.add_argument("--eval_size", type=int, default=1000)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_attempts", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # quick debug print
    debug_torch_device_info()

    modes = ["standard", "restrict_only", "curriculum_1", "curriculum_2"]
    aggregated: dict[str, Any] = {}
    out_runs = PROJECT_ROOT / "runs" / "curriculum_diameter_dynamics"
    ensure_dir(out_runs)

    for mode in modes:
        cfg = TrainConfig(
            output_dir="",
            n=args.n,
            p=args.p,
            train_size=0,
            val_size=args.eval_size,
            batch_size=args.batch_size,
            epochs=0,
            lr=args.lr,
            weight_decay=args.weight_decay,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            n_layers=args.n_layers,
            dropout=0.0,
            train_mode="er",
            val_mode="er",
            max_diameter_train=None,
            seed=args.seed,
            device=args.device,
            num_workers=0,
            max_attempts=args.max_attempts,
        )
        try:
            hist = train_mode_run(mode, cfg, total_steps=args.total_steps, eval_every_steps=args.eval_every_steps, t1=args.t1, eval_size=args.eval_size, force=args.force_recompute)
        except Exception as e:
            print(f"Mode {mode} failed: {e}")
            hist = {"error": str(e)}
        aggregated[mode] = hist

    agg_json = out_runs / "curriculum_diameter_dynamics_aggregated.json"
    save_json(agg_json, aggregated)
    png = out_runs / "curriculum_diameter_dynamics.png"
    plot_curriculum_diameter_dynamics(agg_json, png)
    print(f"Saved aggregated results to {agg_json} and plot to {png}")


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    main()
