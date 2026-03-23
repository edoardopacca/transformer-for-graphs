from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import DatasetConfig, GraphMatrixDataset
from eval import evaluate_distance_conditioned_accuracy, load_checkpoint
from plots import plot_distance_accuracy
from train import TrainConfig, train_model
from utils import ensure_dir, get_device, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capacity test by shortest-path distance.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "baseline" / "best.pt"),
    )
    p.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "runs" / "capacity_test"))
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--p", type=float, default=0.08)
    p.add_argument("--eval_size", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--reliable_threshold", type=float, default=0.99)
    p.add_argument("--train_if_missing", action="store_true")
    return p.parse_args()


def maybe_train_baseline(checkpoint: Path) -> None:
    if checkpoint.exists():
        return
    cfg = TrainConfig(output_dir=str(PROJECT_ROOT / "runs" / "baseline"))
    train_model(cfg)


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if args.train_if_missing:
        maybe_train_baseline(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Run baseline first or pass --train_if_missing."
        )

    out_dir = ensure_dir(args.output_dir)
    device = get_device(args.device)
    loaded = load_checkpoint(checkpoint, device)

    ds = GraphMatrixDataset(
        DatasetConfig(mode="er", n=args.n, p=args.p, size=args.eval_size, seed=args.seed)
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    metrics = evaluate_distance_conditioned_accuracy(
        loaded.model,
        dl,
        device=device,
        reliable_threshold=args.reliable_threshold,
    )
    save_json(Path(out_dir) / "distance_metrics.json", metrics)
    plot_distance_accuracy(Path(out_dir) / "distance_metrics.json", Path(out_dir) / "distance_plot.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
