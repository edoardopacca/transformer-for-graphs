from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import DatasetConfig, GraphMatrixDataset
from eval import evaluate_model
from model import GraphConnectivityTransformer, ModelConfig
from utils import ensure_dir, get_device, save_json, set_seed


@dataclass
class TrainConfig:
    output_dir: str = "runs/default"
    n: int = 20
    p: float = 0.08
    train_size: int = 20000
    val_size: int = 2000
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-2
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    n_layers: int = 2
    dropout: float = 0.0
    train_mode: str = "er"
    val_mode: str = "er"
    max_diameter_train: int | None = None
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    use_cosine_scheduler: bool = True
    grad_clip_norm: float = 1.0
    threshold: float = 0.0


def _build_loader(
    *,
    mode: str,
    n: int,
    p: float,
    size: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    max_diameter: int | None = None,
) -> DataLoader:
    ds = GraphMatrixDataset(
        DatasetConfig(
            mode=mode,
            n=n,
            p=p,
            size=size,
            seed=seed,
            max_diameter=max_diameter,
            k=n // 2,
        )
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=(mode == "er"), num_workers=num_workers)


def train_model(config: TrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    out_dir = ensure_dir(config.output_dir)
    device = get_device(config.device)
    print(f"Using device: {device}")

    train_loader = _build_loader(
        mode=config.train_mode,
        n=config.n,
        p=config.p,
        size=config.train_size,
        seed=config.seed,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_diameter=config.max_diameter_train,
    )
    val_loader = _build_loader(
        mode=config.val_mode,
        n=config.n,
        p=config.p,
        size=config.val_size,
        seed=config.seed + 1,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_diameter=None,
    )

    model_cfg = ModelConfig(
        n=config.n,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        n_layers=config.n_layers,
        dropout=config.dropout,
    )
    model = GraphConnectivityTransformer(model_cfg).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = None
    if config.use_cosine_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_exact_match_acc": [],
        "val_pairwise_acc": [],
        "lr": [],
    }
    best_exact = -1.0
    best_ckpt = Path(out_dir) / "best.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            x = batch["adj"].to(device)
            y = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if config.grad_clip_norm and config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            bs = x.size(0)
            running += float(loss.item()) * bs
            seen += bs

        if scheduler is not None:
            scheduler.step()

        train_loss = running / max(seen, 1)
        val_metrics = evaluate_model(
            model, val_loader, device=device, split_name="val", threshold=config.threshold
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_exact_match_acc"].append(float(val_metrics["exact_match_acc"]))
        history["val_pairwise_acc"].append(float(val_metrics["pairwise_acc"]))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_exact={val_metrics['exact_match_acc']:.4f} "
            f"val_pairwise={val_metrics['pairwise_acc']:.4f}"
        )

        if val_metrics["exact_match_acc"] > best_exact:
            best_exact = float(val_metrics["exact_match_acc"])
            payload = {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(model_cfg),
                "train_config": asdict(config),
                "best_val_exact_match_acc": best_exact,
                "epoch": epoch,
            }
            torch.save(payload, best_ckpt)

    save_json(Path(out_dir) / "config.json", asdict(config))
    save_json(Path(out_dir) / "history.json", history)

    final_payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_cfg),
        "train_config": asdict(config),
        "best_val_exact_match_acc": best_exact,
        "epoch": config.epochs,
    }
    torch.save(final_payload, Path(out_dir) / "last.pt")

    return {
        "output_dir": str(out_dir),
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(Path(out_dir) / "last.pt"),
        "history": history,
        "best_val_exact_match_acc": best_exact,
    }


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train graph connectivity transformer.")
    p.add_argument("--output_dir", type=str, default="runs/default")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--p", type=float, default=0.08)
    p.add_argument("--train_size", type=int, default=20000)
    p.add_argument("--val_size", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--train_mode", type=str, default="er")
    p.add_argument("--val_mode", type=str, default="er")
    p.add_argument("--max_diameter_train", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--no_cosine_scheduler", action="store_true")
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.0)
    args = p.parse_args()
    return TrainConfig(
        output_dir=args.output_dir,
        n=args.n,
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
        dropout=args.dropout,
        train_mode=args.train_mode,
        val_mode=args.val_mode,
        max_diameter_train=args.max_diameter_train,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        use_cosine_scheduler=not args.no_cosine_scheduler,
        grad_clip_norm=args.grad_clip_norm,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_model(cfg)
