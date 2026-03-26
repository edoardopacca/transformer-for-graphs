from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval import load_checkpoint
from train import TrainConfig, train_model
from data import DatasetConfig, GraphMatrixDataset
from utils import get_device, save_json, ensure_dir, canonical_run_id, get_training_dir


def find_checkpoint_for_step(out_dir: Path, target_step: int = 10000) -> Path | None:
    # prefer explicit epoch_{N}.pt files if present, otherwise best.pt / last.pt
    epoch_ckpts = []
    for p in out_dir.glob("epoch_*.pt"):
        m = re.search(r"epoch_(\d+)\.pt$", p.name)
        if m:
            epoch_ckpts.append((int(m.group(1)), p))
    if epoch_ckpts:
        # choose epoch with number closest to target_step
        epoch_ckpts.sort()
        closest = min(epoch_ckpts, key=lambda t: abs(t[0] - target_step))
        return closest[1]
    # fallback
    best = out_dir / "best.pt"
    last = out_dir / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    return None


def project_to_IJ(W: np.ndarray) -> tuple[np.ndarray, float]:
    # Project matrix W onto space span{I, J} where J is all-ones
    n = W.shape[0]
    I = np.eye(n)
    J = np.ones((n, n))
    # solve least squares for coefficients a, b minimizing ||W - a I - b J||_F
    A = np.stack([I.ravel(), J.ravel()], axis=1)  # (n*n, 2)
    b = W.ravel()
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, bb = coeffs
    W_proj = a * I + bb * J
    rel_err = np.linalg.norm(W - W_proj) / max(np.linalg.norm(W), 1e-12)
    return W_proj, float(rel_err)


def plot_weights(W: np.ndarray, W_proj: np.ndarray, rel_err: float, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vmin = min(W.min(), W_proj.min())
    vmax = max(W.max(), W_proj.max())
    im0 = axes[0].imshow(W, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[0].set_title("Original W")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(W_proj, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Projected W (rel err={rel_err:.3f})")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle("Weight Matrix and Projection to a*I + b*J")
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

    # look for checkpoint near step 10000 (epoch number heuristic)
    ckpt = find_checkpoint_for_step(out_dir, target_step=10000)
    if ckpt is None:
        raise RuntimeError("No checkpoint found to evaluate.")
    print(f"Using checkpoint: {ckpt}")
    loaded = load_checkpoint(ckpt, device)
    model = loaded.model

    # Build effective W as read_out.weight @ read_in.weight (shape n x n)
    try:
        W_t = model.read_out.weight.detach().cpu().numpy() @ model.read_in.weight.detach().cpu().numpy()
    except Exception as e:
        raise RuntimeError("Could not form read_out @ read_in matrix from model") from e

    W_proj, rel_err = project_to_IJ(W_t)

    out_png = out_dir / "dicotomy_weights.png"
    plot_weights(W_t, W_proj, rel_err, out_png)

    # copy into runs/dicotomy_weights for quick inspection
    try:
        runs_dir = PROJECT_ROOT / "runs" / "dicotomy_weights"
        runs_dir.mkdir(parents=True, exist_ok=True)
        if out_png.exists():
            shutil.copy2(out_png, runs_dir / "dicotomy_weights.png")
    except Exception:
        pass

    summary = {"checkpoint": str(ckpt), "relative_projection_error": rel_err}
    save_json(out_dir / "dicotomy_weights_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    main()
