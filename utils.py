from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pathlib import Path


def canonical_run_id(config: Any) -> str:
    """Create a canonical run id string from a training config object or mapping.

    The function accepts objects with attributes or dictionaries containing keys.
    Fields encoded: n, p, d_model, n_layers, n_heads, d_ff, batch_size,
    lr, weight_decay, dropout, train_mode, val_mode, epochs, seed,
    max_diameter_train (encoded as diam<val> or diamNone).
    """
    def _get(k: str, default=None):
        if isinstance(config, dict):
            return config.get(k, default)
        return getattr(config, k, default)

    n = int(_get("n", 20))
    p = float(_get("p", 0.08))
    d_model = int(_get("d_model", 128))
    n_layers = int(_get("n_layers", 2))
    n_heads = int(_get("n_heads", 4))
    d_ff = int(_get("d_ff", 256))
    batch_size = int(_get("batch_size", 64))
    lr = float(_get("lr", 1e-3))
    weight_decay = float(_get("weight_decay", 1e-2))
    dropout = float(_get("dropout", 0.0))
    train_mode = str(_get("train_mode", "er"))
    val_mode = str(_get("val_mode", "er"))
    epochs = int(_get("epochs", 20))
    seed = int(_get("seed", 42))
    max_diam = _get("max_diameter_train", _get("max_diameter", None))
    diam_part = f"diam{max_diam}" if max_diam is not None else "diamNone"

    # compact float formatting
    def fmtf(x: float) -> str:
        return f"{x:.3g}" if x != int(x) else f"{int(x)}"

    parts = [
        f"n{n}",
        f"p{fmtf(p)}",
        f"d{d_model}",
        f"layers{n_layers}",
        f"heads{n_heads}",
        f"dff{d_ff}",
        f"bs{batch_size}",
        f"lr{fmtf(lr)}",
        f"wd{fmtf(weight_decay)}",
        f"drop{fmtf(dropout)}",
        f"mode{train_mode}",
        f"val{val_mode}",
        f"ep{epochs}",
        f"seed{seed}",
        diam_part,
    ]
    return "_".join(parts)


def get_training_dir(config: Any, project_root: str | Path) -> Path:
    pr = Path(project_root)
    rid = canonical_run_id(config)
    return pr / "trainings" / rid


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(device_str: str | None = None) -> torch.device:
    if device_str is None or device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
