from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import DatasetConfig, GraphMatrixDataset
from model import GraphConnectivityTransformer, ModelConfig


def threshold_predictions(logits: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    return (logits > threshold).to(torch.int64)


def exact_match_accuracy(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.0) -> float:
    pred = threshold_predictions(logits, threshold)
    tgt = target.to(torch.int64)
    exact = (pred == tgt).reshape(pred.shape[0], -1).all(dim=1).float().mean()
    return float(exact.item())


def pairwise_accuracy(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.0) -> float:
    pred = threshold_predictions(logits, threshold)
    tgt = target.to(torch.int64)
    return float((pred == tgt).float().mean().item())


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str,
    threshold: float = 0.0,
) -> dict[str, Any]:
    bce = nn.BCEWithLogitsLoss()
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    total_exact = 0.0
    total_pairwise_correct = 0.0
    total_entries = 0
    for batch in dataloader:
        x = batch["adj"].to(device)
        y = batch["target"].to(device)
        logits = model(x)
        loss = bce(logits, y)
        bs = x.size(0)
        total_graphs += bs
        total_loss += float(loss.item()) * bs
        pred = threshold_predictions(logits, threshold)
        tgt = y.to(torch.int64)
        total_exact += float((pred == tgt).reshape(bs, -1).all(dim=1).float().sum().item())
        total_pairwise_correct += float((pred == tgt).sum().item())
        total_entries += int(tgt.numel())
    return {
        "split": split_name,
        "loss": total_loss / max(total_graphs, 1),
        "exact_match_acc": total_exact / max(total_graphs, 1),
        "pairwise_acc": total_pairwise_correct / max(total_entries, 1),
        "num_graphs": total_graphs,
    }


@torch.no_grad()
def evaluate_distance_conditioned_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.0,
    reliable_threshold: float = 0.99,
) -> dict[str, Any]:
    model.eval()
    correct_per_dist: dict[int, int] = defaultdict(int)
    count_per_dist: dict[int, int] = defaultdict(int)
    disc_correct = 0
    disc_count = 0
    for batch in dataloader:
        x = batch["adj"].to(device)
        y = batch["target"].to(device).to(torch.int64)
        d = batch["dist"].to(device).to(torch.int64)
        logits = model(x)
        pred = threshold_predictions(logits, threshold)
        eq = (pred == y)

        finite = d >= 0
        disconnected = d < 0
        if disconnected.any():
            disc_correct += int(eq[disconnected].sum().item())
            disc_count += int(disconnected.sum().item())

        # Exclude diagonal for path-length analysis.
        n = d.size(-1)
        eye = torch.eye(n, dtype=torch.bool, device=device).unsqueeze(0).expand_as(d)
        finite_offdiag = finite & (~eye)
        max_d = int(d[finite_offdiag].max().item()) if finite_offdiag.any() else 0
        for dist_val in range(1, max_d + 1):
            mask = finite_offdiag & (d == dist_val)
            if mask.any():
                count_per_dist[dist_val] += int(mask.sum().item())
                correct_per_dist[dist_val] += int(eq[mask].sum().item())
    connected_correct = sum(correct_per_dist.values())
    connected_count = sum(count_per_dist.values())
    connected_pair_accuracy = (
        connected_correct / connected_count if connected_count > 0 else None
    )
    per_dist_acc = {
        str(k): (correct_per_dist[k] / count_per_dist[k]) for k in sorted(count_per_dist.keys())
    }
    per_dist_count = {str(k): count_per_dist[k] for k in sorted(count_per_dist.keys())}

    cumulative_acc: dict[str, float] = {}
    running_correct = 0
    running_count = 0
    max_rel_exact = 0
    max_rel_cum = 0
    for k in sorted(count_per_dist.keys()):
        acc = per_dist_acc[str(k)]
        if acc >= reliable_threshold:
            max_rel_exact = k
        running_correct += correct_per_dist[k]
        running_count += count_per_dist[k]
        cum = running_correct / max(running_count, 1)
        cumulative_acc[str(k)] = cum
        if cum >= reliable_threshold:
            max_rel_cum = k

    return {
        "per_distance_pairwise_accuracy": per_dist_acc,
        "per_distance_count": per_dist_count,
        "cumulative_accuracy_leq_distance": cumulative_acc,
        "disconnected_pair_accuracy": (disc_correct / disc_count) if disc_count > 0 else None,
        "max_reliable_path_length_exact": max_rel_exact,
        "max_reliable_path_length_cumulative": max_rel_cum,
        "reliable_threshold": reliable_threshold,
        "note": "Per-distance and reliable-length metrics exclude diagonal (distance 0).",
        "connected_pair_accuracy": connected_pair_accuracy,
        "connected_pair_count": connected_count,
    }


def evaluate_ood_suites(
    model: nn.Module,
    n: int,
    k: int,
    size: int,
    batch_size: int,
    device: torch.device,
    threshold: float = 0.0,
) -> dict[str, Any]:
    two_chains = GraphMatrixDataset(
        DatasetConfig(mode="two_chains", n=n, k=k, size=size, seed=1234)
    )
    two_cliques = GraphMatrixDataset(
        DatasetConfig(mode="two_cliques", n=n, k=k, size=size, seed=5678)
    )
    dl_chains = DataLoader(two_chains, batch_size=batch_size, shuffle=False)
    dl_cliques = DataLoader(two_cliques, batch_size=batch_size, shuffle=False)
    return {
        "two_chains": evaluate_model(model, dl_chains, device, "two_chains", threshold),
        "two_cliques": evaluate_model(model, dl_cliques, device, "two_cliques", threshold),
    }


@dataclass
class LoadedCheckpoint:
    model: GraphConnectivityTransformer
    model_config: ModelConfig
    train_config: dict[str, Any]


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> LoadedCheckpoint:
    payload = torch.load(checkpoint_path, map_location=device)
    model_cfg = ModelConfig(**payload["model_config"])
    model = GraphConnectivityTransformer(model_cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return LoadedCheckpoint(
        model=model,
        model_config=model_cfg,
        train_config=payload.get("train_config", {}),
    )
