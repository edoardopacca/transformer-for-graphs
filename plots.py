from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from utils import load_json


def plot_training_history(history_path: str | Path, output_png: str | Path) -> None:
    history = load_json(history_path)
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if "train_loss" in df and "val_loss" in df:
        axes[0].plot(df["train_loss"], label="train_loss")
        axes[0].plot(df["val_loss"], label="val_loss")
        axes[0].set_title("Loss")
        axes[0].legend()
    if "val_exact_match_acc" in df and "val_pairwise_acc" in df:
        axes[1].plot(df["val_exact_match_acc"], label="val_exact")
        axes[1].plot(df["val_pairwise_acc"], label="val_pairwise")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("Validation Accuracy")
        axes[1].legend()
    fig.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def plot_distance_accuracy(distance_metrics_json: str | Path, output_png: str | Path) -> None:
    metrics = load_json(distance_metrics_json)
    per_dist = metrics["per_distance_pairwise_accuracy"]
    cumulative = metrics["cumulative_accuracy_leq_distance"]
    xs = sorted(int(k) for k in per_dist.keys())
    y1 = [per_dist[str(k)] for k in xs]
    y2 = [cumulative.get(str(k), 0.0) for k in xs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, y1, marker="o", label="exact distance accuracy")
    ax.plot(xs, y2, marker="s", label="cumulative <= d accuracy")
    ax.set_xlabel("Shortest-path distance d")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def plot_compare_ood(
    results_a: dict[str, Any],
    results_b: dict[str, Any],
    output_png: str | Path,
) -> None:
    labels = ["two_chains", "two_cliques"]
    a_vals = [results_a[label]["exact_match_acc"] for label in labels]
    b_vals = [results_b[label]["exact_match_acc"] for label in labels]
    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width / 2 for i in x], a_vals, width=width, label="baseline")
    ax.bar([i + width / 2 for i in x], b_vals, width=width, label="restricted")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Exact Match Accuracy")
    ax.set_title("OOD Exact Match Comparison")
    ax.legend()
    fig.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
