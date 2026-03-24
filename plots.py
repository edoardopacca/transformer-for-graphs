from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from utils import load_json


def plot_training_history(history_path: str | Path, output_png: str | Path) -> None:
    history = load_json(history_path)
    df = pd.DataFrame(history)

    # Determine number of epochs from available lists
    num_epochs = 0
    if "train_loss" in df:
        num_epochs = len(df["train_loss"])  # train_loss recorded per epoch
    elif "val_exact_match_acc" in df:
        num_epochs = len(df["val_exact_match_acc"])
    epochs = list(range(1, max(1, num_epochs) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    # ER Validation (per-epoch curves)
    er_ax = axes[0]
    if "val_exact_match_acc" in df:
        er_ax.plot(epochs[: len(df["val_exact_match_acc"])], df["val_exact_match_acc"],
                   marker="o", label="Val Exact Match", color="#1f77b4")
    if "val_pairwise_acc" in df:
        er_ax.plot(epochs[: len(df["val_pairwise_acc"])], df["val_pairwise_acc"],
                   marker="s", label="Val Pairwise", color="#ff7f0e")
    er_ax.set_title("ER Validation")
    er_ax.set_xlabel("Epoch")
    er_ax.set_ylim(0.0, 1.02)
    er_ax.grid(True, alpha=0.25)
    er_ax.legend(loc="lower right")

    # Two Chains (prefer per-epoch OOD in history, fallback to final ood_metrics.json)
    chains_ax = axes[1]
    if "ood_two_chains_exact" in df and "ood_two_chains_pairwise" in df:
        chains_ax.plot(epochs[: len(df["ood_two_chains_exact"])], df["ood_two_chains_exact"],
                       marker="o", label="Val Exact Match", color="#1f77b4")
        chains_ax.plot(epochs[: len(df["ood_two_chains_pairwise"])], df["ood_two_chains_pairwise"],
                       marker="s", label="Val Pairwise", color="#ff7f0e")
    else:
        # fallback: draw final values from ood_metrics.json if available
        ood_path = Path(history_path).parent / "ood_metrics.json"
        try:
            if ood_path.exists():
                ood = load_json(ood_path)
                if "two_chains" in ood:
                    chains = ood["two_chains"]
                    if "exact_match_acc" in chains:
                        chains_ax.hlines(float(chains["exact_match_acc"]), epochs[0], epochs[-1], colors="#1f77b4", linestyles="-", label="Val Exact Match")
                    if "pairwise_acc" in chains:
                        chains_ax.hlines(float(chains["pairwise_acc"]), epochs[0], epochs[-1], colors="#ff7f0e", linestyles="--", label="Val Pairwise")
        except Exception:
            pass
    chains_ax.set_title("Two Chains (OOD)")
    chains_ax.set_xlabel("Epoch")
    chains_ax.set_ylim(0.0, 1.02)
    chains_ax.grid(True, alpha=0.25)
    chains_ax.legend(loc="lower right")

    # Two Cliques
    cliques_ax = axes[2]
    if "ood_two_cliques_exact" in df and "ood_two_cliques_pairwise" in df:
        cliques_ax.plot(epochs[: len(df["ood_two_cliques_exact"])], df["ood_two_cliques_exact"],
                        marker="o", label="Val Exact Match", color="#1f77b4")
        cliques_ax.plot(epochs[: len(df["ood_two_cliques_pairwise"])], df["ood_two_cliques_pairwise"],
                        marker="s", label="Val Pairwise", color="#ff7f0e")
    else:
        ood_path = Path(history_path).parent / "ood_metrics.json"
        try:
            if ood_path.exists():
                ood = load_json(ood_path)
                if "two_cliques" in ood:
                    cliques = ood["two_cliques"]
                    if "exact_match_acc" in cliques:
                        cliques_ax.hlines(float(cliques["exact_match_acc"]), epochs[0], epochs[-1], colors="#1f77b4", linestyles="-", label="Val Exact Match")
                    if "pairwise_acc" in cliques:
                        cliques_ax.hlines(float(cliques["pairwise_acc"]), epochs[0], epochs[-1], colors="#ff7f0e", linestyles="--", label="Val Pairwise")
        except Exception:
            pass
    cliques_ax.set_title("Two Cliques (OOD)")
    cliques_ax.set_xlabel("Epoch")
    cliques_ax.set_ylim(0.0, 1.02)
    cliques_ax.grid(True, alpha=0.25)
    cliques_ax.legend(loc="lower right")

    fig.suptitle("Validation Exact Match and Pairwise Accuracy")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
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
