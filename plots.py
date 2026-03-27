from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from utils import load_json


def plot_training_history(history_path: str | Path, output_png: str | Path, *, title: str | None = None) -> None:
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

    # Add optional title carrying model / dataset info
    base_title = "Validation Exact Match and Pairwise Accuracy"
    if title:
        fig.suptitle(f"{title} — {base_title}")
    else:
        fig.suptitle(base_title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def plot_distance_accuracy(distance_metrics_json: str | Path, output_png: str | Path) -> None:
    metrics = load_json(distance_metrics_json)
    per_dist = metrics["per_distance_pairwise_accuracy"]
    xs = sorted(int(k) for k in per_dist.keys())
    y1 = [per_dist[str(k)] for k in xs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, y1, marker="o", label="Exact-distance pairwise accuracy")
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


def plot_restrict_diameter_sweep(results_json: str | Path, output_png: str | Path) -> None:
    data = load_json(results_json)
    # data expected: dict with keys being str(p) and values per diam mapping
    ps = sorted({float(k) for k in data.keys()})
    # collect diameters (they are stored as strings like 'None', '7', '9') and normalize
    diam_set = set()
    for p in ps:
        for diam in data[str(p)].keys():
            if diam in ("None", "NoneType", "null"):
                diam_set.add(None)
            else:
                try:
                    diam_set.add(int(diam))
                except Exception:
                    diam_set.add(diam)
    diameters = sorted(diam_set, key=lambda x: (x is not None, x if x is None else int(x)))

    # Build mapping diam -> list of accuracies for k=2 and k=3
    curves_k2 = {d: [] for d in diameters}
    curves_k3 = {d: [] for d in diameters}
    for p in ps:
        entry = data[str(p)]
        for d in diameters:
            key = "None" if d is None else str(d)
            row = entry.get(key)
            if row is None:
                curves_k2[d].append(float('nan'))
                curves_k3[d].append(float('nan'))
            else:
                # prefer two_chains and two_cliques exact match keys
                chains_key = next((k for k in row.keys() if k.startswith("two_chains") and k.endswith("_exact")), None)
                cliques_key = next((k for k in row.keys() if k.startswith("two_cliques") and k.endswith("_exact")), None)
                curves_k2[d].append(row.get(chains_key, float('nan')) if chains_key else float('nan'))
                curves_k3[d].append(row.get(cliques_key, float('nan')) if cliques_key else float('nan'))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = plt.get_cmap('tab10')
    for i, d in enumerate(diameters):
        label = "unrestricted" if d is None else f"diam <= {d}"
        axes[0].plot(ps, curves_k2[d], marker='o', label=label, color=colors(i))
        axes[1].plot(ps, curves_k3[d], marker='o', label=label, color=colors(i))

    axes[0].set_xlabel('ER edge probability p')
    axes[0].set_ylabel('Exact match accuracy (TwoChains k=2)')
    axes[1].set_xlabel('ER edge probability p')
    axes[1].set_ylabel('Exact match accuracy (TwoChains k=3)')
    axes[0].set_title('Diameter restriction sweep: k=2')
    axes[1].set_title('Diameter restriction sweep: k=3')
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(loc='lower right')
    axes[1].legend(loc='lower right')
    fig.suptitle('Effect of training diameter restriction on OOD TwoChains exact-match')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def plot_restrict_diameter_dynamics(aggregated_json: str | Path, output_png: str | Path) -> None:
    data = load_json(aggregated_json)
    # data expected format: {diam: {"steps": [...], "two_chains": [...], "two_cliques": [...]}, ...}
    # normalize diam keys
    def parse_diam(k: str):
        if k in ("None", "NoneType", "null"):
            return None
        try:
            return int(k)
        except Exception:
            return k

    diam_keys = sorted(list(data.keys()), key=lambda x: (parse_diam(x) is not None, parse_diam(x) if parse_diam(x) is None else int(parse_diam(x))))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = plt.get_cmap('tab10')
    for i, dkey in enumerate(diam_keys):
        row = data[dkey]
        steps = row.get("steps", [])
        chains = row.get("two_chains", [])
        cliques = row.get("two_cliques", [])
        d = parse_diam(dkey)
        label = "unrestricted" if d is None else f"diam<= {d}"
        axes[0].plot(steps, chains, label=label, color=colors(i))
        axes[1].plot(steps, cliques, label=label, color=colors(i))

    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel('Exact match accuracy (TwoChains k=2)')
    axes[1].set_xlabel('Training step')
    axes[1].set_ylabel('Exact match accuracy (TwoChains k=3)')
    axes[0].set_title('Training dynamics (k=2)')
    axes[1].set_title('Training dynamics (k=3)')
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(loc='lower right')
    axes[1].legend(loc='lower right')
    fig.suptitle('OOD exact-match accuracy over training under diameter restrictions')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def plot_restrict_diameter_dynamics_pairwise(aggregated_json: str | Path, output_png: str | Path) -> None:
    """Plot pairwise OOD accuracy dynamics for different training diameter restrictions.

    Expected aggregated_json format:
      {"None": {"epochs": [1,..,T], "two_chains_pairwise": [...], "two_cliques_pairwise": [...]},
       "7": {...}, "9": {...}, "11": {...}}
    """
    data = load_json(aggregated_json)

    def parse_diam(k: str):
        if k in ("None", "NoneType", "null"):
            return None
        try:
            return int(k)
        except Exception:
            return k

    diam_keys = sorted(list(data.keys()), key=lambda x: (parse_diam(x) is not None, parse_diam(x) if parse_diam(x) is None else int(parse_diam(x))))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = plt.get_cmap('tab10')

    for i, dkey in enumerate(diam_keys):
        row = data[dkey]
        epochs = row.get("epochs")
        chains = row.get("two_chains_pairwise")
        cliques = row.get("two_cliques_pairwise")
        if epochs is None or chains is None or cliques is None:
            # skip incomplete rows
            continue
        d = parse_diam(dkey)
        label = "Unrestricted" if d is None else f"Train diam <= {d}"
        axes[0].plot(epochs, chains, marker='o', label=label, color=colors(i))
        axes[1].plot(epochs, cliques, marker='o', label=label, color=colors(i))

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Pairwise accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Pairwise accuracy')
    axes[0].set_title('TwoChains OOD pairwise accuracy')
    axes[1].set_title('TwoCliques OOD pairwise accuracy')
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(loc='lower right')
    axes[1].legend(loc='lower right')
    fig.suptitle('OOD pairwise accuracy over training under diameter restrictions')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def plot_curriculum_diameter_dynamics(aggregated_json: str | Path, output_png: str | Path) -> None:
    data = load_json(aggregated_json)
    # data expected: {mode: {"global_step": [...], "two_chains_exact": [...], "two_cliques_exact": [...]}, ...}
    modes = list(data.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = plt.get_cmap('tab10')
    labels_map = {
        "standard": "Standard",
        "restrict_only": "Restrict only (diam<=9)",
        "curriculum_1": "Curriculum 1 (<=9 -> unrestricted)",
        "curriculum_2": "Curriculum 2 (<=7 -> <=9)",
    }
    for i, m in enumerate(modes):
        row = data[m]
        if not row or "global_step" not in row:
            continue
        steps = row.get("global_step", [])
        ch = row.get("two_chains_exact", [])
        cl = row.get("two_cliques_exact", [])
        label = labels_map.get(m, m)
        axes[0].plot(steps, ch, label=label, color=colors(i))
        axes[1].plot(steps, cl, label=label, color=colors(i))

    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel('Exact match accuracy (TwoChains k=10)')
    axes[1].set_xlabel('Training step')
    axes[1].set_ylabel('Exact match accuracy (TwoCliques k=10)')
    axes[0].set_title('Curriculum: TwoChains (OOD)')
    axes[1].set_title('Curriculum: TwoCliques (OOD)')
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(loc='lower right')
    axes[1].legend(loc='lower right')
    fig.suptitle('Curriculum diameter dynamics: OOD exact-match over training')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
