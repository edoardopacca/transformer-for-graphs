"""Generate two cross-experiment comparison plots for the n=40 ER study."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS = {
    "unfiltered": "/Users/edoardopaccagnella/transformer-for-graphs/runs/report2/retrain_er_n40_485350/n40_p005_unfiltered/history.json",
    "D≤11":       "/Users/edoardopaccagnella/transformer-for-graphs/runs/report2/retrain_er_n40_485352/n40_p005_diam11/history.json",
    "D≤9":        "/Users/edoardopaccagnella/transformer-for-graphs/runs/report2/retrain_er_n40_485351/n40_p005_diam9/history.json",
    "D≤7":        "/Users/edoardopaccagnella/transformer-for-graphs/runs/report2/retrain_er_n40_485353/n40_p005_diam7/history.json",
}

# Visual style: distinct, color-blind friendly, ordered from most-restrictive to least
COLORS = {
    "D≤7":        "#1f77b4",   # blue
    "D≤9":        "#2ca02c",   # green
    "D≤11":       "#ff7f0e",   # orange
    "unfiltered": "#d62728",   # red
}
ORDER = ["D≤7", "D≤9", "D≤11", "unfiltered"]

OUT_DIR = Path("/Users/edoardopaccagnella/transformer-for-graphs/runs/report2/n40_cross_experiment")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all() -> dict[str, dict]:
    return {name: json.load(open(path)) for name, path in RUNS.items()}


# ── Plot 1: Convergence (pairwise accuracy vs step, log x) ───────────────────
def plot_convergence(data: dict[str, dict]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: pairwise accuracy
    for name in ORDER:
        h = data[name]
        steps = np.array(h["steps"])
        pa = np.array(h["val_pairwise_acc"])
        ax1.plot(steps, pa, lw=2.0, color=COLORS[name], label=name, alpha=0.9)

    # Mark crossing of 0.99 with vertical dotted lines
    for name in ORDER:
        h = data[name]
        steps = h["steps"]; pa = h["val_pairwise_acc"]
        cross = next((s for s, p in zip(steps, pa) if p >= 0.99), None)
        if cross is not None:
            ax1.axvline(cross, color=COLORS[name], ls=":", alpha=0.5, lw=1)
            ax1.annotate(f"{cross//1000}k", (cross, 0.965),
                         color=COLORS[name], fontsize=9, ha="center",
                         bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                   ec=COLORS[name], alpha=0.9))

    ax1.axhline(0.99, color="gray", ls="--", lw=1, alpha=0.5)
    ax1.set_xscale("log")
    ax1.set_xlim(1000, 500000)
    ax1.set_ylim(0.94, 1.001)
    ax1.set_xlabel("Training step (log scale)", fontsize=12)
    ax1.set_ylabel("Pairwise accuracy", fontsize=12)
    ax1.set_title("Pairwise accuracy convergence", fontsize=13)
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(loc="lower right", fontsize=11, framealpha=0.95, title="Training filter")

    # Right: exact match
    for name in ORDER:
        h = data[name]
        steps = np.array(h["steps"])
        em = np.array(h["val_exact_match"])
        # smooth lightly with rolling max-window for readability
        ax2.plot(steps, em, lw=2.0, color=COLORS[name], label=name, alpha=0.9)

    ax2.set_xscale("log")
    ax2.set_xlim(1000, 500000)
    ax2.set_ylim(0.0, 0.9)
    ax2.set_xlabel("Training step (log scale)", fontsize=12)
    ax2.set_ylabel("Exact-match accuracy", fontsize=12)
    ax2.set_title("Exact-match (per-graph) convergence", fontsize=13)
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(loc="lower right", fontsize=11, framealpha=0.95, title="Training filter")

    fig.suptitle(
        "ER(n=40, p=0.05) — convergence by training-set diameter filter",
        fontsize=14, y=1.00,
    )
    fig.tight_layout()
    out = OUT_DIR / "01_convergence.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ── Plot 2: Per-distance pairwise accuracy at end of training ────────────────
def plot_per_distance(data: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))

    # Use the last (final) per-dist accuracy entry
    max_d = 0
    for name in ORDER:
        h = data[name]
        keys = h["val_per_dist_acc"][-1].keys()
        if keys:
            max_d = max(max_d, max(int(k) for k in keys))

    distances = list(range(1, max_d + 1))

    # Markers + lines
    markers = {"D≤7": "o", "D≤9": "s", "D≤11": "^", "unfiltered": "D"}

    for name in ORDER:
        h = data[name]
        per_dist = h["val_per_dist_acc"][-1]
        xs, ys = [], []
        for d in distances:
            v = per_dist.get(str(d))
            if v is not None:
                xs.append(d)
                ys.append(v)
        ax.plot(xs, ys,
                lw=2.0, marker=markers[name], markersize=7,
                color=COLORS[name], label=name, alpha=0.95,
                markeredgecolor="white", markeredgewidth=0.8)

    ax.axhline(0.99, color="gray", ls="--", lw=1, alpha=0.5)

    # Shade the regions corresponding to each filter range
    ax.axvspan(0.5, 7.5,  alpha=0.04, color=COLORS["D≤7"])
    ax.axvspan(7.5, 9.5,  alpha=0.04, color=COLORS["D≤9"])
    ax.axvspan(9.5, 11.5, alpha=0.04, color=COLORS["D≤11"])

    ax.set_xticks(distances)
    ax.set_xlim(0.5, max_d + 0.5)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Shortest-path distance d", fontsize=12)
    ax.set_ylabel("Pairwise accuracy at final step (500k)", fontsize=12)
    ax.set_title(
        "ER(n=40, p=0.05) — pairwise accuracy by distance, by training filter",
        fontsize=13,
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=11, framealpha=0.95, title="Training filter")

    # Annotation for boundary effect
    ax.annotate(
        "Boundary d=7:\nmore-restrictive\ntraining = lower acc",
        xy=(7, 0.94), xytext=(11, 0.7),
        fontsize=10, ha="left",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0, alpha=0.7),
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.9),
    )
    ax.annotate(
        "Unfiltered model collapses\non long-range pairs (d ≥ 13)",
        xy=(18, 0.45), xytext=(13, 0.15),
        fontsize=10, ha="left",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0, alpha=0.7),
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.9),
    )

    fig.tight_layout()
    out = OUT_DIR / "02_per_distance.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    data = load_all()
    plot_convergence(data)
    plot_per_distance(data)


if __name__ == "__main__":
    main()
