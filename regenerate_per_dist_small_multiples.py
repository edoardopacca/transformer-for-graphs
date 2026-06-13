"""
Standalone script to regenerate the per-distance small-multiples plots
for all retrain experiments, with data-driven y-axis per panel.

Reads history.json from each run directory (does NOT retrain anything),
recomputes the small-multiples plot with auto-adjusted y-limits, and
overwrites the existing PNG.

Handles both history schemas in this repo:
  - older runs (er, 2chains, 2cliques): keys 'val_per_distance_pairwise_acc'
    and 'test_set_distance_counts'
  - newer runs (er_n40):                keys 'val_per_dist_acc'
    and 'test_dist_counts'
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/edoardopaccagnella/transformer-for-graphs/runs/report2")

# (history_path, output_png_name, title)
TARGETS: List[Tuple[Path, str, str]] = [
    (ROOT / "retrain_er_3352759/n10_p0.2_rep0/history.json",
     "per_distance_accuracy_small_multiples_p02.png",
     "Retrain ER (n=10, p=0.2): Pairwise Accuracy by Shortest-Path Distance"),
    (ROOT / "retrain_er_3352759/n10_p0.5_rep0/history.json",
     "per_distance_accuracy_small_multiples_p05.png",
     "Retrain ER (n=10, p=0.5): Pairwise Accuracy by Shortest-Path Distance"),
    (ROOT / "retrain_2chains_3352759/n14_k7_rep0/history_k07.json",
     "per_distance_accuracy_small_multiples_k07.png",
     "Retrain TwoChains (n=14, k=7): Pairwise Accuracy by Shortest-Path Distance"),
    (ROOT / "retrain_2cliques_3352759/n14_k7_rep0/history_k07.json",
     "per_distance_accuracy_small_multiples_k07.png",
     "Retrain TwoCliques (n=14, k=7): Pairwise Accuracy by Shortest-Path Distance"),
    (ROOT / "retrain_er_n40_485350/n40_p005_unfiltered/history.json",
     "er_n40_unfiltered_per_dist_sm.png",
     "ER(n=40, p=0.05) unfiltered: Pairwise Accuracy by Shortest-Path Distance"),
    (ROOT / "retrain_er_n40_485351/n40_p005_diam9/history.json",
     "er_n40_diam9_per_dist_sm.png",
     "ER(n=40, p=0.05) D≤9: Pairwise Accuracy by Shortest-Path Distance"),
    (ROOT / "retrain_er_n40_485352/n40_p005_diam11/history.json",
     "er_n40_diam11_per_dist_sm.png",
     "ER(n=40, p=0.05) D≤11: Pairwise Accuracy by Shortest-Path Distance"),
    (ROOT / "retrain_er_n40_485353/n40_p005_diam7/history.json",
     "er_n40_diam7_per_dist_sm.png",
     "ER(n=40, p=0.05) D≤7: Pairwise Accuracy by Shortest-Path Distance"),
]


def _get_per_dist_and_counts(h: Dict[str, Any]) -> Tuple[List[Dict[str, float]], Dict[str, int]]:
    """Return (per_dist_history, dist_counts) regardless of schema."""
    if "val_per_dist_acc" in h:
        return h["val_per_dist_acc"], h.get("test_dist_counts", {})
    if "val_per_distance_pairwise_acc" in h:
        return h["val_per_distance_pairwise_acc"], h.get("test_set_distance_counts", {})
    raise KeyError("History file is missing per-distance keys")


def _compute_ylim(values: List[float]) -> Tuple[float, float]:
    """Data-driven y-axis range with a small padding."""
    vals = [v for v in values if v is not None]
    if not vals:
        return (0.0, 1.005)
    vmin = min(vals)
    vmax = max(vals)
    if vmin >= 0.99:
        return (0.985, 1.001)        # very tight zoom for near-perfect panels
    pad = max(0.005, (vmax - vmin) * 0.10)
    lo = max(0.0, vmin - pad)
    hi = min(1.005, vmax + pad / 2)
    return (lo, hi)


def plot_small_multiples(history_path: Path, out_name: str, title: str) -> None:
    if not history_path.exists():
        print(f"  SKIP (missing): {history_path}")
        return

    with open(history_path) as f:
        h = json.load(f)

    per_dist_hist, counts_raw = _get_per_dist_and_counts(h)
    steps = h["steps"]

    # Build the ordered list of panels. We display numeric distances first
    # (sorted ascending), then a "disconnected" panel if the schema has one.
    numeric_counts: Dict[int, int] = {}
    disconnected_count: int = 0
    has_disconnected = False
    for k, v in counts_raw.items():
        if k == "disconnected":
            has_disconnected = True
            disconnected_count = int(v)
        else:
            try:
                numeric_counts[int(k)] = int(v)
            except ValueError:
                continue

    numeric_distances = sorted(d for d, c in numeric_counts.items()
                               if d >= 1 and c > 0
                               and any(str(d) in entry for entry in per_dist_hist))

    panels: List[Tuple[str, int, str]] = []      # (key_in_history, count, label)
    for d in numeric_distances:
        panels.append((str(d), numeric_counts[d], f"d = {d}"))
    if (has_disconnected and disconnected_count > 0
            and any("disconnected" in entry for entry in per_dist_hist)):
        panels.append(("disconnected", disconnected_count, "disconnected"))

    if not panels:
        print(f"  SKIP (no per-distance data): {history_path}")
        return

    n_panels = len(panels)
    ncols = 3 if n_panels >= 6 else 2
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.5 * ncols, 2.7 * nrows),
        sharex=True,
    )
    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    else:
        axes_flat = np.array(axes).flatten().tolist()

    for ax, (key, cnt, label) in zip(axes_flat, panels):
        vals = [entry.get(key) for entry in per_dist_hist]
        xs, ys = zip(*[(s, v) for s, v in zip(steps, vals) if v is not None]) \
                 if any(v is not None for v in vals) else ([], [])
        color = "#d62728" if key == "disconnected" else "#1f77b4"
        ax.plot(xs, ys, lw=1.0, color=color)
        ax.set_title(f"{label}  (n = {cnt:,})", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(*_compute_ylim(vals))
        ax.set_xlabel("Training step")
        ax.set_ylabel("Pairwise accuracy")

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = history_path.parent / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    for hpath, out_name, title in TARGETS:
        print(f"Processing {hpath.parent.name}/{out_name}")
        plot_small_multiples(hpath, out_name, title)


if __name__ == "__main__":
    main()
