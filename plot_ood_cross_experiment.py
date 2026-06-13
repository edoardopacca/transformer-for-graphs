"""Two cross-experiment plots for the OOD analysis on n=40 models, plus
copies of the per-experiment OOD plots with unique filenames suitable for
uploading to Overleaf (which uses a flat namespace)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/edoardopaccagnella/transformer-for-graphs/runs/report2")
OOD_N14 = ROOT / "ood_eval_n14_493381"
OOD_N40 = ROOT / "ood_eval_n40_493382"
OUT_DIR = ROOT / "ood_cross_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_TAGS = ["unfiltered", "diam11", "diam9", "diam7"]
MODEL_LABELS = {"unfiltered": "unfiltered", "diam11": "D≤11", "diam9": "D≤9", "diam7": "D≤7"}
MODEL_COLOURS = {
    "diam7":      "#1f77b4",
    "diam9":      "#2ca02c",
    "diam11":     "#ff7f0e",
    "unfiltered": "#d62728",
}
PLOT_ORDER = ["diam7", "diam9", "diam11", "unfiltered"]


def _load(model_tag: str, family: str = "structured") -> dict:
    return json.load(open(OOD_N40 / f"er_n40_{model_tag}_to_{family}" / "results.json"))


# ── Cross-experiment plot 1: per-diameter-bucket on unfiltered ER ────────────
def plot_cross_per_diam_bucket() -> None:
    buckets = ["exact_le7", "exact_le9", "exact_le11", "exact_gt11"]
    bucket_labels = ["D≤7", "D≤9", "D≤11", "D>11"]
    bucket_n_keys = ["n_graphs_le7", "n_graphs_le9", "n_graphs_le11", "n_graphs_gt11"]

    data = {}
    counts = {}
    for tag in PLOT_ORDER:
        r = _load(tag)
        pdb = r["tests"]["unfiltered_er"]["per_diam_bucket"]
        data[tag] = [pdb.get(k, np.nan) for k in buckets]
        counts[tag] = [pdb.get(k, 0) for k in bucket_n_keys]

    n_groups = len(bucket_labels)
    n_models = len(PLOT_ORDER)
    width = 0.20
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, tag in enumerate(PLOT_ORDER):
        offsets = x + (i - (n_models - 1) / 2) * width
        bars = ax.bar(offsets, data[tag], width=width,
                      color=MODEL_COLOURS[tag], label=MODEL_LABELS[tag],
                      edgecolor="black", linewidth=0.4)
        for bar, v in zip(bars, data[tag]):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # Use counts from unfiltered (same per-test test set) for the n-graph annotation
    n_per_bucket = counts[PLOT_ORDER[-1]]
    xtick_labels = [f"{lab}\n(n = {n:,})" for lab, n in zip(bucket_labels, n_per_bucket)]
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Exact-match accuracy", fontsize=11)
    ax.set_xlabel("Test-graph diameter bucket", fontsize=11)
    ax.set_title(
        "OOD on unfiltered ER(n=40, p=0.05): exact match by graph diameter, "
        "across the four training filters",
        fontsize=12,
    )
    ax.legend(title="Training filter", loc="upper right",
              fontsize=10, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "ood_cross_per_diam_bucket.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ── Cross-experiment plot 2: per-distance on 2chains test ─────────────────────
def plot_cross_chains_per_distance() -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    max_d = 0
    series = {}
    for tag in PLOT_ORDER:
        r = _load(tag)
        per = r["tests"]["two_chains_var_n"]["per_dist_acc"]
        numeric_keys = sorted(int(k) for k in per if k != "disconnected")
        max_d = max(max_d, max(numeric_keys))
        series[tag] = (numeric_keys, [per[str(k)] for k in numeric_keys])

    markers = {"diam7": "o", "diam9": "s", "diam11": "^", "unfiltered": "D"}
    for tag in PLOT_ORDER:
        xs, ys = series[tag]
        ax.plot(xs, ys, lw=2.0, color=MODEL_COLOURS[tag],
                marker=markers[tag], markersize=7,
                markeredgecolor="white", markeredgewidth=0.8,
                label=MODEL_LABELS[tag], alpha=0.95)

    ax.axhline(0.99, color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_xticks(range(1, max_d + 1))
    ax.set_xlim(0.5, max_d + 0.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Shortest-path distance d (within-chain pairs)", fontsize=11)
    ax.set_ylabel("Pairwise accuracy", fontsize=11)
    ax.set_title(
        "OOD on 2-chains (variable n_active, padded to 40): "
        "pairwise accuracy by chain-internal distance",
        fontsize=12,
    )
    ax.legend(title="Training filter", loc="upper right",
              fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "ood_cross_chains_per_distance.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ── Rename and copy all per-experiment OOD PNGs to unique filenames ──────────
def collect_per_experiment_pngs() -> None:
    # n14 → ER
    pairs = [
        (OOD_N14 / "2chains_to_er"  / "per_distance_bars.png", "ood_2chains_per_distance_bars.png"),
        (OOD_N14 / "2cliques_to_er" / "per_distance_bars.png", "ood_2cliques_per_distance_bars.png"),
    ]
    # n40 → structured (chains/cliques) + unfiltered ER
    for tag in MODEL_TAGS:
        src_dir = OOD_N40 / f"er_n40_{tag}_to_structured"
        for src_name, prefix in [
            ("per_distance_bars.png",   "per_distance_bars"),
            ("per_n_active.png",        "per_n_active"),
            ("per_diam_bucket_ood.png", "per_diam_bucket"),
        ]:
            pairs.append((src_dir / src_name,
                          f"ood_n40_{tag}_{prefix}.png"))

    for src, dst_name in pairs:
        dst = OUT_DIR / dst_name
        if not src.exists():
            print(f"  MISSING: {src}")
            continue
        shutil.copy(src, dst)
        print(f"  copied {src.parent.name}/{src.name}  →  {dst.name}")


def main() -> None:
    plot_cross_per_diam_bucket()
    plot_cross_chains_per_distance()
    print("\nCopying per-experiment OOD PNGs with unique names…")
    collect_per_experiment_pngs()
    print(f"\nAll OOD plots gathered in: {OUT_DIR}")


if __name__ == "__main__":
    main()
