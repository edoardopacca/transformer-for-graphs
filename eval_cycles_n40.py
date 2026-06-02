"""Exp 1 — test the 4 big ER(n=40) models on cycle graphs (n_active = 40, no
isolated nodes):

  * 1cycle : a single cycle C_40           (1 connected component, diameter 20)
  * 2cycle : two disjoint cycles 2 x C_20  (2 connected components, diam 10 each)

For each of the 4 training filters (unfiltered, D<=11, D<=9, D<=7) we report
exact-match accuracy, pairwise accuracy, and per-distance pairwise accuracy on a
10k-graph test set (random node permutations). Reuses the OOD evaluation
machinery; checkpoints are the same ones used by eval_2chains_pairwise_per_n_active.py.

Runs locally (CPU / MPS / CUDA). Output: runs/n40_cross_experiment/cycles_*.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import (
    add_self_loops,
    compute_connectivity_matrix,
    compute_all_pairs_shortest_paths,
    generate_one_cycle_graph,
    generate_two_cycles_graph,
)
from experiments2.ood_evaluation import evaluate, load_model, _per_dist_bar_panel

CHECKPOINTS = {
    "unfiltered": "runs/retrain_er_n40_big_495903/n40_p005_unfiltered_big/best.pt",
    "D<=11":      "runs/retrain_er_n40_big_494467/n40_p005_diam11_big/best.pt",
    "D<=9":       "runs/retrain_er_n40_big_495198/n40_p005_diam9_big/best.pt",
    "D<=7":       "runs/retrain_er_n40_big_495199/n40_p005_diam7_big/best.pt",
}
ORDER = ["unfiltered", "D<=11", "D<=9", "D<=7"]
KINDS = ["one_cycle", "two_cycles"]
KIND_TITLE = {"one_cycle": "1cycle (C_40)", "two_cycles": "2cycle (2 x C_20)"}

N = 40
N_TEST = 10_000
OUT_DIR = PROJECT_ROOT / "runs" / "n40_cross_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_cycle_test(kind: str, n: int, n_test: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    base = (generate_one_cycle_graph(n) if kind == "one_cycle"
            else generate_two_cycles_graph(n, n // 2))
    xs = np.empty((n_test, n, n), dtype=np.uint8)
    ys = np.empty((n_test, n, n), dtype=np.uint8)
    ds = np.empty((n_test, n, n), dtype=np.int16)
    for i in range(n_test):
        perm = rng.permutation(n)
        adj_no = base[np.ix_(perm, perm)]
        xs[i] = add_self_loops(adj_no).astype(np.uint8)
        ys[i] = compute_connectivity_matrix(adj_no).astype(np.uint8)
        ds[i] = compute_all_pairs_shortest_paths(adj_no).astype(np.int16)
    return {"x": xs, "y": ys, "d": ds}


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"device: {device}")

    # Build the two test sets ONCE; reuse across the 4 models.
    test_sets = {}
    for kind in KINDS:
        t0 = time.perf_counter()
        test_sets[kind] = build_cycle_test(kind, N, N_TEST, seed=777)
        print(f"generated {N_TEST} {kind} graphs in {time.perf_counter()-t0:.1f}s")

    results: dict = {}
    for name in ORDER:
        ckpt = str(PROJECT_ROOT / CHECKPOINTS[name])
        print(f"\n[{name}] loading {CHECKPOINTS[name]}")
        model, _cfg = load_model(ckpt, device)
        results[name] = {}
        for kind in KINDS:
            m = evaluate(model, test_sets[kind], device)
            results[name][kind] = {
                "exact_match": m["exact_match"],
                "pairwise_acc": m["pairwise_acc"],
                "per_dist_acc": m["per_dist_acc"],
                "dist_counts": m["dist_counts"],
            }
            print(f"  {KIND_TITLE[kind]:18s} exact={m['exact_match']:.4f}  "
                  f"pairwise={m['pairwise_acc']:.4f}")
        del model

    # ── Plot: per-distance bars, rows = kinds, cols = models ──
    fig, axes = plt.subplots(len(KINDS), len(ORDER),
                             figsize=(6 * len(ORDER), 5 * len(KINDS)), squeeze=False)
    for r, kind in enumerate(KINDS):
        for c, name in enumerate(ORDER):
            t = results[name][kind]
            _per_dist_bar_panel(
                axes[r][c], t,
                f"{name} | {KIND_TITLE[kind]}\nexact={t['exact_match']:.3f}, "
                f"pairwise={t['pairwise_acc']:.3f}")
    fig.suptitle("Exp 1: n=40 models on cycle graphs — per-distance pairwise accuracy",
                 fontsize=15, y=1.005)
    fig.tight_layout()
    out_png = OUT_DIR / "cycles_per_distance_big.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved figure: {out_png}")

    with open(OUT_DIR / "cycles_results_big.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved data:   {OUT_DIR / 'cycles_results_big.json'}")

    # ── Summary table ──
    print("\nSummary (exact / pairwise):")
    print(f"  {'model':>10} | {'1cycle exact':>12} {'1cycle pair':>11} | "
          f"{'2cycle exact':>12} {'2cycle pair':>11}")
    for name in ORDER:
        a = results[name]["one_cycle"]; b = results[name]["two_cycles"]
        print(f"  {name:>10} | {a['exact_match']:>12.4f} {a['pairwise_acc']:>11.4f} | "
              f"{b['exact_match']:>12.4f} {b['pairwise_acc']:>11.4f}")


if __name__ == "__main__":
    main()
