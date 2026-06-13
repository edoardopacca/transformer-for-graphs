"""Per-n_active pairwise accuracy on the 2chains OOD test set for the big ER models.

Re-evaluates each of the 4 checkpoints (unfiltered, D<=11, D<=9, D<=7) on the
same 10k-graph 2chains test set used by experiments2/ood_evaluation.py, and
produces a single figure plotting pairwise accuracy vs n_active.

Two variants of pairwise accuracy are computed and saved:
  - "full" : computed over the full 40x40 padded canvas (matches the convention
            used in the saved results.json).
  - "active": computed only over the n_active x n_active block (the actual
              graph entries), which is the meaningful quantity per n_active.
The figure plots the "active" variant by default.
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

from experiments2.ood_evaluation import (
    N_TEST,
    N_VALUES_VARIABLE,
    generate_structured_padded_test,
    load_model,
)


CHECKPOINTS = {
    "unfiltered": "runs/report2/retrain_er_n40_big_495903/n40_p005_unfiltered_big/best.pt",
    "D<=11":      "runs/report2/retrain_er_n40_big_494467/n40_p005_diam11_big/best.pt",
    "D<=9":       "runs/report2/retrain_er_n40_big_495198/n40_p005_diam9_big/best.pt",
    "D<=7":       "runs/report2/retrain_er_n40_big_495199/n40_p005_diam7_big/best.pt",
}

COLORS = {
    "D<=7":       "#1f77b4",
    "D<=9":       "#2ca02c",
    "D<=11":      "#ff7f0e",
    "unfiltered": "#d62728",
}
ORDER = ["D<=7", "D<=9", "D<=11", "unfiltered"]

OUT_DIR = PROJECT_ROOT / "runs" / "report3" / "n40_cross_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def predict_all(model, x_np: np.ndarray, device: torch.device,
                batch_size: int = 256) -> np.ndarray:
    n_graphs = x_np.shape[0]
    n = x_np.shape[1]
    out = np.empty((n_graphs, n, n), dtype=np.int8)
    for s in range(0, n_graphs, batch_size):
        e = min(s + batch_size, n_graphs)
        xb = torch.from_numpy(x_np[s:e]).float().to(device)
        logits = model(xb)
        out[s:e] = (logits > 0).cpu().numpy().astype(np.int8)
    return out


def per_n_active_pairwise(pred: np.ndarray, target: np.ndarray,
                           n_actives: np.ndarray) -> dict:
    """Return per-n_active dicts: 'full' (40x40) and 'active' (n_active block)."""
    full: dict[int, dict] = {}
    active: dict[int, dict] = {}
    eq_full = (pred == target)
    for n_a in N_VALUES_VARIABLE:
        mask = n_actives == n_a
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        # full canvas
        full[int(n_a)] = {
            "n_graphs": cnt,
            "pairwise_acc": float(eq_full[mask].mean()),
        }
        # active block only (top-left n_a x n_a of each padded canvas)
        sub_pred = pred[mask][:, :n_a, :n_a]
        sub_tgt  = target[mask][:, :n_a, :n_a]
        active[int(n_a)] = {
            "n_graphs": cnt,
            "pairwise_acc": float((sub_pred == sub_tgt).mean()),
        }
    return {"full": full, "active": active}


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}")

    # Generate the 2chains test set ONCE; reuse for all 4 models.
    t0 = time.perf_counter()
    ds = generate_structured_padded_test(
        kind="chains", n_padded=40, n_values=N_VALUES_VARIABLE,
        n_test=N_TEST, num_workers=4,
    )
    print(f"generated {N_TEST} chains test graphs in {time.perf_counter()-t0:.1f}s")

    x = ds["x"]; y = ds["y"]; n_actives = ds["n_active"]
    target = y.astype(np.int8)

    results: dict[str, dict] = {}
    for name in ORDER:
        ckpt_rel = CHECKPOINTS[name]
        ckpt = str(PROJECT_ROOT / ckpt_rel)
        print(f"\n[{name}] loading {ckpt_rel}")
        model, _cfg = load_model(ckpt, device)
        t0 = time.perf_counter()
        pred = predict_all(model, x, device, batch_size=256)
        print(f"  inference: {time.perf_counter()-t0:.1f}s")
        results[name] = per_n_active_pairwise(pred, target, n_actives)
        # quick summary
        for variant in ("active", "full"):
            avg = np.mean([v["pairwise_acc"] for v in results[name][variant].values()])
            print(f"  mean pairwise ({variant} block) over n_active: {avg:.4f}")
        del model

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    ns = sorted(N_VALUES_VARIABLE)

    for ax, variant, title_suffix in (
        (axes[0], "active", "on n_active × n_active block"),
        (axes[1], "full",   "on full 40 × 40 padded canvas"),
    ):
        for name in ORDER:
            ys = [results[name][variant][n_a]["pairwise_acc"] for n_a in ns]
            lw = 3.0 if name == "unfiltered" else 1.8
            alpha = 1.0 if name == "unfiltered" else 0.85
            marker = "o" if name == "unfiltered" else "s"
            ms = 7 if name == "unfiltered" else 5
            ax.plot(ns, ys, color=COLORS[name], lw=lw, alpha=alpha,
                    marker=marker, markersize=ms, label=name)
        ax.set_xlabel("n_active", fontsize=12)
        ax.set_title(f"Pairwise acc — {title_suffix}", fontsize=12)
        ax.set_xticks(ns)
        ax.grid(alpha=0.3)
        ax.axvline(14, color="gray", ls=":", lw=1, alpha=0.6)
        ax.text(14, ax.get_ylim()[0], " train n", color="gray",
                fontsize=9, va="bottom", ha="left", alpha=0.7)

    axes[0].set_ylabel("Pairwise accuracy", fontsize=12)
    axes[0].legend(loc="lower left", fontsize=11, title="ER training filter",
                   framealpha=0.95)
    fig.suptitle("2chains OOD: pairwise accuracy by n_active — big ER models",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out_path = OUT_DIR / "two_chains_pairwise_by_n_active_big.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"\nsaved figure: {out_path}")

    # Save numeric results
    out_json = OUT_DIR / "two_chains_pairwise_by_n_active_big.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved data:   {out_json}")

    # Print the table for unfiltered (the one of interest)
    print("\nUnfiltered (big) — pairwise per n_active:")
    print(f"  {'n_active':>8}  {'active-block':>13}  {'full-40x40':>13}")
    for n_a in ns:
        a = results["unfiltered"]["active"][n_a]["pairwise_acc"]
        f_ = results["unfiltered"]["full"][n_a]["pairwise_acc"]
        print(f"  {n_a:>8}  {a:>13.4f}  {f_:>13.4f}")


if __name__ == "__main__":
    main()
