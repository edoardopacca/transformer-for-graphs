"""Exp 2 — WHY is pairwise accuracy ~80% on 2chains while exact-match is 0?

We evaluate each big ER(n=40) model on 2chains at n_active = 40 (k = 20, i.e.
two paths of 20 nodes, no isolated padding — the "k = n/2" case the advisor
asked for) and decompose the pairwise accuracy to localise the 20% error.

Two competing explanations:
  (A) Capacity story.  The model implements matrix-powering up to 3^L = 9, so it
      should FAIL on within-component pairs at distance > 9 (the far ends of a
      20-node path), and SUCCEED on everything within distance 9 and on the
      cross-component zeros. Error concentrated in target=1, d>9.
  (B) Heuristic story.  The model learned the degree heuristic and OVER-predicts
      connectivity, so it gets within-component pairs right (even far ones) but
      mislabels cross-component (disconnected) pairs as connected. Error
      concentrated in target=0 (disconnected).

The script prints, per model, the accuracy split by target class, by distance
bucket (d<=9 vs d>9), the full per-distance curve, the disconnected accuracy,
trivial baselines, and the prediction confusion matrix — enough to decide A vs B.

Runs locally. Output: runs/n40_cross_experiment/diagnose_2chains_*.
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
    generate_two_chains_graph,
)
from experiments2.ood_evaluation import load_model

CHECKPOINTS_BY_ROUND = {
    # First training round (seed 1000).
    1: {
        "unfiltered": "runs/retrain_er_n40_big_495903/n40_p005_unfiltered_big/best.pt",
        "D<=11":      "runs/retrain_er_n40_big_494467/n40_p005_diam11_big/best.pt",
        "D<=9":       "runs/retrain_er_n40_big_495198/n40_p005_diam9_big/best.pt",
        "D<=7":       "runs/retrain_er_n40_big_495199/n40_p005_diam7_big/best.pt",
    },
    # Second training round, "exp2" (seed 2000).
    2: {
        "unfiltered": "runs/retrain_er_n40_big_exp2_499357/n40_p005_unfiltered_big/best.pt",
        "D<=11":      "runs/retrain_er_n40_big_exp2_499358/n40_p005_diam11_big/best.pt",
        "D<=9":       "runs/retrain_er_n40_big_exp2_499359/n40_p005_diam9_big/best.pt",
        "D<=7":       "runs/retrain_er_n40_big_exp2_499360/n40_p005_diam7_big/best.pt",
    },
}
ORDER = ["unfiltered", "D<=11", "D<=9", "D<=7"]
CAPACITY = 9          # 3^L, L = 2
N = 40
K = 20                # two chains of 20 -> n_active = 40
N_TEST = 5_000
OUT_DIR = PROJECT_ROOT / "runs" / "n40_cross_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_2chains_full(n: int, k: int, n_test: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    base = generate_two_chains_graph(n, k)
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


@torch.no_grad()
def predict(model, x_np, device, batch_size=256):
    n_graphs, n, _ = x_np.shape
    out = np.empty((n_graphs, n, n), dtype=np.int8)
    for s in range(0, n_graphs, batch_size):
        e = min(s + batch_size, n_graphs)
        xb = torch.from_numpy(x_np[s:e]).float().to(device)
        out[s:e] = (model(xb) > 0).cpu().numpy().astype(np.int8)
    return out


def diagnose(pred, target, dist) -> dict:
    """All metrics computed on OFF-DIAGONAL entries (self-loops excluded)."""
    n_graphs, n, _ = pred.shape
    eye = np.eye(n, dtype=bool)
    off = np.broadcast_to(~eye[None], (n_graphs, n, n))

    eq = (pred == target)
    t1 = off & (target == 1)           # within-component (connected)
    t0 = off & (target == 0)           # cross-component (disconnected)

    # distance buckets within connected pairs
    near = t1 & (dist >= 1) & (dist <= CAPACITY)
    far = t1 & (dist > CAPACITY)

    def acc(mask):
        c = int(mask.sum())
        return (float(eq[mask].mean()) if c else float("nan")), c

    res = {}
    res["overall_offdiag"], res["n_offdiag"] = acc(off)
    res["connected_acc"], res["n_connected"] = acc(t1)        # recall on target=1
    res["disconnected_acc"], res["n_disconnected"] = acc(t0)  # specificity on target=0
    res["connected_near_acc(d<=9)"], res["n_near"] = acc(near)
    res["connected_far_acc(d>9)"], res["n_far"] = acc(far)

    # per-distance pairwise accuracy
    per_dist = {}
    for dv in range(1, int(dist.max()) + 1):
        m = off & (dist == dv)
        c = int(m.sum())
        if c:
            per_dist[dv] = (float(eq[m].mean()), c)
    res["per_dist"] = per_dist

    # trivial baselines on off-diagonal
    frac1 = float((off & (target == 1)).sum()) / float(off.sum())
    res["baseline_all_connected"] = frac1               # predict 1 everywhere
    res["baseline_all_disconnected"] = 1.0 - frac1      # predict 0 everywhere
    # capacity oracle: connected iff 1 <= d <= 9
    oracle = ((dist >= 1) & (dist <= CAPACITY)).astype(np.int8)
    res["baseline_capacity_oracle"] = float((oracle[off] == target[off]).mean())

    # confusion on off-diagonal
    p = pred[off]; t = target[off]
    res["pred_connected_rate"] = float((p == 1).mean())
    res["TP"] = int(((p == 1) & (t == 1)).sum())
    res["FP"] = int(((p == 1) & (t == 0)).sum())
    res["TN"] = int(((p == 0) & (t == 0)).sum())
    res["FN"] = int(((p == 0) & (t == 1)).sum())
    return res


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, choices=[1, 2], default=1,
                    help="1 = first training round (seed 1000); 2 = exp2 round (seed 2000)")
    args = ap.parse_args()
    checkpoints = CHECKPOINTS_BY_ROUND[args.round]
    suffix = "" if args.round == 1 else "_exp2"

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"device: {device}  | round: {args.round}")

    t0 = time.perf_counter()
    ds = build_2chains_full(N, K, N_TEST, seed=4242)
    print(f"generated {N_TEST} 2chains (k={K}) graphs in {time.perf_counter()-t0:.1f}s")
    target = ds["y"].astype(np.int8)
    dist = ds["d"].astype(np.int16)

    results = {}
    for name in ORDER:
        ckpt = str(PROJECT_ROOT / checkpoints[name])
        print(f"\n[{name}] loading {checkpoints[name]}")
        model, _cfg = load_model(ckpt, device)
        pred = predict(model, ds["x"], device)
        r = diagnose(pred, target, dist)
        results[name] = r
        del model

        print(f"  overall off-diag pairwise : {r['overall_offdiag']:.4f}  "
              f"(n={r['n_offdiag']:,})")
        print(f"  connected (target=1)      : {r['connected_acc']:.4f}  "
              f"(n={r['n_connected']:,})   <- recall")
        print(f"     near d<=9              : {r['connected_near_acc(d<=9)']:.4f}  "
              f"(n={r['n_near']:,})")
        print(f"     far  d> 9              : {r['connected_far_acc(d>9)']:.4f}  "
              f"(n={r['n_far']:,})")
        print(f"  disconnected (target=0)   : {r['disconnected_acc']:.4f}  "
              f"(n={r['n_disconnected']:,})   <- specificity")
        print(f"  pred-connected rate       : {r['pred_connected_rate']:.4f}  "
              f"(TP={r['TP']:,} FP={r['FP']:,} TN={r['TN']:,} FN={r['FN']:,})")
        print(f"  baselines: all-conn={r['baseline_all_connected']:.4f}  "
              f"all-disc={r['baseline_all_disconnected']:.4f}  "
              f"capacity-oracle={r['baseline_capacity_oracle']:.4f}")

    # ── Plot: per-distance accuracy + connected/disconnected split per model ──
    fig, axes = plt.subplots(1, len(ORDER), figsize=(5.5 * len(ORDER), 5),
                             sharey=True, squeeze=False)
    for c, name in enumerate(ORDER):
        ax = axes[0][c]
        r = results[name]
        dists = sorted(r["per_dist"].keys())
        vals = [r["per_dist"][d][0] for d in dists]
        ax.bar([str(d) for d in dists], vals, color="#1f77b4")
        ax.axvline(CAPACITY - 0.5, color="red", ls="--", lw=1.2)
        ax.text(CAPACITY - 0.5, 1.02, " 3^L=9", color="red", fontsize=9, ha="left")
        ax.axhline(r["disconnected_acc"], color="#d62728", ls=":", lw=1.5,
                   label=f"disconnected acc={r['disconnected_acc']:.2f}")
        ax.set_title(f"{name}\noverall off-diag={r['overall_offdiag']:.3f}",
                     fontsize=11)
        ax.set_xlabel("within-component distance d")
        ax.set_ylim(0, 1.08); ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    axes[0][0].set_ylabel("pairwise accuracy (off-diagonal)")
    fig.suptitle(f"Exp 2{' (exp2 round)' if suffix else ''}: 2chains (k=20) — "
                 f"where does the 20% pairwise error live?", fontsize=14, y=1.03)
    fig.tight_layout()
    out_png = OUT_DIR / f"diagnose_2chains_k20{suffix}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved figure: {out_png}")

    # JSON (drop the bulky per_dist tuples into plain dict)
    serial = {}
    for name, r in results.items():
        rr = {k: v for k, v in r.items() if k != "per_dist"}
        rr["per_dist"] = {str(d): {"acc": a, "n": c} for d, (a, c) in r["per_dist"].items()}
        serial[name] = rr
    with open(OUT_DIR / f"diagnose_2chains_k20{suffix}.json", "w") as f:
        json.dump(serial, f, indent=2)
    print(f"saved data:   {OUT_DIR / f'diagnose_2chains_k20{suffix}.json'}")


if __name__ == "__main__":
    main()
