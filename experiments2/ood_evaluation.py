"""
Out-of-distribution evaluation for a single trained checkpoint.

Two modes:
  --mode n14_to_er
      For models trained at n=14 (2chains, 2cliques).
      Evaluates the checkpoint on ER(n=14, p) test sets at p in {0.05, 0.2, 0.5}.

  --mode n40_to_structured
      For models trained at n=40 (ER n=40 variants).
      Evaluates on:
        (a) 2chains graphs with random n_active in {14, 16, ..., 40},
            padded to 40x40 with isolated nodes.
        (b) 2cliques graphs with the same n_active distribution.
        (c) Unfiltered ER(n=40, p=0.05) — for the diameter-restricted models
            this is genuinely OOD; for the unfiltered model it coincides with
            its in-distribution evaluation but is included for consistency.
            We additionally report per-diameter-bucket exact-match accuracy.

For each test set we save: exact-match accuracy, pairwise accuracy, per-distance
pairwise accuracy (including disconnected pairs), and a per-n_active or
per-diameter-bucket breakdown where applicable. All metrics go to results.json
in the output dir, and several diagnostic plots are produced alongside.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import (
    add_self_loops,
    compute_connectivity_matrix,
    compute_all_pairs_shortest_paths,
    generate_er_graph,
    generate_two_chains_graph,
    generate_two_cliques_graph,
)
from model import GraphConnectivityTransformer, ModelConfig
from utils import ensure_dir, get_device, save_json


N_TEST = 10_000
N_VALUES_VARIABLE = list(range(14, 41, 2))   # 14, 16, ..., 40
ER_P_VALUES = [0.05, 0.2, 0.5]


# ── Padding utilities ────────────────────────────────────────────────────────

def pad_graph_to_size(adj_no_loops: np.ndarray, dist: np.ndarray,
                      n_padded: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embed an n_active graph into an n_padded canvas, filling the rest with
    isolated nodes. Returns (adj_with_self_loops, connectivity, distances)."""
    n_active = adj_no_loops.shape[0]
    if n_active > n_padded:
        raise ValueError("n_active must be <= n_padded")

    adj_full = np.eye(n_padded, dtype=np.uint8)
    adj_full[:n_active, :n_active] = add_self_loops(adj_no_loops).astype(np.uint8)

    R_full = np.eye(n_padded, dtype=np.uint8)
    R_full[:n_active, :n_active] = compute_connectivity_matrix(adj_no_loops).astype(np.uint8)

    d_full = -np.ones((n_padded, n_padded), dtype=np.int16)
    d_full[:n_active, :n_active] = dist.astype(np.int16)
    np.fill_diagonal(d_full, 0)

    return adj_full, R_full, d_full


# ── Test-set generation (parallel) ───────────────────────────────────────────

def _gen_er_chunk(args) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    start, size, n, p, seed = args
    rng = np.random.default_rng(seed)
    xs = np.empty((size, n, n), dtype=np.uint8)
    ys = np.empty((size, n, n), dtype=np.uint8)
    ds = np.empty((size, n, n), dtype=np.int16)
    for i in range(size):
        adj_no = generate_er_graph(n, p, rng)
        d = compute_all_pairs_shortest_paths(adj_no)
        xs[i] = add_self_loops(adj_no).astype(np.uint8)
        ys[i] = compute_connectivity_matrix(adj_no).astype(np.uint8)
        ds[i] = d.astype(np.int16)
    return start, xs, ys, ds


def _gen_structured_chunk(args):
    start, size, n_padded, n_values, kind, seed = args
    rng = np.random.default_rng(seed)
    xs = np.empty((size, n_padded, n_padded), dtype=np.uint8)
    ys = np.empty((size, n_padded, n_padded), dtype=np.uint8)
    ds = np.empty((size, n_padded, n_padded), dtype=np.int16)
    n_actives = np.empty(size, dtype=np.int16)
    for i in range(size):
        n_active = int(rng.choice(n_values))
        k = n_active // 2
        if kind == "chains":
            adj_no = generate_two_chains_graph(n_active, k)
        else:
            adj_no = generate_two_cliques_graph(n_active, k)
        adj_no = adj_no.astype(np.uint8)
        dist = compute_all_pairs_shortest_paths(adj_no)
        adj_full, R_full, d_full = pad_graph_to_size(adj_no, dist, n_padded)
        xs[i] = adj_full
        ys[i] = R_full
        ds[i] = d_full
        n_actives[i] = n_active
    return start, xs, ys, ds, n_actives


def generate_er_test(n: int, p: float, n_test: int, num_workers: int,
                      seed: int = 0) -> Dict[str, np.ndarray]:
    chunk = max(500, n_test // (num_workers * 4))
    seed_base = int(p * 1000) + 7777 + seed * 100003
    specs = [(s, min(chunk, n_test - s), n, p, seed_base + s)
             for s in range(0, n_test, chunk)]
    xs = np.empty((n_test, n, n), dtype=np.uint8)
    ys = np.empty((n_test, n, n), dtype=np.uint8)
    ds = np.empty((n_test, n, n), dtype=np.int16)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for s, x, y, d in ex.map(_gen_er_chunk, specs):
            e = s + len(x)
            xs[s:e] = x; ys[s:e] = y; ds[s:e] = d
    return {"x": xs, "y": ys, "d": ds}


def generate_structured_padded_test(kind: str, n_padded: int, n_values: List[int],
                                     n_test: int, num_workers: int,
                                     seed: int = 0) -> Dict[str, np.ndarray]:
    chunk = max(500, n_test // (num_workers * 4))
    seed_base = 8888 + (1 if kind == "chains" else 2) * 1000 + seed * 100003
    specs = [(s, min(chunk, n_test - s), n_padded, n_values, kind, seed_base + s)
             for s in range(0, n_test, chunk)]
    xs = np.empty((n_test, n_padded, n_padded), dtype=np.uint8)
    ys = np.empty((n_test, n_padded, n_padded), dtype=np.uint8)
    ds = np.empty((n_test, n_padded, n_padded), dtype=np.int16)
    n_actives = np.empty(n_test, dtype=np.int16)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for s, x, y, d, na in ex.map(_gen_structured_chunk, specs):
            e = s + len(x)
            xs[s:e] = x; ys[s:e] = y; ds[s:e] = d; n_actives[s:e] = na
    return {"x": xs, "y": ys, "d": ds, "n_active": n_actives}


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str,
               device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    mcfg = ModelConfig(
        n=cfg["n"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"], n_layers=cfg["n_layers"],
        dropout=cfg.get("dropout", 0.0),
        # Older checkpoints may not have attn_kind; default to softmax to stay
        # backward-compatible. New big-model checkpoints save normalized_relu.
        attn_kind=cfg.get("attn_kind", "softmax"),
    )
    model = GraphConnectivityTransformer(mcfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: torch.nn.Module, ds: Dict[str, np.ndarray],
             device: torch.device, batch_size: int = 256) -> Dict[str, Any]:
    test_x, test_y, test_d = ds["x"], ds["y"], ds["d"]
    n_graphs, n, _ = test_x.shape

    all_pred = np.empty((n_graphs, n, n), dtype=np.int8)
    for start in range(0, n_graphs, batch_size):
        end = min(start + batch_size, n_graphs)
        xb = torch.from_numpy(test_x[start:end]).float().to(device)
        logits = model(xb)
        all_pred[start:end] = (logits > 0).cpu().numpy().astype(np.int8)

    targets = test_y.astype(np.int8)
    eq = (all_pred == targets)

    exact_per_graph = eq.reshape(n_graphs, -1).all(axis=1)
    exact_match  = float(exact_per_graph.mean())
    pairwise_acc = float(eq.mean())

    eye_mask = ~np.eye(n, dtype=bool)
    offdiag  = np.broadcast_to(eye_mask[None, :, :], (n_graphs, n, n))

    per_dist: Dict[str, float] = {}
    dist_counts: Dict[str, int] = {}
    max_d = int(test_d.max())
    for dv in range(1, max_d + 1):
        mask = offdiag & (test_d == dv)
        cnt = int(mask.sum())
        if cnt > 0:
            per_dist[str(dv)]   = float(eq[mask].mean())
            dist_counts[str(dv)] = cnt
    unreach = offdiag & (test_d == -1)
    if unreach.any():
        per_dist["disconnected"]   = float(eq[unreach].mean())
        dist_counts["disconnected"] = int(unreach.sum())

    return {
        "exact_match": exact_match,
        "pairwise_acc": pairwise_acc,
        "per_dist_acc": per_dist,
        "dist_counts":  dist_counts,
        "exact_per_graph": exact_per_graph,
    }


def compute_per_diam_bucket(exact_per_graph: np.ndarray,
                            test_d: np.ndarray,
                            thresholds: List[int] = (7, 9, 11)) -> Dict[str, Any]:
    """Compute exact-match accuracy broken down by maximum finite distance
    (graph diameter) in each test graph."""
    n_graphs = test_d.shape[0]
    d_clipped = np.where(test_d < 0, 0, test_d)
    per_graph_diam = d_clipped.reshape(n_graphs, -1).max(axis=1)

    out: Dict[str, Any] = {}
    last_thr = thresholds[-1]
    for thr in thresholds:
        mask = per_graph_diam <= thr
        if mask.any():
            out[f"exact_le{thr}"] = float(exact_per_graph[mask].mean())
            out[f"n_graphs_le{thr}"] = int(mask.sum())
    mask_above = per_graph_diam > last_thr
    if mask_above.any():
        out[f"exact_gt{last_thr}"] = float(exact_per_graph[mask_above].mean())
        out[f"n_graphs_gt{last_thr}"] = int(mask_above.sum())
    return out


# ── Plots ────────────────────────────────────────────────────────────────────

def _per_dist_bar_panel(ax, test_data: Dict[str, Any], title: str) -> None:
    """Single bar-chart panel: x = distance, y = pairwise accuracy."""
    per_dist = test_data["per_dist_acc"]
    counts   = test_data["dist_counts"]
    numeric  = sorted(int(k) for k in per_dist if k != "disconnected")
    labels   = [str(d) for d in numeric]
    values   = [per_dist[str(d)] for d in numeric]
    cnts     = [counts[str(d)]    for d in numeric]
    has_disc = "disconnected" in per_dist
    if has_disc:
        labels.append("disc")
        values.append(per_dist["disconnected"])
        cnts.append(counts["disconnected"])
    colors = ["#1f77b4"] * len(numeric) + (["#d62728"] if has_disc else [])
    bars = ax.bar(labels, values, color=colors)
    for bar, c in zip(bars, cnts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{c:,}", ha="center", va="bottom", fontsize=7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Shortest-path distance d")
    ax.set_ylabel("Pairwise accuracy")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)


def make_plots_n14(results: Dict[str, Any], out_dir: Path) -> None:
    tests = [results["tests"][f"er_p{p}"] for p in ER_P_VALUES]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, t, p in zip(axes, tests, ER_P_VALUES):
        _per_dist_bar_panel(
            ax, t,
            f"ER(n=14, p={p})  |  exact={t['exact_match']:.3f}, "
            f"pairwise={t['pairwise_acc']:.3f}",
        )
    fig.suptitle("OOD evaluation: n=14 model tested on ER(n=14, p∈{0.05,0.2,0.5})",
                 fontsize=13)
    fig.tight_layout()
    out = out_dir / "per_distance_bars.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def make_plots_n40(results: Dict[str, Any], out_dir: Path) -> None:
    # ── (1) Per-distance bar chart: chains / cliques / unfiltered ER ──
    test_keys   = ["two_chains_var_n", "two_cliques_var_n", "unfiltered_er"]
    test_titles = ["2chains  (n_active variable)",
                   "2cliques  (n_active variable)",
                   "ER(n=40, p=0.05)  unfiltered"]
    available   = [(k, t) for k, t in zip(test_keys, test_titles)
                   if k in results["tests"]]
    if available:
        ncols = len(available)
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5.5))
        if ncols == 1:
            axes = [axes]
        for ax, (key, title) in zip(axes, available):
            t = results["tests"][key]
            _per_dist_bar_panel(
                ax, t,
                f"{title}  |  exact={t['exact_match']:.3f}",
            )
        fig.suptitle("OOD evaluation: n=40 model — per-distance pairwise accuracy",
                     fontsize=13)
        fig.tight_layout()
        out = out_dir / "per_distance_bars.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")

    # ── (2) Per-n_active line chart for chains and cliques ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    drew_any = False
    for ax, kind in zip(axes, ["chains", "cliques"]):
        t = results["tests"].get(f"two_{kind}_var_n")
        if t is None:
            ax.axis("off"); continue
        drew_any = True
        by_n_raw = t["by_n_active"]
        ns = sorted(int(k) for k in by_n_raw)
        vals = [by_n_raw[str(n) if str(n) in by_n_raw else n]["exact_match"]
                for n in ns]
        ax.plot(ns, vals, marker="o", lw=2, color="#1f77b4", markersize=7,
                markeredgecolor="white", markeredgewidth=0.8)
        for n, v in zip(ns, vals):
            ax.text(n, min(v + 0.025, 1.02), f"{v:.2f}",
                    ha="center", fontsize=8)
        ax.set_title(f"2{kind}  |  overall exact = {t['exact_match']:.3f}",
                     fontsize=11)
        ax.set_xlabel("n_active (number of non-isolated nodes)")
        ax.set_ylabel("Exact-match accuracy")
        ax.set_ylim(0, 1.07)
        ax.set_xticks(ns)
        ax.grid(alpha=0.3)
    if drew_any:
        fig.suptitle("OOD evaluation: exact match as a function of n_active",
                     fontsize=13)
        fig.tight_layout()
        out = out_dir / "per_n_active.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")
    else:
        plt.close(fig)

    # ── (3) Per-diameter-bucket bar chart for unfiltered ER ──
    t = results["tests"].get("unfiltered_er")
    if t is None or "per_diam_bucket" not in t:
        return
    pdb = t["per_diam_bucket"]
    bucket_labels = ["D≤7", "D≤9", "D≤11", "D>11"]
    val_keys      = ["exact_le7", "exact_le9", "exact_le11", "exact_gt11"]
    cnt_keys      = ["n_graphs_le7", "n_graphs_le9", "n_graphs_le11", "n_graphs_gt11"]
    bar_colors    = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    bars_data = [(lab, pdb.get(vk), pdb.get(ck, 0), col)
                 for lab, vk, ck, col in zip(bucket_labels, val_keys, cnt_keys, bar_colors)
                 if pdb.get(vk) is not None]
    if not bars_data:
        return

    fig, ax = plt.subplots(figsize=(9, 5.8))
    bars = ax.bar([b[0] for b in bars_data],
                  [b[1] for b in bars_data],
                  color=[b[3] for b in bars_data],
                  edgecolor="black", linewidth=0.6)
    for bar, (_, val, cnt, _) in zip(bars, bars_data):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{val:.3f}\n(n = {cnt:,})", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Exact-match accuracy")
    ax.set_xlabel("Test-graph diameter bucket")
    ax.set_title(
        f"OOD: ER(n=40, p=0.05) unfiltered — exact match by graph diameter\n"
        f"(overall exact = {t['exact_match']:.3f}, pairwise = {t['pairwise_acc']:.3f})",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "per_diam_bucket_ood.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ── Modes ────────────────────────────────────────────────────────────────────

def mode_n14_to_er(args) -> None:
    device = get_device("auto")
    model, mcfg = load_model(args.checkpoint, device)
    assert mcfg["n"] == 14, f"Expected n=14 model, got n={mcfg['n']}"
    print(f"Loaded n={mcfg['n']} model from {args.checkpoint}")

    out_dir = Path(args.output_dir); ensure_dir(out_dir)
    results: Dict[str, Any] = {
        "checkpoint": args.checkpoint, "model_config": mcfg, "tests": {},
    }

    for p in ER_P_VALUES:
        print(f"\n── Test: ER(n=14, p={p}), {N_TEST:,} graphs ──")
        t0 = time.perf_counter()
        ds = generate_er_test(n=14, p=p, n_test=N_TEST, num_workers=args.num_workers,
                               seed=args.seed)
        gen_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        m = evaluate(model, ds, device)
        eval_s = time.perf_counter() - t0
        print(f"  gen {gen_s:.1f}s  eval {eval_s:.2f}s  "
              f"exact={m['exact_match']:.4f}  pairwise={m['pairwise_acc']:.4f}")
        results["tests"][f"er_p{p}"] = {
            "p": p, "n": 14, "gen_sec": gen_s, "eval_sec": eval_s,
            "exact_match": m["exact_match"],
            "pairwise_acc": m["pairwise_acc"],
            "per_dist_acc": m["per_dist_acc"],
            "dist_counts":  m["dist_counts"],
        }

    save_json(out_dir / "results.json", results)
    print(f"\nResults written to {out_dir / 'results.json'}")
    print("Generating plots …")
    make_plots_n14(results, out_dir)


def mode_n40_to_structured(args) -> None:
    device = get_device("auto")
    model, mcfg = load_model(args.checkpoint, device)
    assert mcfg["n"] == 40, f"Expected n=40 model, got n={mcfg['n']}"
    print(f"Loaded n={mcfg['n']} model from {args.checkpoint}")

    out_dir = Path(args.output_dir); ensure_dir(out_dir)
    results: Dict[str, Any] = {
        "checkpoint": args.checkpoint, "model_config": mcfg, "tests": {},
    }

    # ── (a) and (b): 2chains and 2cliques, n_active variable ──
    for kind in ["chains", "cliques"]:
        print(f"\n── Test: 2{kind}, n_active ∈ {N_VALUES_VARIABLE}, padded to 40, "
              f"{N_TEST:,} graphs ──")
        t0 = time.perf_counter()
        ds = generate_structured_padded_test(
            kind=kind, n_padded=40, n_values=N_VALUES_VARIABLE,
            n_test=N_TEST, num_workers=args.num_workers,
            seed=args.seed,
        )
        gen_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        m = evaluate(model, ds, device)
        eval_s = time.perf_counter() - t0
        print(f"  gen {gen_s:.1f}s  eval {eval_s:.2f}s  "
              f"exact={m['exact_match']:.4f}  pairwise={m['pairwise_acc']:.4f}")

        n_actives = ds["n_active"]
        exact_per_graph = m["exact_per_graph"]
        by_n: Dict[int, Dict[str, float]] = {}
        for n_a in N_VALUES_VARIABLE:
            mask = n_actives == n_a
            if mask.any():
                by_n[int(n_a)] = {
                    "n_graphs": int(mask.sum()),
                    "exact_match": float(exact_per_graph[mask].mean()),
                }
        print("  per-n_active exact match:")
        for n_a, v in sorted(by_n.items()):
            print(f"    n={n_a:>2}: {v['exact_match']:.4f}  ({v['n_graphs']} graphs)")

        results["tests"][f"two_{kind}_var_n"] = {
            "kind": kind, "n_values": N_VALUES_VARIABLE,
            "gen_sec": gen_s, "eval_sec": eval_s,
            "exact_match": m["exact_match"],
            "pairwise_acc": m["pairwise_acc"],
            "per_dist_acc": m["per_dist_acc"],
            "dist_counts":  m["dist_counts"],
            "by_n_active":  by_n,
        }

    # ── (c): unfiltered ER(n=40, p=0.05) with per-diameter-bucket ──
    print(f"\n── Test: unfiltered ER(n=40, p=0.05), {N_TEST:,} graphs ──")
    t0 = time.perf_counter()
    ds = generate_er_test(n=40, p=0.05, n_test=N_TEST, num_workers=args.num_workers,
                           seed=args.seed)
    gen_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    m = evaluate(model, ds, device)
    eval_s = time.perf_counter() - t0
    print(f"  gen {gen_s:.1f}s  eval {eval_s:.2f}s  "
          f"exact={m['exact_match']:.4f}  pairwise={m['pairwise_acc']:.4f}")

    per_diam = compute_per_diam_bucket(m["exact_per_graph"], ds["d"])
    print("  per-diameter-bucket exact match:")
    for thr in [7, 9, 11]:
        if f"exact_le{thr}" in per_diam:
            print(f"    D≤{thr}: {per_diam[f'exact_le{thr}']:.4f}  "
                  f"(n = {per_diam[f'n_graphs_le{thr}']:,})")
    if "exact_gt11" in per_diam:
        print(f"    D>11: {per_diam['exact_gt11']:.4f}  "
              f"(n = {per_diam['n_graphs_gt11']:,})")

    results["tests"]["unfiltered_er"] = {
        "n": 40, "p": 0.05, "gen_sec": gen_s, "eval_sec": eval_s,
        "exact_match": m["exact_match"],
        "pairwise_acc": m["pairwise_acc"],
        "per_dist_acc": m["per_dist_acc"],
        "dist_counts":  m["dist_counts"],
        "per_diam_bucket": per_diam,
    }

    save_json(out_dir / "results.json", results)
    print(f"\nResults written to {out_dir / 'results.json'}")
    print("Generating plots …")
    make_plots_n40(results, out_dir)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["n14_to_er", "n40_to_structured"], required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0,
                   help="shift test-set RNG seeds (0 = original data)")
    args = p.parse_args()

    if args.mode == "n14_to_er":
        mode_n14_to_er(args)
    else:
        mode_n40_to_structured(args)


if __name__ == "__main__":
    main()
