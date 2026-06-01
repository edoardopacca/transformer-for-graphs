"""
Aggregate the reproduction runs under runs/repro_paper_n20/ and produce the
three paper figures (across the 3 seeds / iterations):

  fig1_reproduction.png   — unrestricted ER(n=20,p=0.08): ER / 2Chain / 2Clique
                            exact-match vs step  (paper Figure 1, bottom panel)
  fig7_reproduction.png   — 2Chain exact-match: unrestricted vs restrict-diam≤9
                            (paper Figure 7)
  fig11_reproduction.png  — 2Clique exact-match: unrestricted vs restrict-diam≤9
                            (paper Figure 11)

Each curve is the mean over the available seeds; individual seeds are drawn as
faint lines and a ±1 std band is shaded. Runs that are still in progress are
handled by truncating every group to its common step prefix.

Usage:
    python plot_repro_paper_figures.py [--runs_dir runs/repro_paper_n20]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR_RE = re.compile(r"n\d+_p\d+_(?P<cond>unrestricted|diam\d+)_seed(?P<seed>\d+)")


def load_runs(runs_dir: Path) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        m = DIR_RE.match(d.name)
        hist = d / "history.json"
        if not m or not hist.exists():
            continue
        with hist.open() as f:
            h = json.load(f)
        h["_seed"] = int(m.group("seed"))
        h["_dir"] = d.name
        groups.setdefault(m.group("cond"), []).append(h)
    return groups


def _stack(runs: List[dict], key: str):
    """Align runs to common step prefix; return (steps, array[n_seeds, n_steps])."""
    if not runs:
        return None, None
    L = min(len(r["steps"]) for r in runs)
    if L == 0:
        return None, None
    steps = np.array(runs[0]["steps"][:L])
    mat = np.array([r[key][:L] for r in runs], dtype=float)
    return steps, mat


def _plot_metric(ax, runs, key, label, color):
    steps, mat = _stack(runs, key)
    if steps is None:
        return
    mean = mat.mean(0)
    ax.plot(steps, mean, lw=2.2, color=color, label=label)
    if mat.shape[0] > 1:
        std = mat.std(0)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
        for row in mat:
            ax.plot(steps, row, lw=0.7, color=color, alpha=0.30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs/repro_paper_n20")
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)
    groups = load_runs(runs_dir)

    if not groups:
        raise SystemExit(f"No runs found under {runs_dir}/ "
                         f"(expected subdirs like n20_p008_unrestricted_seed1000/).")
    for cond, runs in groups.items():
        seeds = sorted(r["_seed"] for r in runs)
        print(f"  {cond:14s}: {len(runs)} run(s), seeds={seeds}")

    unr = groups.get("unrestricted", [])
    res = groups.get("diam9", [])

    # ── Figure 1: unrestricted, three eval distributions ─────────────────────
    if unr:
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_metric(ax, unr, "val_er_exact", "Erdős-Rényi (in-dist)", "#1f77b4")
        _plot_metric(ax, unr, "val_2chain_exact", "Two Chains", "#2ca02c")
        _plot_metric(ax, unr, "val_2clique_exact", "Two Cliques", "#ff7f0e")
        ax.set_title("Figure 1 (repro): Exact Match Accuracy — trained on unrestricted ER(n=20, p=0.08)")
        ax.set_xlabel("Training Step"); ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(runs_dir / "fig1_reproduction.png", dpi=180)
        plt.close(fig)
        print("  wrote fig1_reproduction.png")

    # ── Figure 7: 2Chain, unrestricted vs restricted ─────────────────────────
    if unr or res:
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_metric(ax, unr, "val_2chain_exact", "Unrestricted", "#ff7f0e")
        _plot_metric(ax, res, "val_2chain_exact", "Restrict Diameter (<=9)", "#1f77b4")
        ax.set_title("Figure 7 (repro): Exact Match on Two Chains")
        ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(runs_dir / "fig7_reproduction.png", dpi=180)
        plt.close(fig)
        print("  wrote fig7_reproduction.png")

    # ── Figure 11: 2Clique, unrestricted vs restricted ───────────────────────
    if unr or res:
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_metric(ax, unr, "val_2clique_exact", "Unrestricted", "#ff7f0e")
        _plot_metric(ax, res, "val_2clique_exact", "Restrict Diameter (<=9)", "#1f77b4")
        ax.set_title("Figure 11 (repro): Exact Match on Two Cliques")
        ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(runs_dir / "fig11_reproduction.png", dpi=180)
        plt.close(fig)
        print("  wrote fig11_reproduction.png")


if __name__ == "__main__":
    main()
