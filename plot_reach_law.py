"""Summarise the depth->reach sweep: maximum exact-reach distance d* vs depth L,
against the 3^L capacity prediction.

Reads runs/reach_depth/reach_n*_L*_*/history.json and plots d*(L) with the
3^L curve and the n-1 measurement ceiling (d* cannot exceed n-1).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR_RE = re.compile(r"reach_n(?P<n>\d+)_L(?P<L>\d+)_(?P<attn>\w+?)_seed(?P<seed>\d+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs/reach_depth")
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)

    pts = []
    n_val = None
    for d in sorted(runs_dir.iterdir()):
        m = DIR_RE.match(d.name) if d.is_dir() else None
        hist = d / "history.json"
        if not m or not hist.exists():
            continue
        h = json.load(hist.open())
        L = int(m.group("L")); n_val = int(m.group("n"))
        d_star_final = h["d_star"][-1] if h.get("d_star") else 0
        d_star_best = h.get("best_d_star", d_star_final)
        pts.append((L, d_star_final, d_star_best, 3 ** L))
        print(f"L={L}: d*_final={d_star_final}  d*_best={d_star_best}  3^L={3**L}")
    if not pts:
        raise SystemExit(f"No reach runs under {runs_dir}")
    pts.sort()
    Ls = [p[0] for p in pts]
    dfin = [p[1] for p in pts]
    cap = [p[3] for p in pts]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(Ls, dfin, "o-", lw=2.5, ms=9, color="#1f77b4",
            label="measured $d^*$ (reach $\\geq 0.99$)")
    ax.plot(Ls, cap, "s--", lw=2, color="#d62728", label="$3^L$ (capacity prediction)")
    if n_val:
        ax.axhline(n_val - 1, color="gray", ls=":", lw=1.5,
                   label=f"$n-1={n_val-1}$ (measurement ceiling)")
    for L, df in zip(Ls, dfin):
        ax.annotate(str(df), (L, df), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=10, color="#1f77b4")
    ax.set_xlabel("Number of layers $L$")
    ax.set_ylabel("Max exact-reach distance $d^*$")
    ax.set_xticks(Ls)
    ax.set_title(f"Reachability wall vs depth (n={n_val}, path-union training)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    out = runs_dir / "reach_law.png"
    fig.savefig(out, dpi=170); plt.close(fig)
    print(f"\nsaved figure: {out}")


if __name__ == "__main__":
    main()
