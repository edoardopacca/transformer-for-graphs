"""Per-distance reachability of the depth-sweep models (run on HPC: GPU + last.pt).

For each runs/report3/reach_depth/*/last.pt this rebuilds the path-union test set and
computes, per shortest-path distance d, the fraction of within-component pairs
predicted connected ("reach at d") together with the number of such pairs, plus
the in-distribution (path-union) exact-match accuracy. It writes a per-run
reach_by_distance.json and a single combined figure with one panel per depth L
(reach vs d, pair counts above the bars) --- the analogue of the 2Chain/2Clique
per-distance figures, which is far more informative than a single d* number.

    python eval_reach_by_distance.py [--runs_dir runs/report3/reach_depth] [--num_workers 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import GraphConnectivityTransformer, ModelConfig
from experiments2.train_reach_depth import build_test


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ck["model_config"]
    mcfg = ModelConfig(n=c["n"], d_model=c["d_model"], n_heads=c["n_heads"],
                       d_ff=c["d_ff"], n_layers=c["n_layers"],
                       attn_kind=c.get("attn_kind", "normalized_relu"))
    m = GraphConnectivityTransformer(mcfg).to(device); m.load_state_dict(ck["model_state_dict"])
    m.eval(); return m, mcfg, int(ck.get("step", 0))


@torch.no_grad()
def per_distance(model, tx, ty, td, device, batch=256):
    ng, n, _ = tx.shape
    pred = np.empty((ng, n, n), np.int8)
    for s in range(0, ng, batch):
        e = min(s + batch, ng)
        xb = torch.from_numpy(tx[s:e]).to(device, torch.float32)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xb)
        pred[s:e] = (logits > 0).cpu().numpy().astype(np.int8)
    tgt = ty.astype(np.int8); eq = pred == tgt
    exact = float(eq.reshape(ng, -1).all(1).mean())
    off = np.broadcast_to(~np.eye(n, dtype=bool)[None], (ng, n, n))
    conn = off & (tgt == 1); disc = off & (tgt == 0)
    disc_acc = float(eq[disc].mean()) if disc.any() else float("nan")
    per = {}
    for d in range(1, int(td.max()) + 1):
        m = conn & (td == d); c = int(m.sum())
        if c > 0:
            per[d] = (float(eq[m].mean()), c)
    return {"exact": exact, "disc_acc": disc_acc, "per_dist": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs/report3/reach_depth")
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"device: {device}")

    runs = sorted(p for p in Path(args.runs_dir).iterdir()
                  if p.is_dir() and (p / "last.pt").exists())
    cache = {}; results = []
    for d in runs:
        model, mcfg, step = load_model(str(d / "last.pt"), device)
        n, L = mcfg.n, mcfg.n_layers
        if n not in cache:
            cache[n] = build_test(n, args.num_workers, seed=1000)
        tx, ty, td = cache[n]
        r = per_distance(model, tx, ty, td, device); del model
        r.update({"L": L, "n": n, "step": step, "dir": d.name})
        json.dump({"L": L, "n": n, "step": step, "exact": r["exact"],
                   "disc_acc": r["disc_acc"],
                   "per_dist": {str(k): {"reach": v[0], "count": v[1]}
                                for k, v in r["per_dist"].items()}},
                  (d / "reach_by_distance.json").open("w"), indent=2)
        results.append(r)
        print(f"  L={L} step={step} exact={r['exact']:.3f} disc={r['disc_acc']:.3f}")

    results.sort(key=lambda r: r["L"])
    ncols = len(results)
    fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 4.3), sharey=True,
                             squeeze=False)
    for ax, r in zip(axes[0], results):
        ds = sorted(r["per_dist"]); vals = [r["per_dist"][x][0] for x in ds]
        cnts = [r["per_dist"][x][1] for x in ds]
        bars = ax.bar([str(x) for x in ds], vals, color="#1f77b4")
        for b, c in zip(bars, cnts):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{c//1000}k" if c >= 1000 else str(c),
                    ha="center", va="bottom", fontsize=6, rotation=90)
        cap = 3 ** r["L"]
        if cap <= max(ds):
            ax.axvline(cap - 0.5, color="red", ls="--", lw=1.2)
            ax.text(cap - 0.5, 1.02, f" $3^{r['L']}$={cap}", color="red", fontsize=8)
        ax.set_title(f"L={r['L']}  (in-dist exact = {r['exact']:.2f})")
        ax.set_xlabel("within-component distance $d$")
        ax.set_ylim(0, 1.08); ax.grid(axis="y", alpha=0.3)
        for i, lbl in enumerate(ax.get_xticklabels()):
            lbl.set_visible(i % 4 == 0)
    axes[0][0].set_ylabel("reach (pairwise acc on connected pairs)")
    fig.suptitle("Depth sweep (n=64, path-union): reach by shortest-path distance, per depth L",
                 y=1.02)
    fig.tight_layout()
    out = Path(args.runs_dir) / "reach_by_distance.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"\nsaved figure: {out}")


if __name__ == "__main__":
    main()
