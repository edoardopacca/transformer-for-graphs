"""Figures for the density-in-optimisation sweep (Report V, Section 5.3).

Reads the training histories and OOD evals produced by scripts/density_sweep.sbatch:

    runs/report5/density_sweep/p{XX}/n{N}_er_roberta_linear_lam0_seed{S}/
        history.json                  (convergence + final in-dist exact)
        families/families_eval.json   (OOD family battery)

The DFS hypothesis predicts denser TRAINING should make optimisation harder/slower;
matrix powering (shorter distances when dense) predicts the opposite. We plot, against
the training ER edge probability p:

  density_convergence.png : (A) in-dist exact-match trajectories vs step, per-seed +
                            mean, coloured by p; (B) steps to reach 0.99 in-dist exact
                            vs p, per seed (non-converging seeds marked at the ceiling);
                            (C) final in-dist exact vs p, per seed.
  density_ood.png         : OOD exact-match vs p, per seed, for the structured families
                            that actually move (noisy, seed-dominated).

No GPU; run locally after pulling:

  python plot_density_sweep.py --root runs/report5/density_sweep \
      --output_dir runs/report5/report5_figs
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

PTAG_RE = re.compile(r"p(\d+)$")
# families that actually move with p (the all-0 / all-1 ones are uninformative)
OOD_FAMS = ["1cycle", "path_union", "chain_plus"]
SEED_CEILING = 1_050_000  # where to draw "never reached 0.99" markers


def steps_to_threshold(steps, vals, thr=0.99):
    for s, v in zip(steps, vals):
        if v >= thr:
            return s
    return None


def load(root):
    # p -> list of dicts {seed, steps, val_exact, final, steps99, ood{fam:exact}}
    groups = defaultdict(list)
    for hist_f in sorted(Path(root).glob("p*/*/history.json")):
        m = PTAG_RE.search(hist_f.parent.parent.name)
        if not m:
            continue
        p = int(m.group(1)) / 100.0
        seed = int(re.search(r"seed(\d+)", hist_f.parent.name).group(1))
        h = json.loads(hist_f.read_text())
        steps = h.get("steps", [])
        ve = h.get("val_exact", [])
        rec = {"seed": seed, "steps": steps, "val_exact": ve,
               "final": ve[-1] if ve else np.nan,
               "steps99": steps_to_threshold(steps, ve, 0.99),
               "ood": {}}
        fe = hist_f.parent / "families" / "families_eval.json"
        if fe.exists():
            fam = json.loads(fe.read_text())["families"]
            rec["ood"] = {k: fam[k]["exact"] for k in OOD_FAMS if k in fam}
        groups[p].append(rec)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/report5/density_sweep")
    ap.add_argument("--output_dir", default="runs/report5/report5_figs")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    groups = load(args.root)
    if not groups:
        print(f"no histories under {args.root}"); return
    ps = sorted(groups)
    print("densities:", {p: len(groups[p]) for p in ps})
    colors = {p: cm.viridis(i / max(1, len(ps) - 1)) for i, p in enumerate(ps)}

    # ---- Figure 1: convergence ---------------------------------------------
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15, 4.4))

    # (A) in-dist exact trajectories
    for p in ps:
        c = colors[p]
        for r in groups[p]:
            a0.plot(r["steps"], r["val_exact"], color=c, alpha=0.18, lw=0.8)
        # mean trajectory on the common grid
        mat = np.array([r["val_exact"] for r in groups[p]])
        a0.plot(groups[p][0]["steps"], mat.mean(0), color=c, lw=2.4,
                label=f"p={p:.2f}")
    a0.set_xscale("log")
    a0.set_xlabel("training step"); a0.set_ylabel("in-dist exact match")
    a0.set_ylim(0, 1.02); a0.grid(alpha=0.3); a0.legend(fontsize=8, loc="lower right")
    a0.set_title("(A) convergence trajectory: denser climbs faster")

    # (B) steps to 0.99
    for p in ps:
        got = [r["steps99"] for r in groups[p] if r["steps99"] is not None]
        miss = [r for r in groups[p] if r["steps99"] is None]
        if got:
            a1.scatter([p] * len(got), got, color=colors[p], s=42, zorder=3,
                       edgecolor="k", linewidth=0.4)
            a1.scatter([p], [np.mean(got)], color="k", marker="_", s=520, zorder=4)
        if miss:
            a1.scatter([p] * len(miss), [SEED_CEILING] * len(miss),
                       color=colors[p], marker="x", s=55, zorder=3)
    a1.axhline(SEED_CEILING, color="grey", ls=":", lw=0.8)
    a1.text(ps[-1], SEED_CEILING, " never reached\n 0.99", va="center",
            fontsize=7, color="grey")
    a1.set_yscale("log")
    a1.set_xlabel("training density p")
    a1.set_ylabel("steps to 0.99 in-dist exact")
    a1.grid(alpha=0.3)
    a1.set_title("(B) convergence speed (× = never converged)")

    # (C) final in-dist exact
    for p in ps:
        fin = [r["final"] for r in groups[p]]
        a2.scatter([p] * len(fin), fin, color=colors[p], s=42, zorder=3,
                   edgecolor="k", linewidth=0.4)
        a2.scatter([p], [np.nanmean(fin)], color="k", marker="_", s=520, zorder=4)
    a2.set_xlabel("training density p"); a2.set_ylabel("final in-dist exact match")
    a2.set_ylim(0.9, 1.005); a2.grid(alpha=0.3)
    a2.set_title("(C) final in-dist accuracy (sparse seeds straggle)")

    fig.suptitle("Density in optimisation (base RoBERTa, linear read-out, n=20, ER-trained, 4 seeds/p): "
                 "denser training is faster and more reliable, not slower",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    f = out / "density_convergence.png"; fig.savefig(f, dpi=150); plt.close(fig); print("wrote", f)

    # ---- Figure 2: OOD transfer (noisy, per-seed) --------------------------
    fig, axes = plt.subplots(1, len(OOD_FAMS), figsize=(4.6 * len(OOD_FAMS), 4.2),
                             sharey=True)
    for ax, fam in zip(axes, OOD_FAMS):
        for p in ps:
            vals = [r["ood"][fam] for r in groups[p] if fam in r["ood"]]
            if vals:
                ax.scatter([p] * len(vals), vals, color="tab:gray", alpha=0.55, s=30)
                ax.scatter([p], [np.mean(vals)], color="tab:red", marker="_", s=320,
                           zorder=4)
        ax.set_xlabel("training density p"); ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3); ax.set_title(fam)
    axes[0].set_ylabel("OOD exact match")
    fig.suptitle("OOD transfer vs training density (base RoBERTa, linear, n=20, per seed): "
                 "noisy and seed-dominated, at most a weak peak near p=0.12",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    f = out / "density_ood.png"; fig.savefig(f, dpi=150); plt.close(fig); print("wrote", f)


if __name__ == "__main__":
    main()
