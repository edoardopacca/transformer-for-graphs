"""Figures for the density-in-optimisation sweep (Report V). Reads the training
histories and OOD evals produced by scripts/density_sweep.sbatch under

    runs/report5/density_sweep/p{XX}/n{N}_er_roberta_linear_lam0_seed{S}/
        history.json                  (convergence + final in-dist exact)
        families/families_eval.json   (OOD family battery)

and plots, against the training density p: (1) convergence speed (steps to reach
0.99 in-distribution exact match), (2) final in-distribution exact match, and (3)
OOD exact match on a few structured families. No GPU; run locally after pulling.

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

PTAG_RE = re.compile(r"p(\d+)$")
OOD_FAMS = ["2chains", "2cliques", "chain_plus", "path_union"]


def steps_to_threshold(steps, vals, thr=0.99):
    for s, v in zip(steps, vals):
        if v >= thr:
            return s
    return None


def load(root):
    # p -> list of dicts {seed, final_exact, steps99, ood{fam:exact}}
    groups = defaultdict(list)
    for hist_f in sorted(Path(root).glob("p*/*/history.json")):
        m = PTAG_RE.search(hist_f.parent.parent.name)
        if not m:
            continue
        p = int(m.group(1)) / 100.0
        seed = int(re.search(r"seed(\d+)", hist_f.parent.name).group(1))
        h = json.loads(hist_f.read_text())
        rec = {"seed": seed,
               "final_exact": h["val_exact"][-1] if h.get("val_exact") else np.nan,
               "steps99": steps_to_threshold(h.get("steps", []), h.get("val_exact", [])),
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

    # (1) convergence speed + (2) final in-dist exact
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for p in ps:
        s99 = [r["steps99"] for r in groups[p] if r["steps99"] is not None]
        fin = [r["final_exact"] for r in groups[p]]
        if s99:
            a1.scatter([p] * len(s99), s99, color="tab:blue", alpha=0.5, s=20)
            a1.scatter([p], [np.mean(s99)], color="tab:blue", marker="_", s=400)
        a2.scatter([p] * len(fin), fin, color="tab:purple", alpha=0.5, s=20)
        a2.scatter([p], [np.nanmean(fin)], color="tab:purple", marker="_", s=400)
    a1.set_xlabel("training density p"); a1.set_ylabel("steps to 0.99 in-dist exact")
    a1.set_title("convergence speed vs density"); a1.grid(alpha=0.3)
    a2.set_xlabel("training density p"); a2.set_ylabel("final in-dist exact match")
    a2.set_ylim(0, 1.02); a2.set_title("final in-dist accuracy vs density"); a2.grid(alpha=0.3)
    fig.suptitle("Density in optimisation (base model, ER-trained, linear, n=20): "
                 "does denser training help or hurt learning?")
    fig.tight_layout()
    f = out / "density_convergence.png"; fig.savefig(f, dpi=150); plt.close(fig); print("wrote", f)

    # (3) OOD exact per family vs p
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for fam in OOD_FAMS:
        ms, sds, xs = [], [], []
        for p in ps:
            vals = [r["ood"][fam] for r in groups[p] if fam in r["ood"]]
            if vals:
                xs.append(p); ms.append(np.mean(vals)); sds.append(np.std(vals))
        if xs:
            ax.errorbar(xs, ms, yerr=sds, marker="o", capsize=3, label=fam)
    ax.set_xlabel("training density p"); ax.set_ylabel("OOD exact match")
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("OOD generalisation vs training density (base model, n=20)")
    fig.tight_layout()
    f = out / "density_ood.png"; fig.savefig(f, dpi=150); plt.close(fig); print("wrote", f)


if __name__ == "__main__":
    main()
