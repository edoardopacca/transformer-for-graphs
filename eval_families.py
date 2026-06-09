"""Evaluate ONE connectivity checkpoint on many graph FAMILIES and break the
accuracy down three ways:

  (1) pairwise accuracy vs shortest-path distance d   -> the usual capacity plot
  (2) exact-match + pairwise accuracy vs graph DIAMETER
  (3) exact-match + pairwise accuracy vs SPECTRAL GAP (normalised-Laplacian)

per family and aggregated over all families. Works for both architectures
(GraphConnectivityTransformer / RobertaGraphTransformer) and both read-outs
(linear / similarity); the class and read-out are auto-detected from the
checkpoint's state-dict so the same script runs on the older reach/roberta
checkpoints and on the new ones.

Light enough to run locally (MPS) or on HPC (GPU). The checkpoints (.pt) live on
HPC, so in practice this is launched there via sbatch, once per finished run.

    python eval_families.py --checkpoint runs/.../last.pt --output_dir runs/.../families
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
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import (add_self_loops, compute_connectivity_matrix,
                  compute_all_pairs_shortest_paths, compute_spectral_gap,
                  generate_er_graph, generate_two_chains_graph,
                  generate_two_cliques_graph, generate_one_cycle_graph,
                  generate_two_cycles_graph, generate_one_chain_graph,
                  generate_path_union_graph, generate_blocks_graph,
                  generate_barbell_graph, generate_random_regular_graph,
                  generate_chain_plus_graph)
from model import GraphConnectivityTransformer, RobertaGraphTransformer, ModelConfig

# All families we test on. The 'er' density mirrors the n=40 study; at n=20 a
# single ER(20,0.08) graph is the in-distribution family for Set A.
ALL_FAMILIES = ["er", "er_blocks", "clique_blocks", "path_union", "2chains",
                "2cliques", "1cycle", "2cycle", "1chain", "chain_plus", "barbell",
                "barbell_var", "expander", "expander_var"]
# Log-spaced spectral-gap bin edges; finer below 0.01 to resolve the bottleneck
# regime swept by barbell_var (gaps ~0.001 .. ~0.3).
GAP_EDGES = np.array([0.0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.6, 1.01, 2.01])


def _gen(kind, n, rng):
    if kind == "er":            return generate_er_graph(n, 0.08, rng)
    if kind == "er_blocks":     return generate_blocks_graph(n, rng, "er")
    if kind == "clique_blocks": return generate_blocks_graph(n, rng, "clique")
    if kind == "path_union":    return generate_path_union_graph(n, rng, 4)
    if kind == "2chains":       return generate_two_chains_graph(n, n // 2)
    if kind == "2cliques":      return generate_two_cliques_graph(n, n // 2)
    if kind == "1cycle":        return generate_one_cycle_graph(n)
    if kind == "2cycle":        return generate_two_cycles_graph(n, n // 2)
    if kind == "1chain":        return generate_one_chain_graph(n)
    if kind == "chain_plus":    return generate_chain_plus_graph(n, rng)
    if kind == "barbell":       return generate_barbell_graph(n, rng)
    if kind == "barbell_var":   return generate_barbell_graph(n, rng,
                                       clique_size=int(rng.integers(2, n // 2 + 1)))
    if kind == "expander":      return generate_random_regular_graph(n, rng, 3)
    if kind == "expander_var":  return generate_random_regular_graph(n, rng,
                                       degree=int(rng.integers(2, n // 2 + 1)))
    raise ValueError(kind)


def build_family(kind, n, size, seed):
    """Returns adj-with-loops, target, per-graph diameter, per-graph spectral gap,
    and the all-pairs distance tensor (for per-distance accuracy)."""
    rng = np.random.default_rng(seed)
    xs = np.empty((size, n, n), np.float32)
    ys = np.empty((size, n, n), np.int8)
    dist = np.empty((size, n, n), np.int64)
    diam = np.empty(size, np.int64)
    gap = np.empty(size, np.float64)
    for i in range(size):
        a = _gen(kind, n, rng)
        p = rng.permutation(n)
        a = a[np.ix_(p, p)]
        xs[i] = add_self_loops(a)
        ys[i] = compute_connectivity_matrix(a).astype(np.int8)
        d = compute_all_pairs_shortest_paths(a)
        dist[i] = d
        fin = d[d >= 0]
        diam[i] = int(fin.max()) if fin.size else 0
        gap[i] = compute_spectral_gap(a)
    return xs, ys, dist, diam, gap


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ck["model_config"]
    sd = ck["model_state_dict"]
    readout = "similarity" if any(k.endswith("sim_scale") for k in sd) else "linear"
    is_roberta = any("emb_ln" in k for k in sd)
    mcfg = ModelConfig(n=c["n"], d_model=c["d_model"], n_heads=c["n_heads"],
                       d_ff=c["d_ff"], n_layers=c["n_layers"],
                       attn_kind=c.get("attn_kind", "normalized_relu"),
                       norm_style=c.get("norm_style", "post" if is_roberta else "pre"),
                       dropout=0.0, readout=readout)
    Cls = RobertaGraphTransformer if is_roberta else GraphConnectivityTransformer
    m = Cls(mcfg).to(device)
    m.load_state_dict(sd)
    m.eval()
    arch = "roberta" if is_roberta else "minimal"
    return m, mcfg, arch, readout


@torch.no_grad()
def predict(model, xs, device, batch=256):
    ng, n, _ = xs.shape
    pred = np.empty((ng, n, n), np.int8)
    use_cuda = device.type == "cuda"
    for s in range(0, ng, batch):
        e = min(s + batch, ng)
        xb = torch.from_numpy(xs[s:e]).to(device, torch.float32)
        if use_cuda:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
        else:
            logits = model(xb)
        pred[s:e] = (logits > 0).cpu().numpy().astype(np.int8)
    return pred


def family_metrics(pred, ys, dist):
    ng, n, _ = pred.shape
    eq = (pred == ys)
    offdiag = ~np.eye(n, dtype=bool)[None].repeat(ng, 0)
    exact_pg = eq.reshape(ng, -1).all(1)                       # per-graph exact
    pw_pg = (eq & offdiag).reshape(ng, -1).sum(1) / offdiag.reshape(ng, -1).sum(1)
    # per shortest-path distance, on CONNECTED pairs (reach)
    conn = (dist > 0)
    per_dist = {}
    dmax = int(dist.max())
    for d in range(1, dmax + 1):
        m = conn & (dist == d)
        c = int(m.sum())
        if c >= 50:
            per_dist[d] = (float(eq[m].mean()), c)
    # cut quality: accuracy on across-component (target-0) pairs, separated from reach
    disc = offdiag & (dist == -1)
    disc_acc = float(eq[disc].mean()) if disc.any() else None
    conn_acc = float(eq[conn].mean()) if conn.any() else None
    return {
        "exact": float(exact_pg.mean()),
        "pairwise": float(pw_pg.mean()),
        "reach_acc": conn_acc,         # within-component (target-1) pairs
        "disc_acc": disc_acc,          # between-component (target-0) pairs
        "per_dist": per_dist,
        "_exact_pg": exact_pg,
        "_pw_pg": pw_pg,
    }


def bucket_by(values, exact_pg, pw_pg, integer=False, edges=None):
    """Group per-graph exact/pairwise by an integer key (diameter) or by bin (gap)."""
    out = {}
    if integer:
        keys = np.unique(values)
        for k in keys:
            m = values == k
            out[int(k)] = {"exact": float(exact_pg[m].mean()),
                           "pairwise": float(pw_pg[m].mean()), "count": int(m.sum())}
    else:
        idx = np.digitize(values, edges) - 1
        idx = np.clip(idx, 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            m = idx == b
            if m.any():
                out[b] = {"exact": float(exact_pg[m].mean()),
                          "pairwise": float(pw_pg[m].mean()), "count": int(m.sum()),
                          "lo": float(edges[b]), "hi": float(edges[b + 1])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--size", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--families", nargs="+", default=ALL_FAMILIES)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model, mcfg, arch, readout = load_model(args.checkpoint, device)
    n = mcfg.n
    print(f"loaded {arch}/{readout} n={n} L={mcfg.n_layers} from {args.checkpoint}")

    results = {"checkpoint": str(args.checkpoint), "arch": arch, "readout": readout,
               "n": n, "n_layers": mcfg.n_layers, "size": args.size, "families": {}}
    agg_exact, agg_pw, agg_diam, agg_gap = [], [], [], []
    agg_eq_by_d = {}   # d -> [correct, total]

    for fam in args.families:
        xs, ys, dist, diam, gap = build_family(fam, n, args.size, args.seed)
        pred = predict(model, xs, device)
        fm = family_metrics(pred, ys, dist)
        results["families"][fam] = {
            "exact": fm["exact"], "pairwise": fm["pairwise"],
            "reach_acc": fm["reach_acc"], "disc_acc": fm["disc_acc"],
            "per_dist": {str(k): list(v) for k, v in fm["per_dist"].items()},
            "by_diam": bucket_by(diam, fm["_exact_pg"], fm["_pw_pg"], integer=True),
            "by_gap": bucket_by(gap, fm["_exact_pg"], fm["_pw_pg"], edges=GAP_EDGES),
        }
        agg_exact.append(fm["_exact_pg"]); agg_pw.append(fm["_pw_pg"])
        agg_diam.append(diam); agg_gap.append(gap)
        for d, (acc, c) in fm["per_dist"].items():
            cur = agg_eq_by_d.setdefault(d, [0.0, 0]); cur[0] += acc * c; cur[1] += c
        da = "  n/a" if fm["disc_acc"] is None else f"{fm['disc_acc']:.3f}"
        ra = "  n/a" if fm["reach_acc"] is None else f"{fm['reach_acc']:.3f}"
        print(f"  {fam:14s} exact={fm['exact']:.3f} pairwise={fm['pairwise']:.3f} "
              f"reach={ra} disc={da}")

    agg_exact = np.concatenate(agg_exact); agg_pw = np.concatenate(agg_pw)
    agg_diam = np.concatenate(agg_diam); agg_gap = np.concatenate(agg_gap)
    results["aggregate"] = {
        "by_diam": bucket_by(agg_diam, agg_exact, agg_pw, integer=True),
        "by_gap": bucket_by(agg_gap, agg_exact, agg_pw, edges=GAP_EDGES),
        "per_dist": {str(d): [v[0] / v[1], v[1]] for d, v in sorted(agg_eq_by_d.items())},
    }
    (out / "families_eval.json").write_text(json.dumps(results, indent=2))
    _plots(results, out, n)
    print(f"saved -> {out}/families_eval.json (+ png)")


def _plots(results, out, n):
    fams = list(results["families"])
    # (1) capacity: pairwise vs shortest-path distance, one panel per family
    ncol = 4; nrow = int(np.ceil(len(fams) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for ax, fam in zip(axes.ravel(), fams):
        fd = results["families"][fam]
        pd = fd["per_dist"]
        ds = sorted(int(k) for k in pd)
        xticks, xlabels = [], []
        if ds:
            ax.bar(ds, [pd[str(d)][0] for d in ds], color="#3b6ea5")
            xticks = list(ds); xlabels = [str(d) for d in ds]
        # red "disconnected" bar (accuracy on across-component / target-0 pairs)
        disc = fd.get("disc_acc")
        if disc is not None:
            xd = (max(ds) + 2) if ds else 1
            ax.bar([xd], [disc], color="#c0392b")
            xticks.append(xd); xlabels.append("disc")
        ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_title(fam, fontsize=10); ax.set_ylim(0, 1.05)
        ax.set_xlabel("shortest-path d"); ax.set_ylabel("reach (pairwise)")
        ax.grid(axis="y", alpha=0.3)
    for ax in axes.ravel()[len(fams):]:
        ax.axis("off")
    fig.suptitle(f"Reach vs distance per family (n={n}, {results['arch']}/{results['readout']})")
    fig.tight_layout(); fig.savefig(out / "capacity_per_distance.png", dpi=150); plt.close(fig)

    # (2)+(3) aggregate exact+pairwise vs diameter and vs spectral gap
    for key, fname, xlabel, is_gap in [
            ("by_diam", "by_diameter.png", "graph diameter", False),
            ("by_gap", "by_spectral_gap.png", "spectral gap (normalised Laplacian)", True)]:
        agg = results["aggregate"][key]
        if is_gap:
            ks = sorted(agg, key=int)
            xs = [0.5 * (agg[k]["lo"] + agg[k]["hi"]) for k in ks]
            xticklab = [f"{agg[k]['lo']:.2g}-{agg[k]['hi']:.2g}" for k in ks]
        else:
            ks = sorted(agg, key=int); xs = [int(k) for k in ks]; xticklab = xs
        ex = [agg[k]["exact"] for k in ks]; pw = [agg[k]["pairwise"] for k in ks]
        cnt = [agg[k]["count"] for k in ks]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(range(len(xs)), ex, "o-", label="exact match", color="#c44")
        ax.plot(range(len(xs)), pw, "s-", label="pairwise", color="#36c")
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xticklab, rotation=45, ha="right")
        ax.set_ylim(0, 1.05); ax.set_xlabel(xlabel); ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3); ax.legend(loc="lower left")
        for i, c in enumerate(cnt):
            ax.text(i, 1.02, str(c), ha="center", va="bottom", fontsize=7, color="gray")
        ax.set_title(f"Accuracy vs {xlabel} (all families, n={n}, "
                     f"{results['arch']}/{results['readout']})")
        fig.tight_layout(); fig.savefig(out / fname, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
