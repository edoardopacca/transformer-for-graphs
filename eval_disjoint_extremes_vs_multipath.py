"""Report X -- direct comparison asked for after the first multipath result: on the SAME
n=60 K=3 (path_union_k3) checkpoint, is the model any better at "the two ends of a path are
connected" when that pair is queried (a) inside a disjoint 3-way split, the checkpoint's own
training distribution, vs (b) as the shared endpoints of a genuine multipath graph, a structure
the checkpoint never saw in training?

  (a) DISJOINT: generate_multi_path_split_graph(60, (20,20,20)) -- three disjoint 20-node
      paths. Target pair: each path's own two extremes (nodes 0&19, 20&39, 40&59 in
      canonical/unpermuted order) -- the single longest-distance (d=19) pair inside each
      component, pooled across the three components into one accuracy number (plus reported
      individually).
  (b) MULTIPATH: generate_multipath_graph(60, 3, [19,19,20], term_deg=4) -- two shared
      terminals s,t joined by 3 parallel routes of 19/19/20 edges (chosen to mirror (a)'s
      three ~20-node paths as closely as possible: a route of path_len 19 has 20 nodes
      total including s,t, exactly matching a disjoint 20-node path; the third route is
      path_len 20 purely so the construction uses all 60 nodes with only 1 filler node,
      term_deg=4 leaves included). Target pair: s,t themselves.

Both target pairs are the same kind of query (the longest-range pair the model must get right
for that structure to be "solved"), on the same underlying node budget, so a difference between
them isolates what changes when the SAME two-ends-of-a-path question is asked with vs without a
shared-endpoint / multi-route structure around it.

Distinct --seed per checkpoint is REQUIRED for a fair comparison (see report/10
sec:n60-multipath's methodological note: both base graphs here are deterministic given their
parameters, so two checkpoints given the same --seed would see identical relabelled graphs).
--n_repeats > 1 draws independent repeats (seed+0..n_repeats-1) and reports mean+-std.

Eval-only, CPU/GPU-friendly (forward passes only, eval_families.predict).

    python eval_disjoint_extremes_vs_multipath.py --checkpoint runs/.../last.pt \\
        --output_dir runs/report10/disjoint_vs_multipath/<tag> --seed 1000 --n_repeats 5
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from data import (add_self_loops, compute_connectivity_matrix,
                  generate_multi_path_split_graph, generate_multipath_graph)
from eval_families import load_model, predict


def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def _eval_pairs(model, dev, n, base_adj, pairs, rng, n_graphs):
    """pairs: list of (i,j) node-index tuples (canonical/unpermuted order), target=1
    (connected) for all of them. Returns per-pair accuracy list + pooled accuracy, plus
    whole-graph reach/cut for context."""
    base_y = compute_connectivity_matrix(base_adj).astype(np.int8)
    xs = np.empty((n_graphs, n, n), np.float32)
    invs = []
    for i in range(n_graphs):
        p = rng.permutation(n)
        xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
        invs.append(np.argsort(p))
    pred_perm = predict(model, xs, dev)
    pred = np.empty_like(pred_perm)
    for i, inv in enumerate(invs):
        pred[i] = pred_perm[i][np.ix_(inv, inv)]
    eq = (pred == base_y[None])

    per_pair = [float((pred[:, i, j] == 1).mean()) for (i, j) in pairs]
    pooled = float(np.mean([pred[:, i, j] for (i, j) in pairs]))
    overall_pairwise_acc = float(eq.reshape(n_graphs, -1).mean())
    return {"per_pair_predicted_connected": [round(v, 4) for v in per_pair],
            "pooled_predicted_connected": round(pooled, 4),
            "overall_pairwise_accuracy": round(overall_pairwise_acc, 4)}


def eval_disjoint(model, dev, n, sizes, rng, n_graphs):
    base_adj = generate_multi_path_split_graph(n, sizes)
    bounds = [0]
    for s in sizes:
        bounds.append(bounds[-1] + s)
    extreme_pairs = [(bounds[i], bounds[i + 1] - 1) for i in range(len(sizes))]
    r = _eval_pairs(model, dev, n, base_adj, extreme_pairs, rng, n_graphs)
    r["sizes"] = list(sizes); r["extreme_pairs"] = extreme_pairs
    return r


def eval_multipath(model, dev, n, path_lens, term_deg, rng, n_graphs):
    built = generate_multipath_graph(n, len(path_lens), path_lens, rng, term_deg=term_deg)
    if built is None:
        raise ValueError(f"path_lens={path_lens}, term_deg={term_deg} does not fit n={n}")
    base_adj, meta = built
    s, t = meta["s"], meta["t"]
    r = _eval_pairs(model, dev, n, base_adj, [(s, t)], rng, n_graphs)
    r["path_lens"] = path_lens; r["s_t"] = (s, t)
    r["n_nodes_used"] = 2 + sum(len(p) for p in meta["full_paths"]) + len(meta["leaves"])
    r["n_filler"] = len(meta["filler"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=300)
    ap.add_argument("--disjoint_sizes", type=int, nargs=3, default=[20, 20, 20])
    ap.add_argument("--multipath_lens", type=int, nargs=3, default=[19, 19, 20])
    ap.add_argument("--term_deg", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345,
                    help="base seed; MUST differ across checkpoints for an independent "
                         "comparison (both base graphs here are deterministic)")
    ap.add_argument("--n_repeats", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev} "
          f"seed_base={args.seed} n_repeats={args.n_repeats}")

    disjoint_reps, multipath_reps = [], []
    for rep in range(args.n_repeats):
        rng = np.random.default_rng(args.seed + rep)
        rd = eval_disjoint(model, dev, n, tuple(args.disjoint_sizes), rng, args.n_graphs)
        rm = eval_multipath(model, dev, n, list(args.multipath_lens), args.term_deg, rng, args.n_graphs)
        disjoint_reps.append(rd); multipath_reps.append(rm)
        print(f"  repeat={rep} DISJOINT {args.disjoint_sizes} extreme-pairs pooled="
              f"{rd['pooled_predicted_connected']:.3f} per-pair={rd['per_pair_predicted_connected']}")
        print(f"  repeat={rep} MULTIPATH lens={args.multipath_lens} s-t predicted-connected="
              f"{rm['pooled_predicted_connected']:.3f}", flush=True)

    def agg(key, reps):
        vals = [r[key] for r in reps]
        return float(np.mean(vals)), float(np.std(vals))

    d_mean, d_std = agg("pooled_predicted_connected", disjoint_reps)
    m_mean, m_std = agg("pooled_predicted_connected", multipath_reps)
    print(f"\nAGGREGATE over {args.n_repeats} repeat(s):")
    print(f"  disjoint {args.disjoint_sizes} own-path extremes : {d_mean:.4f} +- {d_std:.4f}")
    print(f"  multipath lens={args.multipath_lens} s-t         : {m_mean:.4f} +- {m_std:.4f}")

    result = {"checkpoint": str(args.checkpoint), "n": n, "n_graphs": args.n_graphs,
              "n_repeats": args.n_repeats, "seed_base": args.seed,
              "disjoint_sizes": args.disjoint_sizes, "multipath_lens": args.multipath_lens,
              "disjoint_pooled_mean": round(d_mean, 4), "disjoint_pooled_std": round(d_std, 4),
              "multipath_pooled_mean": round(m_mean, 4), "multipath_pooled_std": round(m_std, 4),
              "disjoint_per_repeat": disjoint_reps, "multipath_per_repeat": multipath_reps}
    (out / "disjoint_vs_multipath.json").write_text(json.dumps(result, indent=2))
    print(f"\nsaved -> {out}/disjoint_vs_multipath.json")


if __name__ == "__main__":
    main()
