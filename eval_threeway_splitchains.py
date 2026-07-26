"""Report IX, Thread A.3 -- test a checkpoint trained EXCLUSIVELY on 2-component splits
(e.g. the split_chains-only n=46 seed-1000 training of this report) on graphs built by taking
that same 2-way split and removing one internal edge from one of the two paths, producing 3
disjoint path components with sizes the training stream never produced by construction (unlike
the three-component tests of Reports VII/VIII, whose training distribution already covered 3-4
components). Every distinct combination of resulting component sizes (every partition of n into
3 positive parts) is tested -- not just a canonical "one long + two small" shape -- since the
point is to see for WHICH size combinations the model still succeeds. Eval-only, no attention/
geometry probes here (those come after inspecting these numbers, on whichever cells turn out
interesting).

    python eval_threeway_splitchains.py --checkpoint runs/.../last.pt \\
        --output_dir runs/report9/threeway_splitchains/<tag>
"""
import argparse, csv
from pathlib import Path

import numpy as np
import torch

from data import (add_self_loops, compute_connectivity_matrix, generate_multi_path_split_graph)
from eval_families import load_model, predict


def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def all_three_way_sizes(n, min_size=1):
    """Every distinct triple (s1<=s2<=s3) of positive ints summing to n -- i.e. every
    partition of n into exactly 3 parts, each >= min_size."""
    out = []
    for s1 in range(min_size, n // 3 + 1):
        for s2 in range(s1, (n - s1) // 2 + 1):
            s3 = n - s1 - s2
            if s3 >= s2:
                out.append((s1, s2, s3))
    return out


def eval_cell(model, dev, n, sizes, rng, n_graphs):
    base_adj = generate_multi_path_split_graph(n, sizes)
    base_y = compute_connectivity_matrix(base_adj).astype(np.int8)

    bounds = [0]
    for s in sizes:
        bounds.append(bounds[-1] + s)
    comps = [np.arange(bounds[i], bounds[i + 1]) for i in range(3)]

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
    ng = n_graphs

    exact = float(eq.reshape(ng, -1).all(1).mean())

    def within(idx):
        if len(idx) <= 1:
            return None
        e = eq[:, idx][:, :, idx]
        off = ~np.eye(len(idx), dtype=bool)
        return float(e[:, off].mean())

    def cross(idx_a, idx_b):
        e = eq[:, idx_a][:, :, idx_b]
        e2 = eq[:, idx_b][:, :, idx_a]
        flat = np.concatenate([e.reshape(ng, -1), e2.reshape(ng, -1)], axis=1)
        return float(flat.mean())

    reach = [within(c) for c in comps]
    cut_12 = cross(comps[0], comps[1])
    cut_13 = cross(comps[0], comps[2])
    cut_23 = cross(comps[1], comps[2])

    return {"s1": sizes[0], "s2": sizes[1], "s3": sizes[2], "n_graphs": ng,
            "exact": round(exact, 4),
            "reach_1": (None if reach[0] is None else round(reach[0], 4)),
            "reach_2": (None if reach[1] is None else round(reach[1], 4)),
            "reach_3": (None if reach[2] is None else round(reach[2], 4)),
            "cut_12": round(cut_12, 4), "cut_13": round(cut_13, 4), "cut_23": round(cut_23, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=200)
    ap.add_argument("--min_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")
    rng = np.random.default_rng(args.seed)

    sizes_list = all_three_way_sizes(n, args.min_size)
    print(f"testing {len(sizes_list)} distinct 3-way size combinations (partitions of {n})")

    rows = []
    for sizes in sizes_list:
        c = eval_cell(model, dev, n, sizes, rng, args.n_graphs)
        rows.append(c)
        print(f"  sizes={sizes} exact={c['exact']:.3f} "
              f"reach=({c['reach_1']},{c['reach_2']},{c['reach_3']}) "
              f"cut=({c['cut_12']},{c['cut_13']},{c['cut_23']})", flush=True)

    with (out / "threeway_split.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved -> {out}/threeway_split.csv")


if __name__ == "__main__":
    main()
