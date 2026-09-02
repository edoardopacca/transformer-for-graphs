"""Report X -- 2026-09-01, Edoardo: OOD test of the path_union_k3 5-seed checkpoints (the
ones behind the paper's route-redundancy section, trained on RANDOM 3-way splits of n=60,
never on this exact shape) against the theta_19_20_19 graph -- disjoint sizes (19,20,19)
stitched into a hub pair (routes 19/20/20, same construction as Construction B but with
different sizes) plus 2 isolated padding nodes. We had already checked Construction B
(sizes (20,20,20)) OOD on these checkpoints; this is the analogous check for the
theta_19_20_19 shape, which so far had only been used as a from-scratch TRAINING target
(new dedicated family), never as an OOD test on the path_union_k3 checkpoints.

Ground truth is exact and needs no assumption about "everything in one component": the
main 58-node piece is one connected component (same reasoning as Construction B), the 2
isolated nodes are disconnected from it and from each other -- compute_connectivity_matrix
gets this right automatically, so real per-pair accuracy (not a mean-Z-heatmap) is used
throughout, same methodology as eval_threeway_perpair.py / eval_redundant_mix_indist.py.

    python eval_theta192019_ood.py --checkpoint runs/.../last.pt \\
        --output_dir runs/report10/theta192019_ood/<tag> --n_graphs 300
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import add_self_loops, compute_connectivity_matrix, generate_stitched_theta_graph
from eval_families import load_model
from stagewise_diagnostics import run_with_stages, _device, _selftest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")
    _selftest(model, dev, n)

    sizes = (19, 20, 19)
    base_adj = generate_stitched_theta_graph(n, sizes, n_isolated=2)
    true_conn = compute_connectivity_matrix(base_adj).astype(bool)
    hub1, hub2 = sizes[0], sizes[0] + sizes[1] - 1  # 19, 38 (0-indexed)
    isolated = [n - 2, n - 1]  # 58, 59
    main_nodes = [i for i in range(n) if i not in isolated]

    target_mask = np.zeros((n, n), dtype=bool); target_mask[hub1, hub2] = True
    main_other_mask = np.zeros((n, n), dtype=bool)
    for i in main_nodes:
        for j in main_nodes:
            if i < j:
                main_other_mask[i, j] = True
    main_other_mask &= ~target_mask
    iso_cross_mask = np.zeros((n, n), dtype=bool)
    for i in main_nodes:
        for j in isolated:
            iso_cross_mask[min(i, j), max(i, j)] = True
    iso_pair_mask = np.zeros((n, n), dtype=bool)
    iso_pair_mask[isolated[0], isolated[1]] = True

    rng = np.random.default_rng(args.seed)
    correct_sum = np.zeros((n, n), dtype=np.float64)
    exact_count = 0
    done = 0
    B = 256
    while done < args.n_graphs:
        g = min(B, args.n_graphs - done)
        xs = np.empty((g, n, n), np.float32)
        invs = []
        for j in range(g):
            p = rng.permutation(n)
            xs[j] = add_self_loops(base_adj[np.ix_(p, p)])
            invs.append(np.argsort(p))
        xb = torch.from_numpy(xs).to(dev, torch.float32)
        with torch.no_grad():
            _, _, logits = run_with_stages(model, xb)
        for j in range(g):
            logit_orig = logits[j][np.ix_(invs[j], invs[j])]
            pred = logit_orig > 0
            eq = pred == true_conn
            correct_sum += eq
            exact_count += int(eq.all())
        done += g
        print(f"  {done}/{args.n_graphs} done", flush=True)

    rate = correct_sum / done

    def cat(mask):
        vals = rate[mask]
        return (float(vals.mean()) if vals.size else None), int(mask.sum())

    target_acc, _ = cat(target_mask)
    main_other_acc, main_other_n = cat(main_other_mask)
    iso_cross_acc, iso_cross_n = cat(iso_cross_mask)
    iso_pair_acc, iso_pair_n = cat(iso_pair_mask)

    summary = {
        "n_graphs": done,
        "overall_exact_match": exact_count / done,
        "target_pair_accuracy": target_acc,
        "target_pair": [hub1, hub2],
        "main_other_accuracy": main_other_acc, "main_other_n": main_other_n,
        "isolated_cross_accuracy": iso_cross_acc, "isolated_cross_n": iso_cross_n,
        "isolated_pair_accuracy": iso_pair_acc, "isolated_pair_n": iso_pair_n,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
