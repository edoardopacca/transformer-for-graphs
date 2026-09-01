"""Report X -- 2026-09-01, Edoardo: the redundant_mix training job (r10redmix, seed 1000,
job 643495) TIMED OUT at the 6h wall -- train_families_n20.py only calls save_json(history)
AFTER its training loop finishes normally, so a SLURM-killed run leaves no history.json at
all (last.pt/best.pt are still written every eval_every, so those are usable). The stdout
log itself (out/r10redmix_643495.out) shows val_exact FROZEN at exactly 0.4942 from the very
first eval point (step 5000) through the last (step 585000, 117 prints), while train loss
kept slowly decreasing the whole time -- i.e. once the model's sign predictions on the fixed
5000-graph val set stabilized, they never flipped again, only got more confident. Since
that val set is a 50/50 mix of two DETERMINISTIC shapes (only the node permutation is
random -- neither generate_stitched_theta_graph nor generate_multipath_graph(term_deg=2,
n_full=2) use rng for structure), a natural hypothesis for why 0.4942 is so suspiciously
exact-frozen is that the model solves one shape ~always and the other ~never, and 0.5 splits
by construction. This script tests that directly: for EACH shape separately (not mixed),
report the real per-pair accuracy broken down by pair category --
  theta:    target = the hub pair (dist 19, the whole point of Construction B), other = every
            other pair (whole graph is one component once stitched)
  twopath:  target = s,t (dist 20), route_other = other pairs within the s/t/route
            component, cross = route component vs the separate filler chain (this is the
            over-connection failure mode found earlier with the deleted Constructions E/F --
            worth checking again here since it's now IN the training distribution, not OOD)
plus whole-graph exact match per shape, to see whether the 0.4942 split really is shape-level.

    python eval_redundant_mix_indist.py --checkpoint runs/.../last.pt \\
        --output_dir runs/report10/redundant_mix_indist/<tag> --n_graphs 500
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import (add_self_loops, compute_connectivity_matrix,
                   generate_stitched_theta_graph, generate_multipath_graph)
from eval_families import load_model
from stagewise_diagnostics import run_with_stages, _device, _selftest


def build_theta(n=60):
    adj = generate_stitched_theta_graph(n, (20, 20, 20))
    return adj, (20, 39)


def build_twopath(n=60):
    dummy_rng = np.random.default_rng(0)  # unused inside (structure is deterministic)
    adj, meta = generate_multipath_graph(n, 2, [20, 20], dummy_rng, term_deg=2, fill=True)
    route_nodes = sorted({meta["s"], meta["t"], *meta["full_paths"][0], *meta["full_paths"][1]})
    filler_nodes = list(meta["filler"])
    return adj, (meta["s"], meta["t"]), route_nodes, filler_nodes


def pair_mask(n, idx_a, idx_b=None):
    """Boolean upper-triangle (i<j) mask over pairs from idx_a x idx_b (idx_b=None -> idx_a x idx_a)."""
    m = np.zeros((n, n), dtype=bool)
    if idx_b is None:
        for i in idx_a:
            for j in idx_a:
                if i < j:
                    m[i, j] = True
    else:
        for i in idx_a:
            for j in idx_b:
                if i != j:
                    m[min(i, j), max(i, j)] = True
    return m


def run_shape(model, dev, n, base_adj, n_graphs, seed, batch=256):
    true_conn = compute_connectivity_matrix(base_adj).astype(bool)
    rng = np.random.default_rng(seed)
    correct_sum = np.zeros((n, n), dtype=np.float64)
    exact_count = 0
    done = 0
    while done < n_graphs:
        g = min(batch, n_graphs - done)
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
        print(f"    {done}/{n_graphs} done", flush=True)
    return correct_sum / done, exact_count / done, done


def cat_acc(correct_rate, mask):
    vals = correct_rate[mask]
    return float(vals.mean()) if vals.size else None, int(mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=500, help="per shape")
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")
    _selftest(model, dev, n)

    summary = {}

    print("== theta shape ==")
    adj_th, (ta, tb) = build_theta(n)
    rate_th, exact_th, ng_th = run_shape(model, dev, n, adj_th, args.n_graphs, args.seed)
    target_mask = np.zeros((n, n), dtype=bool); target_mask[min(ta, tb), max(ta, tb)] = True
    other_mask = pair_mask(n, list(range(n))) & ~target_mask
    t_acc, t_n = cat_acc(rate_th, target_mask)
    o_acc, o_n = cat_acc(rate_th, other_mask)
    summary["theta_exact_match"] = exact_th
    summary["theta_n_graphs"] = ng_th
    summary["theta_target_pair_accuracy"] = t_acc
    summary["theta_other_pairs_accuracy"] = o_acc
    summary["theta_other_pairs_n"] = o_n

    print("== twopath shape ==")
    adj_tp, (sa, sb), route_nodes, filler_nodes = build_twopath(n)
    rate_tp, exact_tp, ng_tp = run_shape(model, dev, n, adj_tp, args.n_graphs, args.seed + 1)
    target_mask = np.zeros((n, n), dtype=bool); target_mask[min(sa, sb), max(sa, sb)] = True
    route_other_mask = pair_mask(n, route_nodes) & ~target_mask
    cross_mask = pair_mask(n, route_nodes, filler_nodes)
    filler_mask = pair_mask(n, filler_nodes)
    tgt_acc, _ = cat_acc(rate_tp, target_mask)
    ro_acc, ro_n = cat_acc(rate_tp, route_other_mask)
    cr_acc, cr_n = cat_acc(rate_tp, cross_mask)
    fi_acc, fi_n = cat_acc(rate_tp, filler_mask)
    summary["twopath_exact_match"] = exact_tp
    summary["twopath_n_graphs"] = ng_tp
    summary["twopath_target_pair_accuracy"] = tgt_acc
    summary["twopath_route_other_accuracy"] = ro_acc
    summary["twopath_route_other_n"] = ro_n
    summary["twopath_cross_component_accuracy"] = cr_acc
    summary["twopath_cross_component_n"] = cr_n
    summary["twopath_filler_internal_accuracy"] = fi_acc
    summary["twopath_filler_internal_n"] = fi_n

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
