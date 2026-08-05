"""Report IX -- prof's request (2026-07-27 email, verbatim): "mi ricordo che avevi
fatto degli esperimenti su grafi con paths parallele, e la presenza di piu paths
sembrava rendere l'accuratezza migliore rispetto al caso con singola path. Potresti
visualizzare gli attention scores imparati in questi casi? [...] bisognerebbe
confrontare con training sugli stessi grafi."

Report VI Thread A found that, at a fixed beyond-capacity distance, adding parallel
routes between two terminals s,t rescues the connection (k=1 fails, k=2/3 succeed) --
but never looked at the real attention weights, only aggregate accuracy/leak
statistics. This script builds ONE canonical multipath graph per k (deterministic
given n/k/ell/term_deg -- data.py's generate_multipath_graph) and dumps the real
attention score/alpha matrices (layer 0 and 1), averaged over many independent
random relabellings mapped back to base node order, exactly like every other
heatmap script in this project (mechanistic_heatmaps.py). Meant to be run twice per
(k, ell) pair of interest: once on a checkpoint that never saw multipath in training
(e.g. the Report VI Thread A.1 ER-trained or path_union-trained checkpoints -- the
OOD-generalisation reading) and once on a checkpoint trained directly on the
multipath stream (Report VI Thread A.2 -- the in-distribution reading), so the two
attention pictures can be compared side by side.

Eval-only, CPU-friendly (forward passes only, no backward/Jacobian).

  python mechanistic_multipath_heatmaps.py --checkpoint <ckpt.pt> \
      --output_dir runs/report9/heatmaps_multipath/<tag> --ks 1 2 3 --ell 13
"""
import argparse, math
from pathlib import Path

import numpy as np
import torch

from data import add_self_loops, generate_multipath_graph
from eval_families import load_model
from mechanistic_asym_chains import run_with_cache, _device


def _route_bounds(meta):
    """Base-order node-index boundaries: [0]=start (node 0=s), then after node 1 (=t),
    then after each successive full route's internal nodes -- generate_multipath_graph
    lays routes out contiguously in this exact order, so no remapping is needed."""
    bounds = [0, 2]
    for route in meta["full_paths"]:
        bounds.append(bounds[-1] + len(route))
    return bounds


def multipath_heatmap_probe(model, dev, n, ks, ell, term_deg, rng, n_graphs):
    out = {}
    for k in ks:
        built = generate_multipath_graph(n, k, ell, rng, term_deg=term_deg)
        if built is None:
            print(f"  k={k} ell={ell}: does not fit n={n} (term_deg={term_deg}) -- skipped")
            continue
        base_adj, meta = built
        xs = np.empty((n_graphs, n, n), np.float32)
        invs = []
        for i in range(n_graphs):
            p = rng.permutation(n)
            xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
            invs.append(np.argsort(p))
        xb = torch.from_numpy(xs).to(dev, torch.float32)
        cache, h_final, logits = run_with_cache(model, xb)

        def unperm_pair(mat):
            o = np.empty_like(mat)
            for i, inv in enumerate(invs):
                o[i] = mat[i][np.ix_(inv, inv)]
            return o.mean(0)

        d = {}
        for li in range(len(model.blocks)):
            q, kk, v = cache[f"layer{li}_q"], cache[f"layer{li}_k"], cache[f"layer{li}_v"]
            alpha = cache[f"layer{li}_alpha"]
            head_dim = q.shape[-1]
            scores = np.einsum("gid,gjd->gij", q, kk) / math.sqrt(head_dim)
            d[f"scores{li}"] = unperm_pair(scores)
            d[f"alpha{li}"] = unperm_pair(alpha)
            d[f"row_mass{li}"] = unperm_pair(alpha).sum(-1)
        d["s"] = np.array([meta["s"]]); d["t"] = np.array([meta["t"]])
        d["route_bounds"] = np.array(_route_bounds(meta))
        d["n_full"] = np.array([meta["n_full"]]); d["path_len"] = np.array([meta["path_len"]])
        d["n_active"] = np.array([2 + meta["n_full"] * (meta["path_len"] - 1)
                                  + len(meta["leaves"])])
        out[f"k{k}"] = d
        print(f"  multipath probe k={k} ell={ell} done "
              f"(n_active={int(d['n_active'][0])}/{n})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3],
                    help="number of parallel routes to compare (Report VI Thread A "
                         "convention: k=1 fails beyond capacity, k>=2 rescues)")
    ap.add_argument("--ell", type=int, default=13,
                    help="route length (edges); beyond the 3^L=9 capacity by default, "
                         "matching Report VI's a1_profile_far.png condition")
    ap.add_argument("--term_deg", type=int, default=4,
                    help="terminal degree padding (keeps s,t at a fixed degree "
                         "regardless of k, Report VI's confound-free construction)")
    ap.add_argument("--n_graphs", type=int, default=80,
                    help="independent random relabellings averaged per k")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} "
          f"attn_kind={mcfg.attn_kind} device={dev}")
    if arch != "roberta":
        raise NotImplementedError("run_with_cache is written for RobertaGraphTransformer only")

    rng = np.random.default_rng(args.seed)
    result = multipath_heatmap_probe(model, dev, n, args.ks, args.ell, args.term_deg,
                                     rng, args.n_graphs)

    flat = {}
    for kkey, d in result.items():
        for name, arr in d.items():
            flat[f"{kkey}__{name}"] = arr
    flat["ks_present"] = np.array([int(kk[1:]) for kk in result.keys()])
    np.savez(out / "multipath_heatmap_data.npz", **flat)
    print(f"  saved -> {out}/multipath_heatmap_data.npz")


if __name__ == "__main__":
    main()
