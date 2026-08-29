"""Report X -- the actual redundant-chain rescue test for the directed_chain checkpoint
(scripts/r10_n40_directed_chain_seed1000.sbatch), Edoardo's 2026-08-29 request: does
reachability(s,t) get rescued when there are K=1,2,3,4 independent directed chains from s to
t instead of one, once each individual chain is longer than the 3^L=9 capacity wall?

Construction: s=node0, t=node1, K independent directed chains s->...->t (route_len edges
each, all-distinct internal nodes across routes -- no shared nodes, no shortcuts between
routes), leftover canvas wired into ONE separate directed filler chain (never left as
isolated/degree-0 nodes: the training distribution is always a single connected directed
chain, so isolated nodes would be a distribution-shape confound the filler avoids).

route_len=11 edges (10 internal nodes/route) for K=1,2,3 -- fits n=40 with room to spare.
route_len=10 edges (9 internal nodes/route) for K=4 only, since 4x11 does not fit in n=40
(2+4*10=42>40); still comfortably past the wall (distance 10>9). Flagged explicitly in the
printed/saved results so the K=4 number is never compared to K=1..3 as if same route length.

    python eval_directed_chain_redundancy.py --checkpoint runs/.../last.pt \\
        --output_dir runs/report10/directed_chain_redundancy/<tag> --seed 1000
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from eval_families import load_model
from stagewise_diagnostics import _device, _selftest


def generate_directed_multipath_graph(n, n_routes, route_len):
    """s=0, t=1, n_routes independent directed chains s->...->t, each route_len edges
    (route_len-1 internal nodes, all distinct across routes, no shortcuts). Leftover canvas
    wired into one separate directed filler chain (disjoint from the s-t structure)."""
    need = 2 + n_routes * (route_len - 1)
    if need > n:
        raise ValueError(f"n_routes={n_routes} route_len={route_len} needs {need} nodes > n={n}")
    adj = np.zeros((n, n), dtype=np.float32)
    s, t = 0, 1
    cur = 2
    routes = []
    for _ in range(n_routes):
        prev = s
        nodes = []
        for _ in range(route_len - 1):
            adj[prev, cur] = 1.0
            nodes.append(cur); prev = cur; cur += 1
        adj[prev, t] = 1.0
        routes.append(nodes)
    filler = list(range(cur, n))
    for i in range(len(filler) - 1):
        adj[filler[i], filler[i + 1]] = 1.0
    return adj, s, t, routes, filler


def _forward_logits(model, dev, xb):
    with torch.no_grad():
        logits, _ = model.forward_and_embeddings(xb)
    return logits.cpu().numpy()


def eval_redundancy(model, dev, n, n_routes, route_len, n_graphs, rng):
    adj, s, t, routes, filler = generate_directed_multipath_graph(n, n_routes, route_len)
    xs = np.empty((n_graphs, n, n), np.float32)
    st_pos = np.empty((n_graphs, 2), dtype=np.int64)
    for g in range(n_graphs):
        p = rng.permutation(n)
        inv = np.argsort(p)
        a_perm = adj[np.ix_(p, p)]
        x = a_perm.copy(); np.fill_diagonal(x, 1.0)  # add_self_loops inline
        xs[g] = x
        st_pos[g] = [inv[s], inv[t]]
    xb = torch.from_numpy(xs).to(dev, torch.float32)
    logits = np.empty((n_graphs, n, n), np.float32)
    B = 256
    for i in range(0, n_graphs, B):
        logits[i:i + B] = _forward_logits(model, dev, xb[i:i + B])
    pred_st = np.array([logits[g, st_pos[g, 0], st_pos[g, 1]] > 0 for g in range(n_graphs)])
    return {
        "n_routes": n_routes, "route_len": route_len, "distance": route_len,
        "s_t_connected_acc": float(pred_st.mean()),
        "n_graphs": n_graphs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")
    if readout != "linear":
        raise NotImplementedError("this script assumes the asymmetric linear read-out")
    _selftest(model, dev, n)

    configs = [(1, 11), (2, 11), (3, 11), (4, 10)]
    results = []
    for k, route_len in configs:
        rng = np.random.default_rng(args.seed + k)
        r = eval_redundancy(model, dev, n, k, route_len, args.n_graphs, rng)
        print(f"  K={k} route_len={route_len} (distance {route_len}): "
              f"s-t connected accuracy = {r['s_t_connected_acc']:.4f} "
              f"(n={r['n_graphs']} graphs)", flush=True)
        results.append(r)

    (out / "directed_chain_redundancy.json").write_text(json.dumps(results, indent=2))
    print(f"saved -> {out}/directed_chain_redundancy.json")


if __name__ == "__main__":
    main()
