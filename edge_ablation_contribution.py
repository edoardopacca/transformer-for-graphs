"""Report VIII -- causal edge-ablation contribution (eval-only, no training).

The Jacobian-based node-to-node contribution used since Report VII,
C_ik = ||d h_i^(L) / d h_k^(0)||_F (mechanistic_asym_chains.py::exact_contribution),
measures internal token-to-token SENSITIVITY: how much h_i^(L) moves if h_k^(0)
is perturbed. But h_k^(0) is already an internal representation --
LayerNorm(A' W_in + b) -- built from node k's WHOLE neighbourhood, not node k
alone, and an arbitrary perturbation of it is not a change realisable by
editing the graph. A large C_ik therefore does not, by itself, show that node
k's actual incident edges have a causal effect on node i's final embedding or
prediction. This script intervenes directly on the input adjacency A' instead:
for every edge in the graph, remove it (both directions; self-loops
untouched), re-run the forward pass, and measure how much node i's final
embedding / logits change. The old measure is NOT replaced -- it stays valid
as an internal-sensitivity diagnostic -- this is a separate, causal
complement, used wherever the report makes a causal claim about an edge or a
component's role.

Three 40x40 matrices are produced, per split, row i = query node whose final
state we observe, column k = node whose INCIDENT edges are ablated (mean, not
sum, over the edges incident to k -- a path endpoint has degree 1, an
internal node degree 2; summing would confound "more edges" with "more
influence"):

  C_edge[i,k]   = mean over edges e incident to k of || h_i^(L)(A'\\e) - h_i^(L)(A') ||_2
  C_logit[i,k]  = same, but the mean ABSOLUTE change over node i's whole logit row
  C_far[i,k]    = same, but the logit change is averaged only over pairs (i,j)
                  with i AND j in the long component and true graph distance
                  d(i,j) > cap (the beyond-capacity pairs the Report VI/VII
                  puzzle is about) -- zero for i outside the long component or
                  with no such partner.

Caveat kept here, not repeated at length in the rendered report: on a
path/cycle EVERY edge is a bridge for at least one pair, so removing it also
changes the TRUE connectivity target for that pair -- this measures the
effect of a real structural perturbation, not an additive/independent
contribution. A chord-addition control (add an edge INSIDE a component,
leaving the true target unchanged) is a natural follow-up, not implemented
here.

Node order is always BASE graph position (the split's own node indices,
short component first if --topology cycle/chain places it there); every
per-graph relabelling is unpermuted back to base order before the edges are
attributed to node k or the far-pair distances are looked up, exactly the
convention every other Report VII/VIII script uses.

    python edge_ablation_contribution.py --checkpoint runs/.../last.pt \\
        --output_dir runs/report8/edge_contrib/<tag> --topology chain
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from data import add_self_loops, compute_all_pairs_shortest_paths
from eval_families import load_model
from mechanistic_asym_chains import _build_split_graph  # chain/cycle dispatch


def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def _base_edges(base_adj):
    n = base_adj.shape[0]
    return [(u, v) for u in range(n) for v in range(u + 1, n) if base_adj[u, v] > 0]


def edge_ablation_probe(model, dev, n, splits, rng, n_graphs, topology="chain", cap=9):
    out = {}
    for a in splits:
        base_adj = _build_split_graph(n, a, topology)
        base_dist = compute_all_pairs_shortest_paths(base_adj)
        seg0, seg1 = np.arange(0, a), np.arange(a, n)
        L, S = (seg1, seg0) if (n - a) >= a else (seg0, seg1)
        edges = _base_edges(base_adj)
        E = len(edges)

        # F(i): partners j, both i and j in L, true distance > cap. Zero rows
        # outside L (or with no such partner) by construction.
        far_mask = np.zeros((n, n), dtype=bool)
        far_mask[np.ix_(L, L)] = base_dist[np.ix_(L, L)] > cap
        far_counts = far_mask.sum(1)          # |F(i)| per query node i

        deg = np.zeros(n, dtype=np.int64)
        for u, v in edges:
            deg[u] += 1; deg[v] += 1

        C_edge = np.zeros((n, n), dtype=np.float64)
        C_logit = np.zeros((n, n), dtype=np.float64)
        C_far = np.zeros((n, n), dtype=np.float64)

        for _ in range(n_graphs):
            p = rng.permutation(n)
            inv = np.argsort(p)               # inv[base_node] = network position
            net_adj = base_adj[np.ix_(p, p)]
            x0 = add_self_loops(net_adj)
            batch = np.empty((1 + E, n, n), dtype=np.float32)
            batch[0] = x0
            for ei, (u, v) in enumerate(edges):
                xu, xv = inv[u], inv[v]
                xe = x0.copy()
                xe[xu, xv] = 0.0; xe[xv, xu] = 0.0   # self-loops (diagonal) untouched
                batch[1 + ei] = xe
            xb = torch.from_numpy(batch).to(dev, torch.float32)
            with torch.no_grad():
                logits_net, h_net = model.forward_and_embeddings(xb)
            h_net = h_net.detach().cpu().numpy()
            logits_net = logits_net.detach().cpu().numpy()
            # unpermute every instance back to base node order
            h_base = h_net[:, inv, :]
            logits_base = logits_net[:, inv, :][:, :, inv]

            h0 = h_base[0]; z0 = logits_base[0]
            delta_h = np.linalg.norm(h_base[1:] - h0[None], axis=-1)          # [E, n]
            diff_full = np.abs(logits_base[1:] - z0[None])                   # [E, n, n]
            delta_logit = diff_full.mean(-1)                                 # [E, n]
            far_num = (diff_full * far_mask[None]).sum(-1)                   # [E, n]
            with np.errstate(invalid="ignore", divide="ignore"):
                delta_far = np.where(far_counts[None] > 0, far_num / np.maximum(far_counts[None], 1), 0.0)

            for ei, (u, v) in enumerate(edges):
                C_edge[:, u] += delta_h[ei]; C_edge[:, v] += delta_h[ei]
                C_logit[:, u] += delta_logit[ei]; C_logit[:, v] += delta_logit[ei]
                C_far[:, u] += delta_far[ei]; C_far[:, v] += delta_far[ei]

        denom = (n_graphs * np.maximum(deg, 1))[None, :]
        C_edge = C_edge / denom
        C_logit = C_logit / denom
        C_far = C_far / denom

        def leak(mat):
            num = mat[np.ix_(L, S)].sum()
            den = mat[L, :].sum()
            return float(num / den) if den > 0 else float("nan")

        e_leak, l_leak, f_leak = leak(C_edge), leak(C_logit), leak(C_far)
        out[f"a{a}"] = {
            "C_edge": C_edge, "C_logit": C_logit, "C_far": C_far,
            "long_idx": L, "short_idx": S,
            "edge_leak": np.array([e_leak]), "logit_leak": np.array([l_leak]),
            "far_leak": np.array([f_leak]),
        }
        print(f"  edge-ablation a={a:>2d} done (edges={E}, edge_leak={e_leak:.3f}, "
              f"logit_leak={l_leak:.3f}, far_leak={f_leak:.3f})", flush=True)
    return out


def _selftest(model, dev, n, topology="chain"):
    """Sanity checks on a toy model before trusting a real checkpoint:
    (1) the baseline row of the batch reproduces a standalone forward pass;
    (2) ablating an edge leaves every self-loop (diagonal) entry untouched;
    (3) F(i) is empty (far_counts==0) for every node outside the long
    component, for a representative split."""
    a = max(3, n // 4) if topology == "cycle" else max(1, n // 4)
    base_adj = _build_split_graph(n, a, topology)
    rng = np.random.default_rng(0)
    p = rng.permutation(n)
    net_adj = base_adj[np.ix_(p, p)]
    x0 = add_self_loops(net_adj)
    edges = _base_edges(base_adj)
    inv = np.argsort(p)
    u, v = edges[0]
    xu, xv = inv[u], inv[v]
    xe = x0.copy(); xe[xu, xv] = 0.0; xe[xv, xu] = 0.0
    assert np.allclose(np.diag(xe), 1.0), "self-loops must survive an edge ablation"
    xb = torch.from_numpy(np.stack([x0, xe])).to(dev, torch.float32)
    with torch.no_grad():
        logits_batch, h_batch = model.forward_and_embeddings(xb)
    logits_solo, h_solo = model.forward_and_embeddings(
        torch.from_numpy(x0[None]).to(dev, torch.float32))
    assert torch.allclose(h_batch[0], h_solo[0], atol=1e-5), \
        "batched baseline forward must match a standalone forward pass"
    assert torch.allclose(logits_batch[0], logits_solo[0], atol=1e-4)
    seg0, seg1 = np.arange(0, a), np.arange(a, n)
    L, S = (seg1, seg0) if (n - a) >= a else (seg0, seg1)
    dist = compute_all_pairs_shortest_paths(base_adj)
    far_mask = np.zeros((n, n), dtype=bool)
    far_mask[np.ix_(L, L)] = dist[np.ix_(L, L)] > 9
    assert far_mask[S, :].sum() == 0 and far_mask[:, S].sum() == 0, \
        "F(i) must be confined to the long component on both sides"
    print("  [selftest] batched baseline matches standalone forward; self-loops "
          "survive ablation; F(i) confined to the long component")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=64)
    ap.add_argument("--splits", type=int, nargs="+", default=None,
                    help="representative splits; default the same set used "
                         "throughout Report VII/VIII (1,4,7,8,10,11,12,13,14,17,n//2)")
    ap.add_argument("--topology", choices=["chain", "cycle"], default="chain",
                    help="chain = two disjoint paths; cycle = two disjoint cycles "
                         "(Report VIII sec:cycles)")
    ap.add_argument("--cap", type=int, default=9, help="architectural capacity 3^L")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--skip_selftest", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} "
          f"device={dev} topology={args.topology}")
    if arch != "roberta":
        raise NotImplementedError("this script targets the RobertaGraphTransformer only")
    if not args.skip_selftest:
        _selftest(model, dev, n, topology=args.topology)

    min_a = 3 if args.topology == "cycle" else 1
    splits = args.splits if args.splits is not None else \
        sorted({s for s in (1, 4, 7, 8, 10, 11, 12, 13, 14, 17, n // 2) if min_a <= s <= n // 2})
    rng = np.random.default_rng(args.seed)

    probe = edge_ablation_probe(model, dev, n, splits, rng, args.n_graphs,
                                 topology=args.topology, cap=args.cap)
    flat = {}
    for a_key, d in probe.items():
        for k, v in d.items():
            flat[f"{a_key}__{k}"] = np.asarray(v)
    np.savez_compressed(out / "edge_contrib.npz", **flat)
    print(f"  saved -> {out}/edge_contrib.npz")


if __name__ == "__main__":
    main()
