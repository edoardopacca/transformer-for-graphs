"""Report VII -- raw material for weight- and attention-heatmaps (eval-only).

Tier-1's mechanistic_asym_chains.py already computed AGGREGATE statistics
(leak fraction, mean logits). This script keeps the underlying MATRICES
themselves -- the actual attention-score heatmaps, Q/K/V per node, and the
raw weight matrices -- so they can be looked at directly, as the user asked.

Everything per-node/per-pair is computed in BASE (unpermuted) node coordinates,
averaged over many independent random relabellings of the SAME underlying
graph (the network is permutation-equivariant, so this converges to a stable,
low-variance "expected attention/embedding for structural position i", not an
arbitrary single instance). Pairwise quantities (scores, alpha, rollout,
contribution) are unpermuted with the double np.ix_(inv, inv) used everywhere
else in this project; per-node quantities (q, k, v) are unpermuted by
indexing rows with `inv` alone -- both patterns already validated in
mechanistic_asym_chains.py (the readout-decomposition bugfix).

Outputs, per checkpoint:
  heatmap_data.npz   -- per representative split a: scores0/1, alpha0/1,
                         rollout, contrib0/1 (all [n,n]), row_mass0/1,
                         q0/k0/v0/q1/k1/v1 (all [n, head_dim]), long_idx,
                         short_idx.
  raw_weights.npz     -- W_in, W_out, and per layer W_Q/W_K/W_V/W_O (all
                         [d,d] or [d,n]/[n,d]), plus their singular values.

    python mechanistic_heatmaps.py --checkpoint runs/.../last.pt \
        --output_dir runs/report7/heatmaps/<tag>
"""
import argparse, math
from pathlib import Path

import numpy as np
import torch

from data import add_self_loops
from eval_families import load_model
from mechanistic_asym_chains import run_with_cache, exact_contribution, _TOPOLOGY_GENERATORS


def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def heatmap_probe(model, dev, n, splits, rng, n_graphs, contrib_n_graphs=8, topology="chain"):
    out = {}
    for a in splits:
        base_adj = _TOPOLOGY_GENERATORS[topology](n, a)
        seg0, seg1 = np.arange(0, a), np.arange(a, n)
        L, S = (seg1, seg0) if (n - a) >= a else (seg0, seg1)
        xs = np.empty((n_graphs, n, n), np.float32)
        invs = []
        for i in range(n_graphs):
            p = rng.permutation(n)
            xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
            invs.append(np.argsort(p))
        xb = torch.from_numpy(xs).to(dev, torch.float32)
        cache, h_final, logits = run_with_cache(model, xb)

        def unperm_pair(mat):     # [G,n,n] -> mean over G in base coords
            o = np.empty_like(mat)
            for i, inv in enumerate(invs):
                o[i] = mat[i][np.ix_(inv, inv)]
            return o.mean(0)

        def unperm_node(mat):     # [G,n,d] -> mean over G in base coords
            o = np.empty_like(mat)
            for i, inv in enumerate(invs):
                o[i] = mat[i][inv]
            return o.mean(0)

        d = {}
        for li in range(len(model.blocks)):
            q, k, v = cache[f"layer{li}_q"], cache[f"layer{li}_k"], cache[f"layer{li}_v"]
            alpha = cache[f"layer{li}_alpha"]
            wo_v = cache[f"layer{li}_wo_v"]
            head_dim = q.shape[-1]
            scores = np.einsum("gid,gjd->gij", q, k) / math.sqrt(head_dim)
            contrib = alpha * np.linalg.norm(wo_v, axis=-1)[:, None, :]
            d[f"scores{li}"] = unperm_pair(scores)
            d[f"alpha{li}"] = unperm_pair(alpha)
            d[f"contrib{li}"] = unperm_pair(contrib)
            d[f"row_mass{li}"] = unperm_pair(alpha).sum(-1)
            d[f"q{li}"] = unperm_node(q)
            d[f"k{li}"] = unperm_node(k)
            d[f"v{li}"] = unperm_node(v)
        # real node-to-node contribution: exact Jacobian norm through the
        # whole real forward pass (V, W_O, residual, MLP, LayerNorm), a
        # smaller sample of graphs (O(n) backward passes each).
        h0_all = cache["h0"]
        g_idx = np.arange(min(contrib_n_graphs, n_graphs))
        contrib_exact_list = []
        for gi in g_idx:
            h0_row = torch.from_numpy(h0_all[gi]).to(dev, torch.float32)
            C = exact_contribution(model, h0_row)                    # [n,n], network coords
            contrib_exact_list.append(C[np.ix_(invs[gi], invs[gi])])  # -> base coords
        d["contrib_exact"] = np.stack(contrib_exact_list).mean(0)
        d["long_idx"] = L
        d["short_idx"] = S
        out[f"a{a}"] = d
        print(f"  heatmap probe a={a:>2d} done", flush=True)
    return out


def raw_weights(model):
    d = {}
    d["W_in"] = model.read_in.weight.detach().cpu().numpy()     # [d_model, n]
    d["b_in"] = model.read_in.bias.detach().cpu().numpy()
    if model.readout_kind == "similarity":
        # no W_out for this read-out -- just the two learned scalars.
        d["sim_scale"] = np.array([float(model.sim_scale.detach().cpu())])
        d["sim_bias"] = np.array([float(model.sim_bias.detach().cpu())])
    else:
        d["W_out"] = model.read_out.weight.detach().cpu().numpy()   # [n, d_model]
        d["b_out"] = model.read_out.bias.detach().cpu().numpy()
        d["sv_W_out"] = np.linalg.svd(d["W_out"], compute_uv=False)
    for li, blk in enumerate(model.blocks):
        d[f"WQ{li}"] = blk.attn.q_proj.weight.detach().cpu().numpy()
        d[f"WK{li}"] = blk.attn.k_proj.weight.detach().cpu().numpy()
        d[f"WV{li}"] = blk.attn.v_proj.weight.detach().cpu().numpy()
        d[f"WO{li}"] = blk.attn_dense.weight.detach().cpu().numpy()
        for name in (f"WQ{li}", f"WK{li}", f"WV{li}", f"WO{li}"):
            d[f"sv_{name}"] = np.linalg.svd(d[name], compute_uv=False)
    d["sv_W_in"] = np.linalg.svd(d["W_in"], compute_uv=False)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=80)
    ap.add_argument("--contrib_n_graphs", type=int, default=8,
                    help="graphs averaged for the EXACT node-to-node contribution "
                         "(Jacobian norm) -- kept small, it is O(n) backward passes/graph")
    ap.add_argument("--splits", type=int, nargs="+", default=None,
                    help="representative splits; default 1,4,7,8,10,14,17,20")
    ap.add_argument("--topology",
                    choices=["chain", "cycle", "split_cliques", "chorded_cycles", "split_regular3"],
                    default="chain",
                    help="chain = two disjoint paths (Report VI/VII); cycle = two "
                         "disjoint cycles, same split, no path endpoints (Report VIII); "
                         "split_cliques/chorded_cycles/split_regular3 = Report IX "
                         "controlled-distribution battery")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")
    if arch != "roberta":
        raise NotImplementedError("this script targets the RobertaGraphTransformer only")
    min_a = 3 if args.topology == "cycle" else 1
    splits = args.splits if args.splits is not None else \
        sorted({s for s in (1, 4, 7, 8, 10, 14, 17, n // 2) if min_a <= s <= n // 2})
    rng = np.random.default_rng(args.seed)

    print("== attention/Q/K/V heatmap probe ==")
    probe = heatmap_probe(model, dev, n, splits, rng, args.n_graphs,
                           contrib_n_graphs=args.contrib_n_graphs, topology=args.topology)
    flat = {}
    for a_key, d in probe.items():
        for k, v in d.items():
            flat[f"{a_key}__{k}"] = np.asarray(v)
    np.savez_compressed(out / "heatmap_data.npz", **flat)
    print(f"  saved -> {out}/heatmap_data.npz")

    print("== raw weight matrices ==")
    w = raw_weights(model)
    np.savez_compressed(out / "raw_weights.npz", **w)
    print(f"  saved -> {out}/raw_weights.npz")


if __name__ == "__main__":
    main()
