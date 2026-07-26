"""Report VII -- mechanistic audit of the asymmetric two-chains split
(Report VI, Thread B). Eval-only, no training.

Report VI found that, on the SAME trained model, a two-chain test graph split
(a, n-a) is solved perfectly for small a and fails completely once a nears the
architectural capacity 3^L=9 -- but never opened the model to say why. This
script is the first thing in the whole project to actually consume
`RobertaGraphTransformer.attention_maps` / the read-out weight matrix for real
analysis (grep confirms attention_maps was defined but never used anywhere).

Key architectural fact driving the design (see report/7 sec 1): with the
LINEAR read-out, z_ij = h_i^T w_j + b_j, where w_j is row j of W_out -- a fixed
learned vector attached to the TARGET node, not derived from h_j. So the two
central objects to inspect are (a) the h_i^T w_j decomposition itself, and
(b) the geometry of W_out's rows -- not a symmetric embedding distance (that
quantity belongs to the SIMILARITY read-out, a different mechanism, out of
scope here; this script raises if handed a similarity checkpoint).

Random relabelling: every one of the n_graphs test graphs per split is
independently permuted before being fed to the model and predictions are
unpermuted before scoring -- the SAME methodology eval_asym_chains.py already
uses for Report VI's tab:basplit. Pass --fixed_label to disable this (identity
permutation) for the one explicit relabelling-comparison figure the report
needs; the default (permuted) is the condition all the headline numbers use.

Outputs, per checkpoint, under --output_dir:
  metrics.csv          -- one row per (split a, seed[implicit: one checkpoint
                           per call], mode) with exact/reach/cut/per-distance/
                           predicted-positive-rate (Tier-1 #1, #2).
  readout.csv           -- h_i^T w_j decomposition aggregated by pair type
                           (within-short/within-long/cut), true distance, split
                           a (Tier-1 #3).
  weights_summary.json  -- ||w_j|| per target node, cosine matrix of W_out's
                           rows, and the read-in/read-out alignment
                           E_in @ W_out^T (Tier-1 #4).
  attn_cache.npz        -- real per-layer attention, the 2-layer rollout,
                           row-mass, and message contribution alpha_ij*(W_O v_j),
                           for a handful of REPRESENTATIVE splits only
                           (Tier-1 #5) -- this is heavier, kept separate from
                           the cheap sweep above.

    python mechanistic_asym_chains.py --checkpoint runs/.../last.pt \
        --output_dir runs/report7/mechanistic/<tag>
"""
import argparse, csv, json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import (add_self_loops, compute_connectivity_matrix,
                  compute_all_pairs_shortest_paths, generate_split_chains_graph,
                  generate_split_cycles_graph, generate_split_cliques_asym_graph,
                  generate_chorded_cycles_graph, generate_split_regular_graph)
from eval_families import load_model


def _split_regular3(n, a):
    """Fixed-seed wrapper so the (random) 3-regular generator fits the (n, a) ->
    adjacency signature every other topology here uses -- ONE canonical 3-regular
    pair of blocks per split size, permuted for relabelling like every other
    topology, not a fresh random draw per call (this is a mechanistic PROBE on a
    representative graph, not the online training stream)."""
    return generate_split_regular_graph(n, 3, a, np.random.default_rng(12345))


# Report VIII: --topology cycle swaps every "two disjoint paths, split (a, n-a)"
# graph in this script for "two disjoint CYCLES, split (a, n-a)" (same split
# knob, same node-index ranges, only the two segments are closed into cycles).
# Kept as a single dispatch point so every downstream function (behavioural
# sweep, readout decomposition, attention probe) works unchanged for either
# topology, rather than duplicating this whole file for one graph-generator swap.
# Report IX (controlled-distribution battery): three more topologies added the
# same way -- split_cliques/chorded_cycles/split_regular3.
_TOPOLOGY_GENERATORS = {"chain": generate_split_chains_graph,
                        "cycle": generate_split_cycles_graph,
                        "split_cliques": generate_split_cliques_asym_graph,
                        "chorded_cycles": generate_chorded_cycles_graph,
                        "split_regular3": _split_regular3}


def _build_split_graph(n, a, topology):
    return _TOPOLOGY_GENERATORS[topology](n, a)


def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Manual forward-with-cache. Re-derives RobertaGraphTransformer's forward pass
# block by block so we can keep Q/K/V/attention/message-contribution, none of
# which model.py exposes on their own (attention_maps() only keeps the
# attention weights, as a side effect of a normal forward call).
# --------------------------------------------------------------------------
@torch.no_grad()
def run_with_cache(model, x):
    """x: [B, n, n] float tensor (self-loop-augmented adjacency).
    Returns (cache, h_final, logits). cache holds per-layer q/k/v/alpha/
    attn_out/wo_v (= W_O @ v_j, the per-node value AFTER the output
    projection, so alpha_ij * wo_v[j] is the exact additive message j sends to
    i's residual stream), each as float32 numpy arrays with the batch and
    (single) head dims squeezed out -- shapes [B,n,d]/[B,n,n] as noted below."""
    h = model.emb_drop(model.emb_ln(model.read_in(x)))
    cache = {"h0": h.detach().cpu().numpy()}
    for li, blk in enumerate(model.blocks):
        inp = blk.attn_ln(h) if blk.norm_style != "post" else h
        att = blk.attn
        b, n, d = inp.shape
        q = att.q_proj(inp).view(b, n, att.n_heads, att.head_dim).transpose(1, 2)
        k = att.k_proj(inp).view(b, n, att.n_heads, att.head_dim).transpose(1, 2)
        v = att.v_proj(inp).view(b, n, att.n_heads, att.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(att.head_dim)
        alpha = F.softmax(scores, dim=-1) if att.attn_kind == "softmax" else F.relu(scores) / n
        ctx = torch.matmul(alpha, v).transpose(1, 2).contiguous().view(b, n, d)
        a_out = blk.attn_dense(ctx)
        # W_O applied to each node's OWN value vector (before the alpha-weighted
        # sum) -- linearity lets us decompose a_out_i = sum_j alpha_ij * wo_v_j.
        wo_v = F.linear(v[:, 0], blk.attn_dense.weight)     # [b,n,d], n_heads=1 so v[:,0]=v squeezed
        if blk.norm_style == "post":
            h = blk.attn_ln(h + a_out)
            f = blk.output(blk.act(blk.intermediate(h)))
            h = blk.out_ln(h + f)
        else:
            h = h + a_out
            f = blk.output(blk.act(blk.intermediate(blk.out_ln(h))))
            h = h + f
        cache[f"layer{li}_q"] = q[:, 0].detach().cpu().numpy()
        cache[f"layer{li}_k"] = k[:, 0].detach().cpu().numpy()
        cache[f"layer{li}_v"] = v[:, 0].detach().cpu().numpy()
        cache[f"layer{li}_alpha"] = alpha[:, 0].detach().cpu().numpy()
        cache[f"layer{li}_wo_v"] = wo_v.detach().cpu().numpy()
        cache[f"h{li + 1}"] = h.detach().cpu().numpy()
    if model.readout_kind == "similarity":
        hn = F.normalize(h, dim=-1)
        logits = model.sim_scale * torch.matmul(hn, hn.transpose(-1, -2)) + model.sim_bias
    else:
        logits = model.read_out(h)
    return cache, h.detach().cpu().numpy(), logits.detach().cpu().numpy()


# --------------------------------------------------------------------------
# Real node-to-node contribution: replaces the alpha1@alpha0 attention-only
# "rollout" approximation. C[i,k] = || d h_i^(L) / d h_k^(0) ||_F, the exact
# Jacobian-norm sensitivity of node i's final embedding to node k's read-in
# embedding, computed by literal backprop through the REAL forward pass (real
# V, W_O, the residual identity, the MLP, LayerNorm's true local derivative)
# -- not a linearised proxy that drops any of those. Single instance (one
# graph), which is what "how much does k contribute to i, here" means.
# --------------------------------------------------------------------------
def exact_contribution(model, h0_row):
    """h0_row: [n, d] read-in embedding of ONE graph (no batch dim; already
    past emb_ln/emb_drop). Returns C: [n, n] numpy, C[i, k] =
    ||d h_i^(L) / d h_k^(0)||_F. Loops over query row i (n backward calls,
    each vmapped over the d output coordinates of that row via
    is_grads_batched) -- exact, not approximated, but O(n) backward passes,
    so keep the caller's graph count small (this is far more expensive than
    the cheap alpha/scores/message-contribution quantities)."""
    h0 = h0_row.detach().clone().unsqueeze(0).requires_grad_(True)   # [1,n,d]
    h = h0
    for block in model.blocks:
        h = block(h)
    n, d = h.shape[1], h.shape[2]
    basis = torch.eye(d, device=h.device)
    C = torch.zeros(n, n)
    for i in range(n):
        grad_outputs = torch.zeros(d, 1, n, d, device=h.device)
        grad_outputs[:, 0, i, :] = basis
        grads = torch.autograd.grad(h, h0, grad_outputs=grad_outputs,
                                     retain_graph=True, is_grads_batched=True)[0]
        # grads: [d, 1, n, d] = d h_{i,r}^(L) / d h0_{k,s}, r the batch dim
        C[i] = grads.squeeze(1).pow(2).sum(dim=(0, 2)).sqrt().detach().cpu()
    return C.numpy()


def _selftest_exact_contribution(dev):
    """exact_contribution's batched autograd must match an independent,
    unbatched double-loop computation (and, loosely, finite differences) on a
    small random toy model -- guards against any batching/indexing bug before
    this touches a real checkpoint."""
    from model import ModelConfig, RobertaGraphTransformer
    torch.manual_seed(0)
    n, d = 6, 8
    cfg = ModelConfig(n=n, d_model=d, n_heads=1, d_ff=16, n_layers=2, dropout=0.0,
                       attn_kind="normalized_relu", norm_style="post",
                       layer_norm_eps=1e-5, init_std=0.5, readout="linear")
    m = RobertaGraphTransformer(cfg).to(dev).eval()
    x = (torch.rand(1, n, n, device=dev) > 0.6).float()
    x = ((x + x.transpose(-1, -2)) > 0).float()
    for i in range(n):
        x[0, i, i] = 1.0
    with torch.no_grad():
        h0 = m.emb_drop(m.emb_ln(m.read_in(x)))[0]
    C_fast = exact_contribution(m, h0)

    h0b = h0.detach().clone().unsqueeze(0).requires_grad_(True)
    h = h0b
    for block in m.blocks:
        h = block(h)
    C_brute = torch.zeros(n, n)
    for i in range(n):
        acc = torch.zeros(n, device=dev)
        for r in range(d):
            g = torch.autograd.grad(h[0, i, r], h0b, retain_graph=True)[0][0]  # [n,d]
            acc += g.pow(2).sum(dim=-1)
        C_brute[i] = acc.sqrt()
    diff = np.abs(C_fast - C_brute.detach().cpu().numpy()).max()
    assert diff < 1e-4, f"exact_contribution selftest failed, max diff {diff}"
    print(f"  [selftest] exact_contribution matches brute-force double loop (max diff {diff:.2e})")


def _selftest(model, dev, n):
    """Cache's final h/logits must match model.forward_and_embeddings exactly."""
    x = torch.rand(2, n, n, device=dev)
    x = ((x + x.transpose(-1, -2)) > 1.3).float()
    for i in range(2):
        x[i].fill_diagonal_(1.0)
    _, h_cache, logits_cache = run_with_cache(model, x)
    logits_ref, h_ref = model.forward_and_embeddings(x)
    h_ref, logits_ref = h_ref.detach().cpu().numpy(), logits_ref.detach().cpu().numpy()
    assert np.allclose(h_cache, h_ref, atol=1e-4), "cache h != forward_and_embeddings h"
    assert np.allclose(logits_cache, logits_ref, atol=1e-3), "cache logits != forward_and_embeddings logits"
    print("  [selftest] run_with_cache matches model.forward_and_embeddings exactly")


# --------------------------------------------------------------------------
# Tier-1 #1/#2: dense behavioural sweep (+ relabelling toggle)
# --------------------------------------------------------------------------
def behavioural_sweep(model, dev, n, splits, rng, n_graphs, fixed_label, topology="chain"):
    rows = []
    for a in splits:
        base_adj = _build_split_graph(n, a, topology)
        base_y = compute_connectivity_matrix(base_adj).astype(np.int8)
        base_dist = compute_all_pairs_shortest_paths(base_adj)
        seg0, seg1 = np.arange(0, a), np.arange(a, n)
        L, S = (seg1, seg0) if (n - a) >= a else (seg0, seg1)

        xs = np.empty((n_graphs, n, n), np.float32)
        invs = []
        for i in range(n_graphs):
            p = np.arange(n) if fixed_label else rng.permutation(n)
            xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
            invs.append(np.argsort(p))
        xb = torch.from_numpy(xs).to(dev, torch.float32)
        preds = np.empty((n_graphs, n, n), np.int8)
        for s in range(0, n_graphs, 128):
            e = min(s + 128, n_graphs)
            logits = model.forward(xb[s:e])
            preds[s:e] = (logits > 0).detach().cpu().numpy().astype(np.int8)
        pred = np.empty_like(preds)
        for i, inv in enumerate(invs):
            pred[i] = preds[i][np.ix_(inv, inv)]
        eq = (pred == base_y[None])

        exact = float(eq.reshape(n_graphs, -1).all(1).mean())
        eqL = eq[:, L][:, :, L]; offL = ~np.eye(len(L), dtype=bool)
        reach_long = float(eqL[:, offL].mean())
        if len(S) > 1:
            eqS = eq[:, S][:, :, S]; offS = ~np.eye(len(S), dtype=bool)
            reach_short = float(eqS[:, offS].mean())
        else:
            reach_short = float("nan")
        memb = np.zeros(n, dtype=np.int8); memb[L] = 1
        cmask = memb[:, None] != memb[None, :]
        cut = float(eq[:, cmask].mean())
        pos_rate = float(pred.mean())
        dL = base_dist[np.ix_(L, L)]
        for d in sorted(set(int(v) for v in dL[dL > 0])):
            m = dL == d
            rows.append({"split_a": a, "mode": "fixed" if fixed_label else "random",
                         "metric": "reach_long_by_dist", "distance": d,
                         "value": round(float(eqL[:, m].mean()), 4), "n_pairs": int(m.sum())})
        rows.append({"split_a": a, "mode": "fixed" if fixed_label else "random",
                     "metric": "exact", "distance": None, "value": round(exact, 4), "n_pairs": None})
        rows.append({"split_a": a, "mode": "fixed" if fixed_label else "random",
                     "metric": "reach_long", "distance": None, "value": round(reach_long, 4), "n_pairs": None})
        rows.append({"split_a": a, "mode": "fixed" if fixed_label else "random",
                     "metric": "reach_short", "distance": None, "value": round(reach_short, 4), "n_pairs": None})
        rows.append({"split_a": a, "mode": "fixed" if fixed_label else "random",
                     "metric": "cut", "distance": None, "value": round(cut, 4), "n_pairs": None})
        rows.append({"split_a": a, "mode": "fixed" if fixed_label else "random",
                     "metric": "pred_positive_rate", "distance": None, "value": round(pos_rate, 4), "n_pairs": None})
        print(f"  a={a:>2d} exact={exact:.3f} reach_long={reach_long:.3f} "
              f"reach_short={reach_short:.3f} cut={cut:.3f} mode={'fixed' if fixed_label else 'random'}",
              flush=True)
    return rows


# --------------------------------------------------------------------------
# Tier-1 #3: h_i^T w_j readout decomposition, aggregated by pair type/distance/split
# --------------------------------------------------------------------------
def readout_decomposition(model, dev, n, splits, rng, n_graphs, topology="chain"):
    W_out = model.read_out.weight.detach().cpu().numpy()      # [n, d_model], row j = w_j
    b_out = model.read_out.bias.detach().cpu().numpy()        # [n]
    rows = []
    for a in splits:
        base_adj = _build_split_graph(n, a, topology)
        base_dist = compute_all_pairs_shortest_paths(base_adj)
        seg0, seg1 = np.arange(0, a), np.arange(a, n)
        L, S = (seg1, seg0) if (n - a) >= a else (seg0, seg1)
        memb = np.zeros(n, dtype=np.int8); memb[L] = 1

        xs = np.empty((n_graphs, n, n), np.float32)
        invs = []
        for i in range(n_graphs):
            p = rng.permutation(n)
            xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
            invs.append(np.argsort(p))
        xb = torch.from_numpy(xs).to(dev, torch.float32)
        h_all = np.empty((n_graphs, n, W_out.shape[1]), np.float32)
        for s in range(0, n_graphs, 128):
            e = min(s + 128, n_graphs)
            h = model.embeddings(xb[s:e])
            h_all[s:e] = h.detach().cpu().numpy()
        # h_all and W_out are BOTH in network/permuted coordinates (w_j is fixed
        # to network position j, not to the base-graph node j) -- so h_i^T w_j
        # must be computed in network coordinates first, and ONLY THEN unpermuted
        # as a pairwise [n,n] matrix (exactly like `pred` elsewhere in this
        # project), never by unpermuting h's rows alone against a fixed W_out.
        raw_perm = np.einsum("gid,jd->gij", h_all, W_out)   # [n_graphs, n, n], network coords
        raw = np.empty_like(raw_perm)
        z = np.empty_like(raw_perm)                          # raw + bias, i.e. the true logit
        for i, inv in enumerate(invs):
            raw[i] = raw_perm[i][np.ix_(inv, inv)]          # -> base node coordinates
            z[i] = raw[i] + b_out[inv][None, :]              # bias indexed by TARGET's network position
        # aggregate the mean by pair type / distance, over all graphs and node pairs
        for i_set, j_set, label in ((L, L, "within_long"), (S, S, "within_short"),
                                     (L, S, "cut"), (S, L, "cut")):
            if len(i_set) == 0 or len(j_set) == 0:
                continue
            if label in ("within_long", "within_short") and len(i_set) <= 1:
                continue  # a single-node component has no within-pairs
            sub = raw[:, i_set][:, :, j_set]
            subz = z[:, i_set][:, :, j_set]
            if label == "within_long" or label == "within_short":
                off = ~np.eye(len(i_set), dtype=bool)
                vals, valsz = sub[:, off], subz[:, off]
            else:
                vals, valsz = sub.reshape(sub.shape[0], -1), subz.reshape(subz.shape[0], -1)
            rows.append({"split_a": a, "pair_type": label,
                         "mean_hTw": round(float(vals.mean()), 4),
                         "frac_positive": round(float((valsz > 0).mean()), 4),
                         "n_pairs": int(vals.size)})
        # within-long, by true distance (the profile that matters for the wall)
        dL = base_dist[np.ix_(L, L)]
        rawL = raw[:, L][:, :, L]
        zL = z[:, L][:, :, L]
        for d in sorted(set(int(v) for v in dL[dL > 0])):
            m = dL == d
            rows.append({"split_a": a, "pair_type": f"within_long_d{d}",
                         "mean_hTw": round(float(rawL[:, m].mean()), 4),
                         "frac_positive": round(float((zL[:, m] > 0).mean()), 4),
                         "n_pairs": int(m.sum())})
        print(f"  readout a={a:>2d} done", flush=True)
    return rows


# --------------------------------------------------------------------------
# Tier-1 #3, SIMILARITY variant. There is no per-target vector w_j for this
# read-out -- z_ij = scale*cos(h_i,h_j)+bias is a genuinely symmetric function
# of BOTH embeddings, so the natural analogue of the h_i^T w_j decomposition
# is cos(h_i,h_j) itself (the raw, pre-scale/bias quantity that the read-out
# actually thresholds). Unlike the linear version, both indices here refer to
# real node embeddings, so unpermuting needs the ordinary double np.ix_(inv,inv)
# used elsewhere in this project for symmetric pairwise quantities (rollout/
# contrib), not the row-only unpermute the linear w_j-is-fixed-in-network-
# coordinates case requires.
# --------------------------------------------------------------------------
def readout_decomposition_similarity(model, dev, n, splits, rng, n_graphs, topology="chain"):
    rows = []
    for a in splits:
        base_adj = _build_split_graph(n, a, topology)
        base_dist = compute_all_pairs_shortest_paths(base_adj)
        seg0, seg1 = np.arange(0, a), np.arange(a, n)
        L, S = (seg1, seg0) if (n - a) >= a else (seg0, seg1)

        xs = np.empty((n_graphs, n, n), np.float32)
        invs = []
        for i in range(n_graphs):
            p = rng.permutation(n)
            xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
            invs.append(np.argsort(p))
        xb = torch.from_numpy(xs).to(dev, torch.float32)
        h_all = np.empty((n_graphs, n, model.config.d_model), np.float32)
        for s in range(0, n_graphs, 128):
            e = min(s + 128, n_graphs)
            h = model.embeddings(xb[s:e])
            h_all[s:e] = h.detach().cpu().numpy()
        hn = h_all / (np.linalg.norm(h_all, axis=-1, keepdims=True) + 1e-9)
        cos_perm = np.einsum("gid,gjd->gij", hn, hn)               # [n_graphs,n,n], network coords
        scale = float(model.sim_scale.detach().cpu()); bias = float(model.sim_bias.detach().cpu())
        cos = np.empty_like(cos_perm)
        for i, inv in enumerate(invs):
            cos[i] = cos_perm[i][np.ix_(inv, inv)]                  # -> base node coordinates
        z = scale * cos + bias                                       # the actual logit
        for i_set, j_set, label in ((L, L, "within_long"), (S, S, "within_short"),
                                     (L, S, "cut"), (S, L, "cut")):
            if len(i_set) == 0 or len(j_set) == 0:
                continue
            if label in ("within_long", "within_short") and len(i_set) <= 1:
                continue
            sub = cos[:, i_set][:, :, j_set]
            subz = z[:, i_set][:, :, j_set]
            if label == "within_long" or label == "within_short":
                off = ~np.eye(len(i_set), dtype=bool)
                vals, valsz = sub[:, off], subz[:, off]
            else:
                vals, valsz = sub.reshape(sub.shape[0], -1), subz.reshape(subz.shape[0], -1)
            rows.append({"split_a": a, "pair_type": label,
                         "mean_cos": round(float(vals.mean()), 4),
                         "frac_positive": round(float((valsz > 0).mean()), 4),
                         "n_pairs": int(vals.size)})
        dL = base_dist[np.ix_(L, L)]
        cosL = cos[:, L][:, :, L]
        zL = z[:, L][:, :, L]
        for d in sorted(set(int(v) for v in dL[dL > 0])):
            m = dL == d
            rows.append({"split_a": a, "pair_type": f"within_long_d{d}",
                         "mean_cos": round(float(cosL[:, m].mean()), 4),
                         "frac_positive": round(float((zL[:, m] > 0).mean()), 4),
                         "n_pairs": int(m.sum())})
        print(f"  readout(similarity) a={a:>2d} done", flush=True)
    return rows


# --------------------------------------------------------------------------
# Tier-1 #4: W_out / W_in geometry
# --------------------------------------------------------------------------
def weights_geometry(model):
    W_out = model.read_out.weight.detach().cpu().numpy()      # [n, d_model]
    W_in = model.read_in.weight.detach().cpu().numpy()        # [d_model, n] -> E_in row k = W_in[:,k]
    E_in = W_in.T                                              # [n, d_model]
    norms_out = np.linalg.norm(W_out, axis=1)                  # ||w_j||
    norms_in = np.linalg.norm(E_in, axis=1)                    # ||e_k||

    def cos_matrix(M):
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        return Mn @ Mn.T

    cos_out = cos_matrix(W_out)
    cos_in = cos_matrix(E_in)
    alignment = E_in @ W_out.T          # M[k,j] = e_k . w_j  -- direct read-in-to-target skip view
    return {"norms_out": norms_out.tolist(), "norms_in": norms_in.tolist(),
            "cos_out": cos_out.tolist(), "cos_in": cos_in.tolist(),
            "alignment_ein_wout": alignment.tolist()}


# --------------------------------------------------------------------------
# Tier-1 #4, SIMILARITY variant. There is no W_out for this read-out (only the
# two learned scalars sim_scale, sim_bias) and therefore no per-target row w_j
# to align E_in against -- the skip-connection check (E_in @ W_out^T) and the
# W_out row-cosine check have no counterpart here. What DOES carry over is
# W_in/E_in itself (shared with the linear model, since read-in is identical
# architecture): if hypothesis 3 (label/order shortcut) were true it would
# still have to show up as structure in E_in, so we report exactly that, plus
# the two read-out scalars for completeness.
# --------------------------------------------------------------------------
def weights_geometry_similarity(model):
    W_in = model.read_in.weight.detach().cpu().numpy()
    E_in = W_in.T
    norms_in = np.linalg.norm(E_in, axis=1)

    def cos_matrix(M):
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        return Mn @ Mn.T

    cos_in = cos_matrix(E_in)
    return {"norms_in": norms_in.tolist(), "cos_in": cos_in.tolist(),
            "sim_scale": float(model.sim_scale.detach().cpu()),
            "sim_bias": float(model.sim_bias.detach().cpu())}


# --------------------------------------------------------------------------
# Tier-1 #5: real node-to-node contribution + row-mass + message contribution,
# representative splits. `contrib_n_graphs` is separate from (and much
# smaller than) `n_graphs`: exact_contribution needs one backward pass per
# query node per graph, so it is far more expensive than the plain
# alpha/message-contribution quantities below.
# --------------------------------------------------------------------------
def attention_probe(model, dev, n, splits, rng, n_graphs, contrib_n_graphs=8, topology="chain"):
    out = {}
    for a in splits:
        base_adj = _build_split_graph(n, a, topology)
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
        alpha0, alpha1 = cache["layer0_alpha"], cache["layer1_alpha"]   # [G,n,n]
        row_mass0 = alpha0.sum(-1); row_mass1 = alpha1.sum(-1)          # normalized-ReLU: doesn't sum to 1
        # message contribution node j sends to i at layer 1: alpha1_ij * ||wo_v1_j||
        wo_v1 = cache["layer1_wo_v"]                                    # [G,n,d]
        contrib1 = alpha1 * np.linalg.norm(wo_v1, axis=-1)[:, None, :]  # [G,n,n]

        # unpermute everything back to base node order (rows AND cols)
        def unperm(mat):
            out_m = np.empty_like(mat)
            for i, inv in enumerate(invs):
                out_m[i] = mat[i][np.ix_(inv, inv)]
            return out_m

        contrib1_b = unperm(contrib1)
        # row-mass is per-node, unpermute rows only
        row_mass0_b = np.stack([row_mass0[i][inv] for i, inv in enumerate(invs)])
        row_mass1_b = np.stack([row_mass1[i][inv] for i, inv in enumerate(invs)])

        # real node-to-node contribution: EXACT Jacobian norm through the
        # whole real forward pass (V, W_O, residual, MLP, LayerNorm), a
        # separate, smaller sample of graphs since it is O(n) backward
        # passes per graph.
        h0_all = cache["h0"]                                            # [G,n,d]
        g_idx = np.arange(min(contrib_n_graphs, n_graphs))
        contrib_exact_list = []
        for gi in g_idx:
            h0_row = torch.from_numpy(h0_all[gi]).to(dev, torch.float32)
            C = exact_contribution(model, h0_row)                       # [n,n], network coords
            contrib_exact_list.append(C[np.ix_(invs[gi], invs[gi])])    # -> base coords
        contrib_exact_b = np.stack(contrib_exact_list)

        out[f"a{a}"] = {
            "contrib_exact_mean": contrib_exact_b.mean(0),
            "row_mass0_mean": row_mass0_b.mean(0), "row_mass1_mean": row_mass1_b.mean(0),
            "contrib1_mean": contrib1_b.mean(0),
            "long_idx": L, "short_idx": S,
        }
        print(f"  attention a={a:>2d} done (row_mass1 mean={row_mass1_b.mean():.3f}, "
              f"exact-contribution over {len(g_idx)} graphs)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=200)
    ap.add_argument("--attn_n_graphs", type=int, default=40)
    ap.add_argument("--contrib_n_graphs", type=int, default=8,
                    help="graphs averaged for the EXACT node-to-node contribution "
                         "(Jacobian norm) -- kept small, it is O(n) backward passes/graph")
    ap.add_argument("--splits", type=int, nargs="+", default=None,
                    help="split range for the dense behavioural/readout sweep; default 1..n//2")
    ap.add_argument("--attn_splits", type=int, nargs="+", default=None,
                    help="representative splits for the (heavier) attention probe")
    ap.add_argument("--topology",
                    choices=["chain", "cycle", "split_cliques", "chorded_cycles", "split_regular3"],
                    default="chain",
                    help="chain = two disjoint paths, split (a, n-a) (Report VI/VII); "
                         "cycle = the same split but each segment closed into a cycle, "
                         "so neither component has a degree-1 path endpoint (Report VIII); "
                         "split_cliques/chorded_cycles/split_regular3 = Report IX controlled-"
                         "distribution battery (degree signature / single landmark / no "
                         "landmark at all controls)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--skip_selftest", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")
    if arch != "roberta":
        raise NotImplementedError("run_with_cache is written for RobertaGraphTransformer only "
                                   "(the base model behind Report VI's Thread B)")
    if not args.skip_selftest:
        _selftest(model, dev, n)
        _selftest_exact_contribution(dev)

    # a cycle needs >= 3 nodes per component; a chorded cycle / 3-regular block needs >= 4;
    # a chain's smallest component is 1 node. split_regular3 additionally requires EVEN
    # block sizes (d*size must be even for d=3) -- filtered out of every candidate list below.
    _MIN_A = {"cycle": 3, "chorded_cycles": 4, "split_regular3": 4}
    min_a = _MIN_A.get(args.topology, 1)

    def _feasible(candidates):
        if args.topology == "split_regular3":
            return [s for s in candidates if s % 2 == 0 and (n - s) % 2 == 0]
        return list(candidates)

    splits = args.splits if args.splits is not None else _feasible(range(min_a, n // 2 + 1))
    # 11, 12, 13 added (istruzioni.md errore 62): the leak-fraction spike found at a=10
    # (Report VII fig:r7attn) was left unresolved because the next probed split jumped
    # straight to 14 -- always probe the immediate neighbours of an interesting point in
    # a sweep, not just the point itself.
    attn_splits = args.attn_splits if args.attn_splits is not None else \
        sorted({s for s in (1, 4, 7, 8, 10, 11, 12, 13, 14, 17, n // 2) if s in splits})

    rng = np.random.default_rng(args.seed)
    print("\n== behavioural sweep (random relabelling, the primary condition) ==")
    rows = behavioural_sweep(model, dev, n, splits, rng, args.n_graphs, fixed_label=False,
                              topology=args.topology)
    print("\n== fixed-label comparison (identity permutation, confirmatory only) ==")
    fixed_splits = sorted({s for s in _feasible((1, 4, 7, 8, 10, 17, n // 2))
                           if min_a <= s <= n // 2})
    rows += behavioural_sweep(model, dev, n, fixed_splits, rng, args.n_graphs, fixed_label=True,
                              topology=args.topology)
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split_a", "mode", "metric", "distance", "value", "n_pairs"])
        w.writeheader(); w.writerows(rows)
    print(f"  saved -> {out}/metrics.csv")

    if readout == "similarity":
        print("\n== readout decomposition cos(h_i,h_j) (similarity read-out) ==")
        rrows = readout_decomposition_similarity(model, dev, n, splits, rng, args.n_graphs,
                                                  topology=args.topology)
        with (out / "readout.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["split_a", "pair_type", "mean_cos", "frac_positive", "n_pairs"])
            w.writeheader(); w.writerows(rrows)
        print(f"  saved -> {out}/readout.csv")

        print("\n== W_in geometry (similarity read-out has no W_out) ==")
        wg = weights_geometry_similarity(model)
    else:
        print("\n== readout decomposition h_i^T w_j ==")
        rrows = readout_decomposition(model, dev, n, splits, rng, args.n_graphs,
                                       topology=args.topology)
        with (out / "readout.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["split_a", "pair_type", "mean_hTw", "frac_positive", "n_pairs"])
            w.writeheader(); w.writerows(rrows)
        print(f"  saved -> {out}/readout.csv")

        print("\n== W_out / W_in geometry ==")
        wg = weights_geometry(model)
    wg["readout_kind"] = readout
    (out / "weights_summary.json").write_text(json.dumps(wg))
    print(f"  saved -> {out}/weights_summary.json")

    print("\n== attention probe (representative splits) ==")
    ap_out = attention_probe(model, dev, n, attn_splits, rng, args.attn_n_graphs,
                              contrib_n_graphs=args.contrib_n_graphs, topology=args.topology)
    npz_dict = {}
    existing_path = out / "attn_cache.npz"
    if existing_path.exists():
        # merge with whatever splits were already cached (e.g. a prior run computed
        # 1,4,7,8,10,14,17,20 and this run only adds 11,12,13) instead of overwriting
        # them -- recomputing the expensive exact-contribution splits from scratch
        # every time a new split is added would waste the earlier, still-valid work.
        with np.load(existing_path) as old:
            npz_dict = {k: old[k] for k in old.files}
    for a_key, d in ap_out.items():
        for k, v in d.items():
            npz_dict[f"{a_key}__{k}"] = np.asarray(v)
    np.savez_compressed(existing_path, **npz_dict)
    print(f"  saved -> {existing_path} ({len(npz_dict)} arrays)")


if __name__ == "__main__":
    main()
