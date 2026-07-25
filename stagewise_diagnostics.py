"""Report VIII -- stagewise diagnostics: WHERE in the forward pass is
connectivity information combined? (eval-only, no training, Priority 1 of the
"where is information combined" experimental thread; see istruzioni.md sec 15
for the full 12-section brief this implements Priority 1 of.)

Saves every intermediate state of the two-layer trunk (not only H^(1)/H^(2)):
for layer ell in {1,2}, AttnOut^(ell) = (alpha V) W_O, H_attn^(ell) =
LayerNorm(H^(ell-1) + AttnOut^(ell)), FFNOut^(ell) = FFN(H_attn^(ell)),
H^(ell) = LayerNorm(H_attn^(ell) + FFNOut^(ell)) -- this is `run_with_stages`,
a manual re-derivation of RobertaGraphTransformer's forward pass (mirrors
mechanistic_asym_chains.py::run_with_cache, extended with the two intra-layer
states that script did not keep). Does not touch model.py or change the
model's normal forward behaviour.

For the five MAIN stages (H^(0), H_attn^(1), H^(1), H_attn^(2), H^(2) ==
final embeddings) computes, per split:
  * the node-node cosine geometry G^X_ij = cos(x_i, x_j) -- the quantity the
    similarity read-out actually uses, at every stage, not only the last;
  * an "intermediate read-out" probe Z^X = scale*G^X + bias (the model's own
    trained scale/bias applied to an untrained stage, diagnostic only, never
    a real prediction the model makes) and its accuracy (exact match, reach
    short/long, reach long split into near <=9 / far >9, cut, positive rate);
  * the margin M_far^X = mean(G^X over far within-long pairs) - mean(G^X
    over cross pairs) -- the model's own trained decision boundary is
    scale*cos+bias=0, so this margin is exactly what would have to grow
    positive for that decision to already be right at stage X;
  * the four inter-stage logit deltas dZ_attn1, dZ_mlp1, dZ_attn2, dZ_mlp2
    (each a signed 40x40 heatmap) and their category means (within-short,
    within-long-near, within-long-far, cross).

Only the similarity read-out is supported (this report's standard, sec:setup)
-- the cosine-geometry framing is specific to it.

    python stagewise_diagnostics.py --checkpoint runs/.../last.pt \\
        --output_dir runs/report8/stagewise/<tag>
"""
import argparse, csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import (add_self_loops, compute_connectivity_matrix,
                  compute_all_pairs_shortest_paths)
from eval_families import load_model
from mechanistic_asym_chains import _build_split_graph

MAIN_STAGES = ["H0", "Hattn1", "H1", "Hattn2", "H2"]
SUBBLOCKS = [("dZ_attn1", "Hattn1", "H0"), ("dZ_mlp1", "H1", "Hattn1"),
             ("dZ_attn2", "Hattn2", "H1"), ("dZ_mlp2", "H2", "Hattn2")]


def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def run_with_stages(model, x):
    """x: [B,n,n] self-loop-augmented adjacency. Returns (stages, attn_cache,
    logits): stages holds H0/AttnOut{1,2}/Hattn{1,2}/FFNOut{1,2}/H{1,2} as
    [B,n,d] numpy arrays; attn_cache holds per-layer q/k/v/scores/alpha
    ([B,n,d] or [B,n,n], head dim squeezed -- single head); logits is [B,n,n]."""
    h = model.emb_drop(model.emb_ln(model.read_in(x)))
    stages = {"H0": h.detach().cpu().numpy()}
    attn_cache = {}
    for li, blk in enumerate(model.blocks):
        att = blk.attn
        b, n, d = h.shape
        q = att.q_proj(h).view(b, n, att.n_heads, att.head_dim).transpose(1, 2)
        k = att.k_proj(h).view(b, n, att.n_heads, att.head_dim).transpose(1, 2)
        v = att.v_proj(h).view(b, n, att.n_heads, att.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (att.head_dim ** 0.5)
        alpha = F.softmax(scores, dim=-1) if att.attn_kind == "softmax" else F.relu(scores) / n
        ctx = torch.matmul(alpha, v).transpose(1, 2).contiguous().view(b, n, d)
        attn_out = blk.attn_dense(ctx)
        h_attn = blk.attn_ln(h + attn_out)
        ffn_out = blk.output(blk.act(blk.intermediate(h_attn)))
        h_new = blk.out_ln(h_attn + ffn_out)

        attn_cache[f"layer{li}_q"] = q[:, 0].detach().cpu().numpy()
        attn_cache[f"layer{li}_k"] = k[:, 0].detach().cpu().numpy()
        attn_cache[f"layer{li}_v"] = v[:, 0].detach().cpu().numpy()
        attn_cache[f"layer{li}_scores"] = scores[:, 0].detach().cpu().numpy()
        attn_cache[f"layer{li}_alpha"] = alpha[:, 0].detach().cpu().numpy()
        stages[f"AttnOut{li + 1}"] = attn_out.detach().cpu().numpy()
        stages[f"Hattn{li + 1}"] = h_attn.detach().cpu().numpy()
        stages[f"FFNOut{li + 1}"] = ffn_out.detach().cpu().numpy()
        stages[f"H{li + 1}"] = h_new.detach().cpu().numpy()
        h = h_new
    if model.readout_kind == "similarity":
        hn = F.normalize(h, dim=-1)
        logits = model.sim_scale * torch.matmul(hn, hn.transpose(-1, -2)) + model.sim_bias
    else:
        logits = model.read_out(h)
    return stages, attn_cache, logits.detach().cpu().numpy()


def _selftest(model, dev, n):
    x = torch.rand(2, n, n, device=dev)
    x = ((x + x.transpose(-1, -2)) > 1.3).float()
    for i in range(2):
        x[i].fill_diagonal_(1.0)
    stages, _, logits = run_with_stages(model, x)
    logits_ref, h_ref = model.forward_and_embeddings(x)
    assert np.allclose(stages["H2"], h_ref.detach().cpu().numpy(), atol=1e-4), \
        "run_with_stages H2 != model.forward_and_embeddings"
    assert np.allclose(logits, logits_ref.detach().cpu().numpy(), atol=1e-3), \
        "run_with_stages logits != model.forward_and_embeddings"
    print("  [selftest] run_with_stages H2/logits match model.forward_and_embeddings exactly")


def _cosine_batch(arr):
    """arr: [G,n,d] -> [G,n,n] cosine similarity matrices."""
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    normed = arr / np.maximum(norm, 1e-8)
    return np.einsum("gid,gjd->gij", normed, normed)


def stagewise_probe(model, dev, n, splits, rng, n_graphs, cap=9, topology="chain"):
    if model.readout_kind != "similarity":
        raise NotImplementedError("this script is specific to the similarity read-out "
                                   "(report/8 sec:setup standardizes on it)")
    scale = float(model.sim_scale.detach().cpu())
    bias = float(model.sim_bias.detach().cpu())
    out = {}
    for a in splits:
        base_adj = _build_split_graph(n, a, topology)
        base_y = compute_connectivity_matrix(base_adj).astype(np.int8)
        base_dist = compute_all_pairs_shortest_paths(base_adj)
        seg0, seg1 = np.arange(0, a), np.arange(a, n)
        L, S = (seg1, seg0) if (n - a) >= a else (seg0, seg1)
        offL = ~np.eye(len(L), dtype=bool)
        offS = ~np.eye(len(S), dtype=bool) if len(S) > 1 else None
        dL = base_dist[np.ix_(L, L)]
        near_mask = offL & (dL <= cap)
        far_mask = offL & (dL > cap)
        memb = np.zeros(n, dtype=np.int8); memb[L] = 1
        cmask = memb[:, None] != memb[None, :]

        xs = np.empty((n_graphs, n, n), np.float32)
        invs = []
        for i in range(n_graphs):
            p = rng.permutation(n)
            xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
            invs.append(np.argsort(p))
        xb = torch.from_numpy(xs).to(dev, torch.float32)
        stages, _, _ = run_with_stages(model, xb)

        def unperm(arr):  # [G,n,d] -> unpermuted to base node order
            return np.stack([arr[i][inv] for i, inv in enumerate(invs)])

        G = {}   # stage -> [n,n] mean cosine matrix (base order)
        for X in MAIN_STAGES:
            arr_base = unperm(stages[X])           # [G,n,d]
            G_all = _cosine_batch(arr_base)         # [G,n,n]
            G[X] = G_all.mean(0)

        metrics_rows, margin_rows = [], []
        for X in MAIN_STAGES:
            arr_base = unperm(stages[X])
            G_all = _cosine_batch(arr_base)                     # [G,n,n]
            Z_all = scale * G_all + bias
            Rhat = (Z_all > 0)
            eq = (Rhat == base_y[None])
            exact = float(eq.reshape(n_graphs, -1).all(1).mean())
            reach_long = float(eq[:, L][:, :, L][:, offL].mean())
            reach_long_near = float(eq[:, L][:, :, L][:, near_mask].mean()) if near_mask.any() else float("nan")
            reach_long_far = float(eq[:, L][:, :, L][:, far_mask].mean()) if far_mask.any() else float("nan")
            reach_short = float(eq[:, S][:, :, S][:, offS].mean()) if offS is not None else float("nan")
            cut = float(eq[:, cmask].mean())
            pos_rate = float(Rhat.mean())
            for metric, value in [("exact", exact), ("reach_long", reach_long),
                                   ("reach_long_near", reach_long_near),
                                   ("reach_long_far", reach_long_far),
                                   ("reach_short", reach_short), ("cut", cut),
                                   ("pred_positive_rate", pos_rate)]:
                metrics_rows.append({"split_a": a, "stage": X, "metric": metric,
                                      "value": round(value, 5) if value == value else value})

            Gx = G[X]
            mu_short = float(Gx[np.ix_(S, S)][offS].mean()) if offS is not None else float("nan")
            mu_long_near = float(Gx[np.ix_(L, L)][near_mask].mean()) if near_mask.any() else float("nan")
            mu_long_far = float(Gx[np.ix_(L, L)][far_mask].mean()) if far_mask.any() else float("nan")
            mu_cross = float(Gx[np.ix_(L, S)].mean())
            m_far = mu_long_far - mu_cross if mu_long_far == mu_long_far else float("nan")
            for quantity, value in [("mu_short", mu_short), ("mu_long_near", mu_long_near),
                                     ("mu_long_far", mu_long_far), ("mu_cross", mu_cross),
                                     ("M_far", m_far)]:
                margin_rows.append({"split_a": a, "stage": X, "quantity": quantity,
                                     "value": round(value, 5) if value == value else value})

        Z = {X: scale * G[X] + bias for X in MAIN_STAGES}
        dZ = {name: Z[to] - Z[frm] for name, to, frm in SUBBLOCKS}
        deltaz_rows = []
        for branch, mat in dZ.items():
            cats = [("within_short", mat[np.ix_(S, S)][offS].mean() if offS is not None else float("nan")),
                    ("within_long_near", mat[np.ix_(L, L)][near_mask].mean() if near_mask.any() else float("nan")),
                    ("within_long_far", mat[np.ix_(L, L)][far_mask].mean() if far_mask.any() else float("nan")),
                    ("cross", mat[np.ix_(L, S)].mean())]
            for cat, value in cats:
                deltaz_rows.append({"split_a": a, "branch": branch, "category": cat,
                                     "value": round(float(value), 5) if value == value else value})

        out[f"a{a}"] = {"G": G, "dZ": dZ, "long_idx": L, "short_idx": S,
                        "metrics_rows": metrics_rows, "margin_rows": margin_rows,
                        "deltaz_rows": deltaz_rows}
        print(f"  stagewise a={a:>2d} done "
              f"(M_far H0={ [r['value'] for r in margin_rows if r['stage']=='H0' and r['quantity']=='M_far'][0]:.3f}, "
              f"H2={[r['value'] for r in margin_rows if r['stage']=='H2' and r['quantity']=='M_far'][0]:.3f})",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=64)
    ap.add_argument("--splits", type=int, nargs="+", default=[4, 7, 8, 10, 14, 20])
    ap.add_argument("--cap", type=int, default=9)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--skip_selftest", action="store_true")
    ap.add_argument("--topology", choices=["chain", "cycle"], default="chain",
                    help="chain (default, every two-chain result since Report VI) or cycle "
                         "(report/8 sec:cycles -- each segment closed into a ring)")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")
    if arch != "roberta":
        raise NotImplementedError("this script targets the RobertaGraphTransformer only")
    if not args.skip_selftest:
        _selftest(model, dev, n)

    rng = np.random.default_rng(args.seed)
    probe = stagewise_probe(model, dev, n, args.splits, rng, args.n_graphs, cap=args.cap,
                            topology=args.topology)

    # Saved once per checkpoint (a single global scalar, not per split/stage) so the plot
    # script can render scale*cos+bias -- the model's own decision boundary at 0 -- instead
    # of raw cosine, whose "connected" threshold sits at an arbitrary cos>-bias/scale and
    # makes a cosine-centred diverging colormap misleading (istruzioni.md, Report IX request).
    npz_dict = {"scale": np.float64(float(model.sim_scale.detach().cpu())),
                "bias": np.float64(float(model.sim_bias.detach().cpu()))}
    metrics_rows, margin_rows, deltaz_rows = [], [], []
    for a_key, d in probe.items():
        for X, mat in d["G"].items():
            npz_dict[f"{a_key}__G_{X}"] = mat
        for branch, mat in d["dZ"].items():
            npz_dict[f"{a_key}__{branch}"] = mat
        npz_dict[f"{a_key}__long_idx"] = d["long_idx"]
        npz_dict[f"{a_key}__short_idx"] = d["short_idx"]
        metrics_rows += d["metrics_rows"]
        margin_rows += d["margin_rows"]
        deltaz_rows += d["deltaz_rows"]
    np.savez_compressed(out / "stagewise_geometry.npz", **npz_dict)
    with (out / "stagewise_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split_a", "stage", "metric", "value"])
        w.writeheader(); w.writerows(metrics_rows)
    with (out / "stagewise_margins.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split_a", "stage", "quantity", "value"])
        w.writeheader(); w.writerows(margin_rows)
    with (out / "stagewise_deltaz.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split_a", "branch", "category", "value"])
        w.writeheader(); w.writerows(deltaz_rows)
    print(f"  saved -> {out}/stagewise_{{geometry.npz,metrics.csv,margins.csv,deltaz.csv}}")


if __name__ == "__main__":
    main()
