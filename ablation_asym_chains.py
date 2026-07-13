"""Report VII, Tier 2 -- whole-component ablations (eval-only, no retraining).

Localises WHERE the two-component completion signal (Report VII sec 4.1-4.5)
is built, by zeroing one sub-computation at a time in an otherwise normal
forward pass and re-measuring the split sweep. Conditions:

  baseline        -- unmodified model.
  zero_attn{0,1}  -- the attention branch's output is zeroed at that layer
                     (the residual stream skips straight to the FFN branch),
                     i.e. that layer stops mixing information ACROSS nodes.
  zero_mlp{0,1}   -- the feed-forward branch's output is zeroed at that layer
                     (a purely linear node-mixing step survives, but the
                     per-node nonlinear transform is removed).
  bypass          -- both transformer blocks are skipped entirely: logits are
                     read straight off the read-in embedding (no cross-node
                     mixing at all). If this alone reproduces any of the
                     completion signal, the effect would be a property of
                     read-in/read-out alone, not of attention.

If ablating a component collapses the a<=7 "complete the long path" behaviour
back to (or below) the a>=11 baseline, that component is doing the work;
if the ablated model is unaffected, that component is not where the effect
lives.

    python ablation_asym_chains.py --checkpoint runs/.../last.pt \
        --output_dir runs/report7/ablation/<tag>
"""
import argparse, csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import add_self_loops, compute_connectivity_matrix, generate_split_chains_graph
from eval_families import load_model

CONDITIONS = ("baseline", "zero_attn0", "zero_attn1", "zero_mlp0", "zero_mlp1", "bypass")


def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def _apply_readout(model, h):
    """logits for either read-out kind (mirrors model.forward_and_embeddings)."""
    if model.readout_kind == "similarity":
        hn = F.normalize(h, dim=-1)
        return model.sim_scale * torch.matmul(hn, hn.transpose(-1, -2)) + model.sim_bias
    return model.read_out(h)


@torch.no_grad()
def forward_ablated(model, x, condition):
    """Re-derives RobertaGraphTransformer.forward with one branch optionally
    zeroed. Mirrors mechanistic_asym_chains.run_with_cache's block logic."""
    h = model.emb_drop(model.emb_ln(model.read_in(x)))
    if condition == "bypass":
        return _apply_readout(model, h)
    for li, blk in enumerate(model.blocks):
        inp = blk.attn_ln(h) if blk.norm_style != "post" else h
        att = blk.attn
        b, n, d = inp.shape
        if condition == f"zero_attn{li}":
            a_out = torch.zeros_like(h)
        else:
            q = att.q_proj(inp).view(b, n, att.n_heads, att.head_dim).transpose(1, 2)
            k = att.k_proj(inp).view(b, n, att.n_heads, att.head_dim).transpose(1, 2)
            v = att.v_proj(inp).view(b, n, att.n_heads, att.head_dim).transpose(1, 2)
            import math
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(att.head_dim)
            alpha = F.softmax(scores, dim=-1) if att.attn_kind == "softmax" else F.relu(scores) / n
            ctx = torch.matmul(alpha, v).transpose(1, 2).contiguous().view(b, n, d)
            a_out = blk.attn_dense(ctx)
        if blk.norm_style == "post":
            h = blk.attn_ln(h + a_out)
            if condition == f"zero_mlp{li}":
                f = torch.zeros_like(h)
            else:
                f = blk.output(blk.act(blk.intermediate(h)))
            h = blk.out_ln(h + f)
        else:
            h = h + a_out
            if condition == f"zero_mlp{li}":
                f = torch.zeros_like(h)
            else:
                f = blk.output(blk.act(blk.intermediate(blk.out_ln(h))))
            h = h + f
    return _apply_readout(model, h)


def eval_condition(model, dev, n, a, rng, n_graphs, condition):
    base_adj = generate_split_chains_graph(n, a)
    base_y = compute_connectivity_matrix(base_adj).astype(np.int8)
    seg0, seg1 = np.arange(0, a), np.arange(a, n)
    L, S = (seg1, seg0) if (n - a) >= a else (seg0, seg1)

    xs = np.empty((n_graphs, n, n), np.float32)
    invs = []
    for i in range(n_graphs):
        p = rng.permutation(n)
        xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
        invs.append(np.argsort(p))
    xb = torch.from_numpy(xs).to(dev, torch.float32)
    preds = np.empty((n_graphs, n, n), np.int8)
    for s in range(0, n_graphs, 128):
        e = min(s + 128, n_graphs)
        logits = forward_ablated(model, xb[s:e], condition)
        preds[s:e] = (logits > 0).cpu().numpy().astype(np.int8)
    pred = np.empty_like(preds)
    for i, inv in enumerate(invs):
        pred[i] = preds[i][np.ix_(inv, inv)]
    eq = (pred == base_y[None])

    exact = float(eq.reshape(n_graphs, -1).all(1).mean())
    eqL = eq[:, L][:, :, L]; offL = ~np.eye(len(L), dtype=bool)
    reach_long = float(eqL[:, offL].mean())
    reach_short = float("nan")
    if len(S) > 1:
        eqS = eq[:, S][:, :, S]; offS = ~np.eye(len(S), dtype=bool)
        reach_short = float(eqS[:, offS].mean())
    memb = np.zeros(n, dtype=np.int8); memb[L] = 1
    cmask = memb[:, None] != memb[None, :]
    cut = float(eq[:, cmask].mean())
    return {"exact": round(exact, 4), "reach_long": round(reach_long, 4),
            "reach_short": round(reach_short, 4), "cut": round(cut, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_graphs", type=int, default=150)
    ap.add_argument("--splits", type=int, nargs="+", default=None,
                    help="splits to sweep per condition; default 1,4,7,8,10,14,17,20")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = _device()
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")
    if arch != "roberta":
        raise NotImplementedError("forward_ablated targets the RobertaGraphTransformer only")
    splits = args.splits if args.splits is not None else \
        sorted({s for s in (1, 4, 7, 8, 10, 14, 17, n // 2) if 1 <= s <= n // 2})
    rng = np.random.default_rng(args.seed)

    rows = []
    for cond in args.conditions:
        for a in splits:
            m = eval_condition(model, dev, n, a, rng, args.n_graphs, cond)
            rows.append({"condition": cond, "split_a": a, **m})
            print(f"  {cond:>12} a={a:>2d} exact={m['exact']:.3f} reach_long={m['reach_long']:.3f} "
                  f"reach_short={m['reach_short']:.3f} cut={m['cut']:.3f}", flush=True)

    with (out / "ablation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "split_a", "exact", "reach_long", "reach_short", "cut"])
        w.writeheader(); w.writerows(rows)
    print(f"  saved -> {out}/ablation.csv")


if __name__ == "__main__":
    main()
