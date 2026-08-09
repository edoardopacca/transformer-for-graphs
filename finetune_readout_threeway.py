"""Report 10 (paper prep, §5.3 "Generalization to three paths, finetuning??" of
idea_paper_by_prof.tex) -- targeted fine-tuning of ONLY the similarity read-out's
scale/bias on specific, NAMED three-way split-chain graphs.

Where this comes from: on the split-chains-only n=46 seed-1000 checkpoint (Report IX
Thread A.3, stagewise_threeway.py), two three-component cells stand out --
(15,15,16) (fully balanced -- every stagewise cosine block already looks
"almost learned": within-component blocks are much darker/more positive than the
cross-component ones, yet all three cuts fail) and (7,15,24) (two of the three cuts
succeed, cut(2,3) fails, and pairs inside the 24-node component beyond the doubled
wall 2*3^L=18 are also at stake). In both, the cross-component logits sit just
ABOVE zero -- the decision threshold, not the underlying geometry, looks like the
proximate problem. This asks the narrow, sharp question the earlier Report IX
experiment 6 (finetune_readout_only.py, generic K in {3..6} path_union stream) did
NOT: can two scalars fix these two SPECIFIC hard cells if the fine-tuning stream
contains only them (not a broad, conflicting-gradient K>=3 mixture)?

freeze_trunk is reused UNMODIFIED from finetune_readout_only.py (every parameter
frozen except sim_scale/sim_bias). The training stream draws uniformly from
--target_cells (each an explicit "s1,s2,s3" three-way split built with
generate_multi_path_split_graph -- the data principle of sec 9/istruzioni.md: a
single, explicitly named distribution, here a named FINITE SET of cells, not an
opaque range).

Evaluation (before AND after fine-tuning, same fixed eval seed each time so the two
are comparable), per target cell:
  - whole-graph exact match, per-component reach, the three pairwise cuts, the
    overall predicted-positive rate (as in eval_threeway_splitchains.py / A.3)
  - within the LARGEST component of that cell, reach split into three shortest-path
    distance buckets: within-capacity (d<=9), between-the-walls (9<d<=18), and
    BEYOND the doubled wall (d>18) -- the "did it learn past the bound" question,
    only non-trivial for (7,15,24) (24-node path, diameter 23).
Also evaluates the SAME two cells even if only one was in --target_cells, so a
cell held out of fine-tuning (e.g. train only on (15,15,16), still eval on
(7,15,24)) shows whether the threshold shift transfers.

This script does NOT check the K=2 own-family (split-chains) sweep -- run
eval_asym_chains.py separately on the fine-tuned checkpoint (exactly as the
Report IX sbatch scripts do for finetune_readout_only.py) to check for forgetting.

    python finetune_readout_threeway.py --checkpoint runs/report9/n46_train/n46_split_chains_roberta_similarity_lam0_seed1000/last.pt \\
        --output_dir runs/report10/finetune_readout_threeway/<tag> \\
        --target_cells 15,15,16 7,15,24 --finetune_steps 50000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from data import (add_self_loops, compute_all_pairs_shortest_paths,
                   compute_connectivity_matrix, generate_multi_path_split_graph)
from eval_families import load_model
from finetune_readout_only import freeze_trunk
from utils import ensure_dir, get_device, set_seed

PAIRS = [((0, 1), "12"), ((0, 2), "13"), ((1, 2), "23")]
EVAL_CELLS = [(15, 15, 16), (7, 15, 24)]  # always evaluated, whichever is/isn't trained on


def parse_cell(s: str) -> tuple[int, ...]:
    sizes = tuple(int(x) for x in s.split(","))
    if len(sizes) != 3:
        raise ValueError(f"expected 'a,b,c', got {s!r}")
    return sizes


def sample_batch(n, batch_size, rng, target_cells):
    xs = np.empty((batch_size, n, n), np.float32)
    ys = np.empty((batch_size, n, n), np.float32)
    choice = rng.integers(0, len(target_cells), size=batch_size)
    for i in range(batch_size):
        base = generate_multi_path_split_graph(n, target_cells[choice[i]])
        p = rng.permutation(n)
        a = base[np.ix_(p, p)]
        xs[i] = add_self_loops(a)
        ys[i] = compute_connectivity_matrix(a).astype(np.float32)
    return xs, ys


def _cell_geometry(n, sizes):
    bounds = [0]
    for s in sizes:
        bounds.append(bounds[-1] + s)
    comps = [np.arange(bounds[i], bounds[i + 1]) for i in range(len(sizes))]
    base_adj = generate_multi_path_split_graph(n, sizes)
    base_y = compute_connectivity_matrix(base_adj).astype(np.int64)
    base_dist = compute_all_pairs_shortest_paths(base_adj)
    return comps, base_adj, base_y, base_dist


@torch.no_grad()
def evaluate_cell(model, dev, n, sizes, rng, n_graphs=300, batch=100):
    """Eval-only on ONE three-way cell. Returns a flat dict of metrics."""
    model.eval()
    comps, base_adj, base_y, base_dist = _cell_geometry(n, sizes)
    largest_i = max(range(len(sizes)), key=lambda i: sizes[i])
    lc = comps[largest_i]
    ld = base_dist[np.ix_(lc, lc)]
    off_l = ~np.eye(len(lc), dtype=bool)
    buckets = {
        "within_capacity_d<=9": (ld >= 0) & (ld <= 9) & off_l,
        "between_walls_9<d<=18": (ld > 9) & (ld <= 18) & off_l,
        "beyond_doubled_wall_d>18": (ld > 18) & off_l,
    }

    xs = np.empty((n_graphs, n, n), np.float32)
    invs = np.empty((n_graphs, n), np.int64)
    for i in range(n_graphs):
        p = rng.permutation(n)
        xs[i] = add_self_loops(base_adj[np.ix_(p, p)])
        invs[i] = np.argsort(p)

    exact = np.empty(n_graphs, dtype=bool)
    pos_rate = np.empty(n_graphs, dtype=np.float64)
    reach = {ci: [] for ci in range(len(sizes))}
    cuts = {name: [] for _, name in PAIRS}
    bucket_hits = {name: [] for name in buckets}

    for s in range(0, n_graphs, batch):
        e = min(s + batch, n_graphs)
        xb = torch.from_numpy(xs[s:e]).to(dev, torch.float32)
        pred = (model(xb) > 0).cpu().numpy()
        pred_base = np.stack([pred[k - s][np.ix_(invs[k], invs[k])] for k in range(s, e)])
        eq = pred_base == base_y[None]
        exact[s:e] = eq.reshape(e - s, -1).all(1)
        pos_rate[s:e] = pred_base.reshape(e - s, -1).mean(1)
        for ci, c in enumerate(comps):
            if len(c) > 1:
                offc = ~np.eye(len(c), dtype=bool)
                reach[ci].append(eq[:, c][:, :, c][:, offc])
        for (a_i, b_i), name in PAIRS:
            cuts[name].append(eq[:, comps[a_i]][:, :, comps[b_i]].reshape(e - s, -1))
        sub_eq = eq[:, lc][:, :, lc]
        for name, mask in buckets.items():
            bucket_hits[name].append(sub_eq[:, mask] if mask.any() else np.zeros((e - s, 0)))
    model.train()

    out = {"exact": float(exact.mean()), "pred_positive_rate": float(pos_rate.mean())}
    for ci in range(len(sizes)):
        out[f"reach_comp{ci + 1}"] = (float(np.concatenate(reach[ci]).mean())
                                       if reach[ci] and reach[ci][0].size else float("nan"))
    for _, name in PAIRS:
        out[f"cut_{name}"] = float(np.concatenate(cuts[name]).mean())
    for name, hits in bucket_hits.items():
        cat = np.concatenate(hits, axis=0) if hits else np.zeros((0, 0))
        out[f"largest_comp_{name}"] = float(cat.mean()) if cat.size else float("nan")
    return out


def evaluate_all(model, dev, n, cells, seed, n_graphs=300):
    return {f"s{'_'.join(map(str, sizes))}": evaluate_cell(
        model, dev, n, sizes, np.random.default_rng(seed), n_graphs=n_graphs)
        for sizes in cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="pretrained (full-transformer) checkpoint")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--target_cells", type=str, nargs="+", required=True,
                     help="fine-tuning stream: one or more 's1,s2,s3' cells, drawn uniformly "
                          "per sample, e.g. --target_cells 15,15,16 7,15,24")
    ap.add_argument("--finetune_steps", type=int, default=50000,
                     help="deliberately much larger than the earlier generic K>=3 probe "
                          "(finetune_readout_only.py default 3000): only 2 scalars are "
                          "trainable, so this is still cheap")
    ap.add_argument("--batch_size", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_graphs", type=int, default=300)
    ap.add_argument("--eval_seed", type=int, default=999_999,
                     help="fixed eval stream seed, same before/after/every checkpoint")
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.output_dir); ensure_dir(out)
    set_seed(args.seed)
    dev = get_device("auto")
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    if readout != "similarity":
        raise NotImplementedError("this script targets the similarity read-out only")
    target_cells = [parse_cell(c) for c in args.target_cells]
    for sizes in target_cells:
        if sum(sizes) != n:
            raise ValueError(f"cell {sizes} does not sum to n={n}")
    eval_cells = sorted(set(EVAL_CELLS) | set(target_cells))
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}\n"
          f"  fine-tuning stream (uniform over): {target_cells}\n"
          f"  always evaluated: {eval_cells}")

    trainable, names = freeze_trunk(model, readout)
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"fine-tuning ONLY {names} ({n_trainable} / {n_total} parameters)")

    scale_before = float(model.sim_scale.detach().cpu())
    bias_before = float(model.sim_bias.detach().cpu())
    print(f"scale/bias BEFORE fine-tuning: scale={scale_before:.6f} bias={bias_before:.6f}")

    opt = AdamW(trainable, lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)

    metrics0 = evaluate_all(model, dev, n, eval_cells, args.eval_seed, args.eval_graphs)
    print(f"step      0 (before fine-tuning):")
    for tag, m in metrics0.items():
        print(f"  {tag}: exact={m['exact']:.4f} cut_23={m['cut_23']:.4f} "
              f"pred_pos={m['pred_positive_rate']:.4f}")
    history = [{"step": 0, "loss": None, "scale": scale_before, "bias": bias_before,
                "metrics": metrics0}]

    model.train()
    for step in range(1, args.finetune_steps + 1):
        xs, ys = sample_batch(n, args.batch_size, rng, target_cells)
        xb = torch.from_numpy(xs).to(dev, torch.float32)
        yb = torch.from_numpy(ys).to(dev, torch.float32)
        logits = model(xb)
        loss = criterion(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0 or step == args.finetune_steps:
            metrics = evaluate_all(model, dev, n, eval_cells, args.eval_seed, args.eval_graphs)
            scale_now = float(model.sim_scale.detach().cpu())
            bias_now = float(model.sim_bias.detach().cpu())
            print(f"step {step:>6d} loss={loss.item():.4f} scale={scale_now:.6f} "
                  f"bias={bias_now:.6f}", flush=True)
            for tag, m in metrics.items():
                print(f"  {tag}: exact={m['exact']:.4f} cut_23={m['cut_23']:.4f} "
                      f"pred_pos={m['pred_positive_rate']:.4f}")
            history.append({"step": step, "loss": float(loss.item()), "scale": scale_now,
                             "bias": bias_now, "metrics": metrics})

    scale_after = float(model.sim_scale.detach().cpu())
    bias_after = float(model.sim_bias.detach().cpu())
    print(f"scale/bias AFTER fine-tuning: scale={scale_after:.6f} bias={bias_after:.6f}\n"
          f"  delta: d_scale={scale_after - scale_before:.6f} d_bias={bias_after - bias_before:.6f}")

    ck = {"model_state_dict": model.state_dict(), "model_config": mcfg.__dict__,
          "finetune_args": vars(args), "source_checkpoint": args.checkpoint}
    torch.save(ck, out / "finetuned.pt")
    (out / "finetune_history.json").write_text(json.dumps(history, indent=2))
    (out / "scale_bias_summary.json").write_text(json.dumps({
        "scale_before": scale_before, "bias_before": bias_before,
        "scale_after": scale_after, "bias_after": bias_after,
        "delta_scale": scale_after - scale_before, "delta_bias": bias_after - bias_before,
    }, indent=2))
    print(f"done -> {out}/finetuned.pt")


if __name__ == "__main__":
    main()
