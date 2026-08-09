"""Report 10 (paper prep, sec 5.3) -- the BROADER companion to
finetune_readout_threeway.py. Instead of fine-tuning the read-out on two named
hard cells, this trains on the FULL distribution of exactly-three-component path
splits (every (s1,s2,s3) combination, not just (15,15,16)/(7,15,24)), for a much
longer budget, to ask a different question: does exposure to the whole K=3 space
let the read-out alone learn a genuinely GENERAL three-component rule, rather than
overfitting two specific hard cells?

Training stream: generate_path_union_graph(n, rng, min_paths=3, max_paths=3) --
already implemented and used by finetune_readout_only.py (Report IX experiment 6,
there with min_paths=3/max_paths=6 and only 3000 steps); here min_paths=max_paths=3
(exactly three components, every size combination reachable, since the two cut
points are drawn uniformly) and the step budget is far longer (default 200000 --
only 2 scalars are trainable, so this stays cheap). freeze_trunk is reused
unmodified.

Evaluation, before and every --eval_every steps: (a) the SAME two named cells
(15,15,16) and (7,15,24) tracked by finetune_readout_threeway.py -- for a direct,
apples-to-apples comparison against the targeted experiment; (b) an aggregate
exact/pairwise accuracy over freshly sampled generic three-component graphs (the
same style of number Report IX experiment 6 reported for its K in {3..6} stream).
Own-family (K=2) retention is checked the same way as the targeted experiment: run
eval_asym_chains.py on the fine-tuned checkpoint separately (not in this script).

    python finetune_readout_threeway_full.py --checkpoint runs/report9/n46_train/n46_split_chains_roberta_similarity_lam0_seed1000/last.pt \\
        --output_dir runs/report10/finetune_readout_full3way/<tag> \\
        --finetune_steps 200000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from eval_families import load_model
from finetune_readout_only import evaluate as evaluate_k3_aggregate
from finetune_readout_only import freeze_trunk
from finetune_readout_only import sample_batch as sample_batch_k3
from finetune_readout_threeway import EVAL_CELLS
from finetune_readout_threeway import evaluate_all as evaluate_named_cells
from utils import ensure_dir, get_device, set_seed


def full_eval(model, dev, n, eval_seed, eval_graphs_aggregate, eval_graphs_named):
    agg_exact, agg_pairwise = evaluate_k3_aggregate(
        model, dev, n, np.random.default_rng(eval_seed), min_paths=3, max_paths=3,
        n_graphs=eval_graphs_aggregate)
    named = evaluate_named_cells(model, dev, n, EVAL_CELLS, eval_seed, eval_graphs_named)
    metrics = dict(named)
    metrics["_aggregate_k3_exact"] = agg_exact
    metrics["_aggregate_k3_pairwise"] = agg_pairwise
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="pretrained (full-transformer) checkpoint")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--finetune_steps", type=int, default=200000,
                     help="much longer than the targeted 2-cell experiment (50000): covering "
                          "the full K=3 space needs more exposure per cell")
    ap.add_argument("--batch_size", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--eval_graphs_aggregate", type=int, default=300)
    ap.add_argument("--eval_graphs_named", type=int, default=300)
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
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}\n"
          f"  fine-tuning stream: generate_path_union_graph(min_paths=3, max_paths=3) "
          f"-- the FULL three-component split space\n"
          f"  always evaluated (named cells): {EVAL_CELLS}")

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

    metrics0 = full_eval(model, dev, n, args.eval_seed, args.eval_graphs_aggregate,
                          args.eval_graphs_named)
    print(f"step      0 (before fine-tuning): aggregate_k3 exact={metrics0['_aggregate_k3_exact']:.4f} "
          f"pairwise={metrics0['_aggregate_k3_pairwise']:.4f}")
    for tag in ("s15_15_16", "s7_15_24"):
        m = metrics0[tag]
        print(f"  {tag}: exact={m['exact']:.4f} cut_23={m['cut_23']:.4f} "
              f"pred_pos={m['pred_positive_rate']:.4f}")
    history = [{"step": 0, "loss": None, "scale": scale_before, "bias": bias_before,
                "metrics": metrics0}]

    model.train()
    for step in range(1, args.finetune_steps + 1):
        xs, ys = sample_batch_k3(n, args.batch_size, rng, min_paths=3, max_paths=3)
        xb = torch.from_numpy(xs).to(dev, torch.float32)
        yb = torch.from_numpy(ys).to(dev, torch.float32)
        logits = model(xb)
        loss = criterion(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0 or step == args.finetune_steps:
            metrics = full_eval(model, dev, n, args.eval_seed, args.eval_graphs_aggregate,
                                 args.eval_graphs_named)
            scale_now = float(model.sim_scale.detach().cpu())
            bias_now = float(model.sim_bias.detach().cpu())
            print(f"step {step:>7d} loss={loss.item():.4f} scale={scale_now:.6f} "
                  f"bias={bias_now:.6f} aggregate_k3 exact={metrics['_aggregate_k3_exact']:.4f} "
                  f"pairwise={metrics['_aggregate_k3_pairwise']:.4f}", flush=True)
            for tag in ("s15_15_16", "s7_15_24"):
                m = metrics[tag]
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
