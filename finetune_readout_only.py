"""Report IX -- prof's fine-tuning question (2026-07-27 email, verbatim): "per i
cycles/paths, si potrebbe provare a trainare tutto il transformer su 2 paths e 2
cycles, e poi fare qualche step di fine-tuning sulla distribuzione test (ad esempio,
3 o piu paths) solo del readout."

Loads a checkpoint already pretrained on the FULL transformer (e.g. --families
split_chains,split_cycles), freezes every parameter except the read-out, and runs a
short fine-tuning run on a genuinely different test distribution -- K>=3 disjoint
paths (generate_path_union_graph with min_paths=3), never seen structurally by a
model pretrained only on 2-component splits. Question: does recalibrating only the
read-out (for the similarity read-out, just the two scalars sim_scale/sim_bias; for
the linear read-out, the read_out row vectors) suffice to generalise, or does the
trunk itself have to change? Logs exact/pairwise accuracy on held-out K>=3 graphs
before and periodically during fine-tuning, and saves the fine-tuned checkpoint
(same model_config as the source, so every existing eval/mechanistic script in this
project loads it unchanged) plus a small json history.

  python finetune_readout_only.py --checkpoint <pretrained.pt> \
      --output_dir runs/report9/finetune_readout/<tag> \
      --min_paths 3 --max_paths 6 --finetune_steps 3000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from data import add_self_loops, compute_connectivity_matrix, generate_path_union_graph
from eval_families import load_model
from utils import ensure_dir, get_device, set_seed


def freeze_trunk(model, readout):
    """Freeze every parameter, then re-enable only the read-out ones. Returns the
    list of trainable parameters (for the optimiser) and their names (for logging)."""
    for p in model.parameters():
        p.requires_grad_(False)
    if readout == "similarity":
        model.sim_scale.requires_grad_(True)
        model.sim_bias.requires_grad_(True)
        trainable = [model.sim_scale, model.sim_bias]
        names = ["sim_scale", "sim_bias"]
    else:
        model.read_out.weight.requires_grad_(True)
        model.read_out.bias.requires_grad_(True)
        trainable = [model.read_out.weight, model.read_out.bias]
        names = ["read_out.weight", "read_out.bias"]
    return trainable, names


def sample_batch(n, batch_size, rng, min_paths, max_paths):
    xs = np.empty((batch_size, n, n), np.float32)
    ys = np.empty((batch_size, n, n), np.float32)
    for i in range(batch_size):
        a = generate_path_union_graph(n, rng, max_paths=max_paths, min_paths=min_paths)
        p = rng.permutation(n)
        a = a[np.ix_(p, p)]
        xs[i] = add_self_loops(a)
        ys[i] = compute_connectivity_matrix(a).astype(np.float32)
    return xs, ys


@torch.no_grad()
def evaluate(model, dev, n, rng, min_paths, max_paths, n_graphs=300, batch=100):
    model.eval()
    xs, ys = sample_batch(n, n_graphs, rng, min_paths, max_paths)
    exact_all, pairwise_all = [], []
    for s in range(0, n_graphs, batch):
        e = min(s + batch, n_graphs)
        xb = torch.from_numpy(xs[s:e]).to(dev, torch.float32)
        logits = model(xb)
        pred = (logits > 0).float().cpu().numpy()
        eq = pred == ys[s:e]
        exact_all.append(eq.reshape(e - s, -1).all(1))
        pairwise_all.append(eq.reshape(e - s, -1).mean(1))
    model.train()
    return float(np.concatenate(exact_all).mean()), float(np.concatenate(pairwise_all).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="pretrained (full-transformer) checkpoint")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--min_paths", type=int, default=3,
                    help="K>=this many disjoint-path components (never seen by a "
                         "2-component-only pretrain); default 3")
    ap.add_argument("--max_paths", type=int, default=6)
    ap.add_argument("--finetune_steps", type=int, default=3000,
                    help="'qualche step' -- deliberately small; this is a probe of "
                         "whether the read-out ALONE can adapt, not a full retrain")
    ap.add_argument("--batch_size", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_graphs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.output_dir); ensure_dir(out)
    set_seed(args.seed)
    dev = get_device("auto")
    model, mcfg, arch, readout = load_model(args.checkpoint, dev)
    n = mcfg.n
    print(f"checkpoint={args.checkpoint}\n  arch={arch} readout={readout} n={n} device={dev}")

    trainable, names = freeze_trunk(model, readout)
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"fine-tuning ONLY {names} ({n_trainable} / {n_total} parameters)")

    opt = AdamW(trainable, lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)

    exact0, pairwise0 = evaluate(model, dev, n, np.random.default_rng(args.seed + 999_999),
                                 args.min_paths, args.max_paths, args.eval_graphs)
    print(f"step      0 (before fine-tuning): exact={exact0:.4f} pairwise={pairwise0:.4f}", flush=True)
    history = [{"step": 0, "exact": exact0, "pairwise": pairwise0, "loss": None}]

    model.train()
    for step in range(1, args.finetune_steps + 1):
        xs, ys = sample_batch(n, args.batch_size, rng, args.min_paths, args.max_paths)
        xb = torch.from_numpy(xs).to(dev, torch.float32)
        yb = torch.from_numpy(ys).to(dev, torch.float32)
        logits = model(xb)
        loss = criterion(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0 or step == args.finetune_steps:
            # same fixed eval stream every time (re-seeded), for a comparable curve
            exact, pairwise = evaluate(model, dev, n, np.random.default_rng(args.seed + 999_999),
                                       args.min_paths, args.max_paths, args.eval_graphs)
            print(f"step {step:>6d} loss={loss.item():.4f} exact={exact:.4f} pairwise={pairwise:.4f}",
                  flush=True)
            history.append({"step": step, "exact": exact, "pairwise": pairwise, "loss": float(loss.item())})

    ck = {"model_state_dict": model.state_dict(), "model_config": mcfg.__dict__,
          "finetune_args": vars(args), "source_checkpoint": args.checkpoint}
    torch.save(ck, out / "finetuned.pt")
    (out / "finetune_history.json").write_text(json.dumps(history, indent=2))
    print(f"done -> {out}/finetuned.pt")


if __name__ == "__main__":
    main()
