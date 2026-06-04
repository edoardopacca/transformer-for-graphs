"""Evaluate existing reach_depth checkpoints (last.pt) without retraining.

The depth->reach jobs that hit the SLURM time limit never wrote history.json
(it is only written on clean completion), but they did save last.pt every eval.
L=3 and L=4 had already converged to exact=1.0, so we just need to *evaluate*
their checkpoints; L=1 already completed and L=2 plateaued at its capacity.

For each runs/reach_depth/*/last.pt this loads the model, rebuilds the
path-union test set, recomputes d* and per-distance reach, and writes a
history.json compatible with plot_reach_law.py. Run on HPC (GPU + checkpoints).

    python eval_reach_checkpoint.py [--runs_dir runs/reach_depth] [--num_workers 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from model import GraphConnectivityTransformer, ModelConfig
from experiments2.train_reach_depth import build_test, evaluate


def load_model(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ck["model_config"]
    mcfg = ModelConfig(n=c["n"], d_model=c["d_model"], n_heads=c["n_heads"],
                       d_ff=c["d_ff"], n_layers=c["n_layers"],
                       attn_kind=c.get("attn_kind", "normalized_relu"))
    model = GraphConnectivityTransformer(mcfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, mcfg, int(ck.get("step", 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs/reach_depth")
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"device: {device}")

    runs = sorted(p for p in Path(args.runs_dir).iterdir()
                  if p.is_dir() and (p / "last.pt").exists())
    if not runs:
        raise SystemExit(f"No */last.pt under {args.runs_dir}")

    test_cache: dict = {}   # reuse the test set across runs with the same n
    for d in runs:
        hj = d / "history.json"
        if hj.exists():
            try:
                existing = json.load(hj.open())
                if len(existing.get("steps", [])) > 1 and not existing.get("eval_only"):
                    print(f"  {d.name}: clean history.json already present "
                          f"({len(existing['steps'])} evals) -> skip")
                    continue
            except Exception:
                pass
        model, mcfg, step = load_model(str(d / "last.pt"), device)
        n, L = mcfg.n, mcfg.n_layers
        if n not in test_cache:
            test_cache[n] = build_test(n, args.num_workers, seed=1000)
        tx, ty, td = test_cache[n]
        m = evaluate(model, tx, ty, td, device)
        del model
        hist = {
            "steps": [step], "d_star": [m["d_star"]], "best_d_star": m["d_star"],
            "exact": [m["exact"]], "pairwise": [m["pairwise"]],
            "disc_acc": [m["disc_acc"]],
            "final_per_dist": {d_: v[0] for d_, v in m["per_dist"].items()},
            "n_layers": L, "capacity_3L": 3 ** L, "eval_only": True, "eval_step": step,
        }
        with (d / "history.json").open("w") as f:
            json.dump(hist, f, indent=2)
        print(f"  {d.name}: L={L} step={step} d*={m['d_star']} (3^{L}={3**L}) "
              f"exact={m['exact']:.3f} pair={m['pairwise']:.3f} disc={m['disc_acc']:.3f}"
              f"  -> wrote history.json")


if __name__ == "__main__":
    main()
