"""
Curriculum learning on ER(n=40, p=0.05) using the BIG setup
(d_model=512, normalized-ReLU attention, online data, bf16).

Same four-phase schedule as curriculum_er_n40.py:
  Phase 1 : D ≤ 7
  Phase 2 : D ≤ 9
  Phase 3 : D ≤ 10
  Phase 4 : D ≤ 11   (final target distribution)

Phase-advance criterion (loss on the *current* phase data):
  Phase 1–3 advance when smoothed training loss < 1e-2 for 3 consecutive
            evals, or after 100k steps in that phase (safety cap).
  Phase 4   has threshold 1e-3 and otherwise runs to the total budget.
  Min 5k steps in any phase before an advance can fire.

Implementation:
  Each phase has its own IterableDataset / DataLoader producing online ER
  graphs with rejection sampling at that phase's diameter cutoff. Only ONE
  DataLoader is active at a time; on phase advance the previous one is
  discarded (its worker processes are torn down) and the next one is
  spun up.
"""
from __future__ import annotations

import sys
import time
import resource
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import math as pymath
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model import GraphConnectivityTransformer, ModelConfig
from utils import ensure_dir, get_device, save_json, set_seed

# Reuse the big-setup helpers from retrain_and_test_er_n40_big.py
from experiments2.retrain_and_test_er_n40_big import (
    OnlineERStream, _stream_collate, build_test_dataset, evaluate, lr_at_step,
    _plot_accuracy as _plot_accuracy_basic,
    _plot_loss as _plot_loss_basic,
    _plot_per_dist_sm,
    _plot_per_diam_bucket as _plot_per_diam_bucket_basic,
    N_NODES, P, TEST_SIZE,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Curriculum schedule (same as curriculum_er_n40.py) ───────────────────────
PHASES: List[Dict[str, Any]] = [
    {"id": 1, "name": "phase1_d7",  "max_diam":  7, "loss_threshold": 1e-2},
    {"id": 2, "name": "phase2_d9",  "max_diam":  9, "loss_threshold": 1e-2},
    {"id": 3, "name": "phase3_d10", "max_diam": 10, "loss_threshold": 1e-2},
    {"id": 4, "name": "phase4_d11", "max_diam": 11, "loss_threshold": 1e-3},
]
MIN_STEPS_PER_PHASE       = 5_000
MAX_STEPS_PER_EARLY_PHASE = 100_000
LOSS_HISTORY_K            = 3


def _get_ram_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def make_loader(max_diam: int, batch_size: int, num_workers: int, seed: int) -> DataLoader:
    ds = OnlineERStream(N_NODES, P, max_diam, seed=seed)
    return DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        collate_fn=_stream_collate, pin_memory=True,
        prefetch_factor=4, persistent_workers=True,
    )


class CurriculumState:
    """Tracks which phase we're in + when to advance."""

    def __init__(self):
        self.idx = 0
        self.start_step = 0
        self.transitions: List[Dict[str, Any]] = []

    @property
    def current(self) -> Dict[str, Any]:
        return PHASES[self.idx]

    @property
    def is_final(self) -> bool:
        return self.idx == len(PHASES) - 1

    def maybe_advance(self, step: int, recent_losses: List[float]) -> Optional[str]:
        if self.is_final:
            return None
        steps_in_phase = step - self.start_step
        if steps_in_phase < MIN_STEPS_PER_PHASE:
            return None
        thr = self.current["loss_threshold"]
        loss_ok = (
            len(recent_losses) >= LOSS_HISTORY_K
            and all(l < thr for l in recent_losses[-LOSS_HISTORY_K:])
        )
        cap_hit = steps_in_phase >= MAX_STEPS_PER_EARLY_PHASE
        if loss_ok or cap_hit:
            reason = "loss" if loss_ok else "cap"
            self.transitions.append({
                "step":         step,
                "from_phase":   self.current["name"],
                "to_phase":     PHASES[self.idx + 1]["name"],
                "to_max_diam":  PHASES[self.idx + 1]["max_diam"],
                "steps_in_old": steps_in_phase,
                "reason":       reason,
                "recent_losses": list(recent_losses[-LOSS_HISTORY_K:]),
            })
            self.idx += 1
            self.start_step = step
            return reason
        return None


# ── Curriculum-aware plots (add vertical lines at phase transitions) ─────────

def _annotate_phase_boundaries(ax, history: Dict[str, Any]) -> None:
    for t in history["phase_transitions"]:
        ax.axvline(t["step"], color="gray", ls="--", lw=1.0, alpha=0.7)


def _plot_accuracy(history: Dict, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["steps"], history["val_exact_match"], lw=2, label="Exact Match")
    ax.plot(history["steps"], history["val_pairwise_acc"], lw=2, label="Pairwise Acc")
    ax.set_title(title); ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.01); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    _annotate_phase_boundaries(ax, history)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def _plot_loss(history: Dict, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["steps"], history["train_loss"], lw=2, color="#1f77b4")
    ax.axhline(1e-2, color="orange", ls=":", lw=1, label="threshold 10⁻²")
    ax.axhline(1e-3, color="red", ls=":", lw=1, label="threshold 10⁻³ (phase 4)")
    ax.set_yscale("log")
    ax.set_title(title); ax.set_xlabel("Step"); ax.set_ylabel("Training loss (current phase)")
    ax.grid(alpha=0.3, which="both"); ax.legend(loc="upper right")
    _annotate_phase_boundaries(ax, history)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def _plot_per_diam_bucket(history: Dict, out: Path, title: str) -> None:
    steps = history["steps"]
    buckets = history["val_per_diam_bucket"]
    keys = ["exact_le7", "exact_le9", "exact_le11", "exact_gt11"]
    labels = ["D≤7", "D≤9", "D≤11", "D>11"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label in zip(keys, labels):
        vals = [entry.get(key) for entry in buckets]
        if any(v is not None for v in vals):
            clean = [v if v is not None else float("nan") for v in vals]
            ax.plot(steps, clean, lw=2, label=label)
    ax.set_title(title); ax.set_xlabel("Step"); ax.set_ylabel("Exact-match accuracy")
    ax.set_ylim(0, 1.01); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    _annotate_phase_boundaries(ax, history)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def _plot_phase_timeline(history: Dict, out: Path, title: str) -> None:
    steps = history["steps"]
    phase_ids = history["current_phase_id"]
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(11, 2.5))
    palette = {1: "#1f77b4", 2: "#2ca02c", 3: "#ff7f0e", 4: "#d62728"}
    start = steps[0]; current = phase_ids[0]
    segments = []
    for s, p in zip(steps[1:], phase_ids[1:]):
        if p != current:
            segments.append((start, s, current))
            start = s; current = p
    segments.append((start, steps[-1], current))
    for s0, s1, pid in segments:
        ax.barh(0, s1 - s0, left=s0, height=0.6, color=palette[pid],
                edgecolor="black", linewidth=0.4,
                label=f"phase {pid}: D≤{PHASES[pid-1]['max_diam']}")
    ax.set_yticks([])
    ax.set_xlabel("Training step")
    ax.set_title(title, fontsize=12)
    handles, labels_ = ax.get_legend_handles_labels()
    seen = set(); uh, ul = [], []
    for h, l in zip(handles, labels_):
        if l not in seen:
            uh.append(h); ul.append(l); seen.add(l)
    ax.legend(uh, ul, loc="upper center", bbox_to_anchor=(0.5, -0.4),
              ncol=len(ul), fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ── Training loop ────────────────────────────────────────────────────────────

def train_curriculum_big(out_dir: Path,
                         test_x: np.ndarray, test_y: np.ndarray, test_d: np.ndarray,
                         config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    set_seed(seed)
    device = get_device(config.get("device", "auto"))
    print(f"  device: {device}")

    model_cfg = ModelConfig(
        n=config["n"], d_model=config["d_model"], n_heads=config["n_heads"],
        d_ff=config["d_ff"], n_layers=config["n_layers"],
        dropout=config.get("dropout", 0.0),
        attn_kind=config["attn_kind"],
    )
    model = GraphConnectivityTransformer(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    opt = AdamW(model.parameters(), lr=config["lr"],
                weight_decay=config.get("weight_decay", 1e-4))
    criterion = nn.BCEWithLogitsLoss()

    total_steps = config["train_steps"]
    warmup = config.get("warmup_steps", 1000)
    eval_every = config.get("eval_every", 5000)
    grad_clip = config.get("grad_clip_norm", 1.0)
    peak_lr = config["lr"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

    state = CurriculumState()

    def fresh_loader(phase_idx: int, seed_off: int) -> DataLoader:
        return make_loader(
            max_diam=PHASES[phase_idx]["max_diam"],
            batch_size=batch_size, num_workers=num_workers,
            seed=seed + 13 * (phase_idx + 1) + seed_off,
        )

    current_loader = fresh_loader(state.idx, 0)
    current_iter = iter(current_loader)

    history: Dict[str, Any] = {
        "steps": [], "train_loss": [],
        "val_exact_match": [], "val_pairwise_acc": [],
        "val_per_dist_acc": [], "val_per_diam_bucket": [],
        "current_phase_id":   [], "current_phase_name": [],
        "timing_stats": {"time_per_5000_steps_sec": []},
        "phase_transitions": [],
        "run_stats": {
            "n_parameters": n_params,
            "ram_before_training_gb": _get_ram_gb(),
        },
        "phase_definition": [dict(ph) for ph in PHASES],
        "phase_min_steps":       MIN_STEPS_PER_PHASE,
        "phase_max_steps_early": MAX_STEPS_PER_EARLY_PHASE,
        "phase_loss_history_k":  LOSS_HISTORY_K,
        "test_dist_counts": {},
    }

    loss_window: List[float] = []
    best_exact = -1.0
    t_block = time.perf_counter()
    model.train()
    print(f"  Starting curriculum training: total budget {total_steps:,} steps")
    print(f"  Initial phase: {state.current['name']} (D ≤ {state.current['max_diam']})")

    for step in range(1, total_steps + 1):
        try:
            xb_cpu, yb_cpu = next(current_iter)
        except StopIteration:
            current_iter = iter(current_loader)
            xb_cpu, yb_cpu = next(current_iter)

        lr = lr_at_step(step, warmup, total_steps, peak_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        xb = xb_cpu.to(device, dtype=torch.float32, non_blocking=True)
        yb = yb_cpu.to(device, dtype=torch.float32, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xb)
            loss = criterion(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        loss_window.append(float(loss.item()))
        if len(loss_window) > eval_every:
            loss_window.pop(0)

        if step % eval_every == 0:
            elapsed = time.perf_counter() - t_block
            history["timing_stats"]["time_per_5000_steps_sec"].append(
                {"step": step, "seconds": elapsed}
            )
            metrics = evaluate(model, test_x, test_y, test_d, device)
            avg_loss = sum(loss_window) / max(1, len(loss_window))

            history["steps"].append(step)
            history["train_loss"].append(avg_loss)
            history["val_exact_match"].append(metrics["exact_match"])
            history["val_pairwise_acc"].append(metrics["pairwise_acc"])
            history["val_per_dist_acc"].append(metrics["per_dist_acc"])
            history["val_per_diam_bucket"].append(metrics["per_diam_bucket"])
            history["current_phase_id"].append(state.current["id"])
            history["current_phase_name"].append(state.current["name"])
            history["test_dist_counts"] = metrics["dist_counts"]

            print(f"  step {step:>7d} | phase {state.current['id']} "
                  f"({state.current['name']}) | lr={lr:.2e} | loss={avg_loss:.6f} | "
                  f"exact={metrics['exact_match']:.4f} | pair={metrics['pairwise_acc']:.4f} "
                  f"| {elapsed:.1f}s/5000",
                  flush=True)

            if metrics["exact_match"] > best_exact:
                best_exact = metrics["exact_match"]
                if config.get("save_best", True):
                    torch.save({"model_state_dict": model.state_dict(),
                                "model_config": model_cfg.__dict__,
                                "step": step,
                                "phase_id": state.current["id"]},
                               out_dir / "best.pt")
            torch.save({"model_state_dict": model.state_dict(),
                        "model_config": model_cfg.__dict__,
                        "step": step,
                        "phase_id": state.current["id"]},
                       out_dir / "last.pt")

            advance_reason = state.maybe_advance(step, history["train_loss"])
            if advance_reason is not None:
                tr = state.transitions[-1]
                print(f"  >>> PHASE ADVANCE at step {step}: "
                      f"{tr['from_phase']} -> {tr['to_phase']} "
                      f"(D ≤ {tr['to_max_diam']}) — reason={advance_reason} "
                      f"steps_in_old={tr['steps_in_old']:,}",
                      flush=True)
                # Tear down the old DataLoader's workers, spin up the new one
                del current_iter
                del current_loader
                current_loader = fresh_loader(state.idx, step)
                current_iter = iter(current_loader)

            t_block = time.perf_counter()

    if config.get("save_last", True):
        torch.save({"model_state_dict": model.state_dict(),
                    "model_config": model_cfg.__dict__,
                    "step": total_steps,
                    "phase_id": state.current["id"]},
                   out_dir / "last.pt")

    history["phase_transitions"] = state.transitions
    history["run_stats"]["ram_after_training_gb"] = _get_ram_gb()
    history["best_exact_match"] = best_exact
    return history


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--train_steps", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    run_name = f"n{N_NODES}_p{int(P*100):03d}_curriculum_big"
    out_root = Path(args.output_root)
    out_dir = out_root / run_name
    ensure_dir(out_dir)

    config = {
        "n":             N_NODES,
        "p":             P,
        "d_model":       512,
        "n_heads":       4,
        "d_ff":          2048,
        "n_layers":      2,
        "dropout":       0.0,
        "attn_kind":     "normalized_relu",
        "batch_size":    args.batch_size,
        "lr":            1e-4,
        "weight_decay":  1e-4,
        "train_steps":   args.train_steps,
        "warmup_steps":  1000,
        "eval_every":    5000,
        "grad_clip_norm": 1.0,
        "save_best":     True,
        "save_last":     True,
        "device":        "auto",
        "num_workers":   args.num_workers,
    }

    print(f"\n{'='*72}")
    print(f"  Big ER curriculum: ER(n={N_NODES}, p={P}), 4 phases (D≤7→9→10→11)")
    print(f"  d_model=512, normalized-ReLU, online data, bf16")
    print(f"  Output: {out_dir}")
    print(f"{'='*72}\n")

    # Test set: D≤11 (the final-phase distribution = the target evaluation)
    test_x, test_y, test_d = build_test_dataset(
        N_NODES, P, max_diameter=11, num_workers=args.num_workers,
    )

    history = train_curriculum_big(out_dir, test_x, test_y, test_d, config, args.seed)

    # ── Plots ──
    prefix = "curriculum_er_n40_big"
    _plot_accuracy       (history, out_dir / f"{prefix}_accuracy.png",
                           f"Big curriculum ER(n={N_NODES}, p={P}): Accuracy")
    _plot_loss           (history, out_dir / f"{prefix}_loss.png",
                           f"Big curriculum ER(n={N_NODES}, p={P}): Training Loss")
    _plot_per_diam_bucket(history, out_dir / f"{prefix}_per_diam_bucket.png",
                           f"Big curriculum ER(n={N_NODES}, p={P}): Exact Match by Diameter Bucket")
    _plot_per_dist_sm    (history, out_dir / f"{prefix}_per_dist_sm.png",
                           f"Big curriculum ER(n={N_NODES}, p={P}): Pairwise acc by Distance")
    _plot_phase_timeline (history, out_dir / f"{prefix}_phase_timeline.png",
                           f"Big curriculum schedule (auto-paced, loss-based transitions)")

    summary = {
        "config": config,
        "best_exact_match": history["best_exact_match"],
        "phase_transitions": history["phase_transitions"],
    }
    save_json(out_root / f"summary_{run_name}.json", summary)
    save_json(out_dir / "history.json", history)
    print(f"\nDone. Best exact match: {history['best_exact_match']:.4f}")
    print(f"Phase transitions ({len(history['phase_transitions'])}):")
    for t in history["phase_transitions"]:
        print(f"  step {t['step']:>6}: {t['from_phase']} -> {t['to_phase']} "
              f"(reason={t['reason']}, steps_in_old={t['steps_in_old']:,})")


if __name__ == "__main__":
    main()
