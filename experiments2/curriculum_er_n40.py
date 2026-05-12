"""
Curriculum learning on ER(n=40, p=0.05) with phase-wise diameter restriction.

Protocol (adapted from Abbe, Cornacchia, Lotfi 2023):
  Phase 1: train only on graphs with D <= 7
  Phase 2: train only on graphs with D <= 9
  Phase 3: train only on graphs with D <= 10
  Phase 4: train on all graphs (D <= 11) — the target distribution

Phase advance criterion (loss-based, computed on the *current phase's* data):
  - Phases 1-3: smoothed training loss < 1e-2 for 3 consecutive evals,
                OR a hard safety cap of 100k steps in the phase
  - Phase 4:    smoothed training loss < 1e-3 OR runs until the step budget
  - Min 5k steps per phase before any advancement (prevents flaky early switches)

Single 6,000,000-graph dataset of ER(40, 0.05) with D <= 11 is generated once,
with the per-graph diameter saved. At each phase the dataloader samples only
from the indices that satisfy the phase's diameter constraint.
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
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from data import (
    add_self_loops,
    compute_connectivity_matrix,
    compute_all_pairs_shortest_paths,
    generate_er_graph,
)
from model import GraphConnectivityTransformer, ModelConfig
from utils import ensure_dir, get_device, save_json, set_seed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Constants ────────────────────────────────────────────────────────────────
N_NODES        = 40
P              = 0.05
TRAIN_SIZE     = 6_000_000
TEST_SIZE      = 10_000
CHUNK_SIZE     = 25_000
MAX_DIAM_GLOBAL = 11
MAX_DIST_LOG   = N_NODES - 1

# ── Curriculum schedule ──────────────────────────────────────────────────────
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
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / (1024.0 ** 2)


# ── Data generation ──────────────────────────────────────────────────────────

def _sample_one_with_diam(rng: np.random.Generator, n: int, p: float,
                          max_diameter: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Sample one ER(n, p) graph with rejection until diameter <= max_diameter.
    Returns (adj_with_self_loops, connectivity_matrix, max_finite_distance)."""
    while True:
        adj_no = generate_er_graph(n, p, rng)
        dist   = compute_all_pairs_shortest_paths(adj_no)
        finite = dist[dist >= 0]
        diam   = int(finite.max()) if len(finite) > 0 else -1
        if diam <= max_diameter:
            adj    = add_self_loops(adj_no).astype(np.uint8)
            target = compute_connectivity_matrix(adj_no).astype(np.uint8)
            return adj, target, diam


def _gen_train_chunk_with_diam(
    args: Tuple[int, int, float, int, int, int],
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    start_idx, size, p, n, max_diam, seed = args
    rng = np.random.default_rng(seed)
    xs = np.empty((size, n, n), dtype=np.uint8)
    ys = np.empty((size, n, n), dtype=np.uint8)
    ds = np.empty(size,         dtype=np.int16)
    for i in range(size):
        adj, target, diam = _sample_one_with_diam(rng, n, p, max_diam)
        xs[i] = adj
        ys[i] = target
        ds[i] = diam
    return start_idx, xs, ys, ds


def _gen_test_chunk(
    args: Tuple[int, int, float, int, int, int],
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Same as the training chunk but additionally returns the full distance matrix
    (needed for per-distance evaluation on the test set)."""
    start_idx, size, p, n, max_diam, seed = args
    rng = np.random.default_rng(seed)
    xs = np.empty((size, n, n), dtype=np.uint8)
    ys = np.empty((size, n, n), dtype=np.uint8)
    ds = np.empty((size, n, n), dtype=np.int16)
    for i in range(size):
        while True:
            adj_no = generate_er_graph(n, p, rng)
            dist = compute_all_pairs_shortest_paths(adj_no)
            finite = dist[dist >= 0]
            diam   = int(finite.max()) if len(finite) > 0 else -1
            if diam <= max_diam:
                break
        xs[i] = add_self_loops(adj_no).astype(np.uint8)
        ys[i] = compute_connectivity_matrix(adj_no).astype(np.uint8)
        ds[i] = dist.astype(np.int16)
    return start_idx, xs, ys, ds


def _chunk_specs(total: int, p: float, n: int, max_diam: int,
                 seed_base: int) -> List[Tuple]:
    specs = []
    for start in range(0, total, CHUNK_SIZE):
        size = min(CHUNK_SIZE, total - start)
        specs.append((start, size, p, n, max_diam, seed_base + start))
    return specs


def build_train_dataset(num_workers: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    print(f"Generating {TRAIN_SIZE:,} training graphs ER(n={N_NODES}, p={P}) "
          f"with D ≤ {MAX_DIAM_GLOBAL}, recording per-graph diameter "
          f"(using {num_workers} workers)…")
    t0 = time.perf_counter()
    x = np.empty((TRAIN_SIZE, N_NODES, N_NODES), dtype=np.uint8)
    y = np.empty((TRAIN_SIZE, N_NODES, N_NODES), dtype=np.uint8)
    d = np.empty(TRAIN_SIZE,                     dtype=np.int16)
    specs = _chunk_specs(TRAIN_SIZE, P, N_NODES, MAX_DIAM_GLOBAL, seed_base=12345)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for s, xs, ys, ds in ex.map(_gen_train_chunk_with_diam, specs):
            e = s + len(xs)
            x[s:e] = xs; y[s:e] = ys; d[s:e] = ds
            print(f"  train {e:>9,}/{TRAIN_SIZE:,}", flush=True)
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Quick distribution log
    print("  Per-phase index counts:")
    for ph in PHASES:
        n_idx = int((d <= ph["max_diam"]).sum())
        print(f"    {ph['name']:<12} (D ≤ {ph['max_diam']}):  "
              f"{n_idx:>9,}  ({n_idx / TRAIN_SIZE * 100:.1f}%)")
    return x, y, d, elapsed


def build_test_dataset(num_workers: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    print(f"Generating {TEST_SIZE:,} test graphs with D ≤ {MAX_DIAM_GLOBAL}…")
    t0 = time.perf_counter()
    x = np.empty((TEST_SIZE, N_NODES, N_NODES), dtype=np.uint8)
    y = np.empty((TEST_SIZE, N_NODES, N_NODES), dtype=np.uint8)
    d = np.empty((TEST_SIZE, N_NODES, N_NODES), dtype=np.int16)
    specs = _chunk_specs(TEST_SIZE, P, N_NODES, MAX_DIAM_GLOBAL, seed_base=54321)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for s, xs, ys, ds in ex.map(_gen_test_chunk, specs):
            e = s + len(xs)
            x[s:e] = xs; y[s:e] = ys; d[s:e] = ds
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s")
    return x, y, d, elapsed


# ── Evaluation ───────────────────────────────────────────────────────────────

def _to_device(arr: np.ndarray, device: torch.device,
               dtype: torch.dtype) -> torch.Tensor:
    t = torch.from_numpy(arr)
    if device.type == "cuda":
        return t.pin_memory().to(device=device, dtype=dtype, non_blocking=True)
    return t.to(device=device, dtype=dtype)


@torch.no_grad()
def evaluate(model: nn.Module,
             test_x: np.ndarray, test_y: np.ndarray, test_d: np.ndarray,
             device: torch.device,
             batch_size: int = 512) -> Dict[str, Any]:
    model.eval()
    n_graphs, n, _ = test_x.shape

    all_pred = np.empty((n_graphs, n, n), dtype=np.int8)
    for start in range(0, n_graphs, batch_size):
        end = min(start + batch_size, n_graphs)
        xb = _to_device(test_x[start:end], device, torch.float32)
        logits = model(xb)
        all_pred[start:end] = (logits > 0).cpu().numpy().astype(np.int8)

    targets = test_y.astype(np.int8)
    eq = (all_pred == targets)

    exact_per_graph = eq.reshape(n_graphs, -1).all(axis=1)
    exact_match  = float(exact_per_graph.mean())
    pairwise_acc = float(eq.mean())

    eye_mask = ~np.eye(n, dtype=bool)
    offdiag  = np.broadcast_to(eye_mask[None, :, :], (n_graphs, n, n))
    per_dist: Dict[str, float] = {}
    dist_counts: Dict[str, int] = {}
    for dv in range(1, MAX_DIST_LOG + 1):
        mask = offdiag & (test_d == dv)
        cnt = int(mask.sum())
        if cnt > 0:
            per_dist[str(dv)]   = float(eq[mask].mean())
            dist_counts[str(dv)] = cnt
    unreach = offdiag & (test_d == -1)
    if unreach.any():
        per_dist["disconnected"]   = float(eq[unreach].mean())
        dist_counts["disconnected"] = int(unreach.sum())

    # Per-graph diameter bucket
    d_clipped = np.where(test_d < 0, 0, test_d)
    per_graph_diam = d_clipped.reshape(n_graphs, -1).max(axis=1)
    per_diam: Dict[str, Any] = {}
    for thr in [7, 9, 11]:
        mask = per_graph_diam <= thr
        if mask.any():
            per_diam[f"exact_le{thr}"]    = float(exact_per_graph[mask].mean())
            per_diam[f"n_graphs_le{thr}"] = int(mask.sum())
    mask_gt = per_graph_diam > 11
    if mask_gt.any():
        per_diam["exact_gt11"]    = float(exact_per_graph[mask_gt].mean())
        per_diam["n_graphs_gt11"] = int(mask_gt.sum())

    model.train()
    return {
        "exact_match":     exact_match,
        "pairwise_acc":    pairwise_acc,
        "per_dist_acc":    per_dist,
        "dist_counts":     dist_counts,
        "per_diam_bucket": per_diam,
    }


# ── Curriculum training loop ─────────────────────────────────────────────────

class PhaseState:
    """Encapsulates the curriculum bookkeeping: which phase we are in, the
    per-phase permutation cursor, and the criterion for advancing."""

    def __init__(self, train_diam: np.ndarray, batch_size: int, seed: int):
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed + 7777)

        # Pre-compute the index array for each phase
        self.phase_indices: List[np.ndarray] = []
        for ph in PHASES:
            idx = np.where(train_diam <= ph["max_diam"])[0]
            self.phase_indices.append(idx)
            print(f"    {ph['name']}: {len(idx):,} graphs")

        self.idx          = 0      # current phase index in [0..3]
        self.start_step   = 0      # global step at which current phase began
        self.transitions: List[Dict[str, Any]] = []
        self._refresh_perm()

    @property
    def current(self) -> Dict[str, Any]:
        return PHASES[self.idx]

    @property
    def is_final(self) -> bool:
        return self.idx == len(PHASES) - 1

    def _refresh_perm(self) -> None:
        n = len(self.phase_indices[self.idx])
        self.perm   = self.rng.permutation(n)
        self.cursor = 0

    def sample_batch_global_indices(self) -> np.ndarray:
        pi = self.phase_indices[self.idx]
        n_local = len(pi)
        bs = self.batch_size
        if self.cursor + bs > n_local:
            remaining = n_local - self.cursor
            first  = self.perm[self.cursor:]
            self._refresh_perm()
            second = self.perm[: bs - remaining]
            self.cursor = bs - remaining
            local_sel = np.concatenate([first, second])
        else:
            local_sel = self.perm[self.cursor: self.cursor + bs]
            self.cursor += bs
        return pi[local_sel]

    def maybe_advance(self, step: int, recent_losses: List[float]) -> Optional[str]:
        """Decide whether to advance to the next phase. Returns advance reason
        ('loss' / 'cap') if a transition happened, else None."""
        if self.is_final:
            return None

        steps_in_phase = step - self.start_step
        if steps_in_phase < MIN_STEPS_PER_PHASE:
            return None

        thr = self.current["loss_threshold"]
        # Success criterion: last K eval losses all below threshold
        loss_ok = (
            len(recent_losses) >= LOSS_HISTORY_K
            and all(l < thr for l in recent_losses[-LOSS_HISTORY_K:])
        )
        cap_hit = steps_in_phase >= MAX_STEPS_PER_EARLY_PHASE

        if loss_ok or cap_hit:
            old_phase = self.current["name"]
            self.idx += 1
            self.start_step = step
            self._refresh_perm()
            reason = "loss" if loss_ok else "cap"
            self.transitions.append({
                "step":         step,
                "from_phase":   old_phase,
                "to_phase":     self.current["name"],
                "to_max_diam":  self.current["max_diam"],
                "steps_in_old": steps_in_phase,
                "reason":       reason,
                "recent_losses": list(recent_losses[-LOSS_HISTORY_K:]),
            })
            return reason
        return None


def train_curriculum(
    out_dir: Path,
    train_x: np.ndarray, train_y: np.ndarray, train_diam: np.ndarray,
    test_x: np.ndarray, test_y: np.ndarray, test_d: np.ndarray,
    config: Dict[str, Any], seed: int,
) -> Dict[str, Any]:
    set_seed(seed)
    device = get_device(config.get("device", "auto"))
    print(f"  device: {device}")

    model_cfg = ModelConfig(
        n=config["n"], d_model=config["d_model"], n_heads=config["n_heads"],
        d_ff=config["d_ff"], n_layers=config["n_layers"],
        dropout=config.get("dropout", 0.0),
    )
    model = GraphConnectivityTransformer(model_cfg).to(device)
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = AdamW(model.parameters(), lr=config["lr"],
                weight_decay=config.get("weight_decay", 0.01))
    criterion = nn.BCEWithLogitsLoss()

    batch_size  = config["batch_size"]
    total_steps = config["train_steps"]
    eval_every  = config.get("eval_every", 1000)
    grad_clip   = config.get("grad_clip_norm", 1.0)

    print("  Building phase index arrays from per-graph diameters:")
    state = PhaseState(train_diam, batch_size, seed)

    loss_window: List[float] = []
    history: Dict[str, Any] = {
        "steps": [], "train_loss": [],
        "val_exact_match": [], "val_pairwise_acc": [],
        "val_per_dist_acc": [], "val_per_diam_bucket": [],
        "current_phase_id":   [],
        "current_phase_name": [],
        "timing_stats": {"time_per_1000_steps_sec": []},
        "phase_transitions": [],   # filled at the end
        "run_stats": {
            "train_size": int(TRAIN_SIZE),
            "test_size":  int(TEST_SIZE),
            "ram_before_training_gb": _get_ram_gb(),
        },
        "phase_definition": [
            {k: v for k, v in ph.items()} for ph in PHASES
        ],
        "phase_min_steps":           MIN_STEPS_PER_PHASE,
        "phase_max_steps_early":     MAX_STEPS_PER_EARLY_PHASE,
        "phase_loss_history_k":      LOSS_HISTORY_K,
        "test_dist_counts": {},
    }

    best_exact = -1.0
    t_block    = time.perf_counter()
    model.train()
    print(f"  Starting curriculum training: total budget {total_steps:,} steps")
    print(f"  Initial phase: {state.current['name']} (D ≤ {state.current['max_diam']})")

    for step in range(1, total_steps + 1):
        batch_idx = state.sample_batch_global_indices()
        x = _to_device(train_x[batch_idx], device, torch.float32)
        y = _to_device(train_y[batch_idx], device, torch.float32)

        logits = model(x)
        loss   = criterion(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        loss_window.append(float(loss.item()))
        if len(loss_window) > eval_every:
            loss_window.pop(0)

        if step % 1000 == 0:
            now = time.perf_counter()
            history["timing_stats"]["time_per_1000_steps_sec"].append(
                {"step": step, "seconds": now - t_block}
            )
            t_block = now

        if step % eval_every == 0:
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

            print(f"  step {step:>6d} | phase {state.current['id']} "
                  f"({state.current['name']}) | loss={avg_loss:.5f} | "
                  f"exact={metrics['exact_match']:.4f} | "
                  f"pairwise={metrics['pairwise_acc']:.4f}",
                  flush=True)

            if metrics["exact_match"] > best_exact:
                best_exact = metrics["exact_match"]
                if config.get("save_best", True):
                    torch.save({"model_state_dict": model.state_dict(),
                                "model_config": model_cfg.__dict__,
                                "step": step,
                                "phase_id": state.current["id"]},
                               out_dir / "best.pt")

            # Phase advance check (based on current-phase loss, not full test set)
            advance_reason = state.maybe_advance(step, history["train_loss"])
            if advance_reason is not None:
                tr = state.transitions[-1]
                print(f"  >>> PHASE ADVANCE at step {step}: "
                      f"{tr['from_phase']} -> {tr['to_phase']} "
                      f"(D ≤ {tr['to_max_diam']}) — reason: {advance_reason} "
                      f"(steps_in_old={tr['steps_in_old']:,})", flush=True)

    if config.get("save_last", True):
        torch.save({"model_state_dict": model.state_dict(),
                    "model_config": model_cfg.__dict__,
                    "step": total_steps,
                    "phase_id": state.current["id"]},
                   out_dir / "last.pt")

    history["phase_transitions"]            = state.transitions
    history["run_stats"]["ram_after_training_gb"] = _get_ram_gb()
    history["best_exact_match"]             = best_exact
    return history


# ── Plots ────────────────────────────────────────────────────────────────────

def _phase_boundary_steps(history: Dict[str, Any]) -> List[Tuple[int, str]]:
    """Returns list of (step, label) for vertical lines marking each phase
    transition."""
    return [(t["step"], t["to_phase"]) for t in history["phase_transitions"]]


def _annotate_phases(ax, history: Dict[str, Any], y_label_pos: float = 0.97) -> None:
    boundaries = _phase_boundary_steps(history)
    ymin, ymax = ax.get_ylim()
    for step, _label in boundaries:
        ax.axvline(step, color="gray", ls="--", lw=1.0, alpha=0.7)
    # Phase region labels along the top of the plot
    phase_starts = [0] + [b[0] for b in boundaries]
    phase_names  = []
    for entry_id in history["current_phase_id"]:
        ph = PHASES[entry_id - 1]
        phase_names.append(ph["name"])
    # Use the unique sequence of phases that actually occurred
    seen_ids = []
    for pid in history["current_phase_id"]:
        if not seen_ids or seen_ids[-1] != pid:
            seen_ids.append(pid)
    transitions_steps = [0] + [b[0] for b in boundaries] + [history["steps"][-1]]
    for pi, pid in enumerate(seen_ids):
        ph = PHASES[pid - 1]
        x_mid = 0.5 * (transitions_steps[pi] + transitions_steps[pi + 1])
        ax.text(x_mid, ymax * y_label_pos, f"phase {pid}: D≤{ph['max_diam']}",
                ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec="gray", alpha=0.85))


def _plot_accuracy(history: Dict, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["steps"], history["val_exact_match"],  lw=2, label="Exact Match")
    ax.plot(history["steps"], history["val_pairwise_acc"], lw=2, label="Pairwise Acc")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.01); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    _annotate_phases(ax, history)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def _plot_loss(history: Dict, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["steps"], history["train_loss"], lw=2, color="#1f77b4")
    ax.axhline(1e-2, color="orange", ls=":", lw=1.2, label="threshold 10⁻²")
    ax.axhline(1e-3, color="red",    ls=":", lw=1.2, label="threshold 10⁻³ (phase 4)")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Step"); ax.set_ylabel("Training loss (current-phase data)")
    ax.grid(alpha=0.3, which="both"); ax.legend(loc="upper right")
    _annotate_phases(ax, history, y_label_pos=0.97)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def _plot_per_dist_sm(history: Dict, out: Path, title: str) -> None:
    steps = history["steps"]
    per_dist_list = history["val_per_dist_acc"]
    dist_counts = history["test_dist_counts"]
    numeric = sorted(int(k) for k in dist_counts
                     if k != "disconnected" and dist_counts[k] > 0)
    panels = numeric + (["disconnected"] if dist_counts.get("disconnected", 0) > 0 else [])
    if not panels:
        return
    ncols = 3
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 2.7 * nrows),
                             sharex=True)
    axes_flat = np.array(axes).flatten().tolist()
    for ax, key in zip(axes_flat, panels):
        k = str(key)
        vals = [entry.get(k) for entry in per_dist_list]
        xs, ys = zip(*[(s, v) for s, v in zip(steps, vals) if v is not None]) \
                 if any(v is not None for v in vals) else ([], [])
        ax.plot(xs, ys, lw=1.0, color="#1f77b4")
        # Data-driven y-limits (same logic as regenerate_per_dist_small_multiples.py)
        non_none = [v for v in vals if v is not None]
        if non_none:
            vmin, vmax = min(non_none), max(non_none)
            if vmin >= 0.99:
                ax.set_ylim(0.985, 1.001)
            else:
                pad = max(0.005, (vmax - vmin) * 0.10)
                ax.set_ylim(max(0.0, vmin - pad), min(1.005, vmax + pad / 2))
        label = "disconnected" if key == "disconnected" else f"d = {key}"
        ax.set_title(f"{label}  (n = {dist_counts[k]:,})", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Step"); ax.set_ylabel("Pairwise acc")
    for ax in axes_flat[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


def _plot_per_diam_bucket(history: Dict, out: Path, title: str) -> None:
    steps   = history["steps"]
    buckets = history["val_per_diam_bucket"]
    keys   = ["exact_le7", "exact_le9", "exact_le11", "exact_gt11"]
    labels = ["D≤7", "D≤9", "D≤11", "D>11"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label in zip(keys, labels):
        vals = [entry.get(key) for entry in buckets]
        if any(v is not None for v in vals):
            clean = [v if v is not None else float("nan") for v in vals]
            ax.plot(steps, clean, lw=2, label=label)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Step"); ax.set_ylabel("Exact-match accuracy")
    ax.set_ylim(0, 1.01); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    _annotate_phases(ax, history)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def _plot_phase_timeline(history: Dict, out: Path, title: str) -> None:
    """Horizontal bar showing the active phase across training steps."""
    steps = history["steps"]
    phase_ids = history["current_phase_id"]
    fig, ax = plt.subplots(figsize=(11, 2.5))
    # Build segments: consecutive equal phase_ids
    segments = []
    if steps:
        start = steps[0]; current = phase_ids[0]
        for s, p in zip(steps[1:], phase_ids[1:]):
            if p != current:
                segments.append((start, s, current))
                start = s; current = p
        segments.append((start, steps[-1], current))
    palette = {1: "#1f77b4", 2: "#2ca02c", 3: "#ff7f0e", 4: "#d62728"}
    for s0, s1, pid in segments:
        ax.barh(0, s1 - s0, left=s0, height=0.6,
                color=palette[pid], edgecolor="black", linewidth=0.4,
                label=f"phase {pid}: D≤{PHASES[pid-1]['max_diam']}")
    ax.set_yticks([])
    ax.set_xlabel("Training step")
    ax.set_title(title, fontsize=12)
    # Deduplicate legend
    handles, labels_ = ax.get_legend_handles_labels()
    seen = set(); uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels_):
        if l not in seen:
            uniq_h.append(h); uniq_l.append(l); seen.add(l)
    ax.legend(uniq_h, uniq_l, loc="upper center",
              bbox_to_anchor=(0.5, -0.4), ncol=len(uniq_l), fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    run_name = f"n{N_NODES}_p{int(P*100):03d}_curriculum"
    out_root = Path(args.output_root)
    out_dir  = out_root / run_name
    ensure_dir(out_dir)

    config = {
        "n":              N_NODES,
        "p":              P,
        "d_model":        64,
        "n_heads":        1,
        "d_ff":           128,
        "n_layers":       2,
        "dropout":        0.0,
        "batch_size":     256,
        "lr":             1e-3,
        "weight_decay":   0.01,
        "train_steps":    500_000,
        "eval_every":     1000,
        "grad_clip_norm": 1.0,
        "save_best":      True,
        "save_last":      True,
        "device":         "auto",
        "num_workers":    args.num_workers,
        "max_diam_global": MAX_DIAM_GLOBAL,
        "train_size":      TRAIN_SIZE,
        "test_size":       TEST_SIZE,
    }

    print(f"\n{'='*64}")
    print(f"  Curriculum training on ER(n={N_NODES}, p={P}, D ≤ {MAX_DIAM_GLOBAL})")
    print(f"  Output: {out_dir}")
    print(f"{'='*64}\n")

    train_x, train_y, train_diam, train_gen_sec = build_train_dataset(args.num_workers)
    test_x,  test_y,  test_d,    test_gen_sec  = build_test_dataset(args.num_workers)

    ram_after_gen = _get_ram_gb()
    print(f"RAM after generation: {ram_after_gen:.1f} GB\n")

    history = train_curriculum(
        out_dir=out_dir,
        train_x=train_x, train_y=train_y, train_diam=train_diam,
        test_x=test_x, test_y=test_y, test_d=test_d,
        config=config, seed=1000,
    )

    # ── Plots ──
    print("\nGenerating plots…")
    prefix = "curriculum_er_n40"
    _plot_accuracy       (history, out_dir / f"{prefix}_accuracy.png",
                          f"Curriculum ER(n={N_NODES}, p={P}): Accuracy vs Step")
    _plot_loss           (history, out_dir / f"{prefix}_loss.png",
                          f"Curriculum ER(n={N_NODES}, p={P}): Training Loss vs Step")
    _plot_per_diam_bucket(history, out_dir / f"{prefix}_per_diam_bucket.png",
                          f"Curriculum ER(n={N_NODES}, p={P}): Exact Match by Diameter Bucket")
    _plot_per_dist_sm    (history, out_dir / f"{prefix}_per_dist_sm.png",
                          f"Curriculum ER(n={N_NODES}, p={P}): Pairwise Acc by Distance")
    _plot_phase_timeline (history, out_dir / f"{prefix}_phase_timeline.png",
                          f"Curriculum schedule (auto-paced, loss-based transitions)")

    # ── Summary ──
    summary = {
        "config":            config,
        "best_exact_match":  history["best_exact_match"],
        "phase_transitions": history["phase_transitions"],
        "train_gen_sec":     train_gen_sec,
        "test_gen_sec":      test_gen_sec,
        "ram_after_gen_gb":  ram_after_gen,
    }
    save_json(out_root / "summary.json", summary)
    save_json(out_dir  / "history.json", history)

    print(f"\nDone. Best exact match: {history['best_exact_match']:.4f}")
    print(f"Phase transitions ({len(history['phase_transitions'])}):")
    for t in history["phase_transitions"]:
        print(f"  step {t['step']:>6}: {t['from_phase']} -> {t['to_phase']} "
              f"(reason={t['reason']}, steps_in_old={t['steps_in_old']:,})")
    print(f"\nResults in: {out_dir}")


if __name__ == "__main__":
    main()
