from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils import load_json, save_json, ensure_dir
from plots import plot_restrict_diameter_dynamics_pairwise

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fmtf(x: float) -> str:
    return f"{x:.3g}" if x != int(x) else f"{int(x)}"


def build_run_id(p: float, diam: int | None) -> str:
    p_part = fmtf(p)
    diam_part = "None" if diam is None else str(diam)
    return f"n20_p{p_part}_d128_layers2_heads4_dff256_bs128_lr0.001_wd0.01_drop0_modeer_valer_ep50_seed42_diam{diam_part}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process pairwise OOD dynamics for restrict-diameter runs")
    parser.add_argument("--p", type=float, default=0.2)
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()

    diameters: list[int | None] = [None, 7, 9, 11]

    out_runs = PROJECT_ROOT / "runs" / "restrict_diameter_dynamics_pairwise"
    ensure_dir(out_runs)

    aggregated: dict[str, Any] = {}

    for diam in diameters:
        run_id = build_run_id(args.p, diam)
        run_dir = PROJECT_ROOT / "trainings" / run_id
        history_path = run_dir / "history.json"
        print(f"Reading run dir: {run_dir}")
        if not history_path.exists():
            raise FileNotFoundError(f"Missing history.json for run {run_id} at {history_path}")
        history = load_json(history_path)

        key_ch = "ood_two_chains_pairwise"
        key_cl = "ood_two_cliques_pairwise"

        if key_ch not in history:
            raise KeyError(f"Key '{key_ch}' missing in {history_path}")
        if key_cl not in history:
            raise KeyError(f"Key '{key_cl}' missing in {history_path}")

        ch_vals = history[key_ch]
        cl_vals = history[key_cl]

        if not isinstance(ch_vals, list) or not isinstance(cl_vals, list):
            raise TypeError(f"Expected lists for pairwise metrics in {history_path}")
        if len(ch_vals) != len(cl_vals):
            # allow different lengths but print warning; prefer error per requirements
            raise ValueError(f"Mismatched lengths in {history_path}: two_chains {len(ch_vals)} vs two_cliques {len(cl_vals)}")

        epochs = list(range(1, len(ch_vals) + 1))

        key = "None" if diam is None else str(diam)
        aggregated[key] = {
            "epochs": epochs,
            "two_chains_pairwise": ch_vals,
            "two_cliques_pairwise": cl_vals,
        }

    agg_json = out_runs / "restrict_diameter_dynamics_pairwise_aggregated.json"
    save_json(agg_json, aggregated)
    print(f"Saved aggregated JSON to {agg_json}")

    png = out_runs / "restrict_diameter_dynamics_pairwise.png"
    plot_restrict_diameter_dynamics_pairwise(agg_json, png)
    print(f"Saved plot to {png}")


if __name__ == "__main__":
    main()
