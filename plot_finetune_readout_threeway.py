"""Report 10 -- plot the targeted three-way fine-tuning experiment
(finetune_readout_threeway.py): does fine-tuning ONLY the similarity read-out's
scale/bias, on a stream containing only the (15,15,16) and (7,15,24) three-way
split-chain cells, fix those two cells without forgetting the K=2 own-family sweep?

Three figures:
  * finetune curve: per-cell exact match, cut(2,3) (the failing pair) and the
    predicted-positive rate vs fine-tuning step, one panel per target cell.
  * beyond-the-wall figure: for (7,15,24), reach inside the 24-node component split
    into the three distance buckets (within-capacity/between-walls/beyond-doubled-
    wall) vs fine-tuning step -- did it learn past 2*3^L=18?
  * own-family (K=2 split-chains) sweep, pre (Report IX's already-committed eval on
    the untouched checkpoint) vs post (this fine-tuning's eval_asym_chains.py run).

    python plot_finetune_readout_threeway.py --tag n46_splitchains_seed1000_threeway
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CELLS = [((15, 15, 16), "s15_15_16"), ((7, 15, 24), "s7_15_24")]
BUCKETS = [
    ("largest_comp_within_capacity_d<=9", "d ≤ 9 (within capacity)", "#1b9e77"),
    ("largest_comp_between_walls_9<d<=18", "9 < d ≤ 18 (between the walls)", "#d95f02"),
    ("largest_comp_beyond_doubled_wall_d>18", "d > 18 (beyond the doubled wall)", "#7570b3"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="n46_splitchains_seed1000_threeway")
    ap.add_argument("--finetune_root", default="report10")
    ap.add_argument("--pre_asym_chains",
                     default="runs/report9/asym_chains_n46/n46_splitchains_seed1000/asym_chains.json",
                     help="Report IX's already-committed eval on the UNTOUCHED checkpoint")
    ap.add_argument("--outdir", default="runs/report10/report10_figs")
    args = ap.parse_args()

    runs = Path("runs") / args.finetune_root
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    hist = json.load(open(runs / "finetune_readout_threeway" / args.tag / "finetune_history.json"))
    steps = [r["step"] for r in hist]

    # -------- figure 1: per-cell exact / cut_23 / pred-positive vs step --------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (sizes, key) in zip(axes, CELLS):
        exact = [r["metrics"][key]["exact"] for r in hist]
        cut23 = [r["metrics"][key]["cut_23"] for r in hist]
        pos = [r["metrics"][key]["pred_positive_rate"] for r in hist]
        ax.plot(steps, exact, "o-", color="black", label="exact match", ms=3)
        ax.plot(steps, cut23, "s-", color="#d95f02", label="cut(2,3)", ms=3)
        ax.plot(steps, pos, "^--", color="#7570b3", label="pred.-positive rate", ms=3)
        ax.set_title(f"cell {sizes}")
        ax.set_xlabel("fine-tuning step (read-out only: scale, bias)")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("accuracy / rate")
    axes[0].legend(fontsize=8, loc="center right")
    fig.suptitle("Targeted fine-tuning on {(15,15,16), (7,15,24)}: per-cell metrics vs step")
    fig.tight_layout()
    fig.savefig(outdir / f"r10_finetune_threeway_curve_{args.tag}.png", dpi=150)
    plt.close(fig)

    # -------- figure 2: beyond-the-wall reach for (7,15,24) --------
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for field, label, color in BUCKETS:
        vals = [r["metrics"]["s7_15_24"][field] for r in hist]
        ax.plot(steps, vals, "o-", color=color, label=label, ms=3)
    ax.set_xlabel("fine-tuning step (read-out only: scale, bias)")
    ax.set_ylabel("reach (pairwise accuracy, target connected)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Cell (7,15,24): reach inside the 24-node component, by distance bucket")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"r10_finetune_threeway_beyondwall_{args.tag}.png", dpi=150)
    plt.close(fig)

    # -------- figure 3: own-family (K=2) sweep, pre vs post --------
    pre = json.load(open(args.pre_asym_chains))
    post_path = runs / "asym_chains_n46" / f"{args.tag}_post" / "asym_chains.json"
    post = json.load(open(post_path))
    a_pre = [c["split"] for c in pre["cells"]]
    a_post = [c["split"] for c in post["cells"]]
    exact_pre = [c["exact"] for c in pre["cells"]]
    exact_post = [c["exact"] for c in post["cells"]]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(a_pre, exact_pre, "o-", color="#2c7fb8", label="pre fine-tuning")
    ax.plot(a_post, exact_post, "s--", color="#d95f02", label="post fine-tuning")
    ax.set_xlabel("split $a$ (short-component length)")
    ax.set_ylabel("exact match")
    ax.set_title("Own-family ($K{=}2$ split-chains) sweep, pre vs. post three-way fine-tuning")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / f"r10_finetune_threeway_ownfamily_{args.tag}.png", dpi=150)
    plt.close(fig)

    # -------- scale/bias summary, printed (no figure needed for two numbers) --------
    sb = json.load(open(runs / "finetune_readout_threeway" / args.tag / "scale_bias_summary.json"))
    print(json.dumps(sb, indent=2))
    print("wrote", outdir / f"r10_finetune_threeway_curve_{args.tag}.png")
    print("wrote", outdir / f"r10_finetune_threeway_beyondwall_{args.tag}.png")
    print("wrote", outdir / f"r10_finetune_threeway_ownfamily_{args.tag}.png")


if __name__ == "__main__":
    main()
