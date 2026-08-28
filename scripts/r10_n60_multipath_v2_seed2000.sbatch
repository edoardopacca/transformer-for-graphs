#!/bin/bash
#SBATCH --job-name=r10mpv22000
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- REPLACES the earlier leaf-padded multipath test (Edoardo's objection: leaves
# help the transformer recognise terminals, so they are not a clean test). Three leaf-free
# constructions, all using every one of n=60's nodes as a real route node:
# (A) clean 3 parallel routes s-t, edge-lengths [20,21,20], term_deg=3 (no leaves/filler by
#     construction); (B) the disjoint (20,20,20) split (sec:n60-grid) plus 4 stitching edges
#     (1-indexed (1,21),(20,40),(21,41),(40,60)) turning component B's own endpoints into two
#     degree-3 hubs with 3 routes between them (19/21/21 edges); (C) 2026-08-27 addition --
#     same s,t hub structure as A (degree 3, unchanged) but with one INTERNAL edge deleted
#     from 2 of the 3 routes (the 21-edge route and the first 20-edge route), leaving only ONE
#     route actually connecting s to t -- tests whether A's rescue tracks real path redundancy
#     or just the degree>1 cue at s/t. New script eval_n60_multipath_v2.py (smoke-tested
#     locally on a random-weight n=60 checkpoint; Construction C's edge-cutting logic
#     separately verified with networkx: s/t degree unchanged at 3, graph stays connected via
#     exactly one route, at the same distance 20 as A's target pair); (D) 2026-08-27 --
#     degree-control follow-up to C's result (target pair still 1.000 even with only one real
#     route): moves the tested pair one hop further out to two new nodes B,M with perfectly
#     ORDINARY degree 2 (A-B-[hub s]=3 routes=[hub t]-M-N; hubs s,t keep degree 4, not
#     tested), so any degree-anomaly shortcut is gone. Verified locally: B,M degree 2/2 (same
#     as 54 other nodes), hubs degree 4/4, B-M distance 20 (same as A's original pair).
# On short_gpuh200 deliberately (separate queue from the 10-seed long_gpuh200 trainings
# currently pending, so this does not wait behind them).
#   sbatch scripts/r10_n60_multipath_v2_seed2000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report10/n60_train/n60_path_union_k3_roberta_similarity_lam0_seed2000/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found at $CKPT"
  exit 1
fi

python eval_n60_multipath_v2.py --checkpoint "$CKPT" \
    --output_dir "runs/report10/multipath_v2/n60_pathunion_k3_seed2000" \
    --route_lens_a 20 21 20 --term_deg_a 3 --disjoint_sizes_b 20 20 20 \
    --n_graphs_behaviour 50 --n_repeats 5 --n_graphs_heatmap 64 \
    --seed 2000 --tag "seed 2000"
[ $? -ne 0 ] && { echo "FAILED: eval_n60_multipath_v2.py"; exit 1; }
echo "done: n60_pathunion_k3_seed2000 multipath v2"
