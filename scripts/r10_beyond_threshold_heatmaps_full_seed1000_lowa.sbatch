#!/bin/bash
#SBATCH --job-name=r10bthflowa
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- 2026-09-02, Edoardo: the professor asked for Figure 4 (fig:beyond-diameter-
# heatmap, paper_submission_prigm.tex) to also show some splits with a<12 (the succeeding
# half of the sweep), not just the failing a=12..23 range already computed -- but not ALL of
# a=1..11 (too many panels) and not a hand-picked subset either (would look like cherry-
# picking). Compromise: odd a from 3 to 23 plus a=12 (the exact breakpoint) = 12 panels
# total, replacing the current all-failing set. a=12..23 already exist in
# runs/report10/beyond_threshold_heatmaps/n46_splitchains_full_seed1000/ -- this job adds
# only the missing a=3,5,7,9,11 to that same directory, via eval_beyond_threshold_heatmaps.py's
# new --splits option (same checkpoint/n_graphs/seed as the original full-range job).
#   sbatch scripts/r10_beyond_threshold_heatmaps_full_seed1000_lowa.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report9/n46_train/n46_split_chains_roberta_similarity_lam0_seed1000/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found at $CKPT"
  exit 1
fi

python eval_beyond_threshold_heatmaps.py --checkpoint "$CKPT" \
    --output_dir "runs/report10/beyond_threshold_heatmaps/n46_splitchains_full_seed1000" \
    --n_graphs 64 --seed 1000 --tag "full seed 1000, low-a panels" \
    --splits 3,5,7,9,11
[ $? -ne 0 ] && { echo "FAILED: eval_beyond_threshold_heatmaps.py (low-a)"; exit 1; }
echo "done: n46_splitchains_full_seed1000 low-a beyond-threshold heatmaps"
