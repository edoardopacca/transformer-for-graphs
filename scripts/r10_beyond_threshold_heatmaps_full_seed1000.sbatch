#!/bin/bash
#SBATCH --job-name=r10bthf1000
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X (personal working doc) -- per-split H^2 similarity-geometry heatmaps, requested by
# Edoardo, for every failing split a=12..23 of the two-path FULL split-size distribution
# checkpoint (paper_draft.tex Sec 4.1 / Table 2 in the appendix), seed 1000. Same style as the
# n60 geometry heatmaps (scripts/r10_n60_geometry_heatmap_seed1000.sbatch). New script
# eval_beyond_threshold_heatmaps.py (smoke-tested locally: syntax + import check only, no local
# checkpoint available to run a forward pass on -- watch the per-split log lines below for the
# first real run).
#   sbatch scripts/r10_beyond_threshold_heatmaps_full_seed1000.sbatch

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
    --n_graphs 64 --seed 1000 --tag "full seed 1000"
[ $? -ne 0 ] && { echo "FAILED: eval_beyond_threshold_heatmaps.py (full)"; exit 1; }
echo "done: n46_splitchains_full_seed1000 beyond-threshold heatmaps"
