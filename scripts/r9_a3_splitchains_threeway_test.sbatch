#!/bin/bash
#SBATCH --job-name=r9a3threeway
#SBATCH --account=3352759
#SBATCH --partition=short_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX, Thread A.3 -- test the split_chains-only-trained n=46 seed-1000 checkpoint (this
# report's condition (3), trained EXCLUSIVELY on 2-component splits) on graphs built by taking
# that same 2-way split and removing one internal edge from one of the two paths, producing 3
# disjoint path components -- a component count this training stream never produced by
# construction (unlike the Report VII/VIII three-way tests, whose training already covered 3-4
# components). Tests EVERY distinct combination of resulting component sizes (every partition of
# 46 into 3 positive parts, 176 cells at n=46), not just a canonical "one long + two small"
# shape -- the point is to see for which size combinations the model still succeeds.
#
# Eval-only (forward pass only, no backward, no retraining), CPU is enough -- same order of
# cost as the other CPU-only eval batteries in this project. This is intentionally just the
# basic behavioural sweep (exact/reach/cut per cell) so results come back fast; attention
# scores / layerwise cosine geometry are a deliberate follow-up, run only on whichever specific
# cells turn out interesting once these numbers are back.
#
#   sbatch scripts/r9_a3_splitchains_threeway_test.sbatch

mkdir -p out
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report9/n46_train/n46_split_chains_roberta_similarity_lam0_seed1000/last.pt"
OUTDIR="runs/report9/threeway_splitchains/n46_splitchains_seed1000"

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT"
  exit 1
fi

python eval_threeway_splitchains.py \
    --checkpoint "$CKPT" \
    --output_dir "$OUTDIR" \
    --n_graphs 200 \
    --min_size 1

echo "done: $OUTDIR/threeway_split.csv"
