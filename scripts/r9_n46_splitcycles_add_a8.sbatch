#!/bin/bash
#SBATCH --job-name=r9splcyc8
#SBATCH --account=3352759
#SBATCH --partition=short_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX -- add split a=8 to the existing split_cycles seed-1000 attention-scores and
# layerwise-cosine-geometry battery. a=4 is too small to be a fair "solved" reference point
# for cycles (exact match there is only 0.51, borderline -- see scratch_n46_singleseed_
# experiments.tex), whereas a=6..11 is a clean, fully-solved plateau (exact=1.000). a=8 sits
# in the middle of that plateau. Re-runs BOTH scripts with --splits 4 8 23 (not just 8) so
# the existing a4/a23 data stays in the same npz alongside the new a8 -- cheap, eval-only,
# no retraining, same checkpoint as scripts/r9_n46_splitcycles_seed1000.sbatch. Overwrites
# the SAME output directories.
#
#   sbatch scripts/r9_n46_splitcycles_add_a8.sbatch

mkdir -p out
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report9/n46_train/n46_split_cycles_roberta_similarity_lam0_seed1000/last.pt"
TAG="n46_splitcycles_seed1000"

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT"
  exit 1
fi

FAIL=0

echo "== mechanistic_heatmaps.py (attention scores/alpha, splits 4 8 23, topology=cycle) =="
python mechanistic_heatmaps.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/heatmaps/${TAG}" --splits 4 8 23 --topology cycle
[ $? -ne 0 ] && { echo "FAILED: mechanistic_heatmaps.py"; FAIL=1; }

echo "== stagewise_diagnostics.py (layerwise cosine geometry, splits 4 8 23, topology=cycle) =="
python stagewise_diagnostics.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/stagewise/${TAG}" --splits 4 8 23 --n_graphs 64 --topology cycle
[ $? -ne 0 ] && { echo "FAILED: stagewise_diagnostics.py"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more eval steps FAILED -- see FAILED markers above"
  exit 1
fi
echo "done: $TAG (a=8 added)"
