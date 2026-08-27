#!/bin/bash
#SBATCH --job-name=r10splch8
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# paper_draft.tex Sec. 4.1 -- add split a=8 to the existing full-distribution (n46_splitchains,
# seed 1000) attention-scores battery. The npz currently only has a=4 and a=23; the paper's
# attention figure needs the SAME two splits, (8,38) and (23,23), for both the odd/grid model
# (already has a=8) and the full-distribution model, so the two panels are a fair side-by-side
# comparison rather than an accidental mismatch. Re-runs mechanistic_heatmaps.py with
# --splits 4 8 23 (not just 8) so the existing a4/a23 data stays in the same npz alongside the
# new a8 -- cheap, eval-only, no retraining, same checkpoint as
# scripts/r9_n46_splitchains_seed1000.sbatch. Overwrites the SAME output directory.
#
#   sbatch scripts/r10_n46_splitchains_add_a8.sbatch

mkdir -p out
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report9/n46_train/n46_split_chains_roberta_similarity_lam0_seed1000/last.pt"
TAG="n46_splitchains_seed1000"

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT"
  exit 1
fi

echo "== mechanistic_heatmaps.py (attention scores/alpha, splits 4 8 23, topology=chain) =="
python mechanistic_heatmaps.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/heatmaps/${TAG}" --splits 4 8 23 --topology chain
[ $? -ne 0 ] && { echo "FAILED: mechanistic_heatmaps.py"; exit 1; }

echo "done: $TAG (a=8 added)"
