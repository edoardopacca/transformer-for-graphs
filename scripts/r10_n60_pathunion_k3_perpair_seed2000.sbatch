#!/bin/bash
#SBATCH --job-name=r10pu3pp2000
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- 2026-08-31, Edoardo: complete the per-pair threeway-split data for seeds
# 1000-5000 (already have 1000, 3000; this adds 2000, 4000, 5000), needed for the exact
# same-component d<=18 / d>18 decomposition in paper_draft.tex Table 4
# (tab:n60-beyond-diameter), now updated to 5 seeds instead of 2.
#   sbatch scripts/r10_n60_pathunion_k3_perpair_seed2000.sbatch

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

python eval_threeway_perpair.py --checkpoint "$CKPT" \
    --output_dir "runs/report10/threeway_perpair/n60_pathunion_k3_seed2000" \
    --n_graphs 300 --seed 2000
[ $? -ne 0 ] && { echo "FAILED: eval_threeway_perpair.py"; exit 1; }
echo "done: seed 2000"
