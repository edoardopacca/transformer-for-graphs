#!/bin/bash
#SBATCH --job-name=r10pu3ev9000
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- 2026-08-31, Edoardo: extend the plain-disjoint-(20,20,20)-split baseline
# (exact match for every 3-way split-size combination, eval_threeway_splitchains.py) to
# seed 9000, one of the 8 new K=3 path-union seeds, to reach 10 seeds total for
# paper_draft.tex Sec. 4.3's baseline claim (currently only seeds 1000/2000, scripts/
# r10_n60_pathunion_k3_eval_seed{1000,2000}.sbatch). Same eval, only the checkpoint changes
# -- does NOT re-run mechanistic_multipath_geometry.py (not needed for this update).
#   sbatch scripts/r10_n60_pathunion_k3_eval_seed9000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report10/n60_train/n60_path_union_k3_roberta_similarity_lam0_seed9000/last.pt"
TAG="n60_pathunion_k3_seed9000"

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found at $CKPT"
  exit 1
fi

python eval_threeway_splitchains.py --checkpoint "$CKPT" \
    --output_dir "runs/report10/threeway_splitchains/${TAG}" --n_graphs 300
[ $? -ne 0 ] && { echo "FAILED: eval_threeway_splitchains.py"; exit 1; }
echo "done: $TAG"
