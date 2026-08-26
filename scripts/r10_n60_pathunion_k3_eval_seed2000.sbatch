#!/bin/bash
#SBATCH --job-name=r10pu3ev2000
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- eval-only battery on the n=60 path_union_k3 training checkpoint, seed 2000
# (scripts/r10_n60_pathunion_k3_seed2000.sbatch, COMPLETED 2026-08-26). Two questions:
# (1) exact match for EVERY 3-way split-size combination (eval_threeway_splitchains.py,
#     already generic on n/checkpoint -- same tool used for the n=46 K=3 grid in Report X
#     experiment 1); (2) does this checkpoint, trained only on DISJOINT 3-way splits, still
#     benefit from extra parallel routes on a genuine multipath (shared-endpoint) graph, and
#     what does the final-layer (H^2) similarity geometry look like when it does
#     (mechanistic_multipath_geometry.py, new script).
#   sbatch scripts/r10_n60_pathunion_k3_eval_seed2000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report10/n60_train/n60_path_union_k3_roberta_similarity_lam0_seed2000/last.pt"
TAG="n60_pathunion_k3_seed2000"

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found at $CKPT"
  exit 1
fi

FAIL=0

echo "== eval_threeway_splitchains.py (exact match for every 3-way split combination) =="
python eval_threeway_splitchains.py --checkpoint "$CKPT" \
    --output_dir "runs/report10/threeway_splitchains/${TAG}" --n_graphs 300
[ $? -ne 0 ] && { echo "FAILED: eval_threeway_splitchains.py"; FAIL=1; }

echo "== mechanistic_multipath_geometry.py (multipath accuracy + H2 geometry, k=1,2,3) =="
python mechanistic_multipath_geometry.py --checkpoint "$CKPT" \
    --output_dir "runs/report10/multipath_geometry/${TAG}" --n_graphs 50 --ks 1 2 3 --path_len 19
[ $? -ne 0 ] && { echo "FAILED: mechanistic_multipath_geometry.py"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more eval steps FAILED -- see FAILED markers above"
  exit 1
fi
echo "done: $TAG"
