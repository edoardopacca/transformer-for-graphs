#!/bin/bash
#SBATCH --job-name=r10pu3pp6000
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- 2026-08-31, Edoardo: investigate whether the low-cut seeds on the plain
# disjoint (20,20,20) split (3000/6000/7000/8000/10000: cut 0.85-0.92, vs 1000/2000/4000/
# 5000/9000: cut 1.000) fail in a STRUCTURED way (same node positions every time) or
# diffusely -- new script eval_threeway_perpair.py (smoke-tested locally on a random-weight
# n=60 checkpoint). Includes seed 1000 (a "good", cut=1.000 seed) for contrast.
#   sbatch scripts/r10_n60_pathunion_k3_perpair_seed6000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report10/n60_train/n60_path_union_k3_roberta_similarity_lam0_seed6000/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found at $CKPT"
  exit 1
fi

python eval_threeway_perpair.py --checkpoint "$CKPT" \
    --output_dir "runs/report10/threeway_perpair/n60_pathunion_k3_seed6000" \
    --n_graphs 300 --seed 6000
[ $? -ne 0 ] && { echo "FAILED: eval_threeway_perpair.py"; exit 1; }
echo "done: seed 6000"
