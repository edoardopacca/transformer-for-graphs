#!/bin/bash
#SBATCH --job-name=r10redmixid
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- 2026-09-01, Edoardo: the redundant_mix training job (r10redmix, job 643495)
# TIMED OUT at the 6h wall with val_exact frozen at exactly 0.4942 from step 5000 to 585000
# (117 identical prints in the stdout log) -- new script eval_redundant_mix_indist.py checks
# whether that's a shape-level split (theta shape solved, twopath not, or vice versa) by
# evaluating last.pt on each of the two training shapes SEPARATELY, with a per-pair-category
# breakdown (target long-range pair / other same-component pairs / cross-component for the
# twopath shape's separate filler chain). Smoke-tested locally on a random-weight n=60
# checkpoint (4 graphs per shape, mask sizes verified: theta_other=1769, twopath
# route_other=779, cross=800, filler_internal=190 -- all match the expected C(n,2) counts).
#   sbatch scripts/r10_n60_redundant_mix_indist_seed1000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report10/n60_train/n60_redundant_mix_roberta_similarity_lam0_seed1000/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found at $CKPT"
  exit 1
fi

python eval_redundant_mix_indist.py --checkpoint "$CKPT" \
    --output_dir "runs/report10/redundant_mix_indist/seed1000" \
    --n_graphs 500 --seed 777
[ $? -ne 0 ] && { echo "FAILED: eval_redundant_mix_indist.py"; exit 1; }
echo "done: seed 1000"
