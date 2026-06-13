#!/bin/bash
#SBATCH --job-name=ppclean
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Confound-free parallel-paths probe (Report IV, Thread 2 follow-up), on the n=40
# connectivity checkpoints. Distance fixed (ell), terminal degree fixed, canvas filled
# (sparse), only the number of parallel paths k (= resistance ell/k) varies. Measures the
# terminal-pair accuracy vs k at fixed ell, for the ER- and mixed-trained linear models.

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

run () {
  CKPT="$1"
  if [ -f "$CKPT" ]; then
    echo "=== $CKPT ==="
    python eval_parallel_paths_clean.py --checkpoint "$CKPT" \
        --output_dir "$(dirname "$CKPT")/parallel_paths_clean"
  else
    echo "MISSING $CKPT"
  fi
}

for FAM in er mixed; do
  for S in 1000 2000 3000 4000; do
    run "runs/report4/families_n40/n40_${FAM}_roberta_linear_lam0_seed${S}/last.pt"
  done
done
