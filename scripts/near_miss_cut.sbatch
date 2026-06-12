#!/bin/bash
#SBATCH --job-name=nearmiss
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Near-miss cut test (Report IV, Section "A direct test: the near-miss cut").
# Eval-only: split a single chain into two components by removing one edge and
# measure cut accuracy vs near-miss distance, plus reach vs distance on the same
# graphs. Runs on the n=40 mixed checkpoints (linear + similarity, 4 seeds) and the
# n=64 depth-sweep L=2 checkpoints (linear + similarity) for the depth comparison.

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

run () {  # $1 = checkpoint path
  CKPT="$1"
  if [ -f "$CKPT" ]; then
    echo "=== $CKPT ==="
    python eval_near_miss_cut.py --checkpoint "$CKPT" \
        --output_dir "$(dirname "$CKPT")/near_miss_cut"
  else
    echo "MISSING $CKPT"
  fi
}

# ---- n=40 mixed, linear + similarity, four seeds ----
for RO in linear similarity; do
  for S in 1000 2000 3000 4000; do
    run "runs/families_n40/n40_mixed_roberta_${RO}_lam0_seed${S}/last.pt"
  done
done

# ---- n=64 depth-sweep L=2 (linear + similarity), for the depth comparison ----
run "runs/reach_depth/reach_n64_L2_normalized_relu_seed1000/last.pt"
run "runs/reach_depth/reach_n64_L2_normalized_relu_similarity_seed1000/last.pt"
