#!/bin/bash
#SBATCH --job-name=r9simfx3w
#SBATCH --account=3352759
#SBATCH --partition=short_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX -- layerwise similarity geometry for the three-way split cell (s1,s2,s3)=(7,15,24)
# on the frozen-margin ablation checkpoints (A.2b: scale=1/bias=0, no learnable read-out margin),
# both seeds. This is the same cell already shown for the normal (learnable-margin) split-chains-
# only checkpoint in A.3 (one of the five candidate cells) -- here repeated on the sim_fixed
# checkpoints to see whether removing the learnable margin changes the collapse-then-selective-
# rescue mechanism at this cell. Eval-only, forward passes only (no backward/Jacobian) -> CPU.
# No new code: stagewise_threeway.py already exists and is topology/readout-agnostic.
#   sbatch scripts/r9_a3_simfixed_threeway_stagewise_s7_15_24.sbatch

mkdir -p out
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

FAIL=0

for SEED in 1000 2000; do
  CKPT="runs/report9/n46_train/n46_split_chains_roberta_similarityfixed_lam0_seed${SEED}/last.pt"
  TAG="n46_splitchains_simfixed_seed${SEED}"
  if [ ! -f "$CKPT" ]; then
    echo "-- skip seed $SEED: checkpoint not found at $CKPT"; FAIL=1; continue
  fi
  echo "== stagewise_threeway.py, seed $SEED, cell (7,15,24) =="
  python stagewise_threeway.py --checkpoint "$CKPT" \
      --output_dir "runs/report9/stagewise_threeway/${TAG}" --cells 7,15,24 --n_graphs 64
  [ $? -ne 0 ] && { echo "FAILED: stagewise_threeway.py seed $SEED"; FAIL=1; }
done

if [ "$FAIL" -ne 0 ]; then
  echo "one or more steps FAILED/missing -- see markers above"
  exit 1
fi
echo "done: sim_fixed three-way stagewise, cell (7,15,24), both seeds"
