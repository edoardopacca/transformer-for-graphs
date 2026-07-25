#!/bin/bash
#SBATCH --job-name=r9a4mw
#SBATCH --account=3352759
#SBATCH --partition=short_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --array=0-3
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report IX, Thread A.4 -- 5/6/7-chain out-of-distribution test on the EXISTING n=40
# path_union (disjoint-paths), similarity read-out checkpoints (Report VI/VII/VIII; the
# online training stream only ever draws 1..4 path components, so K=5,6,7 total components
# were never produced in training). Eval-only: no training, no GPU needed -- this is a pure
# forward pass (no backward/Jacobian, unlike the mechanistic scripts of Report VII/VIII),
# so it is run on CPU precisely so it does NOT compete with the two n=46 sanity-check
# training jobs for the project's 4-concurrent-GPU-per-user cap; short_cpu (high priority,
# 1h10 cap) is ample for ~9 cells x 300 graphs of forward passes on this small (L=2, d512)
# model.
#
#   sbatch scripts/r9_a4_multiway_n40.sbatch

mkdir -p out
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEEDS=(1000 2000 3000 4000)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
CKPT="runs/report6/a1_train/n40_path_union_roberta_similarity_lam0_seed${SEED}/last.pt"

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT"
  exit 1
fi
echo "seed=$SEED checkpoint=$CKPT"

python eval_multiway_split.py \
    --checkpoint "$CKPT" \
    --output_dir "runs/report9/multiway_split/n40_pathunion_seed${SEED}" \
    --n_components 5 6 7 \
    --dist_cutoff 18 \
    --n_graphs 300
