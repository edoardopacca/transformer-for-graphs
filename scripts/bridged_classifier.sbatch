#!/bin/bash
#SBATCH --job-name=brclf
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report V: dedicated bridged-vs-split binary classifier (GraphBinaryClassifier).
# Two angles:
#   * clique_size = -1 (random per sample): the DFS test as a TRAINABLE task --
#     accuracy by clique size (flat = matrix-power, falling at large c = DFS-like).
#   * fixed clique_size in {3,6,10}: convergence speed vs density (the density-in-
#     optimisation angle for this task).
# n=20, 200k steps (~20 min each).
#
# Indices (10 tasks): rand x4 seeds, then fixed-c x 2 seeds.
#   sbatch --array=0-9%4 scripts/bridged_classifier.sbatch              # n20, fixed c={3,6,10}
#   sbatch --array=0 scripts/bridged_classifier.sbatch                  # print RUNS table only
#   N=40 FIXED_C="5 10 15 20" sbatch --array=0-11%4 scripts/bridged_classifier.sbatch  # n40 (12 tasks)
# N (default 20) and the fixed-c list FIXED_C (default "3 6 10") are read from the
# environment so n40 needs no separate file; run names encode n{N} so they never collide.

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

N=${N:-20}
FIXED_C=${FIXED_C:-"3 6 10"}
STEPS=200000
OUT_ROOT="runs/report5/bridged_clf"

RUNS=()                                   # "clique_size seed"
for S in 1000 2000 3000 4000; do RUNS+=("-1 $S"); done    # random-c sweep
for C in $FIXED_C; do for S in 1000 2000; do RUNS+=("$C $S"); done; done   # fixed-c convergence

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "RUNS (${#RUNS[@]} tasks):"; for i in "${!RUNS[@]}"; do echo "  $i: ${RUNS[$i]}"; done; exit 0
fi

read CS SEED <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> n=$N clique_size=$CS seed=$SEED"

python experiments2/train_bridged_classifier.py \
    --output_root "$OUT_ROOT" --n_nodes "$N" --clique_size "$CS" \
    --train_steps "$STEPS" --batch_size 1000 --num_workers 16 --seed "$SEED"
