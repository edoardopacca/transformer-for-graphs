#!/bin/bash
#SBATCH --job-name=r6a2
#SBATCH --account=3352759
#SBATCH --partition=medium_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report VI, Thread A2 -- sample efficiency of learning the multipath connectivity.
# Trains the base model (RoBERTa, linear, L=2, d512) on a PURE multipath stream (clean,
# trunc_frac=0) and logs, over training, the (s,t)-pair accuracy AND the whole-matrix
# exact match on a held-out clean multipath val set -> steps (x batch = samples) to learn.
# Grid generalises across n, the number of routes k, and the route length ell:
#   - vary k at fixed (n=40, ell=7):   k = 1,2,3,4
#   - vary ell at fixed (n=40, k=2):   ell = 5,7,9,11,13
#   - vary n at fixed (k=2, ell=7):    n = 20,40,64
# 4 seeds each. 300k steps, eval every 2k. medium partition (each run < 6h).
#
# To print the table run WITHOUT --array (NOT --array=0, which RUNS task 0); then launch in
# blocks respecting the 4-GPU cap (%4):
#   sbatch scripts/r6_a2_samples.sbatch                    # prints the table and exits
#   sbatch --array=0-39%4 scripts/r6_a2_samples.sbatch     # all (or subset the indices)

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

OUT_ROOT="runs/report6/multipath_train"

# ---- ordered RUNS table: "N K ELL SEED" (trunc_frac = 0, clean) ----
CONFIGS=("40 1 7" "40 2 7" "40 3 7" "40 4 7" \
         "40 2 5" "40 2 9" "40 2 11" "40 2 13" \
         "20 2 7" "64 2 7")
RUNS=()
for C in "${CONFIGS[@]}"; do for S in 1000 2000 3000 4000; do RUNS+=("$C $S"); done; done

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "RUNS table (${#RUNS[@]} tasks):"; for i in "${!RUNS[@]}"; do echo "  $i: ${RUNS[$i]}"; done; exit 0
fi

read N K ELL SEED <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> n=$N k=$K ell=$ELL trunc=0 seed=$SEED"

python experiments2/train_multipath.py \
    --output_root "$OUT_ROOT" \
    --n_nodes "$N" --n_full "$K" --path_len "$ELL" --trunc_frac 0 \
    --arch roberta --readout linear \
    --train_steps 300000 --eval_every 2000 --batch_size 1000 --num_workers 16 --seed "$SEED"
