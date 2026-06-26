#!/bin/bash
#SBATCH --job-name=r6a3
#SBATCH --account=3352759
#SBATCH --partition=medium_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report VI, Thread A3 -- do truncated routes in training confuse the model? Trains the
# base model on a multipath stream where a fraction trunc_frac of the graphs have some
# DEAD-END routes (n_full in {0..k-1} full + the rest truncated one hop short of t; labels
# correct, so s-t connected iff n_full>=1). Evaluated on the SAME clean multipath val set
# as Thread A2 -> compare the clean-multipath learning curve vs A2's trunc_frac=0 baseline.
# Same OUT_ROOT as A2: the run-name carries the trunc tag, so the clean (A2) and truncated
# (A3) curves sit side by side for plotting.
#
# Configs (each has its trunc=0 baseline already in A2): n40 k3 ell7, n40 k2 ell9.
# trunc_frac in {0.3, 0.6}; 4 seeds. 300k steps, eval every 2k.
#   sbatch scripts/r6_a3_truncated.sbatch                  # prints the table and exits (no --array)
#   sbatch --array=0-15%4 scripts/r6_a3_truncated.sbatch   # all

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

OUT_ROOT="runs/report6/multipath_train"

# ---- ordered RUNS table: "N K ELL TRUNC SEED" ----
CONFIGS=("40 3 7 0.3" "40 3 7 0.6" "40 2 9 0.3" "40 2 9 0.6")
RUNS=()
for C in "${CONFIGS[@]}"; do for S in 1000 2000 3000 4000; do RUNS+=("$C $S"); done; done

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "RUNS table (${#RUNS[@]} tasks):"; for i in "${!RUNS[@]}"; do echo "  $i: ${RUNS[$i]}"; done; exit 0
fi

read N K ELL TR SEED <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> n=$N k=$K ell=$ELL trunc=$TR seed=$SEED"

python experiments2/train_multipath.py \
    --output_root "$OUT_ROOT" \
    --n_nodes "$N" --n_full "$K" --path_len "$ELL" --trunc_frac "$TR" \
    --arch roberta --readout linear \
    --train_steps 300000 --eval_every 2000 --batch_size 1000 --num_workers 16 --seed "$SEED"
