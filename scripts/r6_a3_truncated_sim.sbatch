#!/bin/bash
#SBATCH --job-name=r6a3S
#SBATCH --account=3352759
#SBATCH --partition=medium_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report VI, Thread A3 -- SIMILARITY read-out variant. Identical to r6_a3_truncated.sbatch but
# with --readout similarity: do truncated (dead-end) routes still confuse the model under the
# similarity head? Evaluated on the same clean multipath val set; the run-name carries
# "_similarity_" and "trunc{f}", so the similarity clean (A2-sim) and truncated curves pair up.
# Each config's similarity clean baseline is the matching r6_a2_samples_sim.sbatch run.
#
#   sbatch scripts/r6_a3_truncated_sim.sbatch                  # prints the table and exits
#   sbatch --array=0-15%4 scripts/r6_a3_truncated_sim.sbatch   # all

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
echo "task $SLURM_ARRAY_TASK_ID -> n=$N k=$K ell=$ELL trunc=$TR seed=$SEED readout=similarity"

python experiments2/train_multipath.py \
    --output_root "$OUT_ROOT" \
    --n_nodes "$N" --n_full "$K" --path_len "$ELL" --trunc_frac "$TR" \
    --arch roberta --readout similarity \
    --train_steps 300000 --eval_every 2000 --batch_size 1000 --num_workers 16 --seed "$SEED"
