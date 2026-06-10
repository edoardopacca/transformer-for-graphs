#!/bin/bash
#SBATCH --job-name=fam40
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Families experiment at n=40 (Report IV, task 2: the 3^L wall). At n=40 the
# components are large enough that 2chains/path_union/chain_plus contain
# within-component distances well past 9, so the capacity wall is visible over a
# wide range (as in Report III) -- unlike n=20 where it is truncated at d=9.
#
# 4 arms x 2 seeds, ordered by seed; each trains then auto-evals on all families.
#   sbatch --array=0-7%2 scripts/train_families_n40.sbatch
# Use gpunew with a REALISTIC time limit, not long_gpunew/36h: long_gpunew has only
# 2 nodes (shared with gpunew) and a 36h job never backfills there. n=40 runs ~6-8h,
# so 12h is plenty and schedules far more easily.
#
# lambda is dropped here: the seed-1000 sweep showed it is inert at convergence
# (the similarity read-out drives the Laplacian energy to ~0 on its own).

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

OUT_ROOT="runs/families_n40"
STEPS=1000000

RUNS=()
for S in 1000 2000; do
  RUNS+=("er linear $S")
  RUNS+=("er similarity $S")
  RUNS+=("mixed linear $S")
  RUNS+=("mixed similarity $S")
done

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "RUNS (${#RUNS[@]}):"; for i in "${!RUNS[@]}"; do echo "  $i: ${RUNS[$i]}"; done; exit 0
fi

read SET RO SEED <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> n=40 set=$SET readout=$RO seed=$SEED"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" --n_nodes 40 --p 0.05 \
    --families "$SET" --arch roberta --readout "$RO" --lambda_lap 0 \
    --train_steps "$STEPS" --batch_size 1000 --num_workers 16 --seed "$SEED"

RUN_NAME="n40_${SET}_roberta_${RO}_lam0_seed${SEED}"
RUN_DIR="$OUT_ROOT/$RUN_NAME"
echo "auto-eval: $RUN_DIR/last.pt"
python eval_families.py --checkpoint "$RUN_DIR/last.pt" --output_dir "$RUN_DIR/families" --size 4000
