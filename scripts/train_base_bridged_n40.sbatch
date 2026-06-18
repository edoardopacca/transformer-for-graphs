#!/bin/bash
#SBATCH --job-name=base40
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report V: retrain the n=40 ER and MIXED linear base models (the older Report-IV
# checkpoints were cleaned for quota) and AUTO-RUN the bridged-cliques probe on each.
# n=40 is where the cliques are large enough that a bounded-traversal (DFS) limit, if
# present, should bite hardest. ~6-8h each; use gpunew with --time 12h (not 36h: see
# the long_gpunew backfill note in istruzioni).
#
#   sbatch --array=0-7%4 scripts/train_base_bridged_n40.sbatch     # er+mixed, seeds 1000-4000
#   sbatch --array=0 scripts/train_base_bridged_n40.sbatch         # print RUNS table

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

N=40
P=0.05
STEPS=1000000
OUT_ROOT="runs/report5/base_n40"
BRIDGED_ROOT="runs/report5/bridged_cliques"

RUNS=()                          # "set seed"  (seed inner)
for S in 1000 2000 3000 4000; do RUNS+=("er $S"); RUNS+=("mixed $S"); done

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "RUNS (${#RUNS[@]}):"; for i in "${!RUNS[@]}"; do echo "  $i: ${RUNS[$i]}"; done; exit 0
fi

read SET SEED <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> n=$N set=$SET seed=$SEED"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" --n_nodes "$N" --p "$P" \
    --families "$SET" --arch roberta --readout linear --lambda_lap 0 \
    --train_steps "$STEPS" --batch_size 1000 --num_workers 16 --seed "$SEED"

RUN_DIR="$OUT_ROOT/n${N}_${SET}_roberta_linear_lam0_seed${SEED}"
echo "auto-eval bridged: $RUN_DIR/last.pt"
python eval_bridged_cliques.py --checkpoint "$RUN_DIR/last.pt" \
    --output_dir "$BRIDGED_ROOT/n${N}_${SET}_seed${SEED}" --n_graphs 2000
python eval_families.py --checkpoint "$RUN_DIR/last.pt" --output_dir "$RUN_DIR/families" --size 4000
