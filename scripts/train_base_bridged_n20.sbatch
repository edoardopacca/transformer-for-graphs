#!/bin/bash
#SBATCH --job-name=base20
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report V: retrain the n=20 MIXED linear base model (the older Report-IV checkpoint
# was cleaned for quota) and AUTO-RUN the bridged-cliques probe on it. n20 ER is NOT
# retrained -- we reuse the density-sweep p08 runs (= ER(20,0.08) linear). ~1h40 each.
#
#   sbatch --array=0-3%4 scripts/train_base_bridged_n20.sbatch     # mixed seeds 1000-4000
#   sbatch --array=0 scripts/train_base_bridged_n20.sbatch         # print RUNS table

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

N=20
P=0.08
STEPS=1000000
OUT_ROOT="runs/report5/base_n20"
BRIDGED_ROOT="runs/report5/bridged_cliques"

RUNS=()                          # "set seed"
for S in 1000 2000 3000 4000; do RUNS+=("mixed $S"); done

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
# OOD family context (optional but cheap)
python eval_families.py --checkpoint "$RUN_DIR/last.pt" --output_dir "$RUN_DIR/families" --size 4000
