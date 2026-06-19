#!/bin/bash
#SBATCH --job-name=trbr
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report V, Exp 2 (the data-prior test). Train the base connectivity model with
# bridged + split cliques ADDED to the mixed stream (random clique size), then run
# the bridged-cliques probe on it. The base model (sec 5.2) never saw bridged cliques
# (held out) and its cross-block collapses with clique size. Question: is that a hard
# CAPACITY limit, or just a DATA gap?
#   - If the cross-block STILL collapses with clique size after training on bridged
#     -> a hard visit-bounded capacity limit at L=2 (the strong conclusion).
#   - If it now propagates the bridge at every clique size -> it was only a data gap.
# This directly attacks the main caveat. n=40 is decisive (clique size up to 20).
#
# Uses train_families_n20.py --include_bridged (fam_tag -> "mixedbr").
#
#   sbatch --array=0-3%4 scripts/train_bridged_in_stream.sbatch   # n40, seeds 1000-4000 (decisive)
#   sbatch --array=4-7%4 scripts/train_bridged_in_stream.sbatch   # n20, seeds 1000-4000 (corroboration)
#   sbatch --array=0 scripts/train_bridged_in_stream.sbatch       # print RUNS table

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

STEPS=1000000
OUT_ROOT="runs/report5/train_bridged_in_stream"
BRIDGED_ROOT="runs/report5/bridged_cliques_trained"

RUNS=()                          # "N P seed"  -- n40 first (decisive), then n20
for S in 1000 2000 3000 4000; do RUNS+=("40 0.05 $S"); done
for S in 1000 2000 3000 4000; do RUNS+=("20 0.08 $S"); done

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "RUNS (${#RUNS[@]}):"; for i in "${!RUNS[@]}"; do echo "  $i: ${RUNS[$i]}"; done; exit 0
fi

read N P SEED <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> n=$N p=$P seed=$SEED  (mixed + bridged/split in stream)"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" --n_nodes "$N" --p "$P" \
    --families mixed --include_bridged --arch roberta --readout linear --lambda_lap 0 \
    --train_steps "$STEPS" --batch_size 1000 --num_workers 16 --seed "$SEED"

RUN_DIR="$OUT_ROOT/n${N}_mixedbr_roberta_linear_lam0_seed${SEED}"
echo "auto-eval bridged: $RUN_DIR/last.pt"
python eval_bridged_cliques.py --checkpoint "$RUN_DIR/last.pt" \
    --output_dir "$BRIDGED_ROOT/n${N}_seed${SEED}" --n_graphs 2000
