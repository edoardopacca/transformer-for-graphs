#!/bin/bash
#SBATCH --job-name=brdepth
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report V, Exp 1 (OPTIONAL, elegant): the causal depth test. Train the MIXED base
# model at depth L=1 and L=3 (L=2 already = the existing base) -- bridged cliques
# stay HELD OUT, exactly as for the L=2 base -- then run the bridged probe on each.
# Question: does the clique-size "node budget" move with depth?
#   - budget grows with L (e.g. ~ tracks 3^L) -> it IS the architectural capacity,
#     the same limit as the distance-9 wall, and the data-prior reading is killed.
#   - budget stays ~6-7 at every L -> it is NOT the 3^L capacity (a surprise worth
#     reporting).
# This is the bonus that ties the node-budget to the 3^L wall causally. n=20 is the
# cheap read of the trend (~2-3h); n=40 confirms it at large cliques but L=3 n40 is
# slow (~12-14h) -- launch n20 first.
#
# Uses train_families_n20.py --n_layers (run-name gets an _L{L} tag for L != 2).
#
#   sbatch --array=0-3%4 scripts/train_bridged_depth_sweep.sbatch   # n20 L1/L3, seeds 1000/2000
#   sbatch --array=4-7%2 scripts/train_bridged_depth_sweep.sbatch   # n40 L1/L3, seeds 1000/2000 (slow)
#   sbatch --array=0 scripts/train_bridged_depth_sweep.sbatch       # print RUNS table

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

STEPS=1000000
OUT_ROOT="runs/report5/depth_sweep"
BRIDGED_ROOT="runs/report5/bridged_cliques_depth"

RUNS=()                          # "N P L seed"  -- n20 first (cheap), then n40
for L in 1 3; do for S in 1000 2000; do RUNS+=("20 0.08 $L $S"); done; done
for L in 1 3; do for S in 1000 2000; do RUNS+=("40 0.05 $L $S"); done; done

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "RUNS (${#RUNS[@]}):"; for i in "${!RUNS[@]}"; do echo "  $i: ${RUNS[$i]}"; done; exit 0
fi

read N P L SEED <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> n=$N p=$P L=$L seed=$SEED  (mixed, bridged held out)"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" --n_nodes "$N" --p "$P" --n_layers "$L" \
    --families mixed --arch roberta --readout linear --lambda_lap 0 \
    --train_steps "$STEPS" --batch_size 1000 --num_workers 16 --seed "$SEED"

RUN_DIR="$OUT_ROOT/n${N}_mixed_roberta_linear_lam0_L${L}_seed${SEED}"
echo "auto-eval bridged: $RUN_DIR/last.pt"
python eval_bridged_cliques.py --checkpoint "$RUN_DIR/last.pt" \
    --output_dir "$BRIDGED_ROOT/n${N}_L${L}_seed${SEED}" --n_graphs 2000
