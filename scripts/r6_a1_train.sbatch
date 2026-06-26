#!/bin/bash
#SBATCH --job-name=r6a1tr
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=14:00:00
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report VI, Thread A1 -- train the clean single-distribution base models needed by the
# multipath probe, then auto-eval each with eval_multipath.py.
#   - ER       at n10 (p=0.20) and n64 (p=0.03)   [n20/n40 ER are REUSED -> r6_a1_eval.sbatch]
#   - path_union at n10, n20, n40, n64            [sees single paths, never parallel routes]
# Base model = RoBERTa-faithful, linear read-out, L=2, single head, d512 (Reports III--V).
#
# Each array task = ONE (distribution, n, seed). Launch size-blocks separately so the SLURM
# --time/partition fit (err. 18: don't inflate --time). Print the table first:
#   sbatch --array=0 scripts/r6_a1_train.sbatch            # (then read the .out for indices)
# Then, respecting the 4-GPU cap (%4):
#   sbatch -p medium_gpuh200 --time=02:00:00 --array=0-7%4   scripts/r6_a1_train.sbatch  # n10 (fast)
#   sbatch -p medium_gpuh200 --time=04:00:00 --array=8-11%4  scripts/r6_a1_train.sbatch  # n20 pathunion
#   sbatch -p gpunew         --time=12:00:00 --array=12-15%4 scripts/r6_a1_train.sbatch  # n40 pathunion
#   sbatch -p gpunew         --time=14:00:00 --array=16-23%4 scripts/r6_a1_train.sbatch  # n64 (long)

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

OUT_ROOT="runs/report6/a1_train"
EVAL_ROOT="runs/report6/multipath"
STEPS=1000000

# ---- ordered RUNS table: "FAMILIES N P SEED" ----
RUNS=()
for S in 1000 2000 3000 4000; do RUNS+=("er 10 0.20 $S"); done          # 0-3
for S in 1000 2000 3000 4000; do RUNS+=("path_union 10 0.05 $S"); done   # 4-7
for S in 1000 2000 3000 4000; do RUNS+=("path_union 20 0.05 $S"); done   # 8-11
for S in 1000 2000 3000 4000; do RUNS+=("path_union 40 0.05 $S"); done   # 12-15
for S in 1000 2000 3000 4000; do RUNS+=("er 64 0.03 $S"); done           # 16-19
for S in 1000 2000 3000 4000; do RUNS+=("path_union 64 0.05 $S"); done   # 20-23

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "RUNS table (${#RUNS[@]} tasks):"; for i in "${!RUNS[@]}"; do echo "  $i: ${RUNS[$i]}"; done; exit 0
fi

read FAM N P SEED <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> families=$FAM n=$N p=$P seed=$SEED"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families "$FAM" --arch roberta --readout linear --lambda_lap 0 \
    --n_nodes "$N" --p "$P" \
    --train_steps "$STEPS" --batch_size 1000 --num_workers 16 --seed "$SEED"

RUN_NAME="n${N}_${FAM}_roberta_linear_lam0_seed${SEED}"
CKPT="$OUT_ROOT/$RUN_NAME/last.pt"
if [ "$FAM" = "path_union" ]; then DTAG="pathunion"; else DTAG="$FAM"; fi
echo "auto-eval multipath: $CKPT"
python eval_multipath.py --checkpoint "$CKPT" \
    --output_dir "$EVAL_ROOT/n${N}_${DTAG}_seed${SEED}" \
    --ks 1 2 3 4 --ells 3 5 7 9 11 13 15 17 19 --n_graphs 400
