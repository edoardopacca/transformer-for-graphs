#!/bin/bash
#SBATCH --job-name=density
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report V: does DENSITY help OPTIMISATION (not just inference)? Train the base
# model (RoBERTa, linear) on ER(n, p) at several edge probabilities p and read off,
# from history.json, the convergence speed and the final in-distribution exact match,
# plus the OOD family battery (auto-eval). Matrix powering predicts denser = easier
# (shorter distances); a bounded traversal predicts denser = harder (more to explore).
#
# n=20 here (fast, ~1h40 each). For n=40 copy this file, set N=40 and use a sparser
# grid (e.g. P_LIST="0.03 0.05 0.08 0.12") and --time 12:00:00.
#
# Indices: 5 p-values x 4 seeds = 20 tasks, ordered (p outer, seed inner):
#   sbatch --array=0-19%4 scripts/density_sweep.sbatch
#   sbatch --array=0 scripts/density_sweep.sbatch     # print the RUNS table only
#
# Then on HPC: git add runs/report5/density_sweep && commit && push
# Locally: git pull ; python plot_density_sweep.py

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

N=20
P_LIST="0.05 0.08 0.12 0.16 0.22"
SEEDS="1000 2000 3000 4000"
STEPS=1000000
OUT_BASE="runs/report5/density_sweep"

RUNS=()
for P in $P_LIST; do
  for S in $SEEDS; do
    RUNS+=("$P $S")
  done
done

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "RUNS (${#RUNS[@]} tasks):"; for i in "${!RUNS[@]}"; do echo "  $i: ${RUNS[$i]}"; done; exit 0
fi

read P SEED <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"
PTAG="p$(echo "$P" | sed 's/^0\.//')"        # 0.05 -> p05, 0.12 -> p12
OUT_ROOT="$OUT_BASE/$PTAG"
echo "task $SLURM_ARRAY_TASK_ID -> n=$N p=$P seed=$SEED  (out: $OUT_ROOT)"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" --n_nodes "$N" --p "$P" \
    --families er --arch roberta --readout linear --lambda_lap 0 \
    --train_steps "$STEPS" --batch_size 1000 --num_workers 16 --seed "$SEED"

RUN_NAME="n${N}_er_roberta_linear_lam0_seed${SEED}"
RUN_DIR="$OUT_ROOT/$RUN_NAME"
echo "auto-eval (OOD families): $RUN_DIR/last.pt"
python eval_families.py --checkpoint "$RUN_DIR/last.pt" --output_dir "$RUN_DIR/families" --size 4000
