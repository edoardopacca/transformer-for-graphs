#!/bin/bash
#SBATCH --job-name=r6ctrS
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report VI, Thread C -- SIMILARITY read-out variant. Identical to r6_c_train.sbatch but trains
# the clean bridged+split single-bridge base model with --readout similarity. Report V (sec 5.5)
# showed the similarity head pushes the bridged node budget out (knee ~7 -> ~11-12 at n40); this
# trains the composition base under that head so the chained / thick-bridge eval can ask whether
# a learned single bridge composes FURTHER under similarity. Checkpoint folders carry
# "_similarity_" so they sit beside the linear ones in the SAME OUT_ROOT with no collision.
#
# Base model = RoBERTa-faithful, similarity read-out, L=2, single head, d512.
# One array task = ONE (n, seed). The eval is a SEPARATE eval-only job (r6_c_eval_sim.sbatch).
#
# Print the RUNS table (run WITHOUT --array; NOT --array=0):
#   sbatch scripts/r6_c_train_sim.sbatch
# Then, respecting the 4-GPU cap (%4):
#   sbatch -p medium_gpuh200 --time=03:00:00 --array=0-3%4  scripts/r6_c_train_sim.sbatch  # n20
#   sbatch -p gpunew         --time=12:00:00 --array=4-7%4  scripts/r6_c_train_sim.sbatch  # n40

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

OUT_ROOT="runs/report6/c_train"
STEPS=1000000

# ordered RUNS table: "N P SEED"
RUNS=()
for S in 1000 2000 3000 4000; do RUNS+=("20 0.08 $S"); done   # 0-3
for S in 1000 2000 3000 4000; do RUNS+=("40 0.05 $S"); done   # 4-7

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "RUNS table (${#RUNS[@]} tasks):"; for i in "${!RUNS[@]}"; do echo "  $i: ${RUNS[$i]}"; done; exit 0
fi

read N P SEED <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> n=$N p=$P seed=$SEED (families=bridged,split) readout=similarity"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families bridged,split --arch roberta --readout similarity --lambda_lap 0 \
    --n_nodes "$N" --p "$P" \
    --train_steps "$STEPS" --batch_size 1000 --num_workers 16 --seed "$SEED"

echo "done -> $OUT_ROOT/n${N}_bridged+split_roberta_similarity_lam0_seed${SEED}/last.pt"
