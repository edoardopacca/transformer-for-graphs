#!/bin/bash
#SBATCH --job-name=r6ctr
#SBATCH --account=3352759
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report VI, Thread C -- train the CLEAN single-bridge base model for the
# "does a learned bridge compose?" thread. Training distribution (data principle, one
# explicitly named distribution per run -- NOT the opaque mixed stream, and NOT the
# Report-V mixedbr checkpoints which mix bridged with the 9-family stream):
#
#   bridged + split cliques  (two cliques, random clique size 2..n/2, WITH or WITHOUT
#   the single bridge edge, 50/50). This is the minimal distribution that contains the
#   "two cliques are one component iff the bridge is there" decision -- positive
#   (bridged) and negative (split) -- and nothing else, so any success on the held-out
#   CHAINS (eval_clique_chain.py) and THICK bridges / dense ER blocks
#   (eval_bridged_cliques.py) is genuine COMPOSITION, not having seen diversity.
#
# Base model = RoBERTa-faithful, linear read-out, L=2, single head, d512 (Reports III--V).
# One array task = ONE (n, seed). The eval is a SEPARATE eval-only job (r6_c_eval.sbatch).
#
# Print the RUNS table (run WITHOUT --array; NOT --array=0, which RUNS task 0):
#   sbatch scripts/r6_c_train.sbatch
# Then, respecting the 4-GPU cap (%4):
#   sbatch -p medium_gpuh200 --time=03:00:00 --array=0-3%4  scripts/r6_c_train.sbatch  # n20
#   sbatch -p gpunew         --time=12:00:00 --array=4-7%4  scripts/r6_c_train.sbatch  # n40

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
echo "task $SLURM_ARRAY_TASK_ID -> n=$N p=$P seed=$SEED (families=bridged,split)"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families bridged,split --arch roberta --readout linear --lambda_lap 0 \
    --n_nodes "$N" --p "$P" \
    --train_steps "$STEPS" --batch_size 1000 --num_workers 16 --seed "$SEED"

echo "done -> $OUT_ROOT/n${N}_bridged+split_roberta_linear_lam0_seed${SEED}/last.pt"
