#!/bin/bash
#SBATCH --job-name=r10hub2arm
#SBATCH --account=3352759
#SBATCH --partition=gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- 2026-09-01, Edoardo: dedicated training on a new truncated-hub ablation at
# n=60, family "hub2arm_truncated_n60" (data.py's generate_hub_truncated_arms_graph). Hubs
# s,t joined by a SINGLE real route through 25 interior nodes (true distance 26), plus TWO
# independent truncated "false route" arms (top/bottom), each a 7-node dead-end off s and
# an 8-node dead-end off t, NOT joined to each other -- so s,t still look like degree-3 hubs
# with 3 apparent routes (as in theta_20_20_20_truncated) but only the direct middle route
# is real. Plus 3 fully isolated padding nodes. Sizes: 2 + 25 + (7+8) + (7+8) + 3 = 60.
# Verified via networkx: deg(s)=deg(t)=3, dist(s,t)=26, main component 57 nodes, 3 isolated
# singleton components, 56 edges (26 middle + 15 + 15 arms). Smoke-tested locally (n=60,
# 20 real steps, batch 8, CPU): runs end-to-end, history.json/checkpoint save correctly.
# --time=16h, gpuh200 (not medium_gpuh200/6h10) -- see r10_n60_theta202020_seed1000.sbatch
# for the timing rationale (redundant_mix timed out at 6h, 58.5% of 1e6 steps).
#   sbatch scripts/r10_n60_hub2arm_truncated_seed1000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEED=1000
OUT_ROOT="runs/report10/n60_train"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families hub2arm_truncated_n60 --arch roberta --readout similarity --lambda_lap 0 \
    --n_nodes 60 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n60_hub2arm_truncated_n60_roberta_similarity_lam0_seed${SEED}/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "training did not produce a checkpoint at $CKPT"
  exit 1
fi
echo "done: training only, no eval battery attached yet"
echo "checkpoint at $CKPT"
