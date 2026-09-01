#!/bin/bash
#SBATCH --job-name=r10chain2020
#SBATCH --account=3352759
#SBATCH --partition=gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- 2026-09-01, Edoardo: dedicated single-shape training on ONLY the plain
# disjoint (20,20,20) 3-chain graph, NO stitching -- the no-redundancy baseline for this
# batch of 4 trainings, fixed at this one exact split (unlike "path_union_k3", which draws
# split sizes uniformly over all combinations). One of four narrower trainings requested to
# see which of three distributions the model can learn on its own (the others:
# theta_20_20_20 = usual stitched theta routes 19/21/21, theta_19_20_19 = routes 19/20/20 +
# 2 isolated nodes, redundant_mix2 = 50/50 of theta_20_20_20 and theta_19_20_19).
# --time=16h, gpuh200 (not medium_gpuh200/6h10) -- see r10_n60_theta202020_seed1000.sbatch
# for the timing rationale (redundant_mix timed out at 6h, 58.5% of 1e6 steps).
# Smoke-tested locally (n=60, 20 real steps, batch 8, CPU, all four new families in one
# mixed stream): runs end-to-end, history.json/checkpoint save correctly.
#   sbatch scripts/r10_n60_chain3202020_seed1000.sbatch

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
    --families chain3_20_20_20 --arch roberta --readout similarity --lambda_lap 0 \
    --n_nodes 60 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n60_chain3_20_20_20_roberta_similarity_lam0_seed${SEED}/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "training did not produce a checkpoint at $CKPT"
  exit 1
fi
echo "done: training only, no eval battery attached yet"
echo "checkpoint at $CKPT"
