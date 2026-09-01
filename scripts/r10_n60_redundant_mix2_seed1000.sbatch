#!/bin/bash
#SBATCH --job-name=r10redmix2
#SBATCH --account=3352759
#SBATCH --partition=gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- 2026-09-01, Edoardo: fifth training of this batch -- 50/50 mix of
# theta_20_20_20 (usual stitched theta, routes 19/21/21) and theta_19_20_19 (routes
# 19/20/20 + 2 isolated nodes), family "redundant_mix2". Unlike the original "redundant_mix"
# (theta vs a structurally different two-route+filler-chain shape, which timed out at 6h
# with val_exact frozen at 0.4942 -- see eval_redundant_mix_indist.py), both halves of this
# mix are the SAME stitched-theta construction, just with different component sizes, so
# it's a cleaner test of whether route-length redundancy transfers between two closely
# related distributions.
# --time=16h, gpuh200 (not medium_gpuh200/6h10) -- redundant_mix's observed rate was
# ~10.3h/1e6 steps extrapolated from 585000 steps in exactly 6h; generous margin here.
# Smoke-tested locally (n=60, 20 real steps, batch 8, CPU, all four new families in one
# mixed stream): runs end-to-end, history.json/checkpoint save correctly.
#   sbatch scripts/r10_n60_redundant_mix2_seed1000.sbatch

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
    --families redundant_mix2 --arch roberta --readout similarity --lambda_lap 0 \
    --n_nodes 60 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n60_redundant_mix2_roberta_similarity_lam0_seed${SEED}/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "training did not produce a checkpoint at $CKPT"
  exit 1
fi
echo "done: training only, no eval battery attached yet"
echo "checkpoint at $CKPT"
