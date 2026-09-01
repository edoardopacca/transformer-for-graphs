#!/bin/bash
#SBATCH --job-name=r10thetatrunc
#SBATCH --account=3352759
#SBATCH --partition=gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- 2026-09-01, Edoardo: dedicated training on the Construction-C-style ablation
# of the usual theta_20_20_20 shape (routes 19/21/21) -- new family
# "theta_20_20_20_truncated" (data.py's generate_stitched_theta_graph_truncated): same hub
# pair, still degree 3 each (still LOOKS locally like a 3-route hub), but the two outer
# routes each have one strictly-internal edge cut, so only the direct through-middle route
# (dist 19) actually connects s,t -- the whole graph stays one connected component (the
# severed outer chains become dead-end arms off each hub, no new isolated nodes). This
# trains FROM SCRATCH on the ablated shape directly, instead of testing an
# already-trained theta_20_20_20 checkpoint OOD on it (eval_n60_multipath_v2.py's
# Construction C did the OOD version already: target-pair accuracy stayed 1.000 there).
# Verified via networkx: hub degree 3/3, hub dist 19 (only path), 60 nodes all one
# component, 59 edges (61 - 2 cuts). Smoke-tested locally (n=60, 20 real steps, batch 8,
# CPU): runs end-to-end, history.json/checkpoint save correctly.
# --time=16h, gpuh200 (not medium_gpuh200/6h10) -- see r10_n60_theta202020_seed1000.sbatch
# for the timing rationale (redundant_mix timed out at 6h, 58.5% of 1e6 steps).
#   sbatch scripts/r10_n60_theta_truncated_seed1000.sbatch

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
    --families theta_20_20_20_truncated --arch roberta --readout similarity --lambda_lap 0 \
    --n_nodes 60 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n60_theta_20_20_20_truncated_roberta_similarity_lam0_seed${SEED}/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "training did not produce a checkpoint at $CKPT"
  exit 1
fi
echo "done: training only, no eval battery attached yet"
echo "checkpoint at $CKPT"
