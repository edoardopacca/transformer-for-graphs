#!/bin/bash
#SBATCH --job-name=r10th1920_3000
#SBATCH --account=3352759
#SBATCH --partition=gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- 2026-09-03, Edoardo: 5-seed batch for theta_19_20_19 (routes 19/20/20 + 2
# isolated nodes, all real). Seed 1000 already COMPLETED (job 645338) -- this is one of the
# remaining 4 seeds.
#   sbatch scripts/r10_n60_theta192019_seed3000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEED=3000
OUT_ROOT="runs/report10/n60_train"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families theta_19_20_19 --arch roberta --readout similarity --lambda_lap 0 \
    --n_nodes 60 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n60_theta_19_20_19_roberta_similarity_lam0_seed${SEED}/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "training did not produce a checkpoint at $CKPT"
  exit 1
fi
echo "done: training only, no eval battery attached yet"
echo "checkpoint at $CKPT"
