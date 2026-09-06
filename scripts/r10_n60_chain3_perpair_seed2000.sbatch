#!/bin/bash
#SBATCH --job-name=r10ch3pp2000
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- 2026-09-06, Edoardo: is the whole-graph exact match=0.000 (all 5 seeds) on
# the DEDICATED chain3_20_20_20 training (fixed (20,20,20) split ONLY, not the random-split
# path_union_k3) caused EXCLUSIVELY by the 3 same-component endpoint pairs (d=19, the "6
# extremes")? Reuses eval_threeway_perpair.py (already built + smoke-tested for exactly
# this fixed-split shape, previously run on path_union_k3 checkpoints) -- same real
# per-pair predicted-connected-rate methodology, now pointed at the chain3_20_20_20 seed
# 2000 checkpoint instead.
#   sbatch scripts/r10_n60_chain3_perpair_seed2000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report10/n60_train/n60_chain3_20_20_20_roberta_similarity_lam0_seed2000/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found at $CKPT"
  exit 1
fi

python eval_threeway_perpair.py --checkpoint "$CKPT" \
    --output_dir "runs/report10/threeway_perpair/n60_chain3_20_20_20_seed2000" \
    --n_graphs 300 --seed 2000
[ $? -ne 0 ] && { echo "FAILED: eval_threeway_perpair.py"; exit 1; }
echo "done: seed 2000"
