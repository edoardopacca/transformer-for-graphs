#!/bin/bash
#SBATCH --job-name=r10dirchred
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- 2026-08-31, Edoardo: the actual redundant-chain rescue test
# (eval_directed_chain_redundancy.py) on the directed_chain n=40 checkpoint (job 640794,
# TIMEOUT at step 990,000/1,000,000 -- using last.pt as-is, since train_loss had already
# plateaued for the last 40,000+ logged steps: 0.53774 -> 0.53771, functionally converged;
# the training script has no --resume/--init_from support, so continuing from the timeout
# is not straightforward, and does not look necessary here).
#
# K=1,2,3,4 independent directed chains s->t, each >9 hops (route_len=11 edges for K=1-3,
# 10 for K=4 -- see eval_directed_chain_redundancy.py docstring for why), testing whether
# reachability(s,t) is rescued by redundancy once a single chain is provably past the
# linear-read-out's 3^L=9 wall (Edoardo, 2026-08-28: confident this wall is exactly 9 for
# this read-out, matching Ye et al.'s un-doubled capacity result).
#   sbatch scripts/r10_n40_directed_chain_redundancy_seed1000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report10/n40_train/n40_directed_chain_roberta_linear_lam0_seed1000/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found at $CKPT"
  exit 1
fi

python eval_directed_chain_redundancy.py --checkpoint "$CKPT" \
    --output_dir "runs/report10/directed_chain_redundancy/n40_seed1000" \
    --n_graphs 200 --seed 1000 --tag "seed 1000"
[ $? -ne 0 ] && { echo "FAILED: eval_directed_chain_redundancy.py"; exit 1; }
echo "done: n40_directed_chain_redundancy seed1000"
