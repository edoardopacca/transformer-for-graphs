#!/bin/bash
#SBATCH --job-name=r10theta1920t
#SBATCH --account=3352759
#SBATCH --partition=gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- 2026-09-01, Edoardo: same truncation ablation as theta_20_20_20_truncated
# (both outer routes severed by an internal edge cut, only the direct through-middle route
# real), now applied to the theta_19_20_19 shape -- new family
# "theta_19_20_19_truncated" (data.py's generate_stitched_theta_graph_truncated with
# sizes=(19,20,19), n_isolated=2). Hub pair still degree 3 each (still looks like a 3-route
# hub), direct middle route dist 19, the two outer routes would be 20 each if not cut, plus
# the same 2 isolated padding nodes as theta_19_20_19. Verified via networkx: deg(hub)=3/3,
# dist=19, main component 58 nodes, 2 isolated singleton components, 57 edges. Smoke-tested
# locally (n=60, 20 real steps, batch 8, CPU): runs end-to-end, history.json/checkpoint
# save correctly.
# --time=16h, gpuh200 (not medium_gpuh200/6h10) -- see r10_n60_theta202020_seed1000.sbatch
# for the timing rationale (redundant_mix timed out at 6h, 58.5% of 1e6 steps).
#   sbatch scripts/r10_n60_theta_192019_truncated_seed1000.sbatch

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
    --families theta_19_20_19_truncated --arch roberta --readout similarity --lambda_lap 0 \
    --n_nodes 60 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n60_theta_19_20_19_truncated_roberta_similarity_lam0_seed${SEED}/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "training did not produce a checkpoint at $CKPT"
  exit 1
fi
echo "done: training only, no eval battery attached yet"
echo "checkpoint at $CKPT"
