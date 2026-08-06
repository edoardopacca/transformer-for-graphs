#!/bin/bash
#SBATCH --job-name=r9ppiso
#SBATCH --account=3352759
#SBATCH --partition=short_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX -- clean parallel-paths rescue test on the n=46 split_chains-only checkpoints
# (Table 3 / A.2's four seeds): two terminals s,t joined by k in {1,2,3,4} internally-
# disjoint paths of 11 edges each, every OTHER node completely ISOLATED (no leaf padding,
# no filler component -- data.py::generate_parallel_paths_graph, NOT the leaf-padded
# generate_multipath_graph used for the earlier prof-requested multipath attention viz).
# 2 + 4*(11-1) = 42 <= 46, so k up to 4 fits at this canvas.
#
# For each seed/k: predicted-connected accuracy on the (s,t) pair (does k routes rescue
# the connection at this fixed, beyond-capacity distance?), within-route reach, isolation
# accuracy (nothing spuriously connects to the isolated filler), predicted-positive rate,
# plus the real attention score/alpha matrices (layer 0/1), eval-only (forward passes only
# -> CPU). Smoke-tested end to end on a 20-step toy checkpoint before this job was written.
#   sbatch scripts/r9_parallel_paths_isolated_splitchains_n46.sbatch

mkdir -p out
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

FAIL=0

for SEED in 1000 2000 3000 4000; do
  CKPT="runs/report9/n46_train/n46_split_chains_roberta_similarity_lam0_seed${SEED}/last.pt"
  TAG="n46_splitchains_seed${SEED}"
  if [ ! -f "$CKPT" ]; then
    echo "-- skip seed $SEED: checkpoint not found at $CKPT"; FAIL=1; continue
  fi
  echo "== mechanistic_parallel_paths_isolated.py, seed $SEED =="
  python mechanistic_parallel_paths_isolated.py --checkpoint "$CKPT" \
      --output_dir "runs/report9/parallel_paths_isolated/${TAG}" \
      --ks 1 2 3 4 --path_len 11 --n_graphs 300
  [ $? -ne 0 ] && { echo "FAILED: seed $SEED"; FAIL=1; }
done

if [ "$FAIL" -ne 0 ]; then
  echo "one or more seeds FAILED/missing -- see markers above"
  exit 1
fi
echo "done: parallel-paths-isolated battery, split_chains-only n=46, seeds 1000/2000/3000/4000"
