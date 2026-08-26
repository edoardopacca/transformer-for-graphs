#!/bin/bash
#SBATCH --job-name=r10dvm2000
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- direct comparison Edoardo asked for: on the n=60 K=3 (path_union_k3) checkpoint,
# seed 2000, accuracy on (a) each path's own two extremes in the disjoint 3-way balanced
# split (20,20,20) vs (b) the shared endpoints s,t of a genuine multipath graph with 3 routes
# of 19,19,20 edges (chosen to mirror (a)'s three ~20-node paths as closely as possible while
# using all 60 nodes). New script eval_disjoint_extremes_vs_multipath.py (smoke-tested
# locally on a random-weight n=60 checkpoint). Distinct --seed per checkpoint (this
# checkpoint's own training seed) + --n_repeats 5, learning from the shared-RNG-seed bug found
# in the first multipath run (out/r10pu3ev*.out).
#   sbatch scripts/r10_n60_disjoint_vs_multipath_seed2000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report10/n60_train/n60_path_union_k3_roberta_similarity_lam0_seed2000/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found at $CKPT"
  exit 1
fi

python eval_disjoint_extremes_vs_multipath.py --checkpoint "$CKPT" \
    --output_dir "runs/report10/disjoint_vs_multipath/n60_pathunion_k3_seed2000" \
    --n_graphs 300 --disjoint_sizes 20 20 20 --multipath_lens 19 19 20 --term_deg 4 \
    --seed 2000 --n_repeats 5
[ $? -ne 0 ] && { echo "FAILED: eval_disjoint_extremes_vs_multipath.py"; exit 1; }
echo "done: n60_pathunion_k3_seed2000 disjoint-vs-multipath"
