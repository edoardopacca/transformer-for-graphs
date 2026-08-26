#!/bin/bash
#SBATCH --job-name=r10mprep1000
#SBATCH --account=3352759
#SBATCH --partition=short_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report X -- rerun of the n=60 K=3 multipath test (mechanistic_multipath_geometry.py) fixing
# a methodological bug found in the first run (out/r10pu3ev1000_637206.out,
# out/r10pu3ev2000_637207.out): both checkpoint seeds were evaluated with the SAME default
# --seed (12345), and since generate_multipath_graph's base graph is itself deterministic,
# this means the two checkpoints saw IDENTICAL relabelled input graphs at every k -- their
# exact numeric agreement at k=2 (term_connect=0.04 both) and k=3 (term_connect=0.10,
# reach_route=0.8734 both, to 4 decimals) reflected identical graphs, not independent model
# behaviour. Fix: --seed base is now the checkpoint's OWN training seed (1000 vs 2000, so the
# two checkpoints draw different graphs from the start), plus --n_repeats 5 (independent
# draws, base_seed+0..4) so term_connect gets a real across-repeat std instead of a single
# n_graphs=50 point estimate. Also newly records the per-graph (not just mean) s-t cosine
# std/positive-logit-fraction at H2, to directly check the mean-vs-term_connect paradox noted
# in report/10 sec:n60-multipath (mean cosine strongly positive while term_connect was only
# 4-10% at k=2,3).
#   sbatch scripts/r10_n60_multipath_geometry_repeats_seed1000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report10/n60_train/n60_path_union_k3_roberta_similarity_lam0_seed1000/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found at $CKPT"
  exit 1
fi

python mechanistic_multipath_geometry.py --checkpoint "$CKPT" \
    --output_dir "runs/report10/multipath_geometry/n60_pathunion_k3_seed1000_repeats5" \
    --n_graphs 50 --ks 1 2 3 --path_len 19 --seed 1000 --n_repeats 5
[ $? -ne 0 ] && { echo "FAILED: mechanistic_multipath_geometry.py"; exit 1; }
echo "done: n60_pathunion_k3_seed1000_repeats5"
