#!/bin/bash
#SBATCH --job-name=r8cycles
#SBATCH --account=3352759
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --array=0-3
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report VIII -- the path-endpoint falsification test (istruzioni.md sec 15,
# report/8 sec:cycles). The mechanistic evidence so far (Report VII's
# far-endpoint-as-source effect under the linear read-out; the layer-1
# high-attention columns at a path's two ends under the similarity read-out,
# report/8 sec:battery-attn) both single out a path's two degree-1 endpoints
# as structurally unusual. This job tests whether that is LOAD-BEARING (the
# model needs a degree-1 "open end" to decide a segment is resolved) or
# INCIDENTAL (any two nodes farthest from the rest of their segment would do)
# by closing BOTH segments of the two-chain split into CYCLES -- same split
# (a, n-a), same total n, same two-component structure, but neither component
# has an endpoint anymore.
#
# Deliberately uses the SAME exact node-to-node contribution C_ik as every
# result since Report VII (Jacobian of h_i^(2) wrt h_k^(0)) -- the revised,
# adjacency-based contribution measure is a SEPARATE open question
# (report/8 sec:outlook) and is not conflated with this experiment.
#
# Checkpoints (existing, similarity read-out -- this project now standardizes
# on it, report/8 sec 2 -- no training):
#   runs/report6/a1_train/n40_path_union_roberta_similarity_lam0_seed{S}/last.pt
#
# Two pieces per seed, both eval-only (no GPU -- dense matmuls + autograd):
#   1. mechanistic_asym_chains.py --topology cycle -- behavioural sweep +
#      readout decomposition + attention probe (leak fraction, exact
#      contribution). Splits default to 3..n//2 automatically for cycles
#      (a cycle needs >=3 nodes per component).
#   2. mechanistic_heatmaps.py --topology cycle -- attention-score/weight
#      heatmaps at two representative splits (a=4, a=20 -- kept as the SAME
#      pair used throughout report/8 sec:battery for a direct comparison).
#
# CPU-only, BLAS multi-threaded across --cpus-per-task, same pattern as every
# other r7_*/r8_* mechanistic sbatch in this project.
#
# Launch (single command, 4 seeds, medium_cpu has 8 nodes):
#   sbatch scripts/r8_two_cycles.sbatch

mkdir -p out
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16 NUMEXPR_NUM_THREADS=16 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEEDS=(1000 2000 3000 4000)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
CKPT="runs/report6/a1_train/n40_path_union_roberta_similarity_lam0_seed${S}/last.pt"
TAG="n40_pathunion_cycle_seed${S}"

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT"
  exit 1
fi

FAIL=0

echo "== mechanistic_asym_chains.py --topology cycle (seed $S) =="
python mechanistic_asym_chains.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report8/mechanistic/${TAG}" \
  --topology cycle \
  --attn_n_graphs 64 \
  --contrib_n_graphs 64
[ $? -ne 0 ] && { echo "FAILED: mechanistic_asym_chains.py"; FAIL=1; }

echo "== mechanistic_heatmaps.py --topology cycle (seed $S) =="
python mechanistic_heatmaps.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report8/heatmaps/${TAG}" \
  --topology cycle \
  --splits 4 20 \
  --contrib_n_graphs 64
[ $? -ne 0 ] && { echo "FAILED: mechanistic_heatmaps.py"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more steps FAILED for seed $S -- see FAILED markers above"
  exit 1
fi
echo "done seed $S"
