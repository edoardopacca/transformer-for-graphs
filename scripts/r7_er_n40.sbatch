#!/bin/bash
#SBATCH --job-name=r7ern40
#SBATCH --account=3352759
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --array=0-3
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report VII -- the SAME Tier-1/2/3 mechanistic audit already run on the n40
# path_union-trained checkpoints (behind sec:res-sweep/-wout/-attention/
# -threeway/-heatmaps/-ablation), now on the n40 ER-trained checkpoints, as a
# second, parallel condition ("b" panels next to the existing "a" ones): ER
# never saw any path/multi-component structure in training, so it is the
# "never saw this shape" cross-check, the same role it plays at n=64
# (sec:res-n64) but now at n=40 too.
#
# Four pieces per seed, all eval-only (no training, no GPU needed):
#   1. mechanistic_asym_chains.py -- split sweep + relabel + read-out geometry
#      + read-out decomposition + attention probe (now with the EXACT
#      node-to-node contribution, not the old rollout).
#   2. mechanistic_heatmaps.py -- attention-score/weight heatmaps, exact
#      contribution, at the two representative splits a=4, a=20 (the only
#      ones any heatmap figure uses).
#   3. eval_three_way_split.py -- the falsification test (three disjoint
#      components, one small + two large not connected to each other).
#   4. ablation_asym_chains.py -- layer-by-layer (attention/MLP) ablations.
#
# CPU-only (dense matmuls + autograd, not model training) -- BLAS is allowed
# to multi-thread across --cpus-per-task, unlike the GPU eval jobs elsewhere
# in this project. contrib_n_graphs=64 to match the path_union run already
# launched (scripts/r7_exact_contribution.sbatch), for a fair "a" vs "b"
# comparison.
#
# Launch (single command, runs all 4 seeds in parallel; medium_cpu has 8
# nodes, so this can run alongside scripts/r7_exact_contribution.sbatch if
# that one hasn't finished yet -- SLURM just queues what doesn't fit):
#   sbatch scripts/r7_er_n40.sbatch

mkdir -p out
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16 NUMEXPR_NUM_THREADS=16 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

find_ckpt () {
  for c in "$@"; do [ -f "$c" ] && { echo "$c"; return; }; done
  echo ""
}

SEEDS=(1000 2000 3000 4000)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
TAG="n40_er_seed${S}"
CKPT="$(find_ckpt \
  "runs/report4/families_n40/n40_er_roberta_linear_lam0_seed${S}/last.pt" \
  "runs/families_n40/n40_er_roberta_linear_lam0_seed${S}/last.pt")"

if [ -z "$CKPT" ]; then
  echo "no checkpoint found for seed $S among the report4/families_n40 candidates"
  exit 1
fi
echo "seed=$S checkpoint=$CKPT"

# Each step's exit status is checked explicitly (not `set -e`, so a failure in
# one step does not skip the later ones) and OR'd into an overall FAIL flag,
# so the job's own exit code reflects reality -- a step that dies (e.g. an
# unpushed dependency: python's own "No such file or directory" still exits
# non-zero) must NOT be able to leave the job reporting COMPLETED, which is
# exactly what let three_way/ablation silently no-op on the first run of this
# script (istruzioni.md sec 14.8/14.9 fix-up note).
FAIL=0

echo "== mechanistic_asym_chains.py (Tier-1 + attention probe, exact contribution) =="
# --attn_n_graphs raised to 64 too: exact_contribution is capped at
# min(contrib_n_graphs, attn_n_graphs), and the currently-running path_union
# job left attn_n_graphs at its default (40), silently capping it there --
# fixed here so this run gets the full 64 requested.
python mechanistic_asym_chains.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report7/mechanistic/${TAG}" \
  --attn_n_graphs 64 \
  --contrib_n_graphs 64
[ $? -ne 0 ] && { echo "FAILED: mechanistic_asym_chains.py"; FAIL=1; }

echo "== mechanistic_heatmaps.py (heatmap probe, exact contribution) =="
python mechanistic_heatmaps.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report7/heatmaps/${TAG}" \
  --splits 4 20 \
  --contrib_n_graphs 64
[ $? -ne 0 ] && { echo "FAILED: mechanistic_heatmaps.py"; FAIL=1; }

echo "== eval_three_way_split.py (falsification test) =="
python eval_three_way_split.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report7/three_way/${TAG}"
[ $? -ne 0 ] && { echo "FAILED: eval_three_way_split.py"; FAIL=1; }

echo "== ablation_asym_chains.py (layer ablations) =="
python ablation_asym_chains.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report7/ablation/${TAG}"
[ $? -ne 0 ] && { echo "FAILED: ablation_asym_chains.py"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more steps FAILED for seed $S -- see FAILED markers above"
  exit 1
fi
echo "done seed $S"
