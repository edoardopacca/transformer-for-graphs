#!/bin/bash
#SBATCH --job-name=r7contrib
#SBATCH --account=3352759
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report VII -- rerun of the Tier-1 attention probe (mechanistic_asym_chains.py) and the
# heatmap probe (mechanistic_heatmaps.py) with the REAL node-to-node contribution
# C_ik = ||d h_i^(2)/d h_k^(0)||_F (exact autograd through V, W_O, the residual, the MLP,
# LayerNorm) in place of the old alpha2@alpha1 attention-only rollout approximation.
# CPU-only (no GPU needed -- this is dense matmuls + autograd, not model training), so a
# CPU partition is used and BLAS is allowed to multi-thread across --cpus-per-task (unlike
# the GPU eval jobs elsewhere in this project, which pin OMP threads to 1).
#
# Writes into the SAME output directories the earlier (rollout-based) run used, so
# metrics.csv/readout.csv/weights_summary.json/raw_weights.npz are harmlessly regenerated
# identical, and attn_cache.npz / heatmap_data.npz are overwritten with the new field
# (contrib_exact_mean / contrib_exact) instead of the old "rollout" field.
#
# Graph counts are intentionally generous (not the small default used for a quick local
# check): 64 graphs averaged per split for the expensive exact-contribution quantity, on
# top of the existing (cheap) attn_n_graphs=40 / heatmap n_graphs=80 for alpha/scores/
# message-contribution/row-mass, which are unaffected by this change.
#
# Launch (single command, runs all 4 seeds in parallel, medium_cpu has 8 nodes):
#   sbatch scripts/r7_exact_contribution.sbatch

mkdir -p out
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16 NUMEXPR_NUM_THREADS=16 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEEDS=(1000 2000 3000 4000)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
CKPT="runs/report6/a1_train/n40_path_union_roberta_linear_lam0_seed${S}/last.pt"
TAG="n40_pathunion_seed${S}"

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT"
  exit 1
fi

FAIL=0

echo "== mechanistic_asym_chains.py (Tier-1 + attention probe, exact contribution) -- seed $S =="
python mechanistic_asym_chains.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report7/mechanistic/${TAG}" \
  --contrib_n_graphs 64
[ $? -ne 0 ] && { echo "FAILED: mechanistic_asym_chains.py"; FAIL=1; }

echo "== mechanistic_heatmaps.py (heatmap probe, exact contribution) -- seed $S =="
python mechanistic_heatmaps.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report7/heatmaps/${TAG}" \
  --splits 4 20 \
  --contrib_n_graphs 64
[ $? -ne 0 ] && { echo "FAILED: mechanistic_heatmaps.py"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more steps FAILED for seed $S -- see FAILED markers above"
  exit 1
fi
echo "done seed $S"
