#!/bin/bash
#SBATCH --job-name=r7simn40
#SBATCH --account=3352759
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --array=0-7
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report VII -- the SAME Tier-1/2/3 mechanistic audit already run on the n40
# LINEAR-read-out checkpoints (path_union "a" + ER "b" sections), now on the
# SIMILARITY read-out, both training distributions, to answer the user's
# question directly: does the two-component completion signal depend on the
# read-out being linear, or does a model with the similarity read-out
# (z_ij = scale*cos(h_i,h_j)+bias, a genuine function of BOTH embeddings, no
# per-target lookup vector) learn the same thing?
#
# Code changes that made this possible (previously these scripts raised
# NotImplementedError on any non-linear checkpoint):
#   - mechanistic_asym_chains.py: run_with_cache's final read-out step now
#     branches on readout_kind (mirrors model.forward_and_embeddings); new
#     readout_decomposition_similarity (cos(h_i,h_j) in place of h_i^T w_j)
#     and weights_geometry_similarity (E_in-only -- there is no W_out for
#     this read-out, only the two scalars sim_scale/sim_bias) added, chosen
#     automatically in main() based on the loaded checkpoint's readout_kind.
#   - ablation_asym_chains.py: forward_ablated's two model.read_out(h) calls
#     replaced with a small _apply_readout() helper that branches the same way.
#   - mechanistic_heatmaps.py: raw_weights() skips W_out/b_out/sv_W_out for a
#     similarity checkpoint (nothing there to extract) and stores sim_scale/
#     sim_bias instead.
#   - eval_three_way_split.py needed NO changes (already readout-agnostic --
#     it only calls model.forward(), which already dispatches on readout_kind).
# All four scripts were smoke-tested end-to-end on a toy similarity checkpoint
# before this sbatch was written (selftest passes, no crashes).
#
# Four pieces per (distribution, seed), all eval-only (no training, no GPU):
#   1. mechanistic_asym_chains.py -- split sweep + relabel + read-out
#      decomposition/geometry (now the similarity-specific versions) +
#      attention probe (exact node-to-node contribution, readout-independent).
#   2. mechanistic_heatmaps.py -- attention-score/weight heatmaps at the two
#      representative splits a=4, a=20.
#   3. eval_three_way_split.py -- the falsification test.
#   4. ablation_asym_chains.py -- layer-by-layer ablations.
#
# Checkpoints (existing, no training -- from Report VI's similarity variant,
# istruzioni.md sec 13.10, and Report IV's families_n40 similarity training):
#   path_union: runs/report6/a1_train/n40_path_union_roberta_similarity_lam0_seed{S}/last.pt
#   ER:         runs/report4/families_n40/n40_er_roberta_similarity_lam0_seed{S}/last.pt
#               (fallback runs/families_n40/... -- non-bucketed path, err. 34)
#
# CPU-only, BLAS multi-threaded across --cpus-per-task (dense matmuls +
# autograd, not model training). Same contrib_n_graphs/attn_n_graphs=64 fix
# used for the ER-n40/n20 runs (not the under-specified first path_union run).
#
# Launch (single command, 8 tasks = 2 distributions x 4 seeds; medium_cpu has
# 8 nodes, so this can run alongside other r7_*.sbatch jobs if still going):
#   sbatch scripts/r7_similarity_n40.sbatch

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
# index 0-3 -> path_union-similarity seeds 1000..4000 ; index 4-7 -> ER-similarity seeds 1000..4000
if [ "$SLURM_ARRAY_TASK_ID" -lt 4 ]; then
  S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
  TAG="n40_pathunion_similarity_seed${S}"
  CKPT="$(find_ckpt "runs/report6/a1_train/n40_path_union_roberta_similarity_lam0_seed${S}/last.pt")"
else
  IDX=$((SLURM_ARRAY_TASK_ID - 4))
  S=${SEEDS[$IDX]}
  TAG="n40_er_similarity_seed${S}"
  CKPT="$(find_ckpt \
    "runs/report4/families_n40/n40_er_roberta_similarity_lam0_seed${S}/last.pt" \
    "runs/families_n40/n40_er_roberta_similarity_lam0_seed${S}/last.pt")"
fi

if [ -z "$CKPT" ]; then
  echo "no checkpoint found for $TAG"
  exit 1
fi
echo "tag=$TAG checkpoint=$CKPT"

FAIL=0

echo "== mechanistic_asym_chains.py (Tier-1 + attention probe, exact contribution) =="
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
  echo "one or more steps FAILED for $TAG -- see FAILED markers above"
  exit 1
fi
echo "done $TAG"
