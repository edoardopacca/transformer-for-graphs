#!/bin/bash
#SBATCH --job-name=r9a3threemech
#SBATCH --account=3352759
#SBATCH --partition=short_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX, Thread A.3 mechanistic follow-up -- attention scores + layerwise cosine geometry
# for the 5 candidate cells picked from the behavioural sweep (scripts/
# r9_a3_splitchains_threeway_test.sbatch, istruzioni.md §16.13/§16.14), on the SAME split_chains-
# only n=46 seed-1000 checkpoint:
#   (1,22,23)  -- cleanest baseline, trivial isolated node
#   (4,15,27), (7,15,24) -- typical "2-out-of-3 cuts succeed, cut(2,3) fails" cells
#   (2,10,34)  -- the s1=2 anomaly (which pair fails flips)
#   (15,15,16) -- fully-balanced, everything-fails regime
#
# New generic (non-K-way-pooled) scripts, mirroring mechanistic_kway_heatmaps.py/
# stagewise_kway.py but for three SEPARATE, arbitrary-size components instead of one
# long + pooled-short: mechanistic_threeway_heatmaps.py, stagewise_threeway.py. Both reuse
# run_with_cache/exact_contribution/run_with_stages/_cosine_batch unmodified -- only the graph
# construction (generate_multi_path_split_graph with 3 distinct sizes) and index bookkeeping is
# new. Smoke-tested end-to-end on a toy n=15 checkpoint in an isolated /tmp working dir before
# touching real weights.
#
# Eval-only (forward pass only, no backward, no retraining), CPU is enough.
#
#   sbatch scripts/r9_a3_splitchains_threeway_mechanistic.sbatch

mkdir -p out
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report9/n46_train/n46_split_chains_roberta_similarity_lam0_seed1000/last.pt"
TAG="n46_splitchains_seed1000"
CELLS="1,22,23 4,15,27 7,15,24 2,10,34 15,15,16"

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT"
  exit 1
fi

FAIL=0

echo "== mechanistic_threeway_heatmaps.py (attention scores/alpha) =="
python mechanistic_threeway_heatmaps.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/heatmaps_threeway/${TAG}" --cells $CELLS
[ $? -ne 0 ] && { echo "FAILED: mechanistic_threeway_heatmaps.py"; FAIL=1; }

echo "== stagewise_threeway.py (layerwise cosine geometry) =="
python stagewise_threeway.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/stagewise_threeway/${TAG}" --cells $CELLS --n_graphs 64
[ $? -ne 0 ] && { echo "FAILED: stagewise_threeway.py"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more eval steps FAILED -- see FAILED markers above"
  exit 1
fi
echo "done: $TAG (5 cells)"
