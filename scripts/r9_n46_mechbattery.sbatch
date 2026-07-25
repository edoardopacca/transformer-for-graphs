#!/bin/bash
#SBATCH --job-name=r9n46mb
#SBATCH --account=3352759
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report IX -- mechanistic battery on the two n=46 sanity-check checkpoints (seeds 1000, 2000
# from scripts/r9_n46_sanity_seed{1000,2000}.sbatch). Reuses the exact same scripts already
# validated on the n=40 checkpoints in Report VII/VIII, with NO code changes -- only the
# checkpoint path and the representative-split pair change (4 and n//2=23, the direct analogue
# of the "4, 20" pair used throughout at n=40).
#
# Produces, per seed:
#   1. mechanistic_asym_chains.py -- the full behavioural split sweep + the raw similarity
#      logit decomposition (Report VIII Table 1 / Figure 1: metrics.csv + readout.csv).
#      --attn_splits restricted to the same two representative splits (4, 23) to keep the
#      attention probe's Jacobian-based exact-contribution step cheap; the sweep itself
#      (Table 1/Figure 1) runs over the FULL default split range regardless -- it is a pure
#      forward-pass computation, not backward, so it stays cheap over many splits.
#   2. mechanistic_heatmaps.py -- the real attention-score/alpha heatmaps at the two
#      representative splits (Report VIII Figure 3), plus the raw weight matrices; left at the
#      script's own (cheap) default contrib_n_graphs=8/n_graphs=80, appropriate for a sanity
#      check, not the higher contrib_n_graphs=64 used for final report figures.
#   3. stagewise_diagnostics.py -- the 5-stage layerwise cosine geometry probe (no backward
#      pass at all, cheap regardless of split count).
#
# CPU-only (no GPU needed for any of these three -- pure forward passes, plus a handful of
# small backward passes for the exact-contribution step): medium_cpu is ample.
#
# EACH seed's eval is gated ONLY on that seed's own training job, not on both together -- the
# two trainings finish at different times, and there is no reason to make seed 1000's eval wait
# on seed 2000's training (or vice versa). This is the SAME array script (SEEDS=(1000 2000),
# index 0/1) submitted TWICE, each time overriding --array to a single index and --dependency to
# that seed's own training job ID (a command-line --array overrides the #SBATCH default above;
# afterany, not afterok -- errore 44: a training job that ends for any reason still leaves a
# checkpoint to evaluate, and afterok would leave this stuck PD forever on a non-zero exit for
# an unrelated reason):
#
#   sbatch --dependency=afterany:604831 --array=0 scripts/r9_n46_mechbattery.sbatch   # seed 1000
#   sbatch --dependency=afterany:604832 --array=1 scripts/r9_n46_mechbattery.sbatch   # seed 2000
#
# (replace the two job IDs if the training jobs were resubmitted under different IDs by the
# time this runs -- check with `squeue -u $(whoami)`).

mkdir -p out
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16 NUMEXPR_NUM_THREADS=16 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEEDS=(1000 2000)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
TAG="n46_pathunion_seed${SEED}"
CKPT="runs/report9/n46_train/n46_path_union_roberta_similarity_lam0_seed${SEED}/last.pt"

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT (did the training job actually finish/save a checkpoint?)"
  exit 1
fi
echo "seed=$SEED checkpoint=$CKPT"

FAIL=0

echo "== mechanistic_asym_chains.py (sweep + logit -> Table 1 / Figure 1) =="
python mechanistic_asym_chains.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report9/mechanistic/${TAG}" \
  --attn_splits 4 23
[ $? -ne 0 ] && { echo "FAILED: mechanistic_asym_chains.py"; FAIL=1; }

echo "== mechanistic_heatmaps.py (attention scores/alpha -> Figure 3) =="
python mechanistic_heatmaps.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report9/heatmaps/${TAG}" \
  --splits 4 23
[ $? -ne 0 ] && { echo "FAILED: mechanistic_heatmaps.py"; FAIL=1; }

echo "== stagewise_diagnostics.py (layerwise cosine geometry) =="
python stagewise_diagnostics.py \
  --checkpoint "$CKPT" \
  --output_dir "runs/report9/stagewise/${TAG}" \
  --splits 4 23 --n_graphs 64
[ $? -ne 0 ] && { echo "FAILED: stagewise_diagnostics.py"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more steps FAILED for seed $SEED -- see FAILED markers above"
  exit 1
fi
echo "done seed $SEED"
