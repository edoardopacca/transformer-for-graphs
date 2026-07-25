#!/bin/bash
#SBATCH --job-name=r9kway40
#SBATCH --account=3352759
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.out
#
# Report IX, Thread A.4 (mechanistic extension) -- the full Report VIII-style battery
# (Table 1/Figure 1 sweep+logit, real attention scores, layerwise cosine geometry) on
# unions of K=5,6,7 disjoint paths -- component counts the path_union training stream never
# produces (it draws only 1..4) -- using the 4 EXISTING n=40 path_union similarity
# checkpoints (Report VI/VII/VIII), eval-only, no training.
#
# CPU-only: every one of the three steps is either a pure forward pass or (for the attention
# heatmaps/exact-contribution) a handful of cheap backward passes at the script's own default
# contrib_n_graphs=8 -- no GPU needed, and this keeps the scarce 4-concurrent-GPU cap free for
# any training jobs running at the same time.
#
#   sbatch scripts/r9_kway_n40_battery.sbatch

mkdir -p out
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16 NUMEXPR_NUM_THREADS=16 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEEDS=(1000 2000 3000 4000)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
TAG="n40_pathunion_seed${SEED}_kway"
CKPT="runs/report6/a1_train/n40_path_union_roberta_similarity_lam0_seed${SEED}/last.pt"

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT"
  exit 1
fi
echo "seed=$SEED checkpoint=$CKPT"

# the same 9 (K, small_size) cells mechanistic_kway.py derives automatically (K in {5,6,7},
# small_size filtered so the long component's diameter exceeds 18 -- Thread A.4, already
# validated in eval_multiway_split.py): k5:{2,3,4,5} k6:{2,3,4} k7:{2,3}
CELLS="5,2 5,3 5,4 5,5 6,2 6,3 6,4 7,2 7,3"

FAIL=0

echo "== mechanistic_kway.py (sweep + logit -> Table 1 / Figure 1) =="
python mechanistic_kway.py --checkpoint "$CKPT" \
  --output_dir "runs/report9/mechanistic_kway/${TAG}"
[ $? -ne 0 ] && { echo "FAILED: mechanistic_kway.py"; FAIL=1; }

echo "== mechanistic_kway_heatmaps.py (attention scores/alpha -> Figure 3) =="
python mechanistic_kway_heatmaps.py --checkpoint "$CKPT" \
  --output_dir "runs/report9/heatmaps_kway/n40_pathunion_seed${SEED}" \
  --cells $CELLS
[ $? -ne 0 ] && { echo "FAILED: mechanistic_kway_heatmaps.py"; FAIL=1; }

echo "== stagewise_kway.py (layerwise cosine geometry) =="
python stagewise_kway.py --checkpoint "$CKPT" \
  --output_dir "runs/report9/stagewise_kway/n40_pathunion_seed${SEED}" \
  --cells $CELLS --n_graphs 64
[ $? -ne 0 ] && { echo "FAILED: stagewise_kway.py"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more steps FAILED for seed $SEED -- see FAILED markers above"
  exit 1
fi
echo "done seed $SEED"
