#!/bin/bash
#SBATCH --job-name=r9splitre2
#SBATCH --account=3352759
#SBATCH --partition=gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX controlled-distribution battery -- family 'split_regular3', n=46, seed 2000.
# Two disjoint, CONNECTED, 3-REGULAR random graphs of asymmetric sizes (a,
# 46-a) -- Report IX's cleanest control: EVERY node has the same degree
# everywhere, no endpoints, no chords, no degree signature of component size at
# all. If imbalanced splits are STILL solved more easily than balanced ones
# here, the effect cannot come from any local landmark or global degree cue --
# it would have to come from the component-SIZE asymmetry itself. Split sizes
# must be EVEN (d=3 regularity feasibility) -- representative splits are 4/42 and
# 22/24 (as close to balanced as evenness allows), not 4/23.
#
# Suggested read order across the three families: cliques (obvious signature)
# -> chorded_cycles (isolates the landmark) -> split_regular3 (checks whether a
# real, general asymmetry effect survives with NO landmark at all).
#   sbatch scripts/r9_n46_split_regular3_seed2000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEED=2000
OUT_ROOT="runs/report9/n46_train"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families split_regular3 --arch roberta --readout similarity --lambda_lap 0 \
    --n_nodes 46 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n46_split_regular3_roberta_similarity_lam0_seed2000/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "training did not produce a checkpoint at $CKPT"
  exit 1
fi

FAIL=0
TAG="n46_split_regular3_seed2000"

echo "== mechanistic_asym_chains.py (sweep+logit, topology=split_regular3) =="
python mechanistic_asym_chains.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/mechanistic/${TAG}" --attn_splits 4 22 --topology split_regular3
[ $? -ne 0 ] && { echo "FAILED: mechanistic_asym_chains.py (sweep+logit, topology=split_regular3)"; FAIL=1; }

echo "== mechanistic_heatmaps.py (attention scores, topology=split_regular3) =="
python mechanistic_heatmaps.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/heatmaps/${TAG}" --splits 4 22 --topology split_regular3
[ $? -ne 0 ] && { echo "FAILED: mechanistic_heatmaps.py (attention scores, topology=split_regular3)"; FAIL=1; }

echo "== stagewise_diagnostics.py (layerwise cosine geometry, topology=split_regular3) =="
python stagewise_diagnostics.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/stagewise/${TAG}" --splits 4 22 --n_graphs 64 --topology split_regular3
[ $? -ne 0 ] && { echo "FAILED: stagewise_diagnostics.py (layerwise cosine geometry, topology=split_regular3)"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more eval steps FAILED -- see FAILED markers above"
  exit 1
fi
echo "done: $TAG"
