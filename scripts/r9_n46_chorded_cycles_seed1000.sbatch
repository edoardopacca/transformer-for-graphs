#!/bin/bash
#SBATCH --job-name=r9chorded1
#SBATCH --account=3352759
#SBATCH --partition=gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX controlled-distribution battery -- family 'chorded_cycles', n=46, seed 1000.
# Two disjoint CYCLES of asymmetric sizes (a, 46-a), each with ONE extra chord
# edge -- exactly 2 nodes per cycle get degree 3 (a landmark breaking rotational
# symmetry), every other node stays degree 2, neither cycle has an open
# endpoint. Isolates whether ANY symmetry-breaking landmark suffices, or
# whether path ENDPOINTS specifically are needed (the middle rung between plain
# cycles and chains).
#
# Suggested read order across the three families: cliques (obvious signature)
# -> chorded_cycles (isolates the landmark) -> split_regular3 (checks whether a
# real, general asymmetry effect survives with NO landmark at all).
#   sbatch scripts/r9_n46_chorded_cycles_seed1000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEED=1000
OUT_ROOT="runs/report9/n46_train"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families chorded_cycles --arch roberta --readout similarity --lambda_lap 0 \
    --n_nodes 46 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n46_chorded_cycles_roberta_similarity_lam0_seed1000/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "training did not produce a checkpoint at $CKPT"
  exit 1
fi

FAIL=0
TAG="n46_chorded_cycles_seed1000"

echo "== mechanistic_asym_chains.py (sweep+logit, topology=chorded_cycles) =="
python mechanistic_asym_chains.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/mechanistic/${TAG}" --attn_splits 4 23 --topology chorded_cycles
[ $? -ne 0 ] && { echo "FAILED: mechanistic_asym_chains.py (sweep+logit, topology=chorded_cycles)"; FAIL=1; }

echo "== mechanistic_heatmaps.py (attention scores, topology=chorded_cycles) =="
python mechanistic_heatmaps.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/heatmaps/${TAG}" --splits 4 23 --topology chorded_cycles
[ $? -ne 0 ] && { echo "FAILED: mechanistic_heatmaps.py (attention scores, topology=chorded_cycles)"; FAIL=1; }

echo "== stagewise_diagnostics.py (layerwise cosine geometry, topology=chorded_cycles) =="
python stagewise_diagnostics.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/stagewise/${TAG}" --splits 4 23 --n_graphs 64 --topology chorded_cycles
[ $? -ne 0 ] && { echo "FAILED: stagewise_diagnostics.py (layerwise cosine geometry, topology=chorded_cycles)"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more eval steps FAILED -- see FAILED markers above"
  exit 1
fi
echo "done: $TAG"
