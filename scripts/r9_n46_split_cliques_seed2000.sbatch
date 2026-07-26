#!/bin/bash
#SBATCH --job-name=r9splitcl2
#SBATCH --account=3352759
#SBATCH --partition=gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX controlled-distribution battery -- family 'split_cliques', n=46, seed 2000.
# Two disjoint COMPLETE CLIQUES of asymmetric sizes (a, 46-a) -- Report IX's
# simplest control: every node's DEGREE directly reveals its component's size,
# an extremely blunt global signature. If imbalanced splits are solved easily
# (as with chains), it shows the model can exploit a trivial degree cue, not
# necessarily anything endpoint- or landmark-specific.
#
# Suggested read order across the three families: cliques (obvious signature)
# -> chorded_cycles (isolates the landmark) -> split_regular3 (checks whether a
# real, general asymmetry effect survives with NO landmark at all).
#   sbatch scripts/r9_n46_split_cliques_seed2000.sbatch

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
    --families split_cliques --arch roberta --readout similarity --lambda_lap 0 \
    --n_nodes 46 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n46_split_cliques_roberta_similarity_lam0_seed2000/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "training did not produce a checkpoint at $CKPT"
  exit 1
fi

FAIL=0
TAG="n46_split_cliques_seed2000"

echo "== mechanistic_asym_chains.py (sweep+logit, topology=split_cliques) =="
python mechanistic_asym_chains.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/mechanistic/${TAG}" --attn_splits 4 23 --topology split_cliques
[ $? -ne 0 ] && { echo "FAILED: mechanistic_asym_chains.py (sweep+logit, topology=split_cliques)"; FAIL=1; }

echo "== mechanistic_heatmaps.py (attention scores, topology=split_cliques) =="
python mechanistic_heatmaps.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/heatmaps/${TAG}" --splits 4 23 --topology split_cliques
[ $? -ne 0 ] && { echo "FAILED: mechanistic_heatmaps.py (attention scores, topology=split_cliques)"; FAIL=1; }

echo "== stagewise_diagnostics.py (layerwise cosine geometry, topology=split_cliques) =="
python stagewise_diagnostics.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/stagewise/${TAG}" --splits 4 23 --n_graphs 64 --topology split_cliques
[ $? -ne 0 ] && { echo "FAILED: stagewise_diagnostics.py (layerwise cosine geometry, topology=split_cliques)"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more eval steps FAILED -- see FAILED markers above"
  exit 1
fi
echo "done: $TAG"
