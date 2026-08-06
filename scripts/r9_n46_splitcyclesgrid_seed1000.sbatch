#!/bin/bash
#SBATCH --job-name=r9gridc1
#SBATCH --account=3352759
#SBATCH --partition=long_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX -- cycle analogue of the memorisation-vs-mechanism grid test
# (scripts/r9_n46_splitchains_grid_seed1000.sbatch). Trains split_cycles-only but with the
# short-cycle length drawn from a SPARSE, FIXED grid {3,5,7,9} instead of the continuous
# 3..n-3 of the usual split_cycles family, then evaluates on the FULL default sweep (topology
# cycle, min feasible split 3), which includes the INTERLEAVED short lengths {4,6,8,10} never
# seen (up to relabelling) in training. Same logic as the chain version: if the completion
# signal at those held-out splits looks the same as at the trained neighbours, that is
# evidence of a genuine mechanism rather than memorisation of the exact trained split sizes.
# n=46, similarity read-out, same architecture/step budget as every other Report IX training.
# One seed (exploratory, to see if it's worth a second before writing this up).
#
# eval_asym_chains.py is CHAIN-only, so (as for every other split_cycles condition in this
# report) it is skipped here; mechanistic_asym_chains.py --topology cycle gives the same
# exact/reach/cut-by-split sweep.
#   sbatch scripts/r9_n46_splitcyclesgrid_seed1000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEED=1000
OUT_ROOT="runs/report9/n46_train"
TAG="n46_splitcyclesgrid_seed${SEED}"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families split_cycles_grid --arch roberta --readout similarity --lambda_lap 0 \
    --n_nodes 46 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n46_split_cycles_grid_roberta_similarity_lam0_seed${SEED}/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "training did not produce a checkpoint at $CKPT"
  exit 1
fi

FAIL=0

echo "== mechanistic_asym_chains.py (sweep + logit -> Table 1 / Figure 1, topology=cycle, full 3..23 -- covers held-out 4/6/8/10) =="
python mechanistic_asym_chains.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/mechanistic/${TAG}" \
    --attn_splits 3 4 5 6 7 8 9 10 11 12 23 --topology cycle
[ $? -ne 0 ] && { echo "FAILED: mechanistic_asym_chains.py"; FAIL=1; }

echo "== mechanistic_heatmaps.py (attention scores/alpha, held-out vs trained vs balanced, topology=cycle) =="
python mechanistic_heatmaps.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/heatmaps/${TAG}" --splits 4 8 10 23 --topology cycle
[ $? -ne 0 ] && { echo "FAILED: mechanistic_heatmaps.py"; FAIL=1; }

echo "== stagewise_diagnostics.py (layerwise cosine geometry, topology=cycle) =="
python stagewise_diagnostics.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/stagewise/${TAG}" --splits 4 8 10 23 --n_graphs 64 --topology cycle
[ $? -ne 0 ] && { echo "FAILED: stagewise_diagnostics.py"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more eval steps FAILED -- see FAILED markers above"
  exit 1
fi
echo "done: $TAG"
