#!/bin/bash
#SBATCH --job-name=r9bthreecyc
#SBATCH --account=3352759
#SBATCH --partition=short_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX, Thread B follow-up -- does the split_cycles-only n=46 seed-1000 checkpoint just
# learn "find the tiny cycle, call everything else connected" (a degenerate shortcut that a
# 2-component test cannot distinguish from genuine cut behaviour, since pred_positive_rate=1.0
# from a=12 onward already -- see istruzioni.md, the split-sweep table)? Test on graphs with
# THREE separate cycles instead of two: if the model has only learned "small vs not-small", it
# should correctly call the tiny cycle disconnected from BOTH others (cut_12, cut_13 high) while
# wrongly calling the two non-tiny cycles connected to each other (cut_23 low) -- the exact
# generalisation of the split_chains-only Thread A.3 finding (istruzioni.md §16.13/16.14), now
# tested for cycles.
#
# New generator generate_multi_cycle_split_graph (data.py) + --topology cycle added to
# eval_threeway_splitchains.py and stagewise_threeway.py (both smoke-tested on a toy n=18
# checkpoint before touching real weights; no change to existing chain-topology behaviour,
# --topology defaults to "chain").
#
# Step 1: FULL behavioural sweep (every partition of 46 into 3 cycle sizes >=3, ~150 cells),
# now including pred_positive_rate per cell (also added to eval_threeway_splitchains.py this
# session -- the whole point of this test).
# Step 2: layerwise cosine geometry on two representative cells: (3,21,22) -- a tiny anchor
# cycle plus two comparably-sized "real" cycles, the direct cycle analogue of the chain
# candidate cells -- and (15,15,16) -- the fully-balanced, no-tiny-anchor regime, for contrast.
#
# Eval-only, CPU is enough.
#
#   sbatch scripts/r9_b_splitcycles_threeway_test.sbatch

mkdir -p out
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

CKPT="runs/report9/n46_train/n46_split_cycles_roberta_similarity_lam0_seed1000/last.pt"
TAG="n46_splitcycles_seed1000"

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT"
  exit 1
fi

FAIL=0

echo "== eval_threeway_splitchains.py --topology cycle (full behavioural sweep) =="
python eval_threeway_splitchains.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/threeway_splitchains/${TAG}" \
    --topology cycle --n_graphs 200
[ $? -ne 0 ] && { echo "FAILED: eval_threeway_splitchains.py"; FAIL=1; }

echo "== stagewise_threeway.py --topology cycle (layerwise cosine geometry, 2 cells) =="
python stagewise_threeway.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/stagewise_threeway/${TAG}" \
    --cells 3,21,22 15,15,16 --topology cycle --n_graphs 64
[ $? -ne 0 ] && { echo "FAILED: stagewise_threeway.py"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more eval steps FAILED -- see FAILED markers above"
  exit 1
fi
echo "done: $TAG (three-cycle test)"
