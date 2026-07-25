#!/bin/bash
#SBATCH --job-name=r9splcyc
#SBATCH --account=3352759
#SBATCH --partition=gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX -- train ONE seed at n=46 on a SINGLE named distribution of two-cycle splits: a
# random two-way split every sample, each segment closed into a cycle instead of left as an
# open path ("split_cycles", data.py::generate_split_cycles_graph, short_len drawn uniformly
# 3..n-3 per sample so each cycle has >=3 nodes). Same architecture/read-out/step budget as
# every other training in this project (RoBERTa L=2/d512/single-head, similarity read-out,
# online stream, 1e6 steps, batch 1000) -- the ONLY thing that changes from
# r9_n46_splitchains_seed1000.sbatch is closing the two chains into cycles.
#
# Purpose: Report VIII's two-cycles test only ever probed a model that NEVER saw a cycle in
# training (an out-of-distribution collapse to "everything connected"). This is the direct
# analogue of the chain check: train on cycles the same systematic way chains were trained,
# then read the same in-distribution battery, to see whether path endpoints are load-bearing
# for the completion signal or whether cycles -- trained on properly -- show an analogous
# signal of their own.
#
# eval_asym_chains.py is CHAIN-only (generate_split_chains_graph), so it is skipped here;
# mechanistic_asym_chains.py's own behavioural sweep (--topology cycle) already gives the
# same exact/reach/cut-by-split table, so nothing is lost.
#
#   sbatch scripts/r9_n46_splitcycles_seed1000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEED=1000
OUT_ROOT="runs/report9/n46_train"
TAG="n46_splitcycles_seed${SEED}"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families split_cycles --arch roberta --readout similarity --lambda_lap 0 \
    --n_nodes 46 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n46_split_cycles_roberta_similarity_lam0_seed${SEED}/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "training did not produce a checkpoint at $CKPT"
  exit 1
fi

FAIL=0

echo "== mechanistic_asym_chains.py (sweep + logit -> Table 1 / Figure 1, topology=cycle) =="
python mechanistic_asym_chains.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/mechanistic/${TAG}" --attn_splits 4 23 --topology cycle
[ $? -ne 0 ] && { echo "FAILED: mechanistic_asym_chains.py"; FAIL=1; }

echo "== mechanistic_heatmaps.py (attention scores/alpha -> Figure 3, topology=cycle) =="
python mechanistic_heatmaps.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/heatmaps/${TAG}" --splits 4 23 --topology cycle
[ $? -ne 0 ] && { echo "FAILED: mechanistic_heatmaps.py"; FAIL=1; }

echo "== stagewise_diagnostics.py (layerwise cosine geometry, topology=cycle) =="
python stagewise_diagnostics.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/stagewise/${TAG}" --splits 4 23 --n_graphs 64 --topology cycle
[ $? -ne 0 ] && { echo "FAILED: stagewise_diagnostics.py"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more eval steps FAILED -- see FAILED markers above"
  exit 1
fi
echo "done: $TAG"
