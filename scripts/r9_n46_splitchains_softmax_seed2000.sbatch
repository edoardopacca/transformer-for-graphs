#!/bin/bash
#SBATCH --job-name=r9smxn46b
#SBATCH --account=3352759
#SBATCH --partition=gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX -- second seed of the (corrected, n=46) softmax-attention condition. See
# scripts/r9_n46_splitchains_softmax_seed1000.sbatch for the full rationale and the note on
# why this replaces the earlier n=64 attempt. Identical setup, seed 2000 only.
#   sbatch scripts/r9_n46_splitchains_softmax_seed2000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEED=2000
OUT_ROOT="runs/report9/n46_train"
TAG="n46_splitchains_softmax_seed${SEED}"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families split_chains --arch roberta --readout similarity --attn_kind softmax --lambda_lap 0 \
    --n_nodes 46 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n46_split_chains_roberta_similarity_lam0_softmax_seed${SEED}/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "training did not produce a checkpoint at $CKPT"
  exit 1
fi

FAIL=0

echo "== eval_asym_chains.py (quick behavioural sweep) =="
python eval_asym_chains.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/asym_chains_n46/${TAG}" --n_graphs 300
[ $? -ne 0 ] && { echo "FAILED: eval_asym_chains.py"; FAIL=1; }

echo "== mechanistic_asym_chains.py (sweep + logit -> Table 1 / Figure 1) =="
python mechanistic_asym_chains.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/mechanistic/${TAG}" --attn_splits 4 23
[ $? -ne 0 ] && { echo "FAILED: mechanistic_asym_chains.py"; FAIL=1; }

echo "== mechanistic_heatmaps.py (attention scores/alpha -> Figure 3) =="
python mechanistic_heatmaps.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/heatmaps/${TAG}" --splits 4 23
[ $? -ne 0 ] && { echo "FAILED: mechanistic_heatmaps.py"; FAIL=1; }

echo "== stagewise_diagnostics.py (layerwise cosine geometry) =="
python stagewise_diagnostics.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/stagewise/${TAG}" --splits 4 23 --n_graphs 64
[ $? -ne 0 ] && { echo "FAILED: stagewise_diagnostics.py"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more eval steps FAILED -- see FAILED markers above"
  exit 1
fi
echo "done: $TAG"
