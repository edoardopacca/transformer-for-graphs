#!/bin/bash
#SBATCH --job-name=r9smax1
#SBATCH --account=3352759
#SBATCH --partition=long_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX -- prof's question: does the completion signal depend on the project's
# normalized-ReLU attention specifically, or does it also appear with classical SOFTMAX
# attention? Same split_chains-only family/readout/step budget as the n=46 condition, but
# n=64 (this project's other canvas historically used for the completion-signal families,
# Report VI/VII) and --attn_kind softmax instead of the default normalized_relu. No code
# change needed on the eval side: every mechanistic script reads attn_kind off the loaded
# checkpoint's model_config (verified: alpha row-sums to 1 for a softmax smoke-test model).
#   sbatch scripts/r9_n64_splitchains_softmax_seed1000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEED=1000
OUT_ROOT="runs/report9/n64_train"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families split_chains --arch roberta --readout similarity --attn_kind softmax --lambda_lap 0 \
    --n_nodes 64 --p 0.03 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n64_split_chains_roberta_similarity_lam0_softmax_seed1000/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "training did not produce a checkpoint at $CKPT"
  exit 1
fi

FAIL=0
TAG="n64_splitchains_softmax_seed1000"

echo "== eval_asym_chains.py (quick behavioural sweep, full 1..32) =="
python eval_asym_chains.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/asym_chains_n64/${TAG}" --n_graphs 300
[ $? -ne 0 ] && { echo "FAILED: eval_asym_chains.py"; FAIL=1; }

echo "== mechanistic_asym_chains.py (sweep + logit -> Table 1 / Figure 1) =="
python mechanistic_asym_chains.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/mechanistic/${TAG}" \
    --attn_splits 1 4 7 8 10 11 12 13 14 15 16 17 18 19 20 21 32
[ $? -ne 0 ] && { echo "FAILED: mechanistic_asym_chains.py"; FAIL=1; }

echo "== mechanistic_heatmaps.py (attention scores/alpha -> Figure 3) =="
python mechanistic_heatmaps.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/heatmaps/${TAG}" --splits 4 12 18 32
[ $? -ne 0 ] && { echo "FAILED: mechanistic_heatmaps.py"; FAIL=1; }

echo "== stagewise_diagnostics.py (layerwise cosine geometry) =="
python stagewise_diagnostics.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/stagewise/${TAG}" --splits 4 12 18 32 --n_graphs 64
[ $? -ne 0 ] && { echo "FAILED: stagewise_diagnostics.py"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more eval steps FAILED -- see FAILED markers above"
  exit 1
fi
echo "done: $TAG"
