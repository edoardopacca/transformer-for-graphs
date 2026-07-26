#!/bin/bash
#SBATCH --job-name=r9splchfix
#SBATCH --account=3352759
#SBATCH --partition=long_gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX, ablation on the split_chains-only condition (this report's condition (3),
# scripts/r9_n46_splitchains_seed1000.sbatch) -- IDENTICAL in every respect (same family
# split_chains, same n=46, same RoBERTa L=2/d512/single-head, same 1e6 steps/batch 1000,
# same seed 1000) EXCEPT the read-out has no learnable affine transform: --sim_fixed freezes
# scale=1/bias=0 for the whole training (model.py's new ModelConfig.sim_learnable=False), so
# the logit is exactly cos(h_i,h_j), never scale*cos+bias. Tests whether the model needs a
# learnable decision margin to build the endpoint-completion / Thread-A.3 three-component
# signature, or whether raw cosine in [-1,1] as the logit is enough.
#
# No new eval code needed: eval_families.py's load_model already reads scale/bias directly
# from the checkpoint (here 1.0/0.0, frozen), so every existing analysis script (mechanistic_
# asym_chains.py, mechanistic_heatmaps.py, stagewise_diagnostics.py, eval_threeway_
# splitchains.py, ...) works completely unmodified on this checkpoint.
#
# Testing: the exact same battery as the original split_chains-only run (behavioural sweep,
# Table-1/Figure-1 equivalent, attention scores, layerwise cosine geometry), PLUS the Thread-
# A.3 three-component sweep from §5.2 (all 176 three-way split-size combinations) -- the
# specific experiment this retrain is for. The follow-up mechanistic dive (attention scores /
# layerwise geometry on specific 3-component cells) is NOT run here -- run it as a separate
# step once these behavioural numbers are back, same as for the original checkpoint.
#
#   sbatch scripts/r9_n46_splitchains_simfixed_seed1000.sbatch

mkdir -p out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
source ~/.bashrc
conda activate graph_tf
cd $SLURM_SUBMIT_DIR

SEED=1000
OUT_ROOT="runs/report9/n46_train"
TAG="n46_splitchains_simfixed_seed${SEED}"

python experiments2/train_families_n20.py \
    --output_root "$OUT_ROOT" \
    --families split_chains --arch roberta --readout similarity --sim_fixed --lambda_lap 0 \
    --n_nodes 46 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n46_split_chains_roberta_similarityfixed_lam0_seed${SEED}/last.pt"
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

echo "== eval_threeway_splitchains.py (Thread A.3, sec 5.2: all 176 three-way size combos) =="
python eval_threeway_splitchains.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/threeway_splitchains/${TAG}" --n_graphs 200
[ $? -ne 0 ] && { echo "FAILED: eval_threeway_splitchains.py"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more eval steps FAILED -- see FAILED markers above"
  exit 1
fi
echo "done: $TAG"
