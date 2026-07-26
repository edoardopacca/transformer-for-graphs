#!/bin/bash
#SBATCH --job-name=r9L3s2
#SBATCH --account=3352759
#SBATCH --partition=gpuh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.out
#
# Report IX -- split_chains-only n=46, seed 2000, but n_layers=3 instead of 2 (the
# only change from scripts/r9_n46_splitchains_seed1000.sbatch besides the seed).
# Purpose: does a deeper model still learn the endpoint-completion signal, and if so,
# via the same mechanism (does it actually use the path endpoints)? Tested two ways:
# (a) own family (split_chains, in-distribution) and (b) OOD on 2-cycles (never seen
# in training) -- attention scores + behavioural sweep for both.
#
# mechanistic_heatmaps.py was generalized this session to save ALL layers (was
# hardcoded for exactly 2) -- verified on toy L=2/L=3 checkpoints, L=2 output is
# byte-identical to before. stagewise_diagnostics.py (layerwise cosine geometry) is NOT
# run here -- its MAIN_STAGES/SUBBLOCKS constants are hardcoded for L=2 and would
# mislabel H2 as final; needs its own generalization first (local/no-GPU follow-up,
# not urgent -- can be done anytime once this checkpoint exists).
#   sbatch scripts/r9_n46_splitchains_L3_seed2000.sbatch

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
    --families split_chains --arch roberta --readout similarity --lambda_lap 0 --n_layers 3 \
    --n_nodes 46 --p 0.05 \
    --train_steps 1000000 --batch_size 1000 --num_workers 16 --seed "$SEED"

CKPT="$OUT_ROOT/n46_split_chains_roberta_similarity_lam0_L3_seed2000/last.pt"
if [ ! -f "$CKPT" ]; then
  echo "training did not produce a checkpoint at $CKPT"
  exit 1
fi

FAIL=0
TAG="n46_splitchains_L3_seed2000"

echo "== eval_asym_chains.py (own family, quick behavioural sweep) =="
python eval_asym_chains.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/asym_chains_n46/${TAG}" --n_graphs 300
[ $? -ne 0 ] && { echo "FAILED: eval_asym_chains.py (own family, quick behavioural sweep)"; FAIL=1; }

echo "== mechanistic_asym_chains.py (own family: sweep+logit -> Table 1/Figure 1) =="
python mechanistic_asym_chains.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/mechanistic/${TAG}" --attn_splits 4 23
[ $? -ne 0 ] && { echo "FAILED: mechanistic_asym_chains.py (own family: sweep+logit -> Table 1/Figure 1)"; FAIL=1; }

echo "== mechanistic_heatmaps.py (own family: attention scores, all 3 layers) =="
python mechanistic_heatmaps.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/heatmaps/${TAG}" --splits 4 23
[ $? -ne 0 ] && { echo "FAILED: mechanistic_heatmaps.py (own family: attention scores, all 3 layers)"; FAIL=1; }

echo "== mechanistic_asym_chains.py (OOD test: sweep+logit, topology=cycle) =="
python mechanistic_asym_chains.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/mechanistic/${TAG}_oodcycle" --attn_splits 4 23 --topology cycle
[ $? -ne 0 ] && { echo "FAILED: mechanistic_asym_chains.py (OOD test: sweep+logit, topology=cycle)"; FAIL=1; }

echo "== mechanistic_heatmaps.py (OOD test: attention scores, topology=cycle) =="
python mechanistic_heatmaps.py --checkpoint "$CKPT" \
    --output_dir "runs/report9/heatmaps/${TAG}_oodcycle" --splits 4 23 --topology cycle
[ $? -ne 0 ] && { echo "FAILED: mechanistic_heatmaps.py (OOD test: attention scores, topology=cycle)"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo "one or more eval steps FAILED -- see FAILED markers above"
  exit 1
fi
echo "done: $TAG"
